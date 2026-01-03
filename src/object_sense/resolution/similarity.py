"""Multi-signal similarity scoring with entity_nature weighting.

This module implements similarity computation between observations and entities.

Key principles (per user spec):
- Signal weights are renormalized among available signals
- Coverage scalar: sqrt(W_present) to penalize sparse evidence
- Low-evidence guardrail: if W_present < 0.25 or < 2 signals, mark as low_evidence
- One resolver with different weight profiles per entity_nature

See design_v2_corrections.md §5 and §6 for specifications.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from object_sense.models.entity import Entity
from object_sense.models.enums import EntityNature


@dataclass
class SignalResult:
    """Result of a single signal computation."""

    signal_name: str
    score: float
    weight: float
    available: bool = True


@dataclass
class SimilarityResult:
    """Result of multi-signal similarity computation."""

    score: float
    """Final weighted score (0-1, higher = more similar)."""

    signals: list[SignalResult]
    """Individual signal contributions."""

    coverage: float
    """Coverage scalar: sqrt(W_present). Lower = sparser evidence."""

    is_low_evidence: bool
    """True if W_present < 0.25 or < 2 signals available."""

    flags: list[str] = field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    """Warning flags: low_evidence, missing_embedding, etc."""


# Signal weight profiles per entity_nature (sum to 1.0)
# These determine which signals matter most for each entity type
SIGNAL_WEIGHTS: dict[EntityNature, dict[str, float]] = {
    EntityNature.INDIVIDUAL: {
        "image_embedding": 0.35,  # Visual re-ID
        "signature": 0.25,  # pHash, rosette patterns
        "location": 0.15,  # Locality prior
        "timestamp": 0.10,  # Temporal proximity
        "deterministic_id": 0.15,  # Hard IDs if present
    },
    EntityNature.CLASS: {
        "text_embedding": 0.40,  # Language/semantic agreement
        "facet_agreement": 0.30,  # Attribute consistency
        "name_match": 0.20,  # Exact/fuzzy name matching
        "deterministic_id": 0.10,  # Taxonomic IDs if present
    },
    EntityNature.GROUP: {
        "member_overlap": 0.35,  # Shared members
        "location": 0.25,  # Spatial clustering
        "temporal_proximity": 0.25,  # Time-based grouping
        "deterministic_id": 0.15,  # Group IDs if present
    },
    EntityNature.EVENT: {
        "timestamp": 0.30,  # Temporal clustering
        "participants": 0.30,  # Entity involvement
        "location": 0.20,  # Where it happened
        "duration": 0.10,  # Temporal extent
        "deterministic_id": 0.10,  # Event IDs if present
    },
}

# Minimum weight threshold for low-evidence flag
MIN_WEIGHT_THRESHOLD = 0.25

# Minimum number of signals for low-evidence flag
MIN_SIGNALS_THRESHOLD = 2


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Returns a value in [0, 1] where 1 is identical.
    """
    if len(a) != len(b):
        return 0.0

    a_np = np.array(a)
    b_np = np.array(b)

    norm_a = np.linalg.norm(a_np)
    norm_b = np.linalg.norm(b_np)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    # Cosine similarity: [-1, 1] -> [0, 1]
    sim = float(np.dot(a_np, b_np) / (norm_a * norm_b))
    return (sim + 1) / 2


class SimilarityScorer:
    """Computes multi-signal similarity between observations and entities.

    This is the core scoring engine for entity resolution. It:
    1. Extracts signals from observation and entity
    2. Applies entity_nature-specific weights
    3. Renormalizes for missing signals
    4. Applies coverage scalar
    5. Flags low-evidence cases

    Usage:
        scorer = SimilarityScorer()
        result = scorer.compute(
            observation_signals=ObservationSignals(...),
            entity=entity,
            entity_nature=EntityNature.INDIVIDUAL
        )
    """

    def __init__(self) -> None:
        """Initialize the scorer."""
        self._weights = SIGNAL_WEIGHTS

    def compute(
        self,
        *,
        observation_signals: ObservationSignals,
        entity: Entity,
        entity_nature: EntityNature,
    ) -> SimilarityResult:
        """Compute similarity between an observation and an entity.

        Args:
            observation_signals: Extracted signals from the observation.
            entity: The candidate entity to compare against.
            entity_nature: The nature of the entity (determines weight profile).

        Returns:
            SimilarityResult with score, signal breakdown, and flags.
        """
        weights = self._weights.get(entity_nature, self._weights[EntityNature.INDIVIDUAL])
        signals: list[SignalResult] = []
        flags: list[str] = []

        # Compute each signal
        for signal_name, weight in weights.items():
            score, available = self._compute_signal(
                signal_name,
                observation_signals,
                entity,
            )
            signals.append(SignalResult(
                signal_name=signal_name,
                score=score,
                weight=weight,
                available=available,
            ))

        # Renormalize weights for available signals
        available_signals = [s for s in signals if s.available]
        w_present = sum(s.weight for s in available_signals)

        if w_present == 0:
            # No signals available at all
            return SimilarityResult(
                score=0.0,
                signals=signals,
                coverage=0.0,
                is_low_evidence=True,
                flags=["no_signals_available"],
            )

        # Compute weighted score with renormalization
        weighted_sum = sum(
            (s.weight / w_present) * s.score
            for s in available_signals
        )

        # Coverage scalar: sqrt(W_present)
        coverage = math.sqrt(w_present)

        # Apply coverage penalty
        final_score = weighted_sum * coverage

        # Check low-evidence conditions
        is_low_evidence = (
            w_present < MIN_WEIGHT_THRESHOLD
            or len(available_signals) < MIN_SIGNALS_THRESHOLD
        )

        if is_low_evidence:
            flags.append("low_evidence")

        return SimilarityResult(
            score=final_score,
            signals=signals,
            coverage=coverage,
            is_low_evidence=is_low_evidence,
            flags=flags,
        )

    def _compute_signal(
        self,
        signal_name: str,
        obs: ObservationSignals,
        entity: Entity,
    ) -> tuple[float, bool]:
        """Compute a single signal score.

        Returns:
            Tuple of (score, available).
        """
        if signal_name == "image_embedding":
            return self._compute_image_embedding(obs, entity)
        elif signal_name == "text_embedding":
            return self._compute_text_embedding(obs, entity)
        elif signal_name == "signature":
            return self._compute_signature(obs, entity)
        elif signal_name == "facet_agreement":
            return self._compute_facet_agreement(obs, entity)
        elif signal_name == "name_match":
            return self._compute_name_match(obs, entity)
        elif signal_name == "location":
            return self._compute_location(obs, entity)
        elif signal_name == "timestamp":
            return self._compute_timestamp(obs, entity)
        elif signal_name == "temporal_proximity":
            return self._compute_temporal_proximity(obs, entity)
        elif signal_name == "member_overlap":
            return self._compute_member_overlap(obs, entity)
        elif signal_name == "participants":
            return self._compute_participants(obs, entity)
        elif signal_name == "duration":
            return self._compute_duration(obs, entity)
        elif signal_name == "deterministic_id":
            # Deterministic IDs are handled separately (short-circuit)
            return 0.0, False
        else:
            # Unknown signal
            return 0.0, False

    def _compute_image_embedding(
        self,
        obs: ObservationSignals,
        entity: Entity,
    ) -> tuple[float, bool]:
        """Compute image embedding similarity."""
        if obs.image_embedding is None or entity.prototype_image_embedding is None:
            return 0.0, False

        score = cosine_similarity(obs.image_embedding, list(entity.prototype_image_embedding))
        return score, True

    def _compute_text_embedding(
        self,
        obs: ObservationSignals,
        entity: Entity,
    ) -> tuple[float, bool]:
        """Compute text embedding similarity."""
        if obs.text_embedding is None or entity.prototype_text_embedding is None:
            return 0.0, False

        score = cosine_similarity(obs.text_embedding, list(entity.prototype_text_embedding))
        return score, True

    def _compute_signature(
        self,
        obs: ObservationSignals,
        entity: Entity,
    ) -> tuple[float, bool]:
        """Compute signature similarity (pHash, etc.)."""
        # v0: Not implemented yet
        # Future: Compare perceptual hashes, etc.
        return 0.0, False

    def _compute_facet_agreement(
        self,
        obs: ObservationSignals,
        entity: Entity,
    ) -> tuple[float, bool]:
        """Compute facet agreement (attribute overlap)."""
        if not obs.facets or not entity.slots:
            return 0.0, False

        # Simple Jaccard-style overlap
        obs_keys = set(obs.facets.keys())
        entity_keys = set(entity.slots.keys())

        if not obs_keys and not entity_keys:
            return 0.0, False

        # Keys that appear in both
        common_keys = obs_keys & entity_keys
        if not common_keys:
            return 0.0, True  # Signal available but no overlap

        # Check value agreement for common keys
        agreements = 0
        for key in common_keys:
            if self._values_match(obs.facets.get(key), entity.slots.get(key)):
                agreements += 1

        score = agreements / len(common_keys) if common_keys else 0.0
        return score, True

    def _values_match(self, a: object, b: object) -> bool:
        """Check if two slot values match (simple equality for v0)."""
        if a == b:
            return True

        # Handle reference values
        if isinstance(a, dict) and isinstance(b, dict):
            a_dict = cast(dict[str, Any], a)
            b_dict = cast(dict[str, Any], b)
            if "ref_entity_id" in a_dict and "ref_entity_id" in b_dict:
                return bool(a_dict["ref_entity_id"] == b_dict["ref_entity_id"])
            if "value" in a_dict and "value" in b_dict:
                return bool(a_dict["value"] == b_dict["value"])

        return False

    def _compute_name_match(
        self,
        obs: ObservationSignals,
        entity: Entity,
    ) -> tuple[float, bool]:
        """Compute name matching score."""
        if not obs.suggested_name or not entity.name:
            return 0.0, False

        # Exact match
        obs_name = obs.suggested_name.lower().strip()
        entity_name = entity.name.lower().strip()

        if obs_name == entity_name:
            return 1.0, True

        # Substring match
        if obs_name in entity_name or entity_name in obs_name:
            return 0.7, True

        # Fuzzy match could be added here
        return 0.0, True

    def _compute_location(
        self,
        obs: ObservationSignals,
        entity: Entity,
    ) -> tuple[float, bool]:
        """Compute location similarity."""
        # v0: Not implemented - requires GPS slots
        return 0.0, False

    def _compute_timestamp(
        self,
        obs: ObservationSignals,
        entity: Entity,
    ) -> tuple[float, bool]:
        """Compute timestamp similarity."""
        # v0: Not implemented - requires temporal slots
        return 0.0, False

    def _compute_temporal_proximity(
        self,
        obs: ObservationSignals,
        entity: Entity,
    ) -> tuple[float, bool]:
        """Compute temporal proximity for groups."""
        # v0: Not implemented
        return 0.0, False

    def _compute_member_overlap(
        self,
        obs: ObservationSignals,
        entity: Entity,
    ) -> tuple[float, bool]:
        """Compute member overlap for groups."""
        # v0: Not implemented
        return 0.0, False

    def _compute_participants(
        self,
        obs: ObservationSignals,
        entity: Entity,
    ) -> tuple[float, bool]:
        """Compute participant overlap for events."""
        # v0: Not implemented
        return 0.0, False

    def _compute_duration(
        self,
        obs: ObservationSignals,
        entity: Entity,
    ) -> tuple[float, bool]:
        """Compute duration similarity for events."""
        # v0: Not implemented
        return 0.0, False


@dataclass
class ObservationSignals:
    """Extracted signals from an observation for similarity computation.

    These are extracted from Observation + Signature + facets.
    """

    image_embedding: list[float] | None = None
    """CLIP image embedding (768-dim)."""

    text_embedding: list[float] | None = None
    """BGE text embedding (1024-dim)."""

    clip_text_embedding: list[float] | None = None
    """CLIP text embedding (768-dim) for cross-modal."""

    facets: dict[str, Any] | None = None
    """Extracted facets from Step 4A."""

    suggested_name: str | None = None
    """Name suggested by LLM for name matching."""

    hash_value: str | None = None
    """Content hash (SHA256, pHash, etc.)."""

    source_id: str | None = None
    """Source identifier for locality."""

    # Future signals
    gps_coords: tuple[float, float] | None = None
    """GPS coordinates (lat, lon) for location."""

    timestamp: float | None = None
    """Unix timestamp for temporal signals."""

    @classmethod
    def from_observation_and_signatures(
        cls,
        observation: Any,  # Observation model
        signatures: list[Any],  # List of Signature models
        entity_hypothesis: Any | None = None,  # EntityHypothesis
    ) -> ObservationSignals:
        """Extract signals from observation and its signatures.

        Args:
            observation: The Observation model instance.
            signatures: List of Signature model instances.
            entity_hypothesis: Optional EntityHypothesis with suggested name.

        Returns:
            ObservationSignals ready for similarity computation.
        """
        image_embedding = None
        text_embedding = None
        clip_text_embedding = None
        hash_value = None

        # Extract embeddings from signatures
        for sig in signatures:
            if sig.image_embedding is not None and image_embedding is None:
                image_embedding = list(sig.image_embedding)
            if sig.text_embedding is not None and text_embedding is None:
                text_embedding = list(sig.text_embedding)
            if sig.clip_text_embedding is not None and clip_text_embedding is None:
                clip_text_embedding = list(sig.clip_text_embedding)
            if sig.hash_value and hash_value is None:
                hash_value = sig.hash_value

        return cls(
            image_embedding=image_embedding,
            text_embedding=text_embedding,
            clip_text_embedding=clip_text_embedding,
            facets=observation.facets if hasattr(observation, "facets") else None,
            suggested_name=(
                entity_hypothesis.suggested_name
                if entity_hypothesis and hasattr(entity_hypothesis, "suggested_name")
                else None
            ),
            hash_value=hash_value,
            source_id=observation.source_id if hasattr(observation, "source_id") else None,
        )
