"""Multi-signal similarity scoring with entity_nature weighting.

This module implements similarity computation between observations and entities.

Key principles:
- Signal weights are renormalized among available signals (ignore missing)
- For sparse signals (1-2 available), return renormalized score directly
- Coverage penalty only applies when 3+ signals expected but few available
- Low-evidence flag for transparency, but doesn't crush score
- Primary signal guardrails (e.g., T_img_min for INDIVIDUAL image matches)

Signal Regimes:
- image-only: Common for wildlife photos. Score ≈ cosine(image_embedding).
- image+facets: Better evidence. Score combines visual + attribute agreement.
- text+facets: For CLASS entities. Score combines semantic + attribute.

Contextual Signals (timestamp, GPS):
- For INDIVIDUAL entities, timestamp/GPS are WEAK PRIORS only
- They can NARROW candidate pool (locality prior) but NEVER gate merges
- Visual similarity MUST pass hard thresholds first (T_img_min, T_link)
- Context signals can only:
  a) Nudge posterior score (small boost if spatiotemporally close)
  b) Create pending/review links (never auto-merge on context alone)
- GPS is NOT a deterministic ID - it has inherent imprecision (~5-10m)
  and causes false splits when treated as hard identity

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
    """True if < 2 signals available. Informational, doesn't crush score."""

    flags: list[str] = field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    """Warning flags: low_evidence, sparse_signal, etc."""

    # Primary signal scores for guardrail checks
    image_similarity: float | None = None
    """Raw cosine similarity for image embedding (before weighting)."""

    text_similarity: float | None = None
    """Raw cosine similarity for text embedding (before weighting)."""

    @property
    def signal_scores(self) -> dict[str, float]:
        """Get available signal scores as a dict."""
        return {s.signal_name: s.score for s in self.signals if s.available}


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


def haversine_km(
    coord1: tuple[float, float],
    coord2: tuple[float, float],
) -> float:
    """Compute haversine distance between two GPS coordinates in kilometers.

    Args:
        coord1: (latitude, longitude) in degrees
        coord2: (latitude, longitude) in degrees

    Returns:
        Distance in kilometers
    """
    R = 6371.0  # Earth's radius in km

    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))

    return R * c


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

        Scoring approach:
        - Renormalize weights among AVAILABLE signals only
        - For sparse signals (1-2), return renormalized weighted sum directly
        - Coverage penalty only for rich-signal cases with missing data
        - Always track raw primary signal scores for guardrail checks

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

        # Track primary signal scores for guardrails
        image_similarity: float | None = None
        text_similarity: float | None = None

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

            # Capture primary signal scores for guardrails
            if signal_name == "image_embedding" and available:
                image_similarity = score
            elif signal_name == "text_embedding" and available:
                text_similarity = score

        # Renormalize weights for available signals
        available_signals = [s for s in signals if s.available]
        n_available = len(available_signals)
        w_present = sum(s.weight for s in available_signals)

        if w_present == 0:
            # No signals available at all
            return SimilarityResult(
                score=0.0,
                signals=signals,
                coverage=0.0,
                is_low_evidence=True,
                flags=["no_signals_available"],
                image_similarity=image_similarity,
                text_similarity=text_similarity,
            )

        # Compute weighted score with renormalization
        # Key fix: renormalized weights sum to 1.0, so we get the true weighted average
        weighted_sum = sum(
            (s.weight / w_present) * s.score
            for s in available_signals
        )

        # Coverage handling:
        # - For sparse signals (1-2), use score directly (no penalty)
        # - For rich evidence (3+), apply mild coverage adjustment
        coverage = math.sqrt(w_present)

        if n_available <= 2:
            # Sparse signal regime: trust the available signals
            # Don't penalize for missing signals that aren't available
            final_score = weighted_sum
            if n_available == 1:
                flags.append("single_signal")
            else:
                flags.append("sparse_signals")
        else:
            # Rich signal regime: mild coverage adjustment
            # Only penalize if we have 3+ signals but low total weight
            if w_present < 0.5:
                final_score = weighted_sum * (0.7 + 0.3 * coverage)
            else:
                final_score = weighted_sum

        # Low-evidence is informational only (doesn't crush score)
        is_low_evidence = n_available < MIN_SIGNALS_THRESHOLD

        if is_low_evidence:
            flags.append("low_evidence")

        return SimilarityResult(
            score=final_score,
            signals=signals,
            coverage=coverage,
            is_low_evidence=is_low_evidence,
            flags=flags,
            image_similarity=image_similarity,
            text_similarity=text_similarity,
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
        """Compute image embedding similarity.

        Supports cross-modal matching: if obs.image_embedding is unavailable,
        uses obs.clip_text_embedding which is in the same 768-dim CLIP space.
        This enables text observations to match against image prototypes.
        """
        if entity.prototype_image_embedding is None:
            return 0.0, False

        # Prefer image embedding, fall back to CLIP text for cross-modal
        query_embedding = obs.image_embedding
        if query_embedding is None:
            query_embedding = obs.clip_text_embedding

        if query_embedding is None:
            return 0.0, False

        score = cosine_similarity(query_embedding, list(entity.prototype_image_embedding))
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
        """Compute location similarity using GPS coordinates.

        Uses haversine distance. Score decays with distance:
        - 0 km: 1.0
        - 5 km: ~0.7
        - 20 km: ~0.3
        - 50+ km: ~0.0
        """
        if obs.gps_coords is None:
            return 0.0, False

        # Try to get entity GPS from slots
        entity_gps = self._get_entity_gps(entity)
        if entity_gps is None:
            return 0.0, False

        dist_km = haversine_km(obs.gps_coords, entity_gps)

        # Exponential decay: score = exp(-dist / scale)
        # scale=10 means ~0.37 at 10km, ~0.14 at 20km
        score = math.exp(-dist_km / 10.0)
        return score, True

    def _get_entity_gps(self, entity: Entity) -> tuple[float, float] | None:
        """Extract GPS coordinates from entity slots."""
        if not entity.slots:
            return None

        lat = entity.slots.get("latitude") or entity.slots.get("gps_latitude")
        lon = entity.slots.get("longitude") or entity.slots.get("gps_longitude")

        if lat is None or lon is None:
            return None

        try:
            # Handle slot format: could be {"value": "..."} or direct value
            if isinstance(lat, dict):
                lat = lat.get("value")
            if isinstance(lon, dict):
                lon = lon.get("value")
            return (float(lat), float(lon))
        except (ValueError, TypeError):
            return None

    def _compute_timestamp(
        self,
        obs: ObservationSignals,
        entity: Entity,
    ) -> tuple[float, bool]:
        """Compute timestamp similarity.

        Score decays with time difference:
        - 0 hours: 1.0
        - 2 hours: ~0.7
        - 12 hours: ~0.3
        - 48+ hours: ~0.0
        """
        if obs.timestamp is None:
            return 0.0, False

        # Try to get entity timestamp from slots
        entity_ts = self._get_entity_timestamp(entity)
        if entity_ts is None:
            return 0.0, False

        delta_hours = abs(obs.timestamp - entity_ts) / 3600.0

        # Exponential decay: score = exp(-delta / scale)
        # scale=6 means ~0.37 at 6h, ~0.14 at 12h
        score = math.exp(-delta_hours / 6.0)
        return score, True

    def _get_entity_timestamp(self, entity: Entity) -> float | None:
        """Extract timestamp from entity slots."""
        if not entity.slots:
            return None

        ts = (
            entity.slots.get("timestamp")
            or entity.slots.get("datetime")
            or entity.slots.get("last_seen")
        )

        if ts is None:
            return None

        try:
            # Handle slot format
            if isinstance(ts, dict):
                ts = ts.get("value")

            if isinstance(ts, (int, float)):
                return float(ts)
            elif isinstance(ts, str):
                from datetime import datetime
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                return dt.timestamp()
        except (ValueError, TypeError):
            pass

        return None

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

        # Extract GPS and timestamp from observation facets
        gps_coords: tuple[float, float] | None = None
        timestamp: float | None = None
        facets = observation.facets if hasattr(observation, "facets") else None

        if facets:
            # Try to extract GPS coordinates
            lat = facets.get("latitude") or facets.get("gps_latitude")
            lon = facets.get("longitude") or facets.get("gps_longitude")
            if lat is not None and lon is not None:
                try:
                    gps_coords = (float(lat), float(lon))
                except (ValueError, TypeError):
                    pass

            # Try to extract timestamp
            ts = facets.get("timestamp") or facets.get("datetime") or facets.get("capture_time")
            if ts is not None:
                try:
                    if isinstance(ts, (int, float)):
                        timestamp = float(ts)
                    elif isinstance(ts, str):
                        # Try parsing ISO format
                        from datetime import datetime
                        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        timestamp = dt.timestamp()
                except (ValueError, TypeError):
                    pass

        return cls(
            image_embedding=image_embedding,
            text_embedding=text_embedding,
            clip_text_embedding=clip_text_embedding,
            facets=facets,
            suggested_name=(
                entity_hypothesis.suggested_name
                if entity_hypothesis and hasattr(entity_hypothesis, "suggested_name")
                else None
            ),
            hash_value=hash_value,
            source_id=observation.source_id if hasattr(observation, "source_id") else None,
            gps_coords=gps_coords,
            timestamp=timestamp,
        )
