"""Main entity resolution algorithm.

This module implements the core Step 5 algorithm from design_v2_corrections.md.

Algorithm overview:
1. Deterministic ID resolution (HIGHEST PRIORITY)
   - IDs handled per-seed, not observation-level
   - If match: hard link (posterior=1.0, status='hard')
   - If NOT found: CREATE entity anchored to ID, then hard link
   - If conflict: create identity_conflicts row

2. For each entity_seed without deterministic ID:
   a. Get candidate pool (ANN over entity prototypes)
   b. Compute multi-signal similarity (weights per entity_nature)
   c. Apply decision thresholds (T_link, T_new, T_margin)
   d. Create links or proto-entities

3. Multi-seed consistency pass

4. Prototype update (running average)

5. Return links for persistence

Utility functions:
- resolve_canonical_entity: Follow canonical_entity_id chain with path compression
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from object_sense.config import settings
from object_sense.models.entity import Entity
from object_sense.models.entity_evolution import EntityEvolution
from object_sense.models.enums import (
    EntityEvolutionKind,
    EntityNature,
    EntityStatus,
    LinkRole,
    LinkStatus,
    Medium,
)
from object_sense.models.identity_conflict import IdentityConflict
from object_sense.models.observation import Observation, ObservationEntityLink
from object_sense.resolution.candidate_pool import CandidatePoolService, EntityCandidate
from object_sense.resolution.similarity import SimilarityResult
from object_sense.resolution.reconciliation import (
    PendingLink,
    deduplicate_links,
    reconcile_multi_seed_links,
)
from object_sense.resolution.similarity import ObservationSignals, SimilarityScorer
from object_sense.utils.slots import normalize_and_record_slot_warnings

if TYPE_CHECKING:
    from object_sense.inference.schemas import DeterministicId, EntityHypothesis
    from object_sense.models.signature import Signature


logger = logging.getLogger(__name__)

# Algorithm version for audit trail
ALGORITHM_VERSION = "resolver-v1.0"


def _slots_to_dict(seed: EntityHypothesis) -> dict[str, Any]:
    """Convert EntityHypothesis slots to a dictionary."""
    result: dict[str, Any] = {}
    for slot in seed.slots:
        if slot.is_reference:
            result[slot.name] = {"ref_name": slot.value, "type": "reference"}
        else:
            result[slot.name] = {"value": slot.value}
    return result


@dataclass
class ResolutionResult:
    """Result of entity resolution for an observation."""

    links: list[ObservationEntityLink]
    """Links created between observation and entities."""

    entities_created: list[Entity]
    """New proto-entities created."""

    conflicts_created: list[IdentityConflict]
    """Identity conflicts flagged for review."""

    evolutions: list[EntityEvolution]
    """Entity evolution records for audit trail."""

    flags: list[str] = field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    """Resolution-level flags and warnings."""


@dataclass
class ResolutionContext:
    """Context for resolving a single observation."""

    observation: Observation
    signatures: list[Signature]
    entity_seeds: list[EntityHypothesis]
    observation_signals: ObservationSignals


class EntityResolver:
    """Main entity resolution service.

    Usage:
        async with AsyncSession(engine) as session:
            resolver = EntityResolver(session)
            result = await resolver.resolve(
                observation=observation,
                signatures=signatures,
                entity_seeds=entity_seeds,
            )
    """

    def __init__(
        self,
        session: AsyncSession,
        *,
        t_link: float | None = None,
        t_new: float | None = None,
        t_margin: float | None = None,
        t_img_min: float | None = None,
        t_margin_img: float | None = None,
        top_k: int = 200,
    ) -> None:
        """Initialize the resolver.

        Args:
            session: Database session.
            t_link: High confidence threshold (default from config).
            t_new: Low confidence threshold (default from config).
            t_margin: Margin for review flag (default from config).
            t_img_min: Minimum image similarity for INDIVIDUAL merges.
            t_margin_img: Margin threshold for image-only decisions.
            top_k: Number of ANN candidates to retrieve.
        """
        self._session = session
        self._t_link = t_link or settings.entity_resolution_t_link
        self._t_new = t_new or settings.entity_resolution_t_new
        self._t_margin = t_margin or settings.entity_resolution_t_margin
        self._t_img_min = t_img_min or settings.entity_resolution_t_img_min
        self._t_margin_img = t_margin_img or settings.entity_resolution_t_margin_img
        self._top_k = top_k

        self._pool = CandidatePoolService(session, default_top_k=top_k)
        self._scorer = SimilarityScorer()

    async def resolve(
        self,
        *,
        observation: Observation,
        signatures: list[Signature],
        entity_seeds: list[EntityHypothesis],
    ) -> ResolutionResult:
        """Resolve entities for an observation.

        This is the main entry point implementing the Step 5 algorithm.

        Args:
            observation: The observation to resolve.
            signatures: Extracted signatures (embeddings, hashes).
            entity_seeds: Entity hypotheses from Step 4.

        Returns:
            ResolutionResult with links, entities, and audit info.
        """
        # Build observation signals for similarity
        obs_signals = ObservationSignals.from_observation_and_signatures(
            observation, signatures
        )

        ctx = ResolutionContext(
            observation=observation,
            signatures=signatures,
            entity_seeds=entity_seeds,
            observation_signals=obs_signals,
        )

        # Collect results
        pending_links: list[PendingLink] = []
        entities_created: list[Entity] = []
        conflicts: list[IdentityConflict] = []
        evolutions: list[EntityEvolution] = []
        result_flags: list[str] = []

        # INVARIANT (bead object-sense-bk9): Observation-level deterministic IDs
        # anchor entities REGARDLESS of what's in entity_seeds.
        # This ensures same SKU in different files → same entity.
        obs_det_ids_result = await self._resolve_observation_deterministic_ids(ctx)
        pending_links.extend(obs_det_ids_result.links)
        entities_created.extend(obs_det_ids_result.entities)
        conflicts.extend(obs_det_ids_result.conflicts)
        evolutions.extend(obs_det_ids_result.evolutions)
        if obs_det_ids_result.links:
            result_flags.append(f"obs_det_ids_resolved:{len(obs_det_ids_result.links)}")

        # Process each entity seed (for additional entities like class/group/event)
        for seed_idx, seed in enumerate(entity_seeds):
            seed_result = await self._resolve_seed(ctx, seed, seed_idx)

            pending_links.extend(seed_result.links)
            entities_created.extend(seed_result.entities)
            conflicts.extend(seed_result.conflicts)
            evolutions.extend(seed_result.evolutions)

        # Cross-modal EVENT alignment (IMAGE observations only)
        # Link image observations to existing text-defined EVENTs using
        # temporal/spatial/species matching. This enables cross-modal coherence
        # without inventing individuals from text.
        if observation.medium == Medium.IMAGE:
            event_links = await self._align_to_events(ctx, pending_links)
            pending_links.extend(event_links)
            if event_links:
                result_flags.append(f"event_aligned:{len(event_links)}")

        # Multi-seed consistency pass
        reconciled = reconcile_multi_seed_links(pending_links)
        if reconciled.conflicts_detected > 0:
            result_flags.append(f"multi_seed_conflicts:{reconciled.conflicts_detected}")
            logger.debug(
                "Multi-seed reconciliation: %d conflicts, %d flags added",
                reconciled.conflicts_detected,
                reconciled.flags_added,
            )

        # Deduplicate final links
        final_links = deduplicate_links(reconciled.links)

        # Convert pending links to ObservationEntityLink
        # Always resolve to canonical entity to handle race conditions where
        # a merge happens between candidate retrieval and link write.
        obs_entity_links: list[ObservationEntityLink] = []
        for link in final_links:
            # Resolve to canonical entity (with path compression)
            canonical_entity = await resolve_canonical_entity(
                self._session, link.entity_id, compress_path=True
            )
            canonical_id = canonical_entity.entity_id

            oel = ObservationEntityLink(
                observation_id=observation.observation_id,
                entity_id=canonical_id,
                posterior=link.posterior,
                status=LinkStatus(link.status),
                role=link.role,
                flags=link.flags,
            )
            obs_entity_links.append(oel)
            self._session.add(oel)

            # Update prototype on the canonical entity
            if self._should_update_prototype(link):
                await self._update_prototype(canonical_entity, ctx.observation_signals, link)

            # Create evolution record for link (to canonical entity)
            evolution = EntityEvolution(
                evolution_id=uuid4(),
                entity_id=canonical_id,
                kind=EntityEvolutionKind.LINK,
                observation_id=observation.observation_id,
                algorithm_version=ALGORITHM_VERSION,
                match_score=link.posterior,
                threshold_used=self._t_link,
                signals_used={},  # Could capture signal breakdown
                reasoning=f"Linked via {link.status} with posterior {link.posterior:.3f}",
            )
            evolutions.append(evolution)
            self._session.add(evolution)

        await self._session.flush()

        return ResolutionResult(
            links=obs_entity_links,
            entities_created=entities_created,
            conflicts_created=conflicts,
            evolutions=evolutions,
            flags=result_flags,
        )

    async def _resolve_observation_deterministic_ids(
        self,
        ctx: ResolutionContext,
    ) -> _SeedResolutionResult:
        """Resolve entities from observation-level deterministic IDs.

        INVARIANT (bead object-sense-bk9): Observation.deterministic_ids anchor
        entities REGARDLESS of entity_seeds. This ensures:
        - Same SKU in different files → same entity
        - Entity deduplication doesn't depend on LLM including IDs in seeds

        Args:
            ctx: Resolution context with observation.

        Returns:
            Results with links/entities for each strong deterministic ID.
        """
        from object_sense.inference.schemas import DeterministicId

        links: list[PendingLink] = []
        entities: list[Entity] = []
        conflicts: list[IdentityConflict] = []
        evolutions: list[EntityEvolution] = []

        obs_det_ids = ctx.observation.deterministic_ids or []

        for det_id_dict in obs_det_ids:
            # INVARIANT: All IDs should have namespaces after normalization
            assert det_id_dict.get("id_namespace"), (
                f"Observation deterministic ID missing namespace: {det_id_dict}"
            )

            # Convert dict to DeterministicId
            det_id = DeterministicId(
                id_type=det_id_dict["id_type"],
                id_value=det_id_dict["id_value"],
                id_namespace=det_id_dict["id_namespace"],
                strength="strong",  # Observation-level IDs are strong
            )

            # Lookup existing entity by deterministic ID
            lookup_result = await self._pool.lookup_by_deterministic_id(det_id)

            if lookup_result.entity:
                # Found existing entity - hard link
                link = PendingLink(
                    entity_id=lookup_result.entity.entity_id,
                    posterior=1.0,
                    status=LinkStatus.HARD,
                    role=LinkRole.SUBJECT,  # Default role for observation-level IDs
                    seed_index=-1,  # Not from a seed
                    facets={},
                    flags=["from_observation_det_id"],
                )
                links.append(link)
                logger.debug(
                    "Linked observation %s to entity %s via det_id %s:%s",
                    ctx.observation.observation_id,
                    lookup_result.entity.entity_id,
                    det_id.id_type,
                    det_id.id_value,
                )
            else:
                # ID not found - create new entity anchored to this ID
                entity = await self._pool.create_entity_for_deterministic_id(
                    det_id,
                    entity_nature=EntityNature.INDIVIDUAL,
                    name=f"{det_id.id_type}:{det_id.id_value}",
                    slots={},
                )
                entities.append(entity)

                link = PendingLink(
                    entity_id=entity.entity_id,
                    posterior=1.0,
                    status=LinkStatus.HARD,
                    role=LinkRole.SUBJECT,
                    seed_index=-1,
                    facets={},
                    flags=["from_observation_det_id", "entity_created"],
                )
                links.append(link)
                logger.debug(
                    "Created entity %s for observation %s via det_id %s:%s",
                    entity.entity_id,
                    ctx.observation.observation_id,
                    det_id.id_type,
                    det_id.id_value,
                )

        return _SeedResolutionResult(
            links=links,
            entities=entities,
            conflicts=conflicts,
            evolutions=evolutions,
        )

    async def _resolve_seed(
        self,
        ctx: ResolutionContext,
        seed: EntityHypothesis,
        seed_idx: int,
    ) -> _SeedResolutionResult:
        """Resolve a single entity seed.

        Args:
            ctx: Resolution context.
            seed: The entity hypothesis to resolve.
            seed_idx: Index of this seed (for tracking).

        Returns:
            Results for this seed.
        """
        links: list[PendingLink] = []
        entities: list[Entity] = []
        conflicts: list[IdentityConflict] = []
        evolutions: list[EntityEvolution] = []

        # Get entity nature from seed
        entity_nature = EntityNature(seed.entity_nature)

        # 1. Check for deterministic IDs first (highest priority)
        if seed.deterministic_ids:
            for det_id in seed.deterministic_ids:
                det_result = await self._resolve_deterministic_id(
                    ctx, seed, det_id, entity_nature, seed_idx
                )
                if det_result.link:
                    links.append(det_result.link)
                if det_result.entity:
                    entities.append(det_result.entity)
                if det_result.conflict:
                    conflicts.append(det_result.conflict)
                if det_result.evolution:
                    evolutions.append(det_result.evolution)

            # If we resolved via deterministic ID, skip similarity search
            if links:
                return _SeedResolutionResult(
                    links=links,
                    entities=entities,
                    conflicts=conflicts,
                    evolutions=evolutions,
                )

        # 2. Fall back to similarity-based resolution
        similarity_result = await self._resolve_by_similarity(
            ctx, seed, entity_nature, seed_idx
        )
        links.extend(similarity_result.links)
        entities.extend(similarity_result.entities)
        evolutions.extend(similarity_result.evolutions)

        return _SeedResolutionResult(
            links=links,
            entities=entities,
            conflicts=conflicts,
            evolutions=evolutions,
        )

    async def _resolve_deterministic_id(
        self,
        ctx: ResolutionContext,
        seed: EntityHypothesis,
        det_id: DeterministicId,
        entity_nature: EntityNature,
        seed_idx: int,
    ) -> _DeterministicIdResult:
        """Resolve via deterministic ID.

        Args:
            ctx: Resolution context.
            seed: Entity hypothesis.
            det_id: The deterministic ID to resolve.
            entity_nature: Nature of the entity.
            seed_idx: Seed index.

        Returns:
            Result with link, entity, or conflict.
        """
        # Lookup existing entity by deterministic ID
        lookup_result = await self._pool.lookup_by_deterministic_id(det_id)

        if lookup_result.entity:
            # Found existing entity - hard link
            link = PendingLink(
                entity_id=lookup_result.entity.entity_id,
                posterior=1.0,
                status=LinkStatus.HARD,
                role=self._infer_role(seed),
                seed_index=seed_idx,
                facets=_slots_to_dict(seed),
            )
            return _DeterministicIdResult(link=link)

        # ID not found - create new entity anchored to this ID
        entity = await self._pool.create_entity_for_deterministic_id(
            det_id,
            entity_nature=entity_nature,
            name=seed.suggested_name,
            slots=_slots_to_dict(seed),
        )

        link = PendingLink(
            entity_id=entity.entity_id,
            posterior=1.0,
            status=LinkStatus.HARD,
            role=self._infer_role(seed),
            seed_index=seed_idx,
        )

        # Create evolution record
        evolution = EntityEvolution(
            evolution_id=uuid4(),
            entity_id=entity.entity_id,
            kind=EntityEvolutionKind.LINK,
            observation_id=ctx.observation.observation_id,
            algorithm_version=ALGORITHM_VERSION,
            match_score=1.0,
            threshold_used=1.0,
            reasoning=f"Created entity for deterministic ID: {det_id.id_type}:{det_id.id_value}",
        )
        self._session.add(evolution)

        return _DeterministicIdResult(link=link, entity=entity, evolution=evolution)

    async def _resolve_by_similarity(
        self,
        ctx: ResolutionContext,
        seed: EntityHypothesis,
        entity_nature: EntityNature,
        seed_idx: int,
    ) -> _SeedResolutionResult:
        """Resolve via similarity-based matching.

        Args:
            ctx: Resolution context.
            seed: Entity hypothesis.
            entity_nature: Nature of the entity.
            seed_idx: Seed index.

        Returns:
            Resolution result with links and/or new entities.
        """
        links: list[PendingLink] = []
        entities: list[Entity] = []
        evolutions: list[EntityEvolution] = []

        # Get query embedding based on entity nature
        query_embedding = self._get_query_embedding(ctx.observation_signals, seed, entity_nature)

        if query_embedding is None:
            # No embedding available - create proto-entity
            logger.debug("No embedding available for seed %d, creating proto-entity", seed_idx)
            entity = await self._create_proto_entity(ctx, seed, entity_nature)
            entities.append(entity)

            link = PendingLink(
                entity_id=entity.entity_id,
                posterior=0.8,
                status=LinkStatus.SOFT,
                role=self._infer_role(seed),
                flags=["no_embedding_available"],
                seed_index=seed_idx,
            )
            links.append(link)
            return _SeedResolutionResult(links=links, entities=entities, evolutions=evolutions)

        # Get candidates via ANN
        candidates = await self._pool.get_candidates_by_embedding(
            embedding=query_embedding,
            entity_nature=entity_nature,
            top_k=self._top_k,
        )

        if not candidates:
            # No candidates - create proto-entity
            logger.debug("No candidates found for seed %d, creating proto-entity", seed_idx)
            entity = await self._create_proto_entity(ctx, seed, entity_nature)
            entities.append(entity)

            link = PendingLink(
                entity_id=entity.entity_id,
                posterior=0.8,
                status=LinkStatus.SOFT,
                role=self._infer_role(seed),
                flags=["no_candidates_found"],
                seed_index=seed_idx,
            )
            links.append(link)
            return _SeedResolutionResult(links=links, entities=entities, evolutions=evolutions)

        # Score candidates - keep full SimilarityResult for guardrail checks
        scored: list[tuple[EntityCandidate, SimilarityResult]] = []
        for candidate in candidates:
            sim_result = self._scorer.compute(
                observation_signals=ctx.observation_signals,
                entity=candidate.entity,
                entity_nature=entity_nature,
            )
            scored.append((candidate, sim_result))

        # Sort by score descending
        scored.sort(key=lambda x: x[1].score, reverse=True)

        best_candidate, best_result = scored[0]
        best_score = best_result.score
        second_best_score = scored[1][1].score if len(scored) > 1 else 0.0
        margin = best_score - second_best_score

        # Log detailed scoring info for debugging
        logger.debug(
            "Similarity scoring: seed=%d candidates=%d best_score=%.4f "
            "second_best=%.4f margin=%.4f image_sim=%s text_sim=%s flags=%s",
            seed_idx,
            len(candidates),
            best_score,
            second_best_score,
            margin,
            f"{best_result.image_similarity:.4f}" if best_result.image_similarity else "N/A",
            f"{best_result.text_similarity:.4f}" if best_result.text_similarity else "N/A",
            best_result.flags,
        )

        # Apply INDIVIDUAL image guardrails
        # For INDIVIDUAL entities in sparse signal regime, apply stricter checks
        passes_image_guardrail = True
        image_guardrail_flags: list[str] = []

        if entity_nature == EntityNature.INDIVIDUAL and best_result.image_similarity is not None:
            # Check T_img_min: minimum raw image similarity
            if best_result.image_similarity < self._t_img_min:
                passes_image_guardrail = False
                image_guardrail_flags.append("below_t_img_min")

            # Check image margin for sparse signal cases
            if "single_signal" in best_result.flags or "sparse_signals" in best_result.flags:
                # For image-only, use stricter margin
                if margin < self._t_margin_img:
                    image_guardrail_flags.append("low_image_margin")

            # SINGLE CANDIDATE GUARDRAIL: If only one candidate in pool,
            # don't auto-merge unless score is VERY high (>= 0.95).
            # This prevents false merges when the pool is sparse.
            if len(candidates) == 1:
                image_guardrail_flags.append("single_candidate")
                if best_score < 0.95:
                    passes_image_guardrail = False
                    image_guardrail_flags.append("single_candidate_low_score")
                    logger.info(
                        "Single candidate guardrail: blocking merge "
                        "(only 1 candidate, score=%.4f < 0.95)",
                        best_score,
                    )

        # Apply decision thresholds
        # Log threshold comparisons for debugging
        logger.debug(
            "Decision thresholds: best_score=%.4f t_link=%.3f t_new=%.3f "
            "passes_guardrail=%s guardrail_flags=%s",
            best_score,
            self._t_link,
            self._t_new,
            passes_image_guardrail,
            image_guardrail_flags,
        )

        if best_score >= self._t_link and passes_image_guardrail:
            # High confidence - soft link to existing entity
            logger.info(
                "LINK DECISION: soft link to existing entity %s "
                "(score=%.4f >= t_link=%.3f, guardrail=PASS)",
                best_candidate.entity.entity_id,
                best_score,
                self._t_link,
            )
            link = PendingLink(
                entity_id=best_candidate.entity.entity_id,
                posterior=best_score,
                status=LinkStatus.SOFT,
                role=self._infer_role(seed),
                seed_index=seed_idx,
            )

            # Check for margin flag
            if best_score - second_best_score < self._t_margin:
                link.flags.append("margin_review")

            # Add image guardrail flags if any
            link.flags.extend(image_guardrail_flags)

            links.append(link)

        elif best_score >= self._t_new:
            # Uncertain - create candidate links to both
            logger.info(
                "LINK DECISION: candidate links (uncertain) "
                "(t_new=%.3f <= score=%.4f < t_link=%.3f)",
                self._t_new,
                best_score,
                self._t_link,
            )

            # Build explainable flags for candidate link
            candidate_flags = ["uncertain_match"]

            # Flag high visual similarity (semantic match but uncertain identity)
            if best_result.image_similarity is not None and best_result.image_similarity >= 0.85:
                candidate_flags.append("high_visual_similarity")

            # Flag context mismatches (GPS/time disagree despite visual match)
            signal_scores = best_result.signal_scores
            if "location" in signal_scores and signal_scores["location"] < 0.3:
                candidate_flags.append("context_mismatch_gps")
            if "timestamp" in signal_scores and signal_scores["timestamp"] < 0.3:
                candidate_flags.append("context_mismatch_time")

            # Flag low evidence
            if best_result.is_low_evidence:
                candidate_flags.append("low_evidence")

            # Link to best candidate
            link_existing = PendingLink(
                entity_id=best_candidate.entity.entity_id,
                posterior=best_score,
                status=LinkStatus.CANDIDATE,
                role=self._infer_role(seed),
                flags=candidate_flags,
                seed_index=seed_idx,
            )
            links.append(link_existing)

            # Also create proto-entity as alternative
            proto = await self._create_proto_entity(ctx, seed, entity_nature)
            entities.append(proto)

            link_proto = PendingLink(
                entity_id=proto.entity_id,
                posterior=1.0 - best_score,
                status=LinkStatus.CANDIDATE,
                role=self._infer_role(seed),
                flags=["proto_alternative"],
                seed_index=seed_idx,
            )
            links.append(link_proto)

        else:
            # Low confidence - create new proto-entity
            logger.info(
                "LINK DECISION: new proto-entity (score=%.4f < t_new=%.3f or guardrail_fail=%s)",
                best_score,
                self._t_new,
                not passes_image_guardrail,
            )
            proto = await self._create_proto_entity(ctx, seed, entity_nature)
            entities.append(proto)

            link = PendingLink(
                entity_id=proto.entity_id,
                posterior=0.8,
                status=LinkStatus.SOFT,
                role=self._infer_role(seed),
                flags=["no_match_found"],
                seed_index=seed_idx,
            )
            links.append(link)

        return _SeedResolutionResult(links=links, entities=entities, evolutions=evolutions)

    def _get_query_embedding(
        self,
        signals: ObservationSignals,
        seed: EntityHypothesis,
        entity_nature: EntityNature,
    ) -> list[float] | None:
        """Get the appropriate query embedding for a seed.

        Cross-modal matching: For INDIVIDUAL entities, if image_embedding is
        unavailable (e.g., text observation), use clip_text_embedding which
        is in the same 768-dim CLIP space as image embeddings.

        Args:
            signals: Observation signals.
            seed: Entity hypothesis.
            entity_nature: Nature of the entity.

        Returns:
            Embedding vector or None if unavailable.
        """
        if entity_nature == EntityNature.INDIVIDUAL:
            # Individuals: prefer image embedding, fall back to CLIP text for cross-modal
            if signals.image_embedding is not None:
                return signals.image_embedding
            # Cross-modal: use CLIP text embedding (same 768-dim space as image)
            return signals.clip_text_embedding
        else:
            # Class/Group/Event: prefer text embedding
            return signals.text_embedding

    async def _create_proto_entity(
        self,
        ctx: ResolutionContext,
        seed: EntityHypothesis,
        entity_nature: EntityNature,
    ) -> Entity:
        """Create a new proto-entity from a seed.

        IMPORTANT: Seeds prototype embeddings at creation time to break the
        "prototype deadlock" where entities with NULL prototypes can never
        be retrieved by ANN search and thus never get linked/updated.

        Seeding strategy (conservative - prefer false split over false merge):
        - Only seed image prototypes for engine-owned subject seeds (entity_type="photo_subject")
          to avoid polluting ANN pool with irrelevant entities (photographer, GPS-derived, etc.)
        - CLASS/GROUP/EVENT: seed prototype_text_embedding if available (semantic)

        Args:
            ctx: Resolution context.
            seed: Entity hypothesis.
            entity_nature: Nature of the entity.

        Returns:
            Newly created Entity.
        """
        entity_id = uuid4()

        # Normalize slots per Slot Hygiene Contract (concept_v2.md §6.3)
        # Also records any warnings as Evidence for debugging
        raw_slots = _slots_to_dict(seed)
        normalized_slots = normalize_and_record_slot_warnings(
            raw_slots, entity_id, self._session, subject_kind="entity"
        )

        # Inject GPS/timestamp from observation for INDIVIDUAL entities.
        # These are used by location/timestamp similarity scoring.
        if entity_nature == EntityNature.INDIVIDUAL:
            obs_signals = ctx.observation_signals
            if obs_signals.gps_coords and "latitude" not in normalized_slots:
                normalized_slots["latitude"] = {"value": obs_signals.gps_coords[0]}
                normalized_slots["longitude"] = {"value": obs_signals.gps_coords[1]}
            if obs_signals.timestamp and "datetime" not in normalized_slots:
                from datetime import datetime, timezone
                dt = datetime.fromtimestamp(obs_signals.timestamp, tz=timezone.utc)
                normalized_slots["datetime"] = {"value": dt.isoformat()}

        # Seed prototype embeddings to break the prototype deadlock.
        # CRITICAL: Only seed image prototypes for engine-owned subject seeds
        # (entity_type="photo_subject"). Other INDIVIDUAL entities (photographer,
        # GPS-derived location, etc.) should NOT get image prototypes - they would
        # pollute the ANN pool and cause wrong merges.
        prototype_image_embedding: list[float] | None = None
        prototype_text_embedding: list[float] | None = None
        prototype_count = 0

        # Engine-owned subject seed for visual re-ID
        is_engine_subject_seed = (
            entity_nature == EntityNature.INDIVIDUAL
            and seed.entity_type == "photo_subject"
        )

        if is_engine_subject_seed:
            # Only the engine-owned subject gets image prototype
            if ctx.observation_signals.image_embedding:
                prototype_image_embedding = ctx.observation_signals.image_embedding
                prototype_count = 1
        elif entity_nature != EntityNature.INDIVIDUAL:
            # CLASS/GROUP/EVENT: seed with text embedding for semantic matching
            if ctx.observation_signals.text_embedding:
                prototype_text_embedding = ctx.observation_signals.text_embedding
                prototype_count = 1
        # else: LLM-proposed INDIVIDUAL entities (photographer, location, etc.)
        # do NOT get image prototypes - they're not for visual re-ID

        entity = Entity(
            entity_id=entity_id,
            entity_nature=entity_nature,
            name=seed.suggested_name,
            slots=normalized_slots,
            status=EntityStatus.PROTO,
            confidence=seed.confidence,
            prototype_count=prototype_count,
            prototype_image_embedding=prototype_image_embedding,
            prototype_text_embedding=prototype_text_embedding,
        )
        self._session.add(entity)
        await self._session.flush()

        logger.debug(
            "Created proto-entity %s (%s) for seed '%s' with prototype_count=%d",
            entity.entity_id,
            entity_nature.value,
            seed.suggested_name,
            prototype_count,
        )

        return entity

    def _infer_role(self, seed: EntityHypothesis) -> LinkRole:
        """Infer the link role from a seed.

        For v0, we use simple heuristics:
        - entity_type containing "primary" or high confidence → subject
        - Otherwise → context
        """
        # Default to subject for now
        # Future: use seed metadata to determine role
        return LinkRole.SUBJECT

    def _should_update_prototype(self, link: PendingLink) -> bool:
        """Determine if a link should update the entity prototype.

        Per user spec:
        - Update on hard links always
        - Update on soft links only if posterior >= T_link
        """
        should_update = False
        reason = "unknown"

        if link.status == LinkStatus.HARD:
            should_update = True
            reason = "hard_link"
        elif link.status == LinkStatus.SOFT and link.posterior >= self._t_link:
            should_update = True
            reason = f"soft_above_t_link({self._t_link})"
        elif link.status == LinkStatus.SOFT:
            reason = f"soft_below_t_link({link.posterior:.3f}<{self._t_link})"
        elif link.status == LinkStatus.CANDIDATE:
            reason = "candidate_status"
        else:
            reason = f"unhandled_status({link.status})"

        logger.debug(
            "Prototype update decision: entity=%s status=%s posterior=%.3f "
            "t_link=%.3f should_update=%s reason=%s",
            link.entity_id,
            link.status,
            link.posterior,
            self._t_link,
            should_update,
            reason,
        )
        return should_update

    async def _update_prototype(
        self,
        entity: Entity,
        signals: ObservationSignals,
        link: PendingLink,
    ) -> None:
        """Update entity prototype embeddings.

        Uses running average:
        new_proto = (proto * count + emb * weight) / (count + weight)

        Weight is capped at 0.5 for soft links to prevent yanking.
        Count always increments by 1 (each observation contributes once).

        IMPORTANT: Only update IMAGE prototypes for photo_subject entities
        (those with GPS in slots or no name). Entities from deterministic IDs
        (camera_serial, lens_serial, image_unique_id) should NOT get image
        prototypes as they pollute the ANN pool for visual re-ID.
        """
        # Determine weight for embedding averaging
        if link.status == LinkStatus.HARD:
            weight = 1.0
        else:
            # Cap soft link weight at 0.5
            weight = min(link.posterior, 0.5)

        old_count = entity.prototype_count
        updated_img = False
        updated_txt = False

        # Check if this entity should get image prototype updates
        # Only photo_subject entities (with GPS or unnamed) get image prototypes
        # Entities from deterministic IDs (camera, lens, image_id) do NOT
        is_photo_subject = False
        if entity.slots:
            # Has GPS → it's a photo_subject we created
            if entity.slots.get("latitude") or entity.slots.get("gps_latitude"):
                is_photo_subject = True
        if not entity.name:
            # Unnamed entity → likely a photo_subject
            is_photo_subject = True

        # Update image prototype (ONLY for photo_subject entities)
        if signals.image_embedding and entity.prototype_image_embedding is not None and is_photo_subject:
            entity.prototype_image_embedding = self._weighted_average(
                list(entity.prototype_image_embedding),
                signals.image_embedding,
                entity.prototype_count,
                weight,
            )
            updated_img = True
        elif signals.image_embedding and entity.prototype_image_embedding is None and is_photo_subject:
            entity.prototype_image_embedding = signals.image_embedding
            updated_img = True

        # Update text prototype
        if signals.text_embedding and entity.prototype_text_embedding is not None:
            entity.prototype_text_embedding = self._weighted_average(
                list(entity.prototype_text_embedding),
                signals.text_embedding,
                entity.prototype_count,
                weight,
            )
            updated_txt = True
        elif signals.text_embedding and entity.prototype_text_embedding is None:
            entity.prototype_text_embedding = signals.text_embedding
            updated_txt = True

        # Update count - always increment by 1 for each contributing observation
        # (not int(weight) which would be 0 for soft links!)
        if updated_img or updated_txt:
            entity.prototype_count += 1

        logger.debug(
            "Prototype update: entity=%s status=%s weight=%.3f "
            "count=%d->%d updated_img=%s updated_txt=%s",
            entity.entity_id,
            link.status,
            weight,
            old_count,
            entity.prototype_count,
            updated_img,
            updated_txt,
        )

    async def _align_to_events(
        self,
        ctx: ResolutionContext,
        existing_links: list[PendingLink],
    ) -> list[PendingLink]:
        """Align image observation to existing EVENT entities.

        Cross-modal EVENT alignment enables linking images to text-defined EVENTs
        using temporal/spatial/species matching. This provides cross-modal coherence
        without inventing individuals from text.

        CONSTRAINTS:
        - Only emits links to EVENT entities (role=CONTEXT)
        - Does NOT affect INDIVIDUAL resolution or posteriors
        - Conservative: uncertain matches → candidate/soft links, never hard

        Args:
            ctx: Resolution context with observation and signals.
            existing_links: Already-created links (for finding photo_subject).

        Returns:
            List of PendingLinks to matched EVENT entities.
        """
        from datetime import datetime, timezone

        from sqlalchemy import select

        from object_sense.resolution.similarity import haversine_km

        event_links: list[PendingLink] = []

        # Extract observation facets for matching
        obs_facets = ctx.observation.facets or {}

        # Get observation timestamp (timezone-aware)
        obs_timestamp = ctx.observation_signals.timestamp
        obs_date: str | None = None
        if obs_timestamp:
            dt = datetime.fromtimestamp(obs_timestamp, tz=timezone.utc)
            obs_date = dt.strftime("%Y-%m-%d")

        # Get observation GPS
        obs_gps = ctx.observation_signals.gps_coords

        # Get observation species (from facets: species_* but not *_presence)
        obs_species: set[str] = set()
        for key, value in obs_facets.items():
            if key.startswith("species_") and not key.endswith("_presence") and value:
                obs_species.add(str(value).lower())

        # Build query with coarse prefilters (by entity_nature and status only)
        # GPS/date filtering done in Python to avoid JSON column issues
        stmt = (
            select(Entity)
            .where(
                Entity.entity_nature == EntityNature.EVENT,
                Entity.canonical_entity_id.is_(None),
                Entity.status != EntityStatus.DEPRECATED,
            )
        )

        # Add date prefilter if observation has date (JSON text comparison)
        if obs_date:
            stmt = stmt.where(
                Entity.slots["event_date"]["value"].astext == obs_date
            )

        result = await self._session.execute(stmt)
        candidate_events = result.scalars().all()

        if not candidate_events:
            return event_links

        # Score each candidate EVENT
        for event_entity in candidate_events:
            event_slots = event_entity.slots or {}
            match_score = 0.0
            match_signals: list[str] = []

            # Date matching (strong signal) - already prefiltered
            if obs_date:
                match_score += 0.4
                match_signals.append("date_match")

            # Location matching (GPS proximity)
            if obs_gps:
                event_lat_slot = event_slots.get("latitude")
                event_lon_slot = event_slots.get("longitude")
                if event_lat_slot and event_lon_slot:
                    try:
                        # Slots store typed values (float for lat/lon)
                        lat_val = event_lat_slot.get("value") if isinstance(event_lat_slot, dict) else event_lat_slot
                        lon_val = event_lon_slot.get("value") if isinstance(event_lon_slot, dict) else event_lon_slot
                        event_gps = (float(lat_val), float(lon_val))
                        dist_km = haversine_km(obs_gps, event_gps)
                        # Within 20km = strong match, decay beyond
                        if dist_km < 20:
                            location_score = 0.3 * (1 - dist_km / 20)
                            match_score += location_score
                            match_signals.append(f"gps_proximity({dist_km:.1f}km)")
                    except (ValueError, TypeError):
                        pass

            # Location name matching (fallback if no GPS on observation)
            if not obs_gps:
                event_loc_name = event_slots.get("location_name")
                if event_loc_name:
                    loc_name_val = event_loc_name.get("value") if isinstance(event_loc_name, dict) else event_loc_name
                    obs_loc = obs_facets.get("location_name", "").lower()
                    if loc_name_val and obs_loc and str(loc_name_val).lower() in obs_loc:
                        match_score += 0.2
                        match_signals.append("location_name_match")

            # Species matching (set overlap)
            if obs_species:
                event_species_slot = event_slots.get("species")
                if event_species_slot:
                    # Slots store typed values (list for species)
                    species_val = event_species_slot.get("value") if isinstance(event_species_slot, dict) else event_species_slot
                    if isinstance(species_val, list):
                        event_species_set = {s.lower() for s in species_val}
                    else:
                        event_species_set = {str(species_val).lower()}
                    overlap = obs_species & event_species_set
                    if overlap:
                        match_score += 0.3 * len(overlap) / max(len(obs_species), len(event_species_set))
                        match_signals.append(f"species_overlap({overlap})")

            # Threshold for alignment (conservative)
            if match_score >= 0.5:
                status = LinkStatus.SOFT
            elif match_score >= 0.3:
                status = LinkStatus.CANDIDATE
            else:
                continue

            link = PendingLink(
                entity_id=event_entity.entity_id,
                posterior=match_score,
                status=status,
                role=LinkRole.CONTEXT,
                seed_index=-1,
                flags=["event_aligned"] + match_signals,
            )
            event_links.append(link)

            logger.info(
                "EVENT alignment: obs=%s → event=%s score=%.3f status=%s signals=%s",
                ctx.observation.observation_id,
                event_entity.entity_id,
                match_score,
                status.value,
                match_signals,
            )

        return event_links

    def _weighted_average(
        self,
        current: list[float],
        new: list[float],
        count: int,
        weight: float,
    ) -> list[float]:
        """Compute weighted running average of embeddings."""
        if count == 0:
            return new

        result: list[float] = []
        for i in range(len(current)):
            avg = (current[i] * count + new[i] * weight) / (count + weight)
            result.append(avg)
        return result


@dataclass
class _SeedResolutionResult:
    """Internal result for resolving a single seed."""

    links: list[PendingLink] = field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    entities: list[Entity] = field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    conflicts: list[IdentityConflict] = field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    evolutions: list[EntityEvolution] = field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]


@dataclass
class _DeterministicIdResult:
    """Internal result for deterministic ID resolution."""

    link: PendingLink | None = None
    entity: Entity | None = None
    conflict: IdentityConflict | None = None
    evolution: EntityEvolution | None = None


async def resolve_canonical_entity(
    session: AsyncSession,
    entity_id: UUID,
    *,
    compress_path: bool = True,
) -> Entity:
    """Resolve an entity to its canonical form, with path compression.

    If an entity has been merged into another, follow the canonical_entity_id
    chain to find the ultimate canonical entity. Optionally compresses the
    path by updating intermediate pointers.

    Args:
        session: Database session.
        entity_id: The entity ID to resolve.
        compress_path: If True, update intermediate entities to point directly
                       to the canonical entity (path compression).

    Returns:
        The canonical Entity (may be the same entity if not merged).

    Raises:
        ValueError: If the entity does not exist.
    """
    visited: list[Entity] = []
    current_id = entity_id

    # Follow the canonical chain
    while True:
        entity = await session.get(Entity, current_id)
        if entity is None:
            msg = f"Entity {current_id} not found"
            raise ValueError(msg)

        visited.append(entity)

        if entity.canonical_entity_id is None:
            # This is the canonical entity
            break

        current_id = entity.canonical_entity_id

    canonical = visited[-1]

    # Path compression: update all visited entities to point directly to canonical
    if compress_path and len(visited) > 1:
        for entity in visited[:-1]:
            if entity.canonical_entity_id != canonical.entity_id:
                entity.canonical_entity_id = canonical.entity_id

        await session.flush()

    return canonical
