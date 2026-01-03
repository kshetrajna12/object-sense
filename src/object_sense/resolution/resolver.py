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
)
from object_sense.models.identity_conflict import IdentityConflict
from object_sense.models.observation import Observation, ObservationEntityLink
from object_sense.resolution.candidate_pool import CandidatePoolService, EntityCandidate
from object_sense.resolution.reconciliation import (
    PendingLink,
    deduplicate_links,
    reconcile_multi_seed_links,
)
from object_sense.resolution.similarity import ObservationSignals, SimilarityScorer

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
        top_k: int = 200,
    ) -> None:
        """Initialize the resolver.

        Args:
            session: Database session.
            t_link: High confidence threshold (default from config).
            t_new: Low confidence threshold (default from config).
            t_margin: Margin for review flag (default from config).
            top_k: Number of ANN candidates to retrieve.
        """
        self._session = session
        self._t_link = t_link or settings.entity_resolution_t_link
        self._t_new = t_new or settings.entity_resolution_t_new
        self._t_margin = t_margin or settings.entity_resolution_t_margin
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

        # Process each entity seed
        for seed_idx, seed in enumerate(entity_seeds):
            seed_result = await self._resolve_seed(ctx, seed, seed_idx)

            pending_links.extend(seed_result.links)
            entities_created.extend(seed_result.entities)
            conflicts.extend(seed_result.conflicts)
            evolutions.extend(seed_result.evolutions)

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
        obs_entity_links: list[ObservationEntityLink] = []
        for link in final_links:
            oel = ObservationEntityLink(
                observation_id=observation.observation_id,
                entity_id=link.entity_id,
                posterior=link.posterior,
                status=LinkStatus(link.status),
                role=link.role,
                flags=link.flags,
            )
            obs_entity_links.append(oel)
            self._session.add(oel)

            # Update prototype if link qualifies
            entity = await self._session.get(Entity, link.entity_id)
            if entity and self._should_update_prototype(link):
                await self._update_prototype(entity, ctx.observation_signals, link)

            # Create evolution record for link
            evolution = EntityEvolution(
                evolution_id=uuid4(),
                entity_id=link.entity_id,
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
                status="hard",
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
            status="hard",
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
                status="soft",
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
                status="soft",
                role=self._infer_role(seed),
                flags=["no_candidates_found"],
                seed_index=seed_idx,
            )
            links.append(link)
            return _SeedResolutionResult(links=links, entities=entities, evolutions=evolutions)

        # Score candidates
        scored: list[tuple[EntityCandidate, float]] = []
        for candidate in candidates:
            sim_result = self._scorer.compute(
                observation_signals=ctx.observation_signals,
                entity=candidate.entity,
                entity_nature=entity_nature,
            )
            scored.append((candidate, sim_result.score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        best_candidate, best_score = scored[0]
        second_best_score = scored[1][1] if len(scored) > 1 else 0.0

        # Apply decision thresholds
        if best_score >= self._t_link:
            # High confidence - soft link to existing entity
            link = PendingLink(
                entity_id=best_candidate.entity.entity_id,
                posterior=best_score,
                status="soft",
                role=self._infer_role(seed),
                seed_index=seed_idx,
            )

            # Check for margin flag
            if best_score - second_best_score < self._t_margin:
                link.flags.append("margin_review")

            links.append(link)

        elif best_score >= self._t_new:
            # Uncertain - create candidate links to both
            # Link to best candidate
            link_existing = PendingLink(
                entity_id=best_candidate.entity.entity_id,
                posterior=best_score,
                status="candidate",
                role=self._infer_role(seed),
                flags=["uncertain_match"],
                seed_index=seed_idx,
            )
            links.append(link_existing)

            # Also create proto-entity as alternative
            proto = await self._create_proto_entity(ctx, seed, entity_nature)
            entities.append(proto)

            link_proto = PendingLink(
                entity_id=proto.entity_id,
                posterior=1.0 - best_score,
                status="candidate",
                role=self._infer_role(seed),
                flags=["proto_alternative"],
                seed_index=seed_idx,
            )
            links.append(link_proto)

        else:
            # Low confidence - create new proto-entity
            proto = await self._create_proto_entity(ctx, seed, entity_nature)
            entities.append(proto)

            link = PendingLink(
                entity_id=proto.entity_id,
                posterior=0.8,
                status="soft",
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

        Args:
            signals: Observation signals.
            seed: Entity hypothesis.
            entity_nature: Nature of the entity.

        Returns:
            Embedding vector or None if unavailable.
        """
        if entity_nature == EntityNature.INDIVIDUAL:
            # Individuals: prefer image embedding
            return signals.image_embedding
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

        Args:
            ctx: Resolution context.
            seed: Entity hypothesis.
            entity_nature: Nature of the entity.

        Returns:
            Newly created Entity.
        """
        entity = Entity(
            entity_id=uuid4(),
            entity_nature=entity_nature,
            name=seed.suggested_name,
            slots=_slots_to_dict(seed),
            status=EntityStatus.PROTO,
            confidence=seed.confidence,
            prototype_count=0,
        )
        self._session.add(entity)
        await self._session.flush()

        logger.debug(
            "Created proto-entity %s (%s) for seed '%s'",
            entity.entity_id,
            entity_nature.value,
            seed.suggested_name,
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
        if link.status == "hard":
            return True
        if link.status == "soft" and link.posterior >= self._t_link:
            return True
        return False

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
        """
        # Determine weight
        if link.status == "hard":
            weight = 1.0
        else:
            # Cap soft link weight at 0.5
            weight = min(link.posterior, 0.5)

        # Update image prototype
        if signals.image_embedding and entity.prototype_image_embedding is not None:
            entity.prototype_image_embedding = self._weighted_average(
                list(entity.prototype_image_embedding),
                signals.image_embedding,
                entity.prototype_count,
                weight,
            )
        elif signals.image_embedding and entity.prototype_image_embedding is None:
            entity.prototype_image_embedding = signals.image_embedding

        # Update text prototype
        if signals.text_embedding and entity.prototype_text_embedding is not None:
            entity.prototype_text_embedding = self._weighted_average(
                list(entity.prototype_text_embedding),
                signals.text_embedding,
                entity.prototype_count,
                weight,
            )
        elif signals.text_embedding and entity.prototype_text_embedding is None:
            entity.prototype_text_embedding = signals.text_embedding

        # Update count
        entity.prototype_count += int(weight)

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
