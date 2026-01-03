"""Candidate pool retrieval via pgvector ANN search.

This module handles Step 5.2a: Get candidate entities for similarity matching.

Strategy (per user spec):
- ANN first with soft facet boosts (not hard blocking)
- topK = 200 (tunable)
- Use embedding appropriate to seed/entity_nature
- Deterministic IDs handled separately (short-circuit before this)

See design_v2_corrections.md §5 and §6 for specifications.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from object_sense.config import settings
from object_sense.models.entity import Entity
from object_sense.models.entity_deterministic_id import EntityDeterministicId
from object_sense.models.enums import EntityNature, EntityStatus
from object_sense.utils.slots import normalize_and_record_slot_warnings

if TYPE_CHECKING:
    from object_sense.inference.schemas import DeterministicId

logger = logging.getLogger(__name__)


@dataclass
class EntityCandidate:
    """An entity candidate from the pool with relevance metadata."""

    entity: Entity
    ann_distance: float
    """Distance from query embedding (lower = more similar)."""


@dataclass
class DeterministicIdLookupResult:
    """Result of deterministic ID lookup."""

    entity: Entity | None
    """The entity if found, None otherwise."""

    conflict_entity_id: UUID | None
    """If the ID exists but conflicts, the existing entity's ID."""


class CandidatePoolService:
    """Service for retrieving entity candidates via pgvector ANN.

    Usage:
        async with AsyncSession(engine) as session:
            pool = CandidatePoolService(session)
            candidates = await pool.get_candidates_by_embedding(
                embedding=observation_embedding,
                entity_nature=EntityNature.INDIVIDUAL,
                top_k=200
            )
    """

    def __init__(
        self,
        session: AsyncSession,
        *,
        default_top_k: int = 200,
    ) -> None:
        """Initialize the candidate pool service.

        Args:
            session: Database session for queries.
            default_top_k: Default number of candidates to retrieve.
        """
        self._session = session
        self._default_top_k = default_top_k

    async def lookup_by_deterministic_id(
        self,
        det_id: DeterministicId,
    ) -> DeterministicIdLookupResult:
        """Lookup entity by deterministic ID (highest priority resolution).

        Args:
            det_id: The deterministic ID to look up.

        Returns:
            DeterministicIdLookupResult with entity if found.
        """
        stmt = (
            select(EntityDeterministicId)
            .where(
                EntityDeterministicId.id_type == det_id.id_type,
                EntityDeterministicId.id_value == det_id.id_value,
                EntityDeterministicId.id_namespace == det_id.id_namespace,
            )
            .limit(1)
        )
        result = await self._session.execute(stmt)
        binding = result.scalar_one_or_none()

        if binding is None:
            return DeterministicIdLookupResult(entity=None, conflict_entity_id=None)

        # Fetch the actual entity
        entity = await self._session.get(Entity, binding.entity_id)
        return DeterministicIdLookupResult(entity=entity, conflict_entity_id=None)

    async def get_candidates_by_embedding(
        self,
        embedding: list[float],
        entity_nature: EntityNature,
        *,
        top_k: int | None = None,
        exclude_entity_ids: set[UUID] | None = None,
        observation_kind: str | None = None,
    ) -> list[EntityCandidate]:
        """Get entity candidates via pgvector ANN search.

        Args:
            embedding: Query embedding vector.
            entity_nature: Nature of the entity we're looking for.
                - individual: Uses image embedding (prototype_image_embedding)
                - class/group/event: Uses text embedding (prototype_text_embedding)
            top_k: Number of candidates to retrieve (default: 200).
            exclude_entity_ids: Entity IDs to exclude from results.
            observation_kind: Optional routing hint for soft boosting (future).

        Returns:
            List of EntityCandidate objects sorted by distance (closest first).
        """
        if top_k is None:
            top_k = self._default_top_k

        # Select prototype column based on entity_nature
        if entity_nature == EntityNature.INDIVIDUAL:
            proto_col = Entity.prototype_image_embedding
            dim = settings.dim_image_embedding
        else:
            proto_col = Entity.prototype_text_embedding
            dim = settings.dim_text_embedding

        # Validate embedding dimension
        if len(embedding) != dim:
            msg = f"Embedding dimension mismatch: got {len(embedding)}, expected {dim}"
            raise ValueError(msg)

        # Build query using pgvector's cosine distance operator (<=>)
        # Lower distance = more similar
        stmt = (
            select(Entity, proto_col.cosine_distance(embedding).label("distance"))
            .where(
                proto_col.isnot(None),  # Must have prototype embedding
                Entity.canonical_entity_id.is_(None),  # Must be canonical
                Entity.status != EntityStatus.DEPRECATED,  # Not deprecated
            )
            .order_by("distance")
            .limit(top_k)
        )

        # Optionally filter by entity_nature (soft constraint)
        # Note: This is a soft filter - we include it for efficiency
        # but the design allows cross-nature matches
        if entity_nature:
            stmt = stmt.where(Entity.entity_nature == entity_nature)

        # Exclude specific entity IDs
        if exclude_entity_ids:
            stmt = stmt.where(Entity.entity_id.notin_(exclude_entity_ids))

        result = await self._session.execute(stmt)
        rows = result.all()

        return [
            EntityCandidate(entity=row.Entity, ann_distance=row.distance)
            for row in rows
        ]

    async def get_all_candidates_for_nature(
        self,
        entity_nature: EntityNature,
        *,
        limit: int = 1000,
    ) -> list[Entity]:
        """Get all entities of a given nature (for fallback/testing).

        This is a fallback for when embeddings aren't available.
        Not recommended for production use.

        Args:
            entity_nature: The entity nature to filter by.
            limit: Maximum entities to return.

        Returns:
            List of Entity objects.
        """
        stmt = (
            select(Entity)
            .where(
                Entity.entity_nature == entity_nature,
                Entity.canonical_entity_id.is_(None),
                Entity.status != EntityStatus.DEPRECATED,
            )
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def create_entity_for_deterministic_id(
        self,
        det_id: DeterministicId,
        *,
        entity_nature: EntityNature,
        name: str | None = None,
        slots: dict[str, Any] | None = None,
    ) -> Entity:
        """Create a new entity anchored to a deterministic ID.

        This is called when a deterministic ID is not found in the system.
        The ID becomes the entity's identity anchor.

        Args:
            det_id: The deterministic ID to anchor the entity to.
            entity_nature: Nature of the new entity.
            name: Optional human-readable name.
            slots: Optional initial slots.

        Returns:
            The newly created Entity.
        """
        from uuid import uuid4

        entity_id = uuid4()

        # Normalize slots per Slot Hygiene Contract (concept_v2.md §6.3)
        # Also records any warnings as Evidence for debugging
        raw_slots = slots or {}
        normalized_slots = normalize_and_record_slot_warnings(
            raw_slots, entity_id, self._session, subject_kind="entity"
        )

        # Create the entity
        entity = Entity(
            entity_id=entity_id,
            entity_nature=entity_nature,
            name=name,
            slots=normalized_slots,
            status=EntityStatus.PROTO,
            confidence=1.0,  # High confidence from deterministic ID
            prototype_count=0,
        )
        self._session.add(entity)

        # Create the deterministic ID binding
        binding = EntityDeterministicId(
            id=uuid4(),
            entity_id=entity_id,
            id_type=det_id.id_type,
            id_value=det_id.id_value,
            id_namespace=det_id.id_namespace,
            confidence=1.0 if det_id.strength == "strong" else 0.8,
        )
        self._session.add(binding)

        await self._session.flush()
        return entity

    async def check_deterministic_id_conflict(
        self,
        det_id: DeterministicId,
        proposed_entity_id: UUID,
    ) -> EntityDeterministicId | None:
        """Check if a deterministic ID already exists for a different entity.

        Args:
            det_id: The deterministic ID to check.
            proposed_entity_id: The entity we want to link to.

        Returns:
            The existing binding if there's a conflict, None otherwise.
        """
        stmt = (
            select(EntityDeterministicId)
            .where(
                EntityDeterministicId.id_type == det_id.id_type,
                EntityDeterministicId.id_value == det_id.id_value,
                EntityDeterministicId.id_namespace == det_id.id_namespace,
                EntityDeterministicId.entity_id != proposed_entity_id,
            )
            .limit(1)
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()
