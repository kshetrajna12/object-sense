"""Type evolution service for managing type lifecycle changes.

Implements the four evolution mechanisms from design_notes.md:
1. Alias - Multiple names → one canonical type
2. Merge - Combine two types into one
3. Split - Divide an overloaded type into subtypes
4. Deprecate - Retire a type and migrate to replacement

All operations create TypeEvolution history records for auditability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from object_sense.models.entity import Entity
from object_sense.models.enums import TypeEvolutionKind, TypeStatus
from object_sense.models.observation import Observation
from object_sense.models.type import Type
from object_sense.models.type_evolution import TypeEvolution


@dataclass
class AliasResult:
    """Result of an alias operation."""

    success: bool
    type_id: UUID
    alias_added: str | None = None
    reason: str | None = None


@dataclass
class MergeResult:
    """Result of a merge operation."""

    success: bool
    source_type_id: UUID
    target_type_id: UUID
    observations_migrated: int = 0
    entities_migrated: int = 0
    reason: str | None = None


@dataclass
class SplitResult:
    """Result of a split operation."""

    success: bool
    source_type_id: UUID
    new_type_ids: list[UUID] | None = None
    observations_redistributed: int = 0
    reason: str | None = None


@dataclass
class DeprecateResult:
    """Result of a deprecate operation."""

    success: bool
    type_id: UUID
    replacement_type_id: UUID | None = None
    observations_migrated: int = 0
    entities_migrated: int = 0
    reason: str | None = None


class TypeEvolutionService:
    """Service for managing type evolution operations.

    All operations:
    1. Validate preconditions
    2. Perform the change
    3. Create a TypeEvolution history record
    4. Migrate affected observations/entities as needed

    Usage:
        async with AsyncSession(engine) as session:
            service = TypeEvolutionService(session)
            result = await service.add_alias(type_id, "new_alias")
            await session.commit()
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize with a database session."""
        self._session = session

    # -------------------------------------------------------------------------
    # 1. ALIAS - Multiple names → one canonical type
    # -------------------------------------------------------------------------

    async def add_alias(
        self,
        type_id: UUID,
        alias: str,
        *,
        reason: str | None = None,
    ) -> AliasResult:
        """Add an alias to an existing type.

        Aliases are alternative names that map to the same canonical type.
        Used when the LLM proposes "nature_image" but "wildlife_photo" exists.

        Args:
            type_id: The type to add the alias to.
            alias: The new alias (will be normalized to lowercase).
            reason: Optional reason for the alias (e.g., "LLM proposal dedup").

        Returns:
            AliasResult with operation details.
        """
        # Normalize alias
        normalized_alias = alias.lower().strip()

        # Load the type
        type_obj = await self._session.get(Type, type_id)
        if type_obj is None:
            return AliasResult(
                success=False,
                type_id=type_id,
                reason=f"Type {type_id} not found",
            )

        # Check if type is in a valid state for modification
        if type_obj.status in (TypeStatus.DEPRECATED, TypeStatus.MERGED_INTO):
            return AliasResult(
                success=False,
                type_id=type_id,
                reason=f"Cannot add alias to {type_obj.status.value} type",
            )

        # Check if alias already exists
        if normalized_alias in [a.lower() for a in type_obj.aliases]:
            return AliasResult(
                success=False,
                type_id=type_id,
                reason=f"Alias '{normalized_alias}' already exists",
            )

        # Check if alias conflicts with another type's canonical name
        conflict = await self._session.execute(
            select(Type).where(Type.canonical_name == normalized_alias)
        )
        if conflict.scalar_one_or_none() is not None:
            return AliasResult(
                success=False,
                type_id=type_id,
                reason=f"Alias '{normalized_alias}' conflicts with another type's canonical name",
            )

        # Add the alias
        type_obj.aliases = [*type_obj.aliases, alias]

        # Record the evolution
        evolution = TypeEvolution(
            evolution_id=uuid4(),
            kind=TypeEvolutionKind.ALIAS,
            source_type_id=type_id,
            target_type_id=None,
            reason=reason or f"Added alias: {alias}",
            details={"alias": alias, "normalized": normalized_alias},
        )
        self._session.add(evolution)

        return AliasResult(
            success=True,
            type_id=type_id,
            alias_added=alias,
            reason=f"Alias '{alias}' added successfully",
        )

    async def remove_alias(
        self,
        type_id: UUID,
        alias: str,
        *,
        reason: str | None = None,
    ) -> AliasResult:
        """Remove an alias from a type.

        Args:
            type_id: The type to remove the alias from.
            alias: The alias to remove.
            reason: Optional reason for removal.

        Returns:
            AliasResult with operation details.
        """
        type_obj = await self._session.get(Type, type_id)
        if type_obj is None:
            return AliasResult(
                success=False,
                type_id=type_id,
                reason=f"Type {type_id} not found",
            )

        normalized_alias = alias.lower().strip()
        matching_aliases = [a for a in type_obj.aliases if a.lower() == normalized_alias]

        if not matching_aliases:
            return AliasResult(
                success=False,
                type_id=type_id,
                reason=f"Alias '{alias}' not found on type",
            )

        # Remove the alias
        type_obj.aliases = [a for a in type_obj.aliases if a.lower() != normalized_alias]

        # Record the evolution (as alias with removal detail)
        evolution = TypeEvolution(
            evolution_id=uuid4(),
            kind=TypeEvolutionKind.ALIAS,
            source_type_id=type_id,
            target_type_id=None,
            reason=reason or f"Removed alias: {alias}",
            details={"alias_removed": alias, "action": "remove"},
        )
        self._session.add(evolution)

        return AliasResult(
            success=True,
            type_id=type_id,
            alias_added=None,
            reason=f"Alias '{alias}' removed successfully",
        )

    # -------------------------------------------------------------------------
    # 2. MERGE - Combine two types into one
    # -------------------------------------------------------------------------

    async def merge_types(
        self,
        source_type_id: UUID,
        target_type_id: UUID,
        *,
        reason: str | None = None,
        migrate_observations: bool = True,
        migrate_entities: bool = True,
    ) -> MergeResult:
        """Merge source type into target type.

        The source type is marked as MERGED_INTO and all its observations/entities
        are migrated to the target type. The source's aliases are preserved in
        the target.

        Args:
            source_type_id: The type being merged (will be marked merged_into).
            target_type_id: The type to merge into (will remain active).
            reason: Optional reason for the merge.
            migrate_observations: Whether to migrate observations to target.
            migrate_entities: Whether to migrate entities to target.

        Returns:
            MergeResult with operation details.
        """
        if source_type_id == target_type_id:
            return MergeResult(
                success=False,
                source_type_id=source_type_id,
                target_type_id=target_type_id,
                reason="Cannot merge a type into itself",
            )

        # Load both types
        source = await self._session.get(Type, source_type_id)
        target = await self._session.get(Type, target_type_id)

        if source is None:
            return MergeResult(
                success=False,
                source_type_id=source_type_id,
                target_type_id=target_type_id,
                reason=f"Source type {source_type_id} not found",
            )

        if target is None:
            return MergeResult(
                success=False,
                source_type_id=source_type_id,
                target_type_id=target_type_id,
                reason=f"Target type {target_type_id} not found",
            )

        if source.status in (TypeStatus.DEPRECATED, TypeStatus.MERGED_INTO):
            return MergeResult(
                success=False,
                source_type_id=source_type_id,
                target_type_id=target_type_id,
                reason=f"Source type is already {source.status.value}",
            )

        if target.status in (TypeStatus.DEPRECATED, TypeStatus.MERGED_INTO):
            return MergeResult(
                success=False,
                source_type_id=source_type_id,
                target_type_id=target_type_id,
                reason=f"Target type is {target.status.value}, cannot merge into it",
            )

        observations_migrated = 0
        entities_migrated = 0

        # Migrate observations
        if migrate_observations:
            observations_migrated = await self._migrate_observations(source_type_id, target_type_id)

        # Migrate entities
        if migrate_entities:
            entities_migrated = await self._migrate_entities(source_type_id, target_type_id)

        # Preserve source aliases in target (including canonical name as alias)
        new_aliases = list(target.aliases)
        new_aliases.append(source.canonical_name)
        for alias in source.aliases:
            if alias.lower() not in [a.lower() for a in new_aliases]:
                new_aliases.append(alias)
        target.aliases = new_aliases

        # Update evidence count
        target.evidence_count += source.evidence_count

        # Mark source as merged
        source.status = TypeStatus.MERGED_INTO
        source.merged_into_type_id = target_type_id

        # Record the evolution
        evolution = TypeEvolution(
            evolution_id=uuid4(),
            kind=TypeEvolutionKind.MERGE,
            source_type_id=source_type_id,
            target_type_id=target_type_id,
            reason=reason or f"Merged {source.canonical_name} into {target.canonical_name}",
            details={
                "source_canonical": source.canonical_name,
                "target_canonical": target.canonical_name,
                "aliases_transferred": list(source.aliases) + [source.canonical_name],
                "observations_migrated": observations_migrated,
                "entities_migrated": entities_migrated,
            },
        )
        self._session.add(evolution)

        return MergeResult(
            success=True,
            source_type_id=source_type_id,
            target_type_id=target_type_id,
            observations_migrated=observations_migrated,
            entities_migrated=entities_migrated,
            reason=f"Successfully merged {source.canonical_name} into {target.canonical_name}",
        )

    async def _migrate_observations(
        self,
        from_type_id: UUID,
        to_type_id: UUID,
    ) -> int:
        """Migrate observations from one type to another.

        Returns:
            Number of observations migrated.
        """
        stmt = (
            update(Observation)
            .where(Observation.stable_type_id == from_type_id)
            .values(stable_type_id=to_type_id)
        )
        result = await self._session.execute(stmt)
        return result.rowcount  # type: ignore[return-value]

    async def _migrate_entities(
        self,
        from_type_id: UUID,
        to_type_id: UUID,
    ) -> int:
        """Migrate entities from one type to another.

        Returns:
            Number of entities migrated.
        """
        stmt = update(Entity).where(Entity.type_id == from_type_id).values(type_id=to_type_id)
        result = await self._session.execute(stmt)
        return result.rowcount  # type: ignore[return-value]

    # -------------------------------------------------------------------------
    # 3. SPLIT - Divide an overloaded type into subtypes
    # -------------------------------------------------------------------------

    async def split_type(
        self,
        source_type_id: UUID,
        new_types: list[dict[str, Any]],
        *,
        reason: str | None = None,
        keep_source: bool = False,
    ) -> SplitResult:
        """Split a type into multiple subtypes.

        Used when a type is discovered to be overloaded (e.g., "photo" should
        become "wildlife_photo", "portrait", "product_photo").

        Args:
            source_type_id: The type to split.
            new_types: List of dicts with keys:
                - name: The new type name (required)
                - parent_type_id: Optional parent (defaults to source's parent)
            reason: Optional reason for the split.
            keep_source: If True, source becomes parent of new types.
                        If False, source is deprecated.

        Returns:
            SplitResult with operation details.

        Note:
            Observation redistribution is NOT automatic. After split, observations
            remain on the source type (if keep_source=True) or need manual/LLM
            re-classification to the new subtypes.
        """
        if not new_types:
            return SplitResult(
                success=False,
                source_type_id=source_type_id,
                reason="Must provide at least one new type",
            )

        source = await self._session.get(Type, source_type_id)
        if source is None:
            return SplitResult(
                success=False,
                source_type_id=source_type_id,
                reason=f"Source type {source_type_id} not found",
            )

        if source.status in (TypeStatus.DEPRECATED, TypeStatus.MERGED_INTO):
            return SplitResult(
                success=False,
                source_type_id=source_type_id,
                reason=f"Cannot split {source.status.value} type",
            )

        # Validate new type names don't conflict
        for type_spec in new_types:
            name = type_spec.get("name")
            if not name:
                return SplitResult(
                    success=False,
                    source_type_id=source_type_id,
                    reason="Each new type must have a 'name' field",
                )
            existing = await self._session.execute(select(Type).where(Type.canonical_name == name))
            if existing.scalar_one_or_none() is not None:
                return SplitResult(
                    success=False,
                    source_type_id=source_type_id,
                    reason=f"Type '{name}' already exists",
                )

        # Create new types
        created_type_ids: list[UUID] = []
        for type_spec in new_types:
            new_type_id = uuid4()
            parent_id = type_spec.get("parent_type_id")

            # If keeping source, new types are children of source
            # Otherwise, new types inherit source's parent
            if keep_source:
                parent_id = source_type_id
            elif parent_id is None:
                parent_id = source.parent_type_id

            new_type = Type(
                type_id=new_type_id,
                canonical_name=type_spec["name"],
                aliases=[],
                parent_type_id=parent_id,
                embedding=None,  # Will be computed later
                status=TypeStatus.PROVISIONAL,
                evidence_count=0,  # Starts fresh
                created_via=source.created_via,  # Inherit creation context
            )
            self._session.add(new_type)
            created_type_ids.append(new_type_id)

        # Handle source type
        if not keep_source:
            source.status = TypeStatus.DEPRECATED

        # Record the evolution
        evolution = TypeEvolution(
            evolution_id=uuid4(),
            kind=TypeEvolutionKind.SPLIT,
            source_type_id=source_type_id,
            target_type_id=None,  # Split has multiple targets
            reason=reason or f"Split {source.canonical_name} into {len(new_types)} subtypes",
            details={
                "source_canonical": source.canonical_name,
                "new_types": [t["name"] for t in new_types],
                "new_type_ids": [str(tid) for tid in created_type_ids],
                "source_kept": keep_source,
            },
        )
        self._session.add(evolution)

        return SplitResult(
            success=True,
            source_type_id=source_type_id,
            new_type_ids=created_type_ids,
            observations_redistributed=0,  # Redistribution is separate
            reason=f"Split {source.canonical_name} into {len(created_type_ids)} subtypes",
        )

    async def redistribute_observations_after_split(
        self,
        source_type_id: UUID,
        assignments: dict[UUID, list[UUID]],
        *,
        reason: str | None = None,
    ) -> int:
        """Redistribute observations to new types after a split.

        This is a manual/LLM-driven step after split_type creates the subtypes.

        Args:
            source_type_id: The original type that was split.
            assignments: Dict mapping new_type_id → list of observation_ids.
            reason: Optional reason for the redistribution.

        Returns:
            Total number of observations redistributed.
        """
        total_redistributed: int = 0

        for new_type_id, observation_ids in assignments.items():
            if not observation_ids:
                continue

            stmt = (
                update(Observation)
                .where(Observation.observation_id.in_(observation_ids))
                .where(Observation.stable_type_id == source_type_id)
                .values(stable_type_id=new_type_id)
            )
            result = await self._session.execute(stmt)
            rowcount = getattr(result, "rowcount", 0) or 0
            total_redistributed += rowcount

            # Update evidence count on new type
            new_type = await self._session.get(Type, new_type_id)
            if new_type:
                new_type.evidence_count += rowcount

        return total_redistributed

    # -------------------------------------------------------------------------
    # 4. DEPRECATE - Retire a type and optionally migrate
    # -------------------------------------------------------------------------

    async def deprecate_type(
        self,
        type_id: UUID,
        *,
        replacement_type_id: UUID | None = None,
        reason: str | None = None,
        migrate_observations: bool = True,
        migrate_entities: bool = True,
    ) -> DeprecateResult:
        """Deprecate a type and optionally migrate to a replacement.

        Deprecation is a soft delete - the type remains in the database for
        history/audit but is no longer used for new observations.

        Args:
            type_id: The type to deprecate.
            replacement_type_id: Optional type to migrate observations/entities to.
            reason: Optional reason for deprecation.
            migrate_observations: Whether to migrate observations (if replacement given).
            migrate_entities: Whether to migrate entities (if replacement given).

        Returns:
            DeprecateResult with operation details.
        """
        type_obj = await self._session.get(Type, type_id)
        if type_obj is None:
            return DeprecateResult(
                success=False,
                type_id=type_id,
                reason=f"Type {type_id} not found",
            )

        if type_obj.status == TypeStatus.DEPRECATED:
            return DeprecateResult(
                success=False,
                type_id=type_id,
                reason="Type is already deprecated",
            )

        if type_obj.status == TypeStatus.MERGED_INTO:
            return DeprecateResult(
                success=False,
                type_id=type_id,
                reason="Type was merged, use the merged-into type",
            )

        observations_migrated = 0
        entities_migrated = 0

        # Validate and migrate to replacement if provided
        if replacement_type_id is not None:
            replacement = await self._session.get(Type, replacement_type_id)
            if replacement is None:
                return DeprecateResult(
                    success=False,
                    type_id=type_id,
                    reason=f"Replacement type {replacement_type_id} not found",
                )

            if replacement.status in (TypeStatus.DEPRECATED, TypeStatus.MERGED_INTO):
                return DeprecateResult(
                    success=False,
                    type_id=type_id,
                    reason=f"Replacement type is {replacement.status.value}",
                )

            if migrate_observations:
                observations_migrated = await self._migrate_observations(
                    type_id, replacement_type_id
                )

            if migrate_entities:
                entities_migrated = await self._migrate_entities(type_id, replacement_type_id)

            # Update replacement evidence count
            replacement.evidence_count += type_obj.evidence_count

        # Mark as deprecated
        type_obj.status = TypeStatus.DEPRECATED

        # Record the evolution
        evolution = TypeEvolution(
            evolution_id=uuid4(),
            kind=TypeEvolutionKind.DEPRECATE,
            source_type_id=type_id,
            target_type_id=replacement_type_id,
            reason=reason or f"Deprecated {type_obj.canonical_name}",
            details={
                "canonical_name": type_obj.canonical_name,
                "replacement_canonical": (
                    (await self._session.get(Type, replacement_type_id)).canonical_name  # type: ignore[union-attr]
                    if replacement_type_id
                    else None
                ),
                "observations_migrated": observations_migrated,
                "entities_migrated": entities_migrated,
            },
        )
        self._session.add(evolution)

        return DeprecateResult(
            success=True,
            type_id=type_id,
            replacement_type_id=replacement_type_id,
            observations_migrated=observations_migrated,
            entities_migrated=entities_migrated,
            reason=f"Successfully deprecated {type_obj.canonical_name}",
        )

    # -------------------------------------------------------------------------
    # Utility methods
    # -------------------------------------------------------------------------

    async def resolve_type(self, type_id: UUID) -> Type | None:
        """Resolve a type, following merge chains to the current canonical type.

        If the type was merged into another, follows the chain to find the
        active type. Useful for lookups that might reference old types.

        Args:
            type_id: The type ID to resolve.

        Returns:
            The resolved Type, or None if not found.
        """
        type_obj = await self._session.get(Type, type_id)
        if type_obj is None:
            return None

        # Follow merge chain (with loop detection)
        seen: set[UUID] = set()
        while type_obj.status == TypeStatus.MERGED_INTO and type_obj.merged_into_type_id:
            if type_obj.merged_into_type_id in seen:
                # Circular merge chain - shouldn't happen but handle gracefully
                break
            seen.add(type_obj.type_id)
            type_obj = await self._session.get(Type, type_obj.merged_into_type_id)
            if type_obj is None:
                break

        return type_obj

    async def find_type_by_alias(self, alias: str) -> Type | None:
        """Find a type by any of its aliases.

        Args:
            alias: The alias to search for (case-insensitive).

        Returns:
            The matching Type, or None if not found.
        """
        normalized = alias.lower().strip()

        # First check canonical names
        result = await self._session.execute(
            select(Type)
            .where(Type.canonical_name == normalized)
            .where(Type.status.notin_([TypeStatus.DEPRECATED, TypeStatus.MERGED_INTO]))
        )
        type_obj = result.scalar_one_or_none()
        if type_obj:
            return type_obj

        # Then check aliases array
        # Note: This uses PostgreSQL array operations
        result = await self._session.execute(
            select(Type)
            .where(Type.aliases.any(normalized))  # type: ignore[attr-defined]
            .where(Type.status.notin_([TypeStatus.DEPRECATED, TypeStatus.MERGED_INTO]))
        )
        return result.scalar_one_or_none()

    async def get_evolution_history(
        self,
        type_id: UUID,
        *,
        limit: int = 100,
    ) -> list[TypeEvolution]:
        """Get the evolution history for a type.

        Args:
            type_id: The type to get history for.
            limit: Maximum number of records to return.

        Returns:
            List of TypeEvolution records, newest first.
        """
        # Get history where this type was source or target
        stmt = (
            select(TypeEvolution)
            .where(
                (TypeEvolution.source_type_id == type_id)
                | (TypeEvolution.target_type_id == type_id)
            )
            .order_by(TypeEvolution.created_at.desc())
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def suggest_merges(
        self,
        *,
        similarity_threshold: float = 0.9,
        limit: int = 10,
    ) -> list[tuple[Type, Type, float]]:
        """Suggest types that might be candidates for merging.

        Uses embedding similarity to find types that may be duplicates.

        Args:
            similarity_threshold: Minimum cosine similarity to suggest.
            limit: Maximum number of suggestions.

        Returns:
            List of tuples (type_a, type_b, similarity_score).

        Note:
            This is a placeholder - actual implementation requires vector
            similarity search in the database.
        """
        # TODO: Implement using pgvector similarity search
        # For now, return empty list
        _ = similarity_threshold, limit
        return []
