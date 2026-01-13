"""Tests for type evolution service."""

from __future__ import annotations

from collections.abc import Callable
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from object_sense.models.enums import (
    Medium,
    TypeCreatedVia,
    TypeEvolutionKind,
    TypeStatus,
)
from object_sense.models.observation import Observation
from object_sense.models.type import Type
from object_sense.models.type_evolution import TypeEvolution
from object_sense.services.type_evolution import TypeEvolutionService

# Type aliases for factory fixtures
MakeType = Callable[..., Type]
MakeObservationWithType = Callable[..., Observation]


@pytest.fixture
def make_type() -> MakeType:
    """Factory fixture for creating Type instances."""

    def _make(
        *,
        type_id: UUID | None = None,
        canonical_name: str | None = None,
        aliases: list[str] | None = None,
        parent_type_id: UUID | None = None,
        status: TypeStatus = TypeStatus.PROVISIONAL,
        evidence_count: int = 1,
        created_via: TypeCreatedVia = TypeCreatedVia.LLM_PROPOSED,
    ) -> Type:
        return Type(
            type_id=type_id or uuid4(),
            canonical_name=canonical_name or f"type_{uuid4().hex[:8]}",
            aliases=aliases or [],
            parent_type_id=parent_type_id,
            embedding=None,
            status=status,
            evidence_count=evidence_count,
            created_via=created_via,
        )

    return _make


@pytest.fixture
def make_observation_with_type() -> MakeObservationWithType:
    """Factory fixture for creating Observation instances with stable_type_id."""

    def _make(
        *,
        observation_id: UUID | None = None,
        stable_type_id: UUID | None = None,
    ) -> Observation:
        return Observation(
            observation_id=observation_id or uuid4(),
            medium=Medium.IMAGE,
            source_id=f"test://source/{uuid4()}",
            observation_kind="test_image",
            facets={},
            deterministic_ids=[],
            stable_type_id=stable_type_id,
        )

    return _make


# -----------------------------------------------------------------------------
# ALIAS TESTS
# -----------------------------------------------------------------------------


class TestAddAlias:
    """Tests for the add_alias operation."""

    async def test_add_alias_success(
        self,
        db_session: AsyncSession,
        make_type: MakeType,
    ) -> None:
        """Should add an alias to a type successfully."""
        type_obj = make_type(canonical_name="wildlife_photo")
        db_session.add(type_obj)
        await db_session.flush()

        service = TypeEvolutionService(db_session)
        result = await service.add_alias(type_obj.type_id, "nature_image")

        assert result.success is True
        assert result.alias_added == "nature_image"
        assert "nature_image" in type_obj.aliases

        # Check evolution record was created
        evolutions = (
            (
                await db_session.execute(
                    select(TypeEvolution).where(TypeEvolution.source_type_id == type_obj.type_id)
                )
            )
            .scalars()
            .all()
        )
        assert len(evolutions) == 1
        assert evolutions[0].kind == TypeEvolutionKind.ALIAS

    async def test_add_alias_type_not_found(
        self,
        db_session: AsyncSession,
    ) -> None:
        """Should fail when type doesn't exist."""
        service = TypeEvolutionService(db_session)
        result = await service.add_alias(uuid4(), "some_alias")

        assert result.success is False
        assert "not found" in result.reason.lower()

    async def test_add_alias_duplicate(
        self,
        db_session: AsyncSession,
        make_type: MakeType,
    ) -> None:
        """Should fail when alias already exists."""
        type_obj = make_type(
            canonical_name="wildlife_photo",
            aliases=["nature_image"],
        )
        db_session.add(type_obj)
        await db_session.flush()

        service = TypeEvolutionService(db_session)
        result = await service.add_alias(type_obj.type_id, "nature_image")

        assert result.success is False
        assert "already exists" in result.reason.lower()

    async def test_add_alias_deprecated_type(
        self,
        db_session: AsyncSession,
        make_type: MakeType,
    ) -> None:
        """Should fail when type is deprecated."""
        type_obj = make_type(
            canonical_name="old_type",
            status=TypeStatus.DEPRECATED,
        )
        db_session.add(type_obj)
        await db_session.flush()

        service = TypeEvolutionService(db_session)
        result = await service.add_alias(type_obj.type_id, "new_alias")

        assert result.success is False
        assert "deprecated" in result.reason.lower()

    async def test_add_alias_conflicts_with_canonical(
        self,
        db_session: AsyncSession,
        make_type: MakeType,
    ) -> None:
        """Should fail when alias conflicts with another type's canonical name."""
        type_a = make_type(canonical_name="wildlife_photo")
        type_b = make_type(canonical_name="portrait")
        db_session.add_all([type_a, type_b])
        await db_session.flush()

        service = TypeEvolutionService(db_session)
        result = await service.add_alias(type_a.type_id, "portrait")

        assert result.success is False
        assert "conflicts" in result.reason.lower()


class TestRemoveAlias:
    """Tests for the remove_alias operation."""

    async def test_remove_alias_success(
        self,
        db_session: AsyncSession,
        make_type: MakeType,
    ) -> None:
        """Should remove an alias successfully."""
        type_obj = make_type(
            canonical_name="wildlife_photo",
            aliases=["nature_image", "animal_pic"],
        )
        db_session.add(type_obj)
        await db_session.flush()

        service = TypeEvolutionService(db_session)
        result = await service.remove_alias(type_obj.type_id, "nature_image")

        assert result.success is True
        assert "nature_image" not in type_obj.aliases
        assert "animal_pic" in type_obj.aliases

    async def test_remove_alias_not_found(
        self,
        db_session: AsyncSession,
        make_type: MakeType,
    ) -> None:
        """Should fail when alias doesn't exist."""
        type_obj = make_type(canonical_name="wildlife_photo")
        db_session.add(type_obj)
        await db_session.flush()

        service = TypeEvolutionService(db_session)
        result = await service.remove_alias(type_obj.type_id, "nonexistent")

        assert result.success is False
        assert "not found" in result.reason.lower()


# -----------------------------------------------------------------------------
# MERGE TESTS
# -----------------------------------------------------------------------------


class TestMergeTypes:
    """Tests for the merge_types operation."""

    async def test_merge_success(
        self,
        db_session: AsyncSession,
        make_type: MakeType,
        make_observation_with_type: MakeObservationWithType,
    ) -> None:
        """Should merge types and migrate observations."""
        source = make_type(
            canonical_name="nature_image",
            aliases=["animal_pic"],
            evidence_count=5,
        )
        target = make_type(
            canonical_name="wildlife_photo",
            evidence_count=10,
        )
        obs = make_observation_with_type(stable_type_id=source.type_id)
        db_session.add_all([source, target, obs])
        await db_session.flush()

        service = TypeEvolutionService(db_session)
        result = await service.merge_types(source.type_id, target.type_id)

        assert result.success is True
        assert result.observations_migrated == 1
        assert source.status == TypeStatus.MERGED_INTO
        assert source.merged_into_type_id == target.type_id
        assert "nature_image" in target.aliases
        assert "animal_pic" in target.aliases
        assert target.evidence_count == 15  # 10 + 5

        # Check observation was migrated
        await db_session.refresh(obs)
        assert obs.stable_type_id == target.type_id

    async def test_merge_self_fails(
        self,
        db_session: AsyncSession,
        make_type: MakeType,
    ) -> None:
        """Should fail when trying to merge a type into itself."""
        type_obj = make_type(canonical_name="wildlife_photo")
        db_session.add(type_obj)
        await db_session.flush()

        service = TypeEvolutionService(db_session)
        result = await service.merge_types(type_obj.type_id, type_obj.type_id)

        assert result.success is False
        assert "itself" in result.reason.lower()

    async def test_merge_deprecated_source_fails(
        self,
        db_session: AsyncSession,
        make_type: MakeType,
    ) -> None:
        """Should fail when source is deprecated."""
        source = make_type(
            canonical_name="old_type",
            status=TypeStatus.DEPRECATED,
        )
        target = make_type(canonical_name="new_type")
        db_session.add_all([source, target])
        await db_session.flush()

        service = TypeEvolutionService(db_session)
        result = await service.merge_types(source.type_id, target.type_id)

        assert result.success is False

    async def test_merge_into_deprecated_fails(
        self,
        db_session: AsyncSession,
        make_type: MakeType,
    ) -> None:
        """Should fail when target is deprecated."""
        source = make_type(canonical_name="good_type")
        target = make_type(
            canonical_name="dead_type",
            status=TypeStatus.DEPRECATED,
        )
        db_session.add_all([source, target])
        await db_session.flush()

        service = TypeEvolutionService(db_session)
        result = await service.merge_types(source.type_id, target.type_id)

        assert result.success is False

    async def test_merge_creates_history(
        self,
        db_session: AsyncSession,
        make_type: MakeType,
    ) -> None:
        """Should create TypeEvolution history record."""
        source = make_type(canonical_name="source_type")
        target = make_type(canonical_name="target_type")
        db_session.add_all([source, target])
        await db_session.flush()

        service = TypeEvolutionService(db_session)
        await service.merge_types(source.type_id, target.type_id)

        evolutions = (
            (
                await db_session.execute(
                    select(TypeEvolution).where(TypeEvolution.source_type_id == source.type_id)
                )
            )
            .scalars()
            .all()
        )
        assert len(evolutions) == 1
        assert evolutions[0].kind == TypeEvolutionKind.MERGE
        assert evolutions[0].target_type_id == target.type_id


# -----------------------------------------------------------------------------
# SPLIT TESTS
# -----------------------------------------------------------------------------


class TestSplitType:
    """Tests for the split_type operation."""

    async def test_split_success_deprecate_source(
        self,
        db_session: AsyncSession,
        make_type: MakeType,
    ) -> None:
        """Should split type and deprecate source."""
        source = make_type(canonical_name="photo")
        db_session.add(source)
        await db_session.flush()

        service = TypeEvolutionService(db_session)
        result = await service.split_type(
            source.type_id,
            [
                {"name": "wildlife_photo"},
                {"name": "portrait_photo"},
                {"name": "product_photo"},
            ],
            keep_source=False,
        )

        assert result.success is True
        assert len(result.new_type_ids) == 3
        assert source.status == TypeStatus.DEPRECATED

        # Check new types were created
        for new_id in result.new_type_ids:
            new_type = await db_session.get(Type, new_id)
            assert new_type is not None
            assert new_type.status == TypeStatus.PROVISIONAL

    async def test_split_success_keep_source_as_parent(
        self,
        db_session: AsyncSession,
        make_type: MakeType,
    ) -> None:
        """Should split type and keep source as parent."""
        source = make_type(canonical_name="photo")
        db_session.add(source)
        await db_session.flush()

        service = TypeEvolutionService(db_session)
        result = await service.split_type(
            source.type_id,
            [{"name": "wildlife_photo"}, {"name": "portrait_photo"}],
            keep_source=True,
        )

        assert result.success is True
        assert source.status == TypeStatus.PROVISIONAL  # Not deprecated

        # New types should have source as parent
        for new_id in result.new_type_ids:
            new_type = await db_session.get(Type, new_id)
            assert new_type.parent_type_id == source.type_id

    async def test_split_empty_types_fails(
        self,
        db_session: AsyncSession,
        make_type: MakeType,
    ) -> None:
        """Should fail when no new types provided."""
        source = make_type(canonical_name="photo")
        db_session.add(source)
        await db_session.flush()

        service = TypeEvolutionService(db_session)
        result = await service.split_type(source.type_id, [])

        assert result.success is False

    async def test_split_duplicate_name_fails(
        self,
        db_session: AsyncSession,
        make_type: MakeType,
    ) -> None:
        """Should fail when new type name already exists."""
        source = make_type(canonical_name="photo")
        existing = make_type(canonical_name="wildlife_photo")
        db_session.add_all([source, existing])
        await db_session.flush()

        service = TypeEvolutionService(db_session)
        result = await service.split_type(
            source.type_id,
            [{"name": "wildlife_photo"}],  # Already exists
        )

        assert result.success is False
        assert "already exists" in result.reason.lower()

    async def test_split_creates_history(
        self,
        db_session: AsyncSession,
        make_type: MakeType,
    ) -> None:
        """Should create TypeEvolution history record."""
        source = make_type(canonical_name="photo")
        db_session.add(source)
        await db_session.flush()

        service = TypeEvolutionService(db_session)
        await service.split_type(
            source.type_id,
            [{"name": "wildlife_photo"}, {"name": "portrait_photo"}],
        )

        evolutions = (
            (
                await db_session.execute(
                    select(TypeEvolution).where(TypeEvolution.source_type_id == source.type_id)
                )
            )
            .scalars()
            .all()
        )
        assert len(evolutions) == 1
        assert evolutions[0].kind == TypeEvolutionKind.SPLIT
        assert "wildlife_photo" in evolutions[0].details["new_types"]


# -----------------------------------------------------------------------------
# DEPRECATE TESTS
# -----------------------------------------------------------------------------


class TestDeprecateType:
    """Tests for the deprecate_type operation."""

    async def test_deprecate_without_replacement(
        self,
        db_session: AsyncSession,
        make_type: MakeType,
    ) -> None:
        """Should deprecate type without migrating."""
        type_obj = make_type(canonical_name="old_type")
        db_session.add(type_obj)
        await db_session.flush()

        service = TypeEvolutionService(db_session)
        result = await service.deprecate_type(type_obj.type_id)

        assert result.success is True
        assert type_obj.status == TypeStatus.DEPRECATED
        assert result.replacement_type_id is None

    async def test_deprecate_with_replacement(
        self,
        db_session: AsyncSession,
        make_type: MakeType,
        make_observation_with_type: MakeObservationWithType,
    ) -> None:
        """Should deprecate and migrate to replacement."""
        old_type = make_type(canonical_name="old_type", evidence_count=5)
        new_type = make_type(canonical_name="new_type", evidence_count=10)
        obs = make_observation_with_type(stable_type_id=old_type.type_id)
        db_session.add_all([old_type, new_type, obs])
        await db_session.flush()

        service = TypeEvolutionService(db_session)
        result = await service.deprecate_type(
            old_type.type_id,
            replacement_type_id=new_type.type_id,
        )

        assert result.success is True
        assert result.observations_migrated == 1
        assert old_type.status == TypeStatus.DEPRECATED
        assert new_type.evidence_count == 15

        await db_session.refresh(obs)
        assert obs.stable_type_id == new_type.type_id

    async def test_deprecate_already_deprecated(
        self,
        db_session: AsyncSession,
        make_type: MakeType,
    ) -> None:
        """Should fail when type already deprecated."""
        type_obj = make_type(
            canonical_name="dead_type",
            status=TypeStatus.DEPRECATED,
        )
        db_session.add(type_obj)
        await db_session.flush()

        service = TypeEvolutionService(db_session)
        result = await service.deprecate_type(type_obj.type_id)

        assert result.success is False

    async def test_deprecate_into_deprecated_fails(
        self,
        db_session: AsyncSession,
        make_type: MakeType,
    ) -> None:
        """Should fail when replacement is deprecated."""
        old_type = make_type(canonical_name="old_type")
        bad_replacement = make_type(
            canonical_name="also_dead",
            status=TypeStatus.DEPRECATED,
        )
        db_session.add_all([old_type, bad_replacement])
        await db_session.flush()

        service = TypeEvolutionService(db_session)
        result = await service.deprecate_type(
            old_type.type_id,
            replacement_type_id=bad_replacement.type_id,
        )

        assert result.success is False


# -----------------------------------------------------------------------------
# UTILITY TESTS
# -----------------------------------------------------------------------------


class TestResolveType:
    """Tests for the resolve_type utility."""

    async def test_resolve_simple(
        self,
        db_session: AsyncSession,
        make_type: MakeType,
    ) -> None:
        """Should return the type directly if not merged."""
        type_obj = make_type(canonical_name="active_type")
        db_session.add(type_obj)
        await db_session.flush()

        service = TypeEvolutionService(db_session)
        resolved = await service.resolve_type(type_obj.type_id)

        assert resolved is not None
        assert resolved.type_id == type_obj.type_id

    async def test_resolve_follows_merge_chain(
        self,
        db_session: AsyncSession,
        make_type: MakeType,
    ) -> None:
        """Should follow merge chain to find current type."""
        # Create chain: A -> B -> C
        type_c = make_type(canonical_name="type_c", status=TypeStatus.STABLE)
        type_b = make_type(
            canonical_name="type_b",
            status=TypeStatus.MERGED_INTO,
        )
        type_a = make_type(
            canonical_name="type_a",
            status=TypeStatus.MERGED_INTO,
        )
        db_session.add_all([type_a, type_b, type_c])
        await db_session.flush()

        type_b.merged_into_type_id = type_c.type_id
        type_a.merged_into_type_id = type_b.type_id
        await db_session.flush()

        service = TypeEvolutionService(db_session)
        resolved = await service.resolve_type(type_a.type_id)

        assert resolved is not None
        assert resolved.type_id == type_c.type_id


class TestFindTypeByAlias:
    """Tests for the find_type_by_alias utility."""

    async def test_find_by_canonical_name(
        self,
        db_session: AsyncSession,
        make_type: MakeType,
    ) -> None:
        """Should find type by canonical name."""
        type_obj = make_type(canonical_name="wildlife_photo")
        db_session.add(type_obj)
        await db_session.flush()

        service = TypeEvolutionService(db_session)
        found = await service.find_type_by_alias("wildlife_photo")

        assert found is not None
        assert found.type_id == type_obj.type_id

    async def test_find_by_alias(
        self,
        db_session: AsyncSession,
        make_type: MakeType,
    ) -> None:
        """Should find type by alias."""
        type_obj = make_type(
            canonical_name="wildlife_photo",
            aliases=["nature_image", "animal_pic"],
        )
        db_session.add(type_obj)
        await db_session.flush()

        service = TypeEvolutionService(db_session)
        found = await service.find_type_by_alias("nature_image")

        assert found is not None
        assert found.type_id == type_obj.type_id

    async def test_find_case_insensitive(
        self,
        db_session: AsyncSession,
        make_type: MakeType,
    ) -> None:
        """Should find type case-insensitively."""
        type_obj = make_type(canonical_name="wildlife_photo")
        db_session.add(type_obj)
        await db_session.flush()

        service = TypeEvolutionService(db_session)
        found = await service.find_type_by_alias("WILDLIFE_PHOTO")

        assert found is not None
        assert found.type_id == type_obj.type_id

    async def test_find_skips_deprecated(
        self,
        db_session: AsyncSession,
        make_type: MakeType,
    ) -> None:
        """Should not find deprecated types."""
        type_obj = make_type(
            canonical_name="dead_type",
            status=TypeStatus.DEPRECATED,
        )
        db_session.add(type_obj)
        await db_session.flush()

        service = TypeEvolutionService(db_session)
        found = await service.find_type_by_alias("dead_type")

        assert found is None


class TestGetEvolutionHistory:
    """Tests for the get_evolution_history utility."""

    async def test_get_history(
        self,
        db_session: AsyncSession,
        make_type: MakeType,
    ) -> None:
        """Should return evolution history."""
        type_obj = make_type(canonical_name="evolving_type")
        db_session.add(type_obj)
        await db_session.flush()

        service = TypeEvolutionService(db_session)

        # Add some aliases to create history
        await service.add_alias(type_obj.type_id, "alias1")
        await service.add_alias(type_obj.type_id, "alias2")

        history = await service.get_evolution_history(type_obj.type_id)

        assert len(history) == 2
        # Both should be alias events (order may vary due to same timestamp)
        aliases_in_history = {h.details["alias"] for h in history}
        assert aliases_in_history == {"alias1", "alias2"}
