"""Tests for slot reference resolution (SlotResolver).

Tests the symbolic-to-canonical slot reference upgrade mechanism.
"""

import pytest
from uuid import uuid4

from object_sense.inference.schemas import DeterministicId, EntityHypothesis, SlotValue
from object_sense.models import Entity, EntityNature, EntityStatus, Medium
from object_sense.models.entity_deterministic_id import EntityDeterministicId
from object_sense.resolution.candidate_pool import CandidatePoolService
from object_sense.resolution.slot_resolver import (
    SlotResolver,
    parse_ref_name,
    _normalize_lexical,
    ParsedRef,
)


class TestParseRefName:
    """Test parse_ref_name function."""

    def test_simple_ref(self):
        """Parse ref:<value> format."""
        result = parse_ref_name("ref:leopard_species")
        assert result is not None
        assert result.namespace == "ref"
        assert result.value == "leopard_species"
        assert result.raw == "ref:leopard_species"

    def test_namespaced_ref(self):
        """Parse ref:<namespace>:<value> format."""
        result = parse_ref_name("ref:wildlife:african_leopard")
        assert result is not None
        assert result.namespace == "wildlife"
        assert result.value == "african_leopard"
        assert result.raw == "ref:wildlife:african_leopard"

    def test_invalid_no_prefix(self):
        """Reject refs without ref: prefix."""
        result = parse_ref_name("just_a_name")
        assert result is None

    def test_invalid_wrong_prefix(self):
        """Reject refs with wrong prefix."""
        result = parse_ref_name("REF:leopard")  # Case-sensitive
        assert result is None

    def test_normalization(self):
        """Values are normalized (lowercase, underscores)."""
        result = parse_ref_name("ref:African Leopard")
        assert result is not None
        assert result.value == "african_leopard"

    def test_normalization_with_namespace(self):
        """Both namespace and value are normalized."""
        result = parse_ref_name("ref:Wildlife Catalog:African Leopard")
        assert result is not None
        assert result.namespace == "wildlife_catalog"
        assert result.value == "african_leopard"

    def test_collapse_underscores(self):
        """Multiple underscores are collapsed."""
        result = parse_ref_name("ref:leopard___species")
        assert result is not None
        assert result.value == "leopard_species"


class TestNormalizeLexical:
    """Test _normalize_lexical function."""

    def test_lowercase(self):
        assert _normalize_lexical("UPPERCASE") == "uppercase"

    def test_trim(self):
        assert _normalize_lexical("  trimmed  ") == "trimmed"

    def test_replace_special_chars(self):
        assert _normalize_lexical("hello-world") == "hello_world"
        assert _normalize_lexical("hello.world") == "hello_world"
        assert _normalize_lexical("hello world") == "hello_world"

    def test_collapse_underscores(self):
        assert _normalize_lexical("a___b") == "a_b"

    def test_strip_trailing_underscores(self):
        assert _normalize_lexical("_leading_") == "leading"


@pytest.mark.integration
class TestSlotResolverIntegration:
    """Integration tests for SlotResolver with database."""

    async def test_resolve_to_existing_entity(
        self,
        db_session,
        make_entity,
    ):
        """Resolve ref to an existing entity via deterministic ID."""
        # Create a CLASS entity with a ref deterministic ID
        species_entity = make_entity(
            entity_nature=EntityNature.CLASS,
            name="leopard",
        )
        db_session.add(species_entity)

        ref_binding = EntityDeterministicId(
            id=uuid4(),
            entity_id=species_entity.entity_id,
            id_type="ref",
            id_namespace="ref",
            id_value="leopard_species",
            confidence=1.0,
        )
        db_session.add(ref_binding)
        await db_session.flush()

        # Create an entity with a symbolic ref slot
        entity = make_entity(
            entity_nature=EntityNature.INDIVIDUAL,
            name="photo_subject",
        )
        entity.slots = {
            "species": {
                "type": "reference",
                "ref_name": "ref:leopard_species",
                "ref_nature": "class",
            }
        }
        db_session.add(entity)
        await db_session.flush()

        # Resolve slot references
        pool = CandidatePoolService(db_session)
        resolver = SlotResolver(db_session, pool)
        result = await resolver.resolve_entity_slots([entity])

        # Verify resolution
        assert result.resolved_count == 1
        assert result.created_count == 0
        assert result.unresolved_count == 0

        # Check slot was upgraded
        species_slot = entity.slots["species"]
        assert "ref_entity_id" in species_slot
        assert species_slot["ref_entity_id"] == str(species_entity.entity_id)
        assert species_slot["ref_raw_name"] == "ref:leopard_species"
        assert species_slot["ref_debug_name"] == "leopard_species"
        assert species_slot["ref_namespace"] == "ref"
        # Original fields preserved
        assert species_slot["type"] == "reference"
        assert species_slot["ref_nature"] == "class"

    async def test_create_on_miss(
        self,
        db_session,
        make_entity,
    ):
        """Create proto entity when ref doesn't exist."""
        entity = make_entity(
            entity_nature=EntityNature.INDIVIDUAL,
            name="photo_subject",
        )
        entity.slots = {
            "species": {
                "type": "reference",
                "ref_name": "ref:unknown_species",
                "ref_nature": "class",
            }
        }
        db_session.add(entity)
        await db_session.flush()

        pool = CandidatePoolService(db_session)
        resolver = SlotResolver(db_session, pool)
        result = await resolver.resolve_entity_slots([entity])

        # Verify creation
        assert result.resolved_count == 0
        assert result.created_count == 1
        assert result.unresolved_count == 0

        # Check slot was upgraded
        species_slot = entity.slots["species"]
        assert "ref_entity_id" in species_slot
        assert species_slot["ref_debug_name"] == "unknown_species"

        # Verify evidence was created
        assert len(result.evidence) == 1
        assert result.evidence[0].predicate == "slot_ref_created"

        # Verify the created entity exists
        from sqlalchemy import select
        stmt = select(Entity).where(Entity.name == "unknown_species")
        created = (await db_session.execute(stmt)).scalar_one_or_none()
        assert created is not None
        assert created.entity_nature == EntityNature.CLASS
        assert created.status == EntityStatus.PROTO
        assert created.confidence == 0.7

    async def test_idempotent_resolution(
        self,
        db_session,
        make_entity,
    ):
        """Running resolution twice produces same result."""
        entity = make_entity(
            entity_nature=EntityNature.INDIVIDUAL,
            name="photo_subject",
        )
        entity.slots = {
            "species": {
                "type": "reference",
                "ref_name": "ref:leopard",
                "ref_nature": "class",
            }
        }
        db_session.add(entity)
        await db_session.flush()

        pool = CandidatePoolService(db_session)
        resolver = SlotResolver(db_session, pool)

        # First resolution
        result1 = await resolver.resolve_entity_slots([entity])
        assert result1.created_count == 1
        first_ref_id = entity.slots["species"]["ref_entity_id"]

        # Second resolution (same entity)
        result2 = await resolver.resolve_entity_slots([entity])
        # Should skip already-resolved slots
        assert result2.resolved_count == 0
        assert result2.created_count == 0
        # ref_entity_id unchanged
        assert entity.slots["species"]["ref_entity_id"] == first_ref_id

    async def test_invalid_ref_name_format(
        self,
        db_session,
        make_entity,
    ):
        """Reject refs without ref: prefix."""
        entity = make_entity()
        entity.slots = {
            "species": {
                "type": "reference",
                "ref_name": "leopard",  # Missing ref: prefix
                "ref_nature": "class",
            }
        }
        db_session.add(entity)
        await db_session.flush()

        pool = CandidatePoolService(db_session)
        resolver = SlotResolver(db_session, pool)
        result = await resolver.resolve_entity_slots([entity])

        assert result.unresolved_count == 1
        assert len(result.evidence) == 1
        assert result.evidence[0].predicate == "slot_ref_unresolved"
        assert result.evidence[0].details["reason"] == "invalid_ref_name"

    async def test_missing_ref_nature(
        self,
        db_session,
        make_entity,
    ):
        """Reject refs without ref_nature (and no fallback)."""
        entity = make_entity()
        entity.slots = {
            "unknown_slot": {  # Not in fallback list
                "type": "reference",
                "ref_name": "ref:something",
                # No ref_nature
            }
        }
        db_session.add(entity)
        await db_session.flush()

        pool = CandidatePoolService(db_session)
        resolver = SlotResolver(db_session, pool)
        result = await resolver.resolve_entity_slots([entity])

        assert result.unresolved_count == 1
        assert result.evidence[0].details["reason"] == "missing_ref_nature"

    async def test_fallback_ref_nature(
        self,
        db_session,
        make_entity,
    ):
        """Use fallback ref_nature for known slots."""
        entity = make_entity()
        entity.slots = {
            "species": {  # In fallback list
                "type": "reference",
                "ref_name": "ref:leopard",
                # No ref_nature - should use fallback
            }
        }
        db_session.add(entity)
        await db_session.flush()

        pool = CandidatePoolService(db_session)
        resolver = SlotResolver(db_session, pool)
        result = await resolver.resolve_entity_slots([entity])

        # Should succeed with fallback
        assert result.created_count == 1
        assert result.unresolved_count == 0

    async def test_nature_mismatch(
        self,
        db_session,
        make_entity,
    ):
        """Reject resolution when target entity has wrong nature."""
        # Create an INDIVIDUAL entity (not CLASS)
        wrong_entity = make_entity(
            entity_nature=EntityNature.INDIVIDUAL,  # Wrong nature
            name="wrong_type",
        )
        db_session.add(wrong_entity)

        ref_binding = EntityDeterministicId(
            id=uuid4(),
            entity_id=wrong_entity.entity_id,
            id_type="ref",
            id_namespace="ref",
            id_value="leopard",
            confidence=1.0,
        )
        db_session.add(ref_binding)
        await db_session.flush()

        entity = make_entity()
        entity.slots = {
            "species": {
                "type": "reference",
                "ref_name": "ref:leopard",
                "ref_nature": "class",  # Expects CLASS
            }
        }
        db_session.add(entity)
        await db_session.flush()

        pool = CandidatePoolService(db_session)
        resolver = SlotResolver(db_session, pool)
        result = await resolver.resolve_entity_slots([entity])

        # Should fail with nature conflict
        assert result.unresolved_count == 1
        assert result.evidence[0].predicate == "slot_ref_nature_conflict"
        assert result.evidence[0].details["reason"] == "nature_mismatch"

    async def test_namespaced_refs(
        self,
        db_session,
        make_entity,
    ):
        """Handle ref:<namespace>:<value> format."""
        # Create entity with custom namespace
        species_entity = make_entity(
            entity_nature=EntityNature.CLASS,
            name="african_leopard",
        )
        db_session.add(species_entity)

        ref_binding = EntityDeterministicId(
            id=uuid4(),
            entity_id=species_entity.entity_id,
            id_type="ref",
            id_namespace="wildlife",  # Custom namespace
            id_value="african_leopard",
            confidence=1.0,
        )
        db_session.add(ref_binding)
        await db_session.flush()

        entity = make_entity()
        entity.slots = {
            "species": {
                "type": "reference",
                "ref_name": "ref:wildlife:african_leopard",  # Namespaced ref
                "ref_nature": "class",
            }
        }
        db_session.add(entity)
        await db_session.flush()

        pool = CandidatePoolService(db_session)
        resolver = SlotResolver(db_session, pool)
        result = await resolver.resolve_entity_slots([entity])

        assert result.resolved_count == 1
        assert entity.slots["species"]["ref_namespace"] == "wildlife"

    async def test_merge_chain_resolution(
        self,
        db_session,
        make_entity,
    ):
        """Follow canonical_entity_id chain to canonical entity."""
        # Create a chain: merged_entity -> canonical_entity
        canonical = make_entity(
            entity_nature=EntityNature.CLASS,
            name="canonical_leopard",
        )
        db_session.add(canonical)
        await db_session.flush()

        merged = make_entity(
            entity_nature=EntityNature.CLASS,
            name="merged_leopard",
            canonical_entity_id=canonical.entity_id,  # Merged into canonical
        )
        db_session.add(merged)

        # Det ID points to merged entity
        ref_binding = EntityDeterministicId(
            id=uuid4(),
            entity_id=merged.entity_id,
            id_type="ref",
            id_namespace="ref",
            id_value="leopard",
            confidence=1.0,
        )
        db_session.add(ref_binding)
        await db_session.flush()

        entity = make_entity()
        entity.slots = {
            "species": {
                "type": "reference",
                "ref_name": "ref:leopard",
                "ref_nature": "class",
            }
        }
        db_session.add(entity)
        await db_session.flush()

        pool = CandidatePoolService(db_session)
        resolver = SlotResolver(db_session, pool)
        result = await resolver.resolve_entity_slots([entity])

        # Should resolve to canonical, not merged
        assert result.resolved_count == 1
        assert entity.slots["species"]["ref_entity_id"] == str(canonical.entity_id)

    async def test_skip_non_reference_slots(
        self,
        db_session,
        make_entity,
    ):
        """Ignore slots that aren't references."""
        entity = make_entity()
        entity.slots = {
            "description": {"value": "A leopard photo", "type": "string"},
            "count": {"value": 5, "type": "number"},
        }
        db_session.add(entity)
        await db_session.flush()

        pool = CandidatePoolService(db_session)
        resolver = SlotResolver(db_session, pool)
        result = await resolver.resolve_entity_slots([entity])

        # Nothing to resolve
        assert result.resolved_count == 0
        assert result.created_count == 0
        assert result.unresolved_count == 0
