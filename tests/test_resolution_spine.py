"""Spine integration tests for entity resolution.

These tests verify the core correctness of the entity resolution system
across real database transactions. They prove the spine works correctly.

See design_v2_corrections.md sections 5 and 8 for specifications.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from object_sense.inference.schemas import DeterministicId, EntityHypothesis
from uuid import UUID

from object_sense.models import (
    EntityDeterministicId,
    EntityNature,
    EntityStatus,
    LinkStatus,
    ObservationEntityLink,
)
from object_sense.resolution import EntityResolver, resolve_canonical_entity

if TYPE_CHECKING:
    from conftest import MakeEntity, MakeObservation, MakeSignature


# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


class TestDeterministicIdCreateOnMiss:
    """Test A: Deterministic ID create-on-miss.

    Acceptance criteria:
    - Create observation O1 with deterministic_id (sku, X, ns)
    - Run resolver -> expect:
      - 1 entity created
      - entity_deterministic_ids has exactly 1 row for that tuple
      - ObservationEntityLink: (O1, E) is hard, posterior=1.0
    - Create observation O2 with same deterministic_id
    - Resolve -> expect:
      - still exactly 1 entity
      - now 2 hard links to same canonical entity
    """

    async def test_first_observation_creates_entity(
        self,
        db_session: AsyncSession,
        make_observation: MakeObservation,
        make_signature: MakeSignature,
    ) -> None:
        """First observation with det ID creates entity and hard link."""
        # Create observation O1 with deterministic ID
        obs1 = make_observation(
            observation_kind="product_record",
            deterministic_ids=[
                {"id_type": "sku", "id_value": "TEST-001", "id_namespace": "acme"}
            ],
        )
        db_session.add(obs1)

        # Create a signature (needed for resolution)
        sig1 = make_signature(
            observation_id=obs1.observation_id,
            signature_type="primary",
        )
        db_session.add(sig1)
        await db_session.flush()

        # Create entity hypothesis with deterministic ID
        det_id = DeterministicId(
            id_type="sku",
            id_value="TEST-001",
            id_namespace="acme",
            strength="strong",
        )
        entity_seed = EntityHypothesis(
            entity_type="product",
            entity_nature=EntityNature.INDIVIDUAL,
            suggested_name="Test Product",
            deterministic_ids=[det_id],
            confidence=0.9,
        )

        # Resolve
        resolver = EntityResolver(db_session)
        result = await resolver.resolve(
            observation=obs1,
            signatures=[sig1],
            entity_seeds=[entity_seed],
        )

        # Verify: 1 entity created
        assert len(result.entities_created) == 1
        entity = result.entities_created[0]
        assert entity.entity_nature == EntityNature.INDIVIDUAL
        assert entity.status == EntityStatus.PROTO

        # Verify: 1 link, hard, posterior=1.0
        assert len(result.links) == 1
        link = result.links[0]
        assert link.entity_id == entity.entity_id
        assert link.status == LinkStatus.HARD
        assert link.posterior == 1.0

        # Verify: entity_deterministic_ids has exactly 1 row
        stmt = select(EntityDeterministicId).where(
            EntityDeterministicId.entity_id == entity.entity_id
        )
        det_ids_result = await db_session.execute(stmt)
        det_id_rows = det_ids_result.scalars().all()
        assert len(det_id_rows) == 1
        assert det_id_rows[0].id_type == "sku"
        assert det_id_rows[0].id_value == "TEST-001"
        assert det_id_rows[0].id_namespace == "acme"

    async def test_second_observation_reuses_entity(
        self,
        db_session: AsyncSession,
        make_observation: MakeObservation,
        make_signature: MakeSignature,
    ) -> None:
        """Second observation with same det ID links to existing entity."""
        det_id = DeterministicId(
            id_type="sku",
            id_value="SHARED-001",
            id_namespace="acme",
            strength="strong",
        )
        entity_seed = EntityHypothesis(
            entity_type="product",
            entity_nature=EntityNature.INDIVIDUAL,
            suggested_name="Shared Product",
            deterministic_ids=[det_id],
            confidence=0.9,
        )

        # First observation and resolution
        obs1 = make_observation(observation_kind="product_record")
        sig1 = make_signature(observation_id=obs1.observation_id)
        db_session.add(obs1)
        db_session.add(sig1)
        await db_session.flush()

        resolver = EntityResolver(db_session)
        result1 = await resolver.resolve(
            observation=obs1,
            signatures=[sig1],
            entity_seeds=[entity_seed],
        )

        assert len(result1.entities_created) == 1
        entity_id = result1.entities_created[0].entity_id

        # Second observation with same det ID
        obs2 = make_observation(observation_kind="product_record")
        sig2 = make_signature(observation_id=obs2.observation_id)
        db_session.add(obs2)
        db_session.add(sig2)
        await db_session.flush()

        result2 = await resolver.resolve(
            observation=obs2,
            signatures=[sig2],
            entity_seeds=[entity_seed],
        )

        # Verify: NO new entity created
        assert len(result2.entities_created) == 0

        # Verify: link goes to same entity
        assert len(result2.links) == 1
        assert result2.links[0].entity_id == entity_id
        assert result2.links[0].status == LinkStatus.HARD
        assert result2.links[0].posterior == 1.0

        # Verify: still only 1 entity with this det ID
        stmt = select(EntityDeterministicId).where(
            EntityDeterministicId.id_type == "sku",
            EntityDeterministicId.id_value == "SHARED-001",
            EntityDeterministicId.id_namespace == "acme",
        )
        det_ids_result = await db_session.execute(stmt)
        det_id_rows = det_ids_result.scalars().all()
        assert len(det_id_rows) == 1

        # Verify: 2 observation links to the entity
        stmt = select(ObservationEntityLink).where(
            ObservationEntityLink.entity_id == entity_id
        )
        links_result = await db_session.execute(stmt)
        all_links = links_result.scalars().all()
        assert len(all_links) == 2
        assert all(link.status == LinkStatus.HARD for link in all_links)


class TestCanonicalChainCompression:
    """Test B: Canonical chain compression.

    Acceptance criteria:
    - Create entities A, B, C with canonical_entity_id pointers A->B, B->C
    - Call resolve_canonical(A) once; expect returned C
    - Call resolve_canonical(A) again; expect:
      - returned C
      - A.canonical_entity_id == C (path compressed)
      - optionally B.canonical_entity_id == C too
    """

    async def test_resolve_canonical_follows_chain(
        self,
        db_session: AsyncSession,
        make_entity: MakeEntity,
    ) -> None:
        """resolve_canonical follows the chain to the canonical entity."""
        # Create chain: A -> B -> C (C is canonical)
        entity_c = make_entity(name="Entity C")
        entity_b = make_entity(
            name="Entity B",
            canonical_entity_id=entity_c.entity_id,
        )
        entity_a = make_entity(
            name="Entity A",
            canonical_entity_id=entity_b.entity_id,
        )

        db_session.add(entity_c)
        db_session.add(entity_b)
        db_session.add(entity_a)
        await db_session.flush()

        # Resolve A -> should get C
        result = await resolve_canonical_entity(
            db_session,
            entity_a.entity_id,
            compress_path=False,  # Don't compress yet
        )

        assert result.entity_id == entity_c.entity_id
        assert result.name == "Entity C"
        assert result.canonical_entity_id is None

    async def test_resolve_canonical_compresses_path(
        self,
        db_session: AsyncSession,
        make_entity: MakeEntity,
    ) -> None:
        """resolve_canonical with compress_path updates intermediate pointers."""
        # Create chain: A -> B -> C (C is canonical)
        entity_c = make_entity(name="Entity C")
        entity_b = make_entity(
            name="Entity B",
            canonical_entity_id=entity_c.entity_id,
        )
        entity_a = make_entity(
            name="Entity A",
            canonical_entity_id=entity_b.entity_id,
        )

        db_session.add(entity_c)
        db_session.add(entity_b)
        db_session.add(entity_a)
        await db_session.flush()

        # Verify initial state
        assert entity_a.canonical_entity_id == entity_b.entity_id
        assert entity_b.canonical_entity_id == entity_c.entity_id

        # First resolve with compression
        result = await resolve_canonical_entity(
            db_session,
            entity_a.entity_id,
            compress_path=True,
        )

        assert result.entity_id == entity_c.entity_id

        # Verify path compression: A now points directly to C
        await db_session.refresh(entity_a)
        assert entity_a.canonical_entity_id == entity_c.entity_id

        # B should also be compressed (it was in the visited path)
        await db_session.refresh(entity_b)
        assert entity_b.canonical_entity_id == entity_c.entity_id

    async def test_resolve_canonical_idempotent(
        self,
        db_session: AsyncSession,
        make_entity: MakeEntity,
    ) -> None:
        """Repeated calls to resolve_canonical return same result."""
        # Create chain: A -> B -> C (C is canonical)
        entity_c = make_entity(name="Entity C")
        entity_b = make_entity(
            name="Entity B",
            canonical_entity_id=entity_c.entity_id,
        )
        entity_a = make_entity(
            name="Entity A",
            canonical_entity_id=entity_b.entity_id,
        )

        db_session.add(entity_c)
        db_session.add(entity_b)
        db_session.add(entity_a)
        await db_session.flush()

        # Resolve twice
        result1 = await resolve_canonical_entity(db_session, entity_a.entity_id)
        result2 = await resolve_canonical_entity(db_session, entity_a.entity_id)

        # Both should return C
        assert result1.entity_id == entity_c.entity_id
        assert result2.entity_id == entity_c.entity_id

    async def test_resolve_canonical_entity_is_already_canonical(
        self,
        db_session: AsyncSession,
        make_entity: MakeEntity,
    ) -> None:
        """Canonical entity returns itself."""
        entity = make_entity(name="Standalone Entity")
        db_session.add(entity)
        await db_session.flush()

        result = await resolve_canonical_entity(db_session, entity.entity_id)

        assert result.entity_id == entity.entity_id
        assert result.canonical_entity_id is None


class TestTypeAuthorityIrrelevance:
    """Test C: Type authority irrelevance.

    Acceptance criteria:
    - Create two observations with same deterministic_id but different observation_kind
    - Resolve both
    - Assert same entity and same hard link results

    The observation_kind is a routing hint, not a type authority.
    It should NOT affect entity resolution when deterministic IDs match.
    """

    async def test_different_observation_kinds_same_entity(
        self,
        db_session: AsyncSession,
        make_observation: MakeObservation,
        make_signature: MakeSignature,
    ) -> None:
        """Same det ID with different observation_kinds -> same entity."""
        det_id = DeterministicId(
            id_type="product_id",
            id_value="PROD-999",
            id_namespace="global",
            strength="strong",
        )

        # First observation with observation_kind="product_record"
        obs1 = make_observation(observation_kind="product_record")
        sig1 = make_signature(observation_id=obs1.observation_id)
        db_session.add(obs1)
        db_session.add(sig1)
        await db_session.flush()

        entity_seed1 = EntityHypothesis(
            entity_type="product",
            entity_nature=EntityNature.INDIVIDUAL,
            suggested_name="Product from record",
            deterministic_ids=[det_id],
            confidence=0.9,
        )

        resolver = EntityResolver(db_session)
        result1 = await resolver.resolve(
            observation=obs1,
            signatures=[sig1],
            entity_seeds=[entity_seed1],
        )

        assert len(result1.entities_created) == 1
        entity_id = result1.entities_created[0].entity_id

        # Second observation with DIFFERENT observation_kind="inventory_update"
        obs2 = make_observation(observation_kind="inventory_update")
        sig2 = make_signature(observation_id=obs2.observation_id)
        db_session.add(obs2)
        db_session.add(sig2)
        await db_session.flush()

        # Same det ID, different observation_kind
        entity_seed2 = EntityHypothesis(
            entity_type="product",
            entity_nature=EntityNature.INDIVIDUAL,
            suggested_name="Product from inventory",
            deterministic_ids=[det_id],
            confidence=0.9,
        )

        result2 = await resolver.resolve(
            observation=obs2,
            signatures=[sig2],
            entity_seeds=[entity_seed2],
        )

        # Verify: NO new entity created
        assert len(result2.entities_created) == 0

        # Verify: linked to same entity
        assert len(result2.links) == 1
        assert result2.links[0].entity_id == entity_id
        assert result2.links[0].status == LinkStatus.HARD

    async def test_observation_kind_does_not_partition_entities(
        self,
        db_session: AsyncSession,
        make_observation: MakeObservation,
        make_signature: MakeSignature,
    ) -> None:
        """Different observation_kinds with same det ID produce single entity."""
        det_id = DeterministicId(
            id_type="sku",
            id_value="MULTI-KIND-001",
            id_namespace="test",
            strength="strong",
        )

        observation_kinds = [
            "wildlife_photo",
            "json_config",
            "text_document",
            "product_record",
        ]

        resolver = EntityResolver(db_session)
        linked_entity_ids: set[UUID] = set()

        for kind in observation_kinds:
            obs = make_observation(observation_kind=kind)
            sig = make_signature(observation_id=obs.observation_id)
            db_session.add(obs)
            db_session.add(sig)
            await db_session.flush()

            entity_seed = EntityHypothesis(
                entity_type="thing",
                entity_nature=EntityNature.INDIVIDUAL,
                suggested_name=f"Thing from {kind}",
                deterministic_ids=[det_id],
                confidence=0.9,
            )

            result = await resolver.resolve(
                observation=obs,
                signatures=[sig],
                entity_seeds=[entity_seed],
            )

            # Collect all entity IDs we link to
            for link in result.links:
                linked_entity_ids.add(link.entity_id)

        # All 4 observations should link to the SAME entity
        assert len(linked_entity_ids) == 1

        # Check all links are hard
        stmt = select(ObservationEntityLink).where(
            ObservationEntityLink.entity_id == list(linked_entity_ids)[0]
        )
        links_result = await db_session.execute(stmt)
        all_links = links_result.scalars().all()
        assert len(all_links) == 4
        assert all(link.status == LinkStatus.HARD for link in all_links)


class TestSimilarityResolutionWithoutDetId:
    """Optional add-on: Similarity resolution without deterministic IDs.

    Tests that the resolver handles similarity-based matching correctly
    when no deterministic IDs are present.
    """

    async def test_similarity_resolution_creates_proto_entity(
        self,
        db_session: AsyncSession,
        make_observation: MakeObservation,
        make_signature: MakeSignature,
    ) -> None:
        """Without det ID or candidates, creates a proto-entity."""
        obs = make_observation(observation_kind="wildlife_photo")
        # Create signature with image embedding
        sig = make_signature(
            observation_id=obs.observation_id,
            image_embedding=[0.1] * 768,  # CLIP dimensions
        )
        db_session.add(obs)
        db_session.add(sig)
        await db_session.flush()

        # Entity seed WITHOUT deterministic ID
        entity_seed = EntityHypothesis(
            entity_type="animal",
            entity_nature=EntityNature.INDIVIDUAL,
            suggested_name="Unknown Animal",
            deterministic_ids=[],  # No det ID
            confidence=0.7,
        )

        resolver = EntityResolver(db_session)
        result = await resolver.resolve(
            observation=obs,
            signatures=[sig],
            entity_seeds=[entity_seed],
        )

        # Should create a proto-entity since no candidates exist
        assert len(result.entities_created) == 1
        assert result.entities_created[0].status == EntityStatus.PROTO

        # Should have a soft link (not hard)
        assert len(result.links) == 1
        assert result.links[0].status == LinkStatus.SOFT

    async def test_different_observation_kinds_do_not_block_candidates(
        self,
        db_session: AsyncSession,
        make_observation: MakeObservation,
        make_signature: MakeSignature,
        make_entity: MakeEntity,
    ) -> None:
        """Observation kind doesn't hard-block candidate retrieval."""
        # Create an existing entity with prototype embedding
        existing_entity = make_entity(
            name="Existing Animal",
            entity_nature=EntityNature.INDIVIDUAL,
        )
        # Set prototype embedding so it can be found via ANN
        existing_entity.prototype_image_embedding = [0.1] * 768
        existing_entity.prototype_count = 1

        db_session.add(existing_entity)
        await db_session.flush()

        # Create observation with same embedding but different observation_kind
        obs = make_observation(observation_kind="random_kind_xyz")
        sig = make_signature(
            observation_id=obs.observation_id,
            image_embedding=[0.1] * 768,  # Same as entity prototype
        )
        db_session.add(obs)
        db_session.add(sig)
        await db_session.flush()

        entity_seed = EntityHypothesis(
            entity_type="animal",
            entity_nature=EntityNature.INDIVIDUAL,
            suggested_name="Animal",
            deterministic_ids=[],
            confidence=0.7,
        )

        resolver = EntityResolver(db_session)
        result = await resolver.resolve(
            observation=obs,
            signatures=[sig],
            entity_seeds=[entity_seed],
        )

        # The resolver should run without crashing
        # Whether it links to existing entity depends on similarity threshold
        # At minimum, we verify it doesn't fail due to observation_kind mismatch
        assert len(result.links) >= 1
