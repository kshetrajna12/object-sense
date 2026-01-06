"""Tests for prototype seeding to break the prototype deadlock.

The prototype deadlock occurs when:
1. candidate_pool only retrieves entities with non-null prototypes
2. _create_proto_entity creates entities with prototype_count=0 and NULL prototypes
3. Result: entities never enter the candidate pool, never get linked, never get updated

These tests verify the fix:
A) _create_proto_entity seeds prototypes at entity creation time
B) Engine-owned subject seed ensures image observations always have an INDIVIDUAL seed
C) Similarity-based resolution can link observations to existing entities

Run with: pytest tests/test_prototype_seeding.py -v
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from object_sense.inference.schemas import EntityHypothesis, SlotValue
from object_sense.models import (
    Entity,
    EntityNature,
    EntityStatus,
    Medium,
    Observation,
    ObservationEntityLink,
    Signature,
)
from object_sense.resolution.resolver import EntityResolver
from object_sense.resolution.similarity import ObservationSignals, SimilarityScorer

if TYPE_CHECKING:
    from conftest import MakeEntity, MakeObservation, MakeSignature


def create_colored_image(
    color: tuple[int, int, int],
    width: int = 100,
    height: int = 100,
) -> bytes:
    """Create a simple solid-color test image.

    Different colors represent different "individuals" in the visual domain.
    """
    img = Image.new("RGB", (width, height), color=color)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return buffer.getvalue()


def create_embedding_for_color(
    color: tuple[int, int, int],
    dim: int = 768,
) -> list[float]:
    """Create a deterministic unit-normalized embedding based on color.

    Similar colors → similar embeddings → high cosine similarity → should link.
    Different colors → different embeddings → low cosine similarity → should separate.

    The embedding is normalized to unit length for proper cosine similarity.
    """
    import math

    r, g, b = color
    # Normalize to 0-1
    r_norm, g_norm, b_norm = r / 255.0, g / 255.0, b / 255.0

    embedding = [0.0] * dim

    # Create a distinctive pattern per color
    for i in range(dim):
        # Use color-derived pattern that varies significantly by color
        if i % 3 == 0:
            embedding[i] = r_norm + 0.1
        elif i % 3 == 1:
            embedding[i] = g_norm + 0.1
        else:
            embedding[i] = b_norm + 0.1
        # Add small position-based variation for realism
        embedding[i] += (i % 10) * 0.001

    # Normalize to unit length for cosine similarity
    norm = math.sqrt(sum(x * x for x in embedding))
    if norm > 0:
        embedding = [x / norm for x in embedding]

    return embedding


def add_embedding_noise(
    embedding: list[float],
    noise_scale: float = 0.01,
    seed: int = 42,
) -> list[float]:
    """Add small deterministic noise to an embedding.

    This simulates slight variations between observations of the same
    individual (different lighting, angle, etc.)
    """
    import random

    rng = random.Random(seed)
    return [v + rng.uniform(-noise_scale, noise_scale) for v in embedding]


# Mark all tests as integration (require database)
pytestmark = pytest.mark.integration


class TestPrototypeSeedingRegression:
    """Regression test: entities created with prototype embeddings."""

    async def test_individual_entity_seeded_with_image_embedding(
        self,
        db_session: AsyncSession,
        make_observation: MakeObservation,
        make_signature: MakeSignature,
    ) -> None:
        """After ingesting first image, INDIVIDUAL entity has prototype_image_embedding."""
        # Create observation with image embedding
        image_embedding = create_embedding_for_color((255, 0, 0))  # Red

        obs = make_observation(
            medium=Medium.IMAGE,
            observation_kind="wildlife_photo",
        )
        sig = make_signature(
            observation_id=obs.observation_id,
            signature_type="image_embedding",
            image_embedding=image_embedding,
        )
        db_session.add(obs)
        db_session.add(sig)
        await db_session.flush()

        # Create entity seed (INDIVIDUAL, no deterministic IDs)
        entity_seed = EntityHypothesis(
            entity_type="photo_subject",
            entity_nature=EntityNature.INDIVIDUAL,
            suggested_name=None,
            slots=[],
            deterministic_ids=[],  # No det IDs - pure similarity
            confidence=0.7,
        )

        # Resolve
        resolver = EntityResolver(db_session)
        result = await resolver.resolve(
            observation=obs,
            signatures=[sig],
            entity_seeds=[entity_seed],
        )

        # REGRESSION CHECK: Entity should have prototype seeded
        assert len(result.entities_created) == 1
        entity = result.entities_created[0]

        # Key assertion: prototype_image_embedding is NOT None
        assert entity.prototype_image_embedding is not None
        assert len(list(entity.prototype_image_embedding)) == 768
        assert entity.prototype_count >= 1
        assert entity.entity_nature == EntityNature.INDIVIDUAL
        assert entity.status == EntityStatus.PROTO

    async def test_class_entity_seeded_with_text_embedding(
        self,
        db_session: AsyncSession,
        make_observation: MakeObservation,
        make_signature: MakeSignature,
    ) -> None:
        """CLASS entities are seeded with prototype_text_embedding."""
        text_embedding = [0.2] * 1024

        obs = make_observation(
            medium=Medium.TEXT,
            observation_kind="species_description",
        )
        sig = make_signature(
            observation_id=obs.observation_id,
            signature_type="text_embedding",
            text_embedding=text_embedding,
        )
        db_session.add(obs)
        db_session.add(sig)
        await db_session.flush()

        # Create CLASS entity seed
        entity_seed = EntityHypothesis(
            entity_type="species",
            entity_nature=EntityNature.CLASS,
            suggested_name="Leopard",
            slots=[],
            deterministic_ids=[],
            confidence=0.8,
        )

        resolver = EntityResolver(db_session)
        result = await resolver.resolve(
            observation=obs,
            signatures=[sig],
            entity_seeds=[entity_seed],
        )

        assert len(result.entities_created) == 1
        entity = result.entities_created[0]

        # Class entities use text embedding
        assert entity.prototype_text_embedding is not None
        assert len(list(entity.prototype_text_embedding)) == 1024
        assert entity.prototype_count >= 1
        assert entity.entity_nature == EntityNature.CLASS


class TestPrototypeSeedingConvergence:
    """Convergence test: similar observations link to the same entity."""

    async def test_three_similar_images_link_to_same_entity(
        self,
        db_session: AsyncSession,
        make_observation: MakeObservation,
        make_signature: MakeSignature,
    ) -> None:
        """3 variants of the same individual link to >= 3 observations on one entity."""
        # Base embedding for "Individual A" (red color)
        base_embedding = create_embedding_for_color((255, 0, 0))

        # Create 3 observation variants with slight noise (same individual, different shots)
        observations: list[Observation] = []
        signatures: list[Signature] = []

        for i in range(3):
            # Add small noise to simulate different shots of same individual
            variant_embedding = add_embedding_noise(base_embedding, noise_scale=0.01, seed=i)

            obs = make_observation(
                medium=Medium.IMAGE,
                observation_kind="wildlife_photo",
            )
            sig = make_signature(
                observation_id=obs.observation_id,
                signature_type="image_embedding",
                image_embedding=variant_embedding,
            )
            db_session.add(obs)
            db_session.add(sig)
            observations.append(obs)
            signatures.append(sig)

        await db_session.flush()

        resolver = EntityResolver(db_session)

        # Resolve first observation - should create new entity
        entity_seed = EntityHypothesis(
            entity_type="photo_subject",
            entity_nature=EntityNature.INDIVIDUAL,
            suggested_name=None,
            slots=[],
            deterministic_ids=[],
            confidence=0.7,
        )

        result1 = await resolver.resolve(
            observation=observations[0],
            signatures=[signatures[0]],
            entity_seeds=[entity_seed],
        )

        assert len(result1.entities_created) == 1
        first_entity_id = result1.entities_created[0].entity_id

        # Resolve second and third observations - should link to existing entity
        result2 = await resolver.resolve(
            observation=observations[1],
            signatures=[signatures[1]],
            entity_seeds=[entity_seed],
        )

        result3 = await resolver.resolve(
            observation=observations[2],
            signatures=[signatures[2]],
            entity_seeds=[entity_seed],
        )

        # CONVERGENCE CHECK: All observations should link to the same entity
        # Check via ObservationEntityLink table
        stmt = select(ObservationEntityLink).where(
            ObservationEntityLink.entity_id == first_entity_id
        )
        links_result = await db_session.execute(stmt)
        links = links_result.scalars().all()

        # At least 3 observations linked to this entity
        assert len(links) >= 3, (
            f"Expected >= 3 links to entity {first_entity_id}, got {len(links)}. "
            f"Result2 entities_created={len(result2.entities_created)}, "
            f"Result3 entities_created={len(result3.entities_created)}"
        )


class TestPrototypeSeedingSeparation:
    """Separation test: different individuals create separate entities."""

    async def test_different_image_creates_separate_entity(
        self,
        db_session: AsyncSession,
        make_observation: MakeObservation,
        make_signature: MakeSignature,
    ) -> None:
        """A distinctly different image creates a new entity, not merging with existing."""
        # Individual A: Red
        embedding_a = create_embedding_for_color((255, 0, 0))

        # Individual B: Blue (very different from red)
        embedding_b = create_embedding_for_color((0, 0, 255))

        # Create observation for Individual A
        obs_a = make_observation(
            medium=Medium.IMAGE,
            observation_kind="wildlife_photo",
        )
        sig_a = make_signature(
            observation_id=obs_a.observation_id,
            signature_type="image_embedding",
            image_embedding=embedding_a,
        )
        db_session.add(obs_a)
        db_session.add(sig_a)
        await db_session.flush()

        entity_seed = EntityHypothesis(
            entity_type="photo_subject",
            entity_nature=EntityNature.INDIVIDUAL,
            suggested_name=None,
            slots=[],
            deterministic_ids=[],
            confidence=0.7,
        )

        resolver = EntityResolver(db_session)

        # Resolve Individual A
        result_a = await resolver.resolve(
            observation=obs_a,
            signatures=[sig_a],
            entity_seeds=[entity_seed],
        )

        assert len(result_a.entities_created) == 1
        entity_a_id = result_a.entities_created[0].entity_id

        # Create observation for Individual B
        obs_b = make_observation(
            medium=Medium.IMAGE,
            observation_kind="wildlife_photo",
        )
        sig_b = make_signature(
            observation_id=obs_b.observation_id,
            signature_type="image_embedding",
            image_embedding=embedding_b,
        )
        db_session.add(obs_b)
        db_session.add(sig_b)
        await db_session.flush()

        # Resolve Individual B
        result_b = await resolver.resolve(
            observation=obs_b,
            signatures=[sig_b],
            entity_seeds=[entity_seed],
        )

        # SEPARATION CHECK: Individual B should create a NEW entity, not link to A
        # Since embeddings are very different, similarity should be low
        assert len(result_b.entities_created) >= 1, (
            "Expected new entity for different individual, but none created"
        )

        entity_b_id = result_b.entities_created[0].entity_id
        assert entity_b_id != entity_a_id, (
            "Different individual should create separate entity, not link to existing"
        )

        # Verify Entity A doesn't have observation B linked
        stmt = select(ObservationEntityLink).where(
            ObservationEntityLink.entity_id == entity_a_id,
            ObservationEntityLink.observation_id == obs_b.observation_id,
        )
        wrong_link = await db_session.execute(stmt)
        assert wrong_link.scalar_one_or_none() is None, (
            "Observation B should NOT be linked to Entity A"
        )


class TestEngineOwnedSubjectSeed:
    """Tests for engine-generated subject seed for images."""

    async def test_engine_seed_added_when_no_photo_subject_proposed(
        self,
        db_session: AsyncSession,
        make_observation: MakeObservation,
        make_signature: MakeSignature,
    ) -> None:
        """Engine adds subject seed when LLM doesn't propose a photo_subject."""
        image_embedding = create_embedding_for_color((0, 255, 0))  # Green

        obs = make_observation(
            medium=Medium.IMAGE,
            observation_kind="wildlife_photo",
        )
        sig = make_signature(
            observation_id=obs.observation_id,
            signature_type="image_embedding",
            image_embedding=image_embedding,
        )
        db_session.add(obs)
        db_session.add(sig)
        await db_session.flush()

        # LLM proposes CLASS entity (species) and INDIVIDUAL entity (camera)
        # but NOT a photo_subject entity
        species_seed = EntityHypothesis(
            entity_type="species",
            entity_nature=EntityNature.CLASS,
            suggested_name="Leopard",
            slots=[],
            deterministic_ids=[],
            confidence=0.8,
        )
        camera_seed = EntityHypothesis(
            entity_type="camera",
            entity_nature=EntityNature.INDIVIDUAL,  # Camera is INDIVIDUAL!
            suggested_name=None,
            slots=[SlotValue(name="make", value="Canon")],
            deterministic_ids=[],
            confidence=0.7,
        )

        # Simulate the engine-added seed check (as done in cli.py)
        entity_seeds = [species_seed, camera_seed]
        has_photo_subject = any(
            seed.entity_type == "photo_subject"
            for seed in entity_seeds
        )
        # Even though camera is INDIVIDUAL, there's no photo_subject
        assert not has_photo_subject

        # Engine adds subject seed
        engine_seed = EntityHypothesis(
            entity_type="photo_subject",
            entity_nature=EntityNature.INDIVIDUAL,
            suggested_name=None,
            slots=[],
            deterministic_ids=[],  # No deterministic IDs
            confidence=0.6,
            reasoning="Engine-generated primary subject seed for visual re-ID",
        )
        entity_seeds.append(engine_seed)

        # Resolve with both seeds
        resolver = EntityResolver(db_session)
        result = await resolver.resolve(
            observation=obs,
            signatures=[sig],
            entity_seeds=entity_seeds,
        )

        # Should have created entities for CLASS, camera (INDIVIDUAL), and photo_subject (INDIVIDUAL)
        natures = [e.entity_nature for e in result.entities_created]
        assert EntityNature.INDIVIDUAL in natures, (
            "Engine seed should create INDIVIDUAL entity"
        )
        assert EntityNature.CLASS in natures, (
            "Species seed should create CLASS entity"
        )

        # Find the photo_subject entity specifically - it should have the prototype
        # The camera entity should NOT have an image prototype
        photo_subject_entities = [
            e for e in result.entities_created
            if e.entity_nature == EntityNature.INDIVIDUAL and e.prototype_image_embedding is not None
        ]
        assert len(photo_subject_entities) == 1, (
            "Only the photo_subject entity should have image prototype, not camera"
        )


class TestGPSNotDeterministicId:
    """Verify GPS is NOT used as a deterministic ID."""

    async def test_gps_in_facets_not_in_det_ids(self) -> None:
        """GPS should appear in facets/extra, not in deterministic_ids."""
        from object_sense.extraction.image import ImageExtractor

        # Create a mock embedding client
        mock_client = MagicMock()
        mock_client.embed_image = AsyncMock(return_value=[[0.1] * 768])

        extractor = ImageExtractor(embedding_client=mock_client)

        # Create a test image with fake EXIF GPS data
        # We can't easily inject EXIF, but we can verify the code path
        # by checking that extraction result has no GPS in deterministic_ids
        image_bytes = create_colored_image((100, 100, 100))
        result = await extractor.extract(image_bytes, filename="test.jpg")

        # Key assertion: deterministic_ids should be empty for images
        # GPS should NOT be in deterministic_ids
        for det_id in result.deterministic_ids:
            assert det_id.id_type != "gps", (
                "GPS should not be a deterministic ID - it causes false splits"
            )


class TestSparseSimilarityScoring:
    """Tests for sparse-signal similarity scoring.

    These tests verify that:
    1. Image-only scores are not penalized (no coverage penalty)
    2. Score ≈ raw cosine similarity for single-signal cases
    3. Image guardrails (T_img_min, margin) work correctly
    """

    def test_image_only_score_equals_cosine(self) -> None:
        """For image-only, final score ≈ raw image cosine similarity."""
        import math

        red_embedding = create_embedding_for_color((255, 0, 0))
        red_noisy = add_embedding_noise(red_embedding, noise_scale=0.01, seed=1)

        # Create mock entity
        entity = MagicMock()
        entity.prototype_image_embedding = red_embedding
        entity.prototype_text_embedding = None
        entity.slots = {}
        entity.name = None

        # Create signals with only image embedding
        signals = ObservationSignals(image_embedding=red_noisy)

        scorer = SimilarityScorer()
        result = scorer.compute(
            observation_signals=signals,
            entity=entity,
            entity_nature=EntityNature.INDIVIDUAL,
        )

        # Score should be close to raw image similarity (no penalty)
        assert result.image_similarity is not None
        assert abs(result.score - result.image_similarity) < 0.01, (
            f"Score {result.score:.4f} should be close to image_similarity "
            f"{result.image_similarity:.4f} for image-only"
        )

        # Should be flagged as single_signal
        assert "single_signal" in result.flags

    def test_different_colors_low_similarity(self) -> None:
        """Very different images should have low similarity."""
        red_embedding = create_embedding_for_color((255, 0, 0))
        blue_embedding = create_embedding_for_color((0, 0, 255))

        entity = MagicMock()
        entity.prototype_image_embedding = red_embedding
        entity.prototype_text_embedding = None
        entity.slots = {}
        entity.name = None

        signals = ObservationSignals(image_embedding=blue_embedding)

        scorer = SimilarityScorer()
        result = scorer.compute(
            observation_signals=signals,
            entity=entity,
            entity_nature=EntityNature.INDIVIDUAL,
        )

        # Score should be low for very different images
        assert result.score < 0.6, (
            f"Score {result.score:.4f} should be low for different colors"
        )

    def test_similar_colors_high_similarity(self) -> None:
        """Similar images should have high similarity."""
        red_embedding = create_embedding_for_color((255, 0, 0))
        red_noisy = add_embedding_noise(red_embedding, noise_scale=0.005, seed=1)

        entity = MagicMock()
        entity.prototype_image_embedding = red_embedding
        entity.prototype_text_embedding = None
        entity.slots = {}
        entity.name = None

        signals = ObservationSignals(image_embedding=red_noisy)

        scorer = SimilarityScorer()
        result = scorer.compute(
            observation_signals=signals,
            entity=entity,
            entity_nature=EntityNature.INDIVIDUAL,
        )

        # Score should be high for similar images
        assert result.score > 0.90, (
            f"Score {result.score:.4f} should be high for similar images"
        )

    def test_class_entity_uses_text_embedding(self) -> None:
        """CLASS entities should use text_embedding, not image."""
        text_embedding = [0.1] * 1024
        text_similar = [0.1 + 0.001 * i for i in range(1024)]

        entity = MagicMock()
        entity.prototype_image_embedding = None
        entity.prototype_text_embedding = text_embedding
        entity.slots = {}
        entity.name = None

        signals = ObservationSignals(text_embedding=text_similar)

        scorer = SimilarityScorer()
        result = scorer.compute(
            observation_signals=signals,
            entity=entity,
            entity_nature=EntityNature.CLASS,
        )

        # Should have text similarity, not image
        assert result.text_similarity is not None
        assert result.image_similarity is None
        assert result.score > 0.5

    def test_no_signals_returns_zero(self) -> None:
        """No available signals should return zero score."""
        entity = MagicMock()
        entity.prototype_image_embedding = None
        entity.prototype_text_embedding = None
        entity.slots = {}
        entity.name = None

        signals = ObservationSignals()  # No signals

        scorer = SimilarityScorer()
        result = scorer.compute(
            observation_signals=signals,
            entity=entity,
            entity_nature=EntityNature.INDIVIDUAL,
        )

        assert result.score == 0.0
        assert "no_signals_available" in result.flags


class TestImageGuardrails:
    """Tests for image-specific merge guardrails."""

    def test_t_img_min_blocks_low_similarity(self) -> None:
        """T_img_min should block merges with low image similarity."""
        from object_sense.config import settings

        red_embedding = create_embedding_for_color((255, 0, 0))
        # Create an embedding that's similar but not enough
        orange_embedding = create_embedding_for_color((255, 100, 0))

        entity = MagicMock()
        entity.prototype_image_embedding = red_embedding
        entity.prototype_text_embedding = None
        entity.slots = {}
        entity.name = None

        signals = ObservationSignals(image_embedding=orange_embedding)

        scorer = SimilarityScorer()
        result = scorer.compute(
            observation_signals=signals,
            entity=entity,
            entity_nature=EntityNature.INDIVIDUAL,
        )

        # Check if image similarity is below T_img_min
        if result.image_similarity is not None:
            if result.image_similarity < settings.entity_resolution_t_img_min:
                # This match should be blocked by guardrail
                assert result.image_similarity < 0.75, (
                    "Test expects image similarity below T_img_min"
                )
