"""End-to-end integration tests for the ingest pipeline.

These tests verify the full ingest flow:
1. Medium detection
2. Feature extraction (embeddings, hashes)
3. Type inference (LLM)
4. Entity resolution
5. Database persistence

Run with: pytest tests/test_e2e_ingest.py -v
For real LLM/embedding tests: pytest tests/test_e2e_ingest.py -v -m integration
"""

from __future__ import annotations

import io
import json
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from PIL import Image
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from object_sense.cli import normalize_deterministic_ids
from object_sense.extraction.base import ExtractedId
from object_sense.extraction.orchestrator import ExtractionOrchestrator
from object_sense.inference.schemas import (
    DeterministicId,
    EntityHypothesis,
    TypeCandidateProposal,
    TypeProposal,
)
from object_sense.models import (
    Blob,
    Entity,
    EntityDeterministicId,
    EntityNature,
    EntityStatus,
    Evidence,
    LinkStatus,
    Medium,
    Observation,
    ObservationEntityLink,
    ObservationStatus,
    Signature,
    TypeCandidate,
)
from object_sense.resolution.resolver import EntityResolver

if TYPE_CHECKING:
    from conftest import MakeEntity, MakeObservation, MakeSignature


def create_test_image(width: int = 100, height: int = 100, format: str = "JPEG") -> bytes:
    """Create a minimal test image."""
    img = Image.new("RGB", (width, height), color="red")
    buffer = io.BytesIO()
    img.save(buffer, format=format)
    return buffer.getvalue()


def create_test_json(data: dict[str, Any] | list[Any]) -> bytes:
    """Create test JSON content."""
    return json.dumps(data).encode("utf-8")


def create_test_text(content: str = "This is test content.") -> bytes:
    """Create test text content."""
    return content.encode("utf-8")


def mock_embedding_client() -> MagicMock:
    """Create a mock embedding client."""
    client = MagicMock()
    client.embed_image = AsyncMock(return_value=[[0.1] * 768])
    client.embed_text = AsyncMock(return_value=[[0.2] * 1024])
    client.embed_text_clip = AsyncMock(return_value=[[0.3] * 768])
    return client


def mock_type_proposal(
    observation_kind: str = "test_observation",
    type_name: str = "test_type",
    entity_seeds: list[EntityHypothesis] | None = None,
    deterministic_ids: list[DeterministicId] | None = None,
) -> TypeProposal:
    """Create a mock type proposal for testing."""
    return TypeProposal(
        observation_kind=observation_kind,
        type_candidate=TypeCandidateProposal(
            proposed_name=type_name,
            rationale="Test type for E2E testing",
            confidence=0.9,
        ),
        entity_seeds=entity_seeds or [],
        deterministic_ids=deterministic_ids or [],
        facets={"test_facet": "value"},
        reasoning="Test reasoning",
    )


# Mark all tests as integration by default (require database)
pytestmark = pytest.mark.integration


class TestMediumDetectionE2E:
    """Test medium detection in the extraction pipeline."""

    async def test_image_medium_detected(self, db_session: AsyncSession) -> None:
        """Image content is correctly detected as IMAGE medium."""
        from object_sense.utils.medium import probe_medium

        image_bytes = create_test_image()
        medium = probe_medium(image_bytes, filename="test.jpg")
        assert medium == Medium.IMAGE

    async def test_json_medium_detected(self, db_session: AsyncSession) -> None:
        """JSON content is correctly detected as JSON medium."""
        from object_sense.utils.medium import probe_medium

        json_bytes = create_test_json({"key": "value"})
        medium = probe_medium(json_bytes, filename="test.json")
        assert medium == Medium.JSON

    async def test_text_medium_detected(self, db_session: AsyncSession) -> None:
        """Plain text is correctly detected as TEXT medium."""
        from object_sense.utils.medium import probe_medium

        text_bytes = create_test_text("Hello world")
        medium = probe_medium(text_bytes, filename="test.txt")
        assert medium == Medium.TEXT

    async def test_binary_medium_for_unknown(self, db_session: AsyncSession) -> None:
        """Unknown binary content falls back to BINARY medium."""
        from object_sense.utils.medium import probe_medium

        binary_bytes = b"\x00\x01\x02\x03\x04\x05"
        medium = probe_medium(binary_bytes, filename="test.bin")
        assert medium == Medium.BINARY


class TestExtractionPipelineE2E:
    """Test extraction produces correct signatures and embeddings."""

    async def test_image_extraction_creates_embedding(self, db_session: AsyncSession) -> None:
        """Image extraction produces image embedding."""
        client = mock_embedding_client()
        orchestrator = ExtractionOrchestrator(embedding_client=client)

        image_bytes = create_test_image()
        result = await orchestrator.extract(image_bytes, filename="test.jpg")

        assert result.image_embedding is not None
        assert len(result.image_embedding) == 768
        assert result.extra.get("medium") == "image"
        assert result.extra.get("width") == 100
        assert result.extra.get("height") == 100

    async def test_text_extraction_creates_embeddings(self, db_session: AsyncSession) -> None:
        """Text extraction produces text and CLIP embeddings."""
        client = mock_embedding_client()
        orchestrator = ExtractionOrchestrator(embedding_client=client)

        result = await orchestrator.extract(
            create_test_text("A wildlife photo description"),
            medium=Medium.TEXT,
        )

        assert result.text_embedding is not None
        assert len(result.text_embedding) == 1024
        assert result.clip_text_embedding is not None
        assert len(result.clip_text_embedding) == 768
        assert result.extra.get("medium") == "text"

    async def test_json_extraction_creates_hash_and_embedding(
        self, db_session: AsyncSession
    ) -> None:
        """JSON extraction produces schema hash and text embedding."""
        client = mock_embedding_client()
        orchestrator = ExtractionOrchestrator(embedding_client=client)

        json_data = {"product_id": "SKU-001", "name": "Test Product", "price": 29.99}
        result = await orchestrator.extract(
            create_test_json(json_data),
            medium=Medium.JSON,
        )

        assert result.hash_value is not None  # Schema hash
        assert result.text_embedding is not None
        assert result.extra.get("medium") == "json"
        assert result.extra.get("is_array") is False
        assert result.extra.get("key_count") == 3


class TestObservationPersistenceE2E:
    """Test that observations are correctly persisted to the database."""

    async def test_observation_created_with_medium(
        self,
        db_session: AsyncSession,
        make_observation: MakeObservation,
    ) -> None:
        """Observation is created with correct medium."""
        obs = make_observation(
            medium=Medium.IMAGE,
            observation_kind="wildlife_photo",
        )
        db_session.add(obs)
        await db_session.flush()

        # Verify persistence
        stmt = select(Observation).where(Observation.observation_id == obs.observation_id)
        result = await db_session.execute(stmt)
        persisted = result.scalar_one()

        assert persisted.medium == Medium.IMAGE
        assert persisted.observation_kind == "wildlife_photo"
        assert persisted.status == ObservationStatus.ACTIVE

    async def test_observation_with_deterministic_ids(
        self,
        db_session: AsyncSession,
        make_observation: MakeObservation,
    ) -> None:
        """Observation stores deterministic IDs correctly."""
        det_ids = [
            {"id_type": "sku", "id_value": "TEST-001", "id_namespace": "test"},
            {"id_type": "upc", "id_value": "123456789", "id_namespace": "global:upc"},
        ]
        obs = make_observation(
            observation_kind="product_record",
            deterministic_ids=det_ids,
        )
        db_session.add(obs)
        await db_session.flush()

        stmt = select(Observation).where(Observation.observation_id == obs.observation_id)
        result = await db_session.execute(stmt)
        persisted = result.scalar_one()

        assert len(persisted.deterministic_ids) == 2
        assert persisted.deterministic_ids[0]["id_type"] == "sku"
        assert persisted.deterministic_ids[1]["id_type"] == "upc"


class TestSignaturePersistenceE2E:
    """Test that signatures are correctly persisted."""

    async def test_image_embedding_signature_persisted(
        self,
        db_session: AsyncSession,
        make_observation: MakeObservation,
        make_signature: MakeSignature,
    ) -> None:
        """Image embedding signature is persisted with correct dimensions."""
        obs = make_observation(medium=Medium.IMAGE)
        db_session.add(obs)
        await db_session.flush()

        sig = make_signature(
            observation_id=obs.observation_id,
            signature_type="image_embedding",
            image_embedding=[0.1] * 768,
        )
        db_session.add(sig)
        await db_session.flush()

        stmt = select(Signature).where(Signature.signature_id == sig.signature_id)
        result = await db_session.execute(stmt)
        persisted = result.scalar_one()

        assert persisted.signature_type == "image_embedding"
        assert persisted.image_embedding is not None
        assert len(list(persisted.image_embedding)) == 768

    async def test_text_embedding_signature_persisted(
        self,
        db_session: AsyncSession,
        make_observation: MakeObservation,
        make_signature: MakeSignature,
    ) -> None:
        """Text embedding signature is persisted with correct dimensions."""
        obs = make_observation(medium=Medium.TEXT)
        db_session.add(obs)
        await db_session.flush()

        sig = make_signature(
            observation_id=obs.observation_id,
            signature_type="text_embedding",
            text_embedding=[0.2] * 1024,
        )
        db_session.add(sig)
        await db_session.flush()

        stmt = select(Signature).where(Signature.signature_id == sig.signature_id)
        result = await db_session.execute(stmt)
        persisted = result.scalar_one()

        assert persisted.signature_type == "text_embedding"
        assert persisted.text_embedding is not None
        assert len(list(persisted.text_embedding)) == 1024


class TestEntityResolutionE2E:
    """Test entity resolution in the full pipeline."""

    async def test_entity_created_for_new_deterministic_id(
        self,
        db_session: AsyncSession,
        make_observation: MakeObservation,
        make_signature: MakeSignature,
    ) -> None:
        """New deterministic ID creates a new entity."""
        obs = make_observation(
            observation_kind="product_record",
            deterministic_ids=[
                {"id_type": "sku", "id_value": "NEW-SKU-001", "id_namespace": "test"}
            ],
        )
        sig = make_signature(observation_id=obs.observation_id)
        db_session.add(obs)
        db_session.add(sig)
        await db_session.flush()

        # Create entity hypothesis with deterministic ID
        det_id = DeterministicId(
            id_type="sku",
            id_value="NEW-SKU-001",
            id_namespace="test",
            strength="strong",
        )
        entity_seed = EntityHypothesis(
            entity_type="product",
            entity_nature=EntityNature.INDIVIDUAL,
            suggested_name="New Product",
            deterministic_ids=[det_id],
            confidence=0.9,
        )

        resolver = EntityResolver(db_session)
        result = await resolver.resolve(
            observation=obs,
            signatures=[sig],
            entity_seeds=[entity_seed],
        )

        # Verify entity created
        assert len(result.entities_created) == 1
        entity = result.entities_created[0]
        assert entity.status == EntityStatus.PROTO
        assert entity.entity_nature == EntityNature.INDIVIDUAL

        # Verify link created
        assert len(result.links) == 1
        assert result.links[0].status == LinkStatus.HARD
        assert result.links[0].posterior == 1.0

        # Verify EntityDeterministicId created
        stmt = select(EntityDeterministicId).where(
            EntityDeterministicId.entity_id == entity.entity_id
        )
        det_id_result = await db_session.execute(stmt)
        det_id_rows = det_id_result.scalars().all()
        assert len(det_id_rows) == 1
        assert det_id_rows[0].id_type == "sku"
        assert det_id_rows[0].id_value == "NEW-SKU-001"

    async def test_existing_entity_reused_for_same_deterministic_id(
        self,
        db_session: AsyncSession,
        make_observation: MakeObservation,
        make_signature: MakeSignature,
    ) -> None:
        """Same deterministic ID links to existing entity."""
        det_id = DeterministicId(
            id_type="sku",
            id_value="SHARED-SKU-001",
            id_namespace="test",
            strength="strong",
        )
        entity_seed = EntityHypothesis(
            entity_type="product",
            entity_nature=EntityNature.INDIVIDUAL,
            suggested_name="Shared Product",
            deterministic_ids=[det_id],
            confidence=0.9,
        )

        # First observation
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

        # No new entity
        assert len(result2.entities_created) == 0

        # Link to same entity
        assert len(result2.links) == 1
        assert result2.links[0].entity_id == entity_id
        assert result2.links[0].status == LinkStatus.HARD

    async def test_proto_entity_created_for_similarity_based_resolution(
        self,
        db_session: AsyncSession,
        make_observation: MakeObservation,
        make_signature: MakeSignature,
    ) -> None:
        """Without deterministic ID, creates proto-entity via similarity."""
        obs = make_observation(observation_kind="wildlife_photo")
        sig = make_signature(
            observation_id=obs.observation_id,
            image_embedding=[0.1] * 768,
        )
        db_session.add(obs)
        db_session.add(sig)
        await db_session.flush()

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

        # Proto-entity should be created
        assert len(result.entities_created) == 1
        assert result.entities_created[0].status == EntityStatus.PROTO

        # Soft link
        assert len(result.links) == 1
        assert result.links[0].status == LinkStatus.SOFT


class TestNamespaceEnforcementE2E:
    """Test namespace enforcement in the pipeline."""

    def test_normalize_deterministic_ids_applies_namespace(self) -> None:
        """normalize_deterministic_ids applies context namespace."""
        ids = [
            DeterministicId(id_type="sku", id_value="TEST-001"),  # Missing namespace
        ]
        result = normalize_deterministic_ids(ids, context_namespace="source:test")

        assert len(result.ids) == 1
        assert result.ids[0].id_namespace == "source:test"
        assert len(result.overrides) == 1
        assert result.overrides[0].reason == "empty_namespace"

    def test_normalize_deterministic_ids_uses_id_type_mapping(self) -> None:
        """ID type mappings take precedence over proposed namespace."""
        ids = [
            DeterministicId(id_type="upc", id_value="123456789", id_namespace="wrong"),
        ]
        result = normalize_deterministic_ids(ids, context_namespace="source:test")

        # UPC should be mapped to global:upc (per id_type_namespace_map)
        assert result.ids[0].id_namespace == "global:upc"
        assert len(result.overrides) == 1
        assert result.overrides[0].reason == "id_type_mapped"

    def test_normalize_extracted_ids(self) -> None:
        """ExtractedId objects are normalized correctly."""
        # ExtractedId requires id_namespace, but test normalization with an invalid one
        ids = [
            ExtractedId(id_type="sku", id_value="EXTRACTED-001", id_namespace="invalid!ns"),
        ]
        result = normalize_deterministic_ids(ids, context_namespace="source:extraction")

        # Invalid namespace should be overridden to context namespace
        assert len(result.ids) == 1
        assert result.ids[0].id_namespace == "source:extraction"
        assert len(result.overrides) == 1
        assert result.overrides[0].reason == "invalid_pattern"


class TestTypeCandidatePersistenceE2E:
    """Test type candidate persistence."""

    async def test_type_candidate_created(
        self,
        db_session: AsyncSession,
    ) -> None:
        """Type candidate is created and persisted."""
        from object_sense.services.type_candidate import TypeCandidateService

        service = TypeCandidateService(db_session)
        candidate, is_new = await service.get_or_create(
            "wildlife_photo",
            details={"rationale": "Photos of wildlife"},
        )

        assert is_new
        assert candidate.proposed_name == "wildlife_photo"
        assert candidate.normalized_name == "wildlife_photo"

        # Verify persistence
        stmt = select(TypeCandidate).where(TypeCandidate.candidate_id == candidate.candidate_id)
        result = await db_session.execute(stmt)
        persisted = result.scalar_one()

        assert persisted.proposed_name == "wildlife_photo"

    async def test_type_candidate_deduplication(
        self,
        db_session: AsyncSession,
    ) -> None:
        """Same type name returns existing candidate."""
        from object_sense.services.type_candidate import TypeCandidateService

        service = TypeCandidateService(db_session)

        # Create first
        candidate1, is_new1 = await service.get_or_create("Wildlife Photo")
        assert is_new1

        # Same normalized name should return existing
        candidate2, is_new2 = await service.get_or_create("wildlife_photo")
        assert not is_new2
        assert candidate1.candidate_id == candidate2.candidate_id


class TestEvidencePersistenceE2E:
    """Test evidence record persistence."""

    async def test_evidence_created_for_type_assignment(
        self,
        db_session: AsyncSession,
        make_observation: MakeObservation,
    ) -> None:
        """Evidence is created when type is assigned."""
        from object_sense.models import EvidenceSource, SubjectKind

        obs = make_observation(observation_kind="test")
        db_session.add(obs)
        await db_session.flush()

        evidence = Evidence(
            evidence_id=uuid4(),
            subject_kind=SubjectKind.OBSERVATION,
            subject_id=obs.observation_id,
            predicate="has_type_candidate",
            target_id=uuid4(),
            source=EvidenceSource.LLM,
            score=0.9,
            details={"reasoning": "Test reasoning"},
        )
        db_session.add(evidence)
        await db_session.flush()

        stmt = select(Evidence).where(Evidence.evidence_id == evidence.evidence_id)
        result = await db_session.execute(stmt)
        persisted = result.scalar_one()

        assert persisted.predicate == "has_type_candidate"
        assert persisted.score == 0.9
        assert persisted.details.get("reasoning") == "Test reasoning"


class TestMultiMediumE2E:
    """Test pipeline handles different mediums correctly."""

    async def test_image_medium_flow(
        self,
        db_session: AsyncSession,
        make_observation: MakeObservation,
        make_signature: MakeSignature,
    ) -> None:
        """Complete flow for IMAGE medium."""
        obs = make_observation(
            medium=Medium.IMAGE,
            observation_kind="wildlife_photo",
        )
        sig = make_signature(
            observation_id=obs.observation_id,
            image_embedding=[0.1] * 768,
        )
        db_session.add(obs)
        db_session.add(sig)
        await db_session.flush()

        assert obs.medium == Medium.IMAGE
        assert sig.image_embedding is not None

    async def test_json_medium_flow(
        self,
        db_session: AsyncSession,
        make_observation: MakeObservation,
        make_signature: MakeSignature,
    ) -> None:
        """Complete flow for JSON medium."""
        obs = make_observation(
            medium=Medium.JSON,
            observation_kind="product_record",
            deterministic_ids=[{"id_type": "sku", "id_value": "JSON-001", "id_namespace": "test"}],
        )
        sig = make_signature(
            observation_id=obs.observation_id,
            text_embedding=[0.2] * 1024,
        )
        db_session.add(obs)
        db_session.add(sig)
        await db_session.flush()

        assert obs.medium == Medium.JSON
        assert len(obs.deterministic_ids) == 1

    async def test_text_medium_flow(
        self,
        db_session: AsyncSession,
        make_observation: MakeObservation,
        make_signature: MakeSignature,
    ) -> None:
        """Complete flow for TEXT medium."""
        obs = make_observation(
            medium=Medium.TEXT,
            observation_kind="document",
        )
        sig = make_signature(
            observation_id=obs.observation_id,
            text_embedding=[0.2] * 1024,
        )
        db_session.add(obs)
        db_session.add(sig)
        await db_session.flush()

        assert obs.medium == Medium.TEXT


class TestBlobDeduplicationE2E:
    """Test blob deduplication."""

    async def test_same_content_same_blob(
        self,
        db_session: AsyncSession,
    ) -> None:
        """Same content hashes to same blob."""
        import hashlib

        content = b"identical content for dedup test"
        sha256 = hashlib.sha256(content).hexdigest()

        # Create first blob
        blob1 = Blob(
            blob_id=uuid4(),
            sha256=sha256,
            size_bytes=len(content),
            storage_path="/test/path1",
        )
        db_session.add(blob1)
        await db_session.flush()

        # Check for existing blob
        stmt = select(Blob).where(Blob.sha256 == sha256)
        result = await db_session.execute(stmt)
        existing = result.scalar_one_or_none()

        assert existing is not None
        assert existing.blob_id == blob1.blob_id


class TestObservationEntityLinkE2E:
    """Test observation-entity links are correctly persisted."""

    async def test_link_persisted_with_correct_status(
        self,
        db_session: AsyncSession,
        make_observation: MakeObservation,
        make_entity: MakeEntity,
    ) -> None:
        """ObservationEntityLink is persisted with correct attributes."""
        from object_sense.models import LinkRole

        obs = make_observation()
        entity = make_entity()
        db_session.add(obs)
        db_session.add(entity)
        await db_session.flush()

        link = ObservationEntityLink(
            observation_id=obs.observation_id,
            entity_id=entity.entity_id,
            posterior=0.95,
            status=LinkStatus.SOFT,
            role=LinkRole.SUBJECT,
            flags=["test_flag"],
        )
        db_session.add(link)
        await db_session.flush()

        stmt = select(ObservationEntityLink).where(
            ObservationEntityLink.observation_id == obs.observation_id,
            ObservationEntityLink.entity_id == entity.entity_id,
        )
        result = await db_session.execute(stmt)
        persisted = result.scalar_one()

        assert persisted.posterior == 0.95
        assert persisted.status == LinkStatus.SOFT
        assert persisted.role == LinkRole.SUBJECT
        assert "test_flag" in persisted.flags


class TestFullPipelineE2E:
    """Test the complete ingest pipeline end-to-end."""

    async def test_image_ingest_creates_all_artifacts(
        self,
        db_session: AsyncSession,
        make_observation: MakeObservation,
        make_signature: MakeSignature,
    ) -> None:
        """Image ingest creates observation, signature, entity, and link."""
        # Create observation
        obs = make_observation(
            medium=Medium.IMAGE,
            observation_kind="wildlife_photo",
            deterministic_ids=[
                {"id_type": "photo_id", "id_value": "IMG-001", "id_namespace": "test"}
            ],
        )
        db_session.add(obs)
        await db_session.flush()

        # Create signature
        sig = make_signature(
            observation_id=obs.observation_id,
            image_embedding=[0.1] * 768,
        )
        db_session.add(sig)
        await db_session.flush()

        # Create entity via resolver
        det_id = DeterministicId(
            id_type="photo_id",
            id_value="IMG-001",
            id_namespace="test",
            strength="strong",
        )
        entity_seed = EntityHypothesis(
            entity_type="wildlife_photo",
            entity_nature=EntityNature.INDIVIDUAL,
            suggested_name="Wildlife Photo IMG-001",
            deterministic_ids=[det_id],
            confidence=0.9,
        )

        resolver = EntityResolver(db_session)
        result = await resolver.resolve(
            observation=obs,
            signatures=[sig],
            entity_seeds=[entity_seed],
        )

        # Verify all artifacts created
        assert len(result.entities_created) == 1
        assert len(result.links) == 1

        # Verify observation persisted
        stmt = select(Observation).where(Observation.observation_id == obs.observation_id)
        obs_result = await db_session.execute(stmt)
        assert obs_result.scalar_one() is not None

        # Verify signature persisted
        stmt = select(Signature).where(Signature.observation_id == obs.observation_id)
        sig_result = await db_session.execute(stmt)
        assert sig_result.scalar_one() is not None

        # Verify entity persisted
        entity_id = result.entities_created[0].entity_id
        stmt = select(Entity).where(Entity.entity_id == entity_id)
        entity_result = await db_session.execute(stmt)
        assert entity_result.scalar_one() is not None

        # Verify link persisted
        stmt = select(ObservationEntityLink).where(
            ObservationEntityLink.observation_id == obs.observation_id
        )
        link_result = await db_session.execute(stmt)
        assert link_result.scalar_one() is not None

    async def test_json_ingest_with_extracted_ids(
        self,
        db_session: AsyncSession,
        make_observation: MakeObservation,
        make_signature: MakeSignature,
    ) -> None:
        """JSON ingest with extracted IDs creates correct entities."""
        # Simulate JSON with product data
        obs = make_observation(
            medium=Medium.JSON,
            observation_kind="product_record",
            deterministic_ids=[
                {"id_type": "sku", "id_value": "PROD-123", "id_namespace": "source:catalog"},
                {"id_type": "upc", "id_value": "012345678901", "id_namespace": "global:upc"},
            ],
        )
        sig = make_signature(
            observation_id=obs.observation_id,
            text_embedding=[0.2] * 1024,
        )
        db_session.add(obs)
        db_session.add(sig)
        await db_session.flush()

        # Both IDs should create entities
        det_id_sku = DeterministicId(
            id_type="sku",
            id_value="PROD-123",
            id_namespace="source:catalog",
            strength="strong",
        )
        det_id_upc = DeterministicId(
            id_type="upc",
            id_value="012345678901",
            id_namespace="global:upc",
            strength="strong",
        )

        entity_seed = EntityHypothesis(
            entity_type="product",
            entity_nature=EntityNature.INDIVIDUAL,
            suggested_name="Product PROD-123",
            deterministic_ids=[det_id_sku, det_id_upc],
            confidence=0.95,
        )

        resolver = EntityResolver(db_session)
        result = await resolver.resolve(
            observation=obs,
            signatures=[sig],
            entity_seeds=[entity_seed],
        )

        # Should create entity with first det ID, second links to same
        assert len(result.entities_created) >= 1
        assert len(result.links) >= 1


@pytest.mark.integration
class TestRealExtractionE2E:
    """Integration tests with real Sparkstation services."""

    async def test_real_image_extraction(self) -> None:
        """Test real image extraction with Sparkstation."""
        orchestrator = ExtractionOrchestrator()
        image_bytes = create_test_image()

        result = await orchestrator.extract(image_bytes, filename="test.jpg")

        assert result.image_embedding is not None
        assert len(result.image_embedding) == 768
        assert result.extra.get("width") == 100

    async def test_real_text_extraction(self) -> None:
        """Test real text extraction with Sparkstation."""
        orchestrator = ExtractionOrchestrator()

        result = await orchestrator.extract(
            b"A majestic leopard prowling through the African savanna",
            medium=Medium.TEXT,
        )

        assert result.text_embedding is not None
        assert len(result.text_embedding) == 1024
        assert result.clip_text_embedding is not None
        assert len(result.clip_text_embedding) == 768

    async def test_real_json_extraction(self) -> None:
        """Test real JSON extraction with Sparkstation."""
        orchestrator = ExtractionOrchestrator()
        json_data = {
            "product_id": "SKU-12345",
            "name": "Wildlife Camera Trap",
            "category": "Photography",
            "price": 299.99,
        }

        result = await orchestrator.extract(
            create_test_json(json_data),
            medium=Medium.JSON,
        )

        assert result.text_embedding is not None
        assert result.hash_value is not None
