"""Tests for the type inference module."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from object_sense.extraction.base import ExtractionResult
from object_sense.inference.schemas import (
    EntityHypothesis,
    SimilarObject,
    SlotValue,
    TypeProposal,
    TypeSearchResult,
)
from object_sense.inference.type_inference import (
    TypeInferenceAgent,
    TypeInferenceDeps,
)


class TestSlotValue:
    """Tests for SlotValue schema."""

    def test_default_values(self) -> None:
        slot = SlotValue(name="species", value="leopard")
        assert slot.name == "species"
        assert slot.value == "leopard"
        assert slot.is_reference is False
        assert slot.confidence == 0.8

    def test_reference_slot(self) -> None:
        slot = SlotValue(
            name="location",
            value="ref:malamala",
            is_reference=True,
            confidence=0.9,
        )
        assert slot.is_reference is True
        assert slot.confidence == 0.9

    def test_confidence_bounds(self) -> None:
        # Valid bounds
        slot = SlotValue(name="test", value="x", confidence=0.0)
        assert slot.confidence == 0.0

        slot = SlotValue(name="test", value="x", confidence=1.0)
        assert slot.confidence == 1.0

        # Out of bounds should fail validation
        with pytest.raises(ValueError):
            SlotValue(name="test", value="x", confidence=1.5)

        with pytest.raises(ValueError):
            SlotValue(name="test", value="x", confidence=-0.1)


class TestEntityHypothesis:
    """Tests for EntityHypothesis schema."""

    def test_minimal(self) -> None:
        entity = EntityHypothesis(entity_type="animal_entity")
        assert entity.entity_type == "animal_entity"
        assert entity.suggested_name is None
        assert entity.slots == []
        assert entity.confidence == 0.7
        assert entity.reasoning is None

    def test_full(self) -> None:
        entity = EntityHypothesis(
            entity_type="animal_entity",
            suggested_name="Marula Leopard",
            slots=[
                SlotValue(name="species", value="leopard"),
                SlotValue(name="pose", value="stalking"),
            ],
            confidence=0.85,
            reasoning="Detected large cat with spotted coat",
        )
        assert entity.suggested_name == "Marula Leopard"
        assert len(entity.slots) == 2
        assert entity.slots[0].name == "species"


class TestTypeProposal:
    """Tests for TypeProposal schema."""

    def test_minimal(self) -> None:
        proposal = TypeProposal(
            primary_type="wildlife_photo",
            reasoning="Image contains wildlife subject",
        )
        assert proposal.primary_type == "wildlife_photo"
        assert proposal.primary_type_confidence == 0.8
        assert proposal.is_existing_type is True
        assert proposal.slots == []
        assert proposal.entity_hypotheses == []
        assert proposal.maybe_new_type is None

    def test_full_proposal(self) -> None:
        proposal = TypeProposal(
            primary_type="wildlife_photo",
            primary_type_confidence=0.95,
            is_existing_type=True,
            slots=[
                SlotValue(name="lighting", value="backlit"),
            ],
            entity_hypotheses=[
                EntityHypothesis(
                    entity_type="animal_entity",
                    suggested_name="Unknown Leopard",
                ),
            ],
            maybe_new_type="backlit_wildlife_photo",
            new_type_rationale="Distinct lighting style worth tracking",
            reasoning="Wildlife photo with dramatic backlighting",
        )

        assert proposal.primary_type_confidence == 0.95
        assert len(proposal.slots) == 1
        assert len(proposal.entity_hypotheses) == 1
        assert proposal.maybe_new_type == "backlit_wildlife_photo"

    def test_new_type_proposal(self) -> None:
        proposal = TypeProposal(
            primary_type="safari_photo",
            is_existing_type=False,
            reasoning="New type for safari context photos",
        )
        assert proposal.is_existing_type is False


class TestTypeSearchResult:
    """Tests for TypeSearchResult schema."""

    def test_defaults(self) -> None:
        result = TypeSearchResult(type_name="wildlife_photo")
        assert result.type_name == "wildlife_photo"
        assert result.aliases == []
        assert result.parent_type is None
        assert result.evidence_count == 0
        assert result.status == "provisional"

    def test_with_all_fields(self) -> None:
        result = TypeSearchResult(
            type_name="leopard_photo",
            aliases=["big_cat_photo"],
            parent_type="wildlife_photo",
            evidence_count=42,
            status="stable",
        )
        assert result.parent_type == "wildlife_photo"
        assert result.evidence_count == 42


class TestSimilarObject:
    """Tests for SimilarObject schema."""

    def test_required_fields(self) -> None:
        obj = SimilarObject(
            object_id="abc-123",
            primary_type="wildlife_photo",
            similarity_score=0.92,
            medium="image",
        )
        assert obj.object_id == "abc-123"
        assert obj.similarity_score == 0.92

    def test_with_extracted_text(self) -> None:
        obj = SimilarObject(
            object_id="def-456",
            primary_type=None,
            similarity_score=0.75,
            medium="text",
            extracted_text="A leopard hunting at dusk",
        )
        assert obj.primary_type is None
        assert obj.extracted_text is not None


class TestTypeInferenceDeps:
    """Tests for TypeInferenceDeps dataclass."""

    def test_defaults(self) -> None:
        deps = TypeInferenceDeps()
        assert deps.search_types_fn is None
        assert deps.get_type_fn is None
        assert deps.find_similar_fn is None
        assert deps.extraction_result is None
        assert deps.medium == "unknown"

    def test_with_callbacks(self) -> None:
        async def mock_search(query: str) -> list[dict]:
            return []

        deps = TypeInferenceDeps(
            search_types_fn=mock_search,
            medium="image",
        )
        assert deps.search_types_fn is not None
        assert deps.medium == "image"


class TestTypeInferenceAgent:
    """Tests for TypeInferenceAgent class."""

    def test_init_without_callbacks(self) -> None:
        agent = TypeInferenceAgent()
        assert agent._search_types_fn is None
        assert agent._get_type_fn is None
        assert agent._find_similar_fn is None

    def test_init_with_callbacks(self) -> None:
        search_fn = AsyncMock(return_value=[])
        get_fn = AsyncMock(return_value=None)
        find_fn = AsyncMock(return_value=[])

        agent = TypeInferenceAgent(
            search_types_fn=search_fn,
            get_type_fn=get_fn,
            find_similar_fn=find_fn,
        )
        assert agent._search_types_fn is search_fn
        assert agent._get_type_fn is get_fn
        assert agent._find_similar_fn is find_fn

    def test_build_prompt_minimal(self) -> None:
        agent = TypeInferenceAgent()
        extraction = ExtractionResult()

        prompt = agent._build_prompt(extraction, "image")

        assert "image observation" in prompt
        assert "Extracted Features" in prompt
        assert "Instructions" in prompt

    def test_build_prompt_with_text(self) -> None:
        agent = TypeInferenceAgent()
        extraction = ExtractionResult(
            extracted_text="A leopard stalking prey in the bush",
        )

        prompt = agent._build_prompt(extraction, "image")

        assert "A leopard stalking" in prompt
        assert "Extracted text/caption" in prompt

    def test_build_prompt_with_metadata(self) -> None:
        agent = TypeInferenceAgent()
        extraction = ExtractionResult(
            extra={"width": 1920, "height": 1080, "format": "JPEG"},
        )

        prompt = agent._build_prompt(extraction, "image")

        assert "width: 1920" in prompt
        assert "height: 1080" in prompt

    def test_build_prompt_with_embeddings(self) -> None:
        agent = TypeInferenceAgent()
        extraction = ExtractionResult(
            text_embedding=[0.1] * 1024,
            image_embedding=[0.2] * 768,
        )

        prompt = agent._build_prompt(extraction, "image")

        assert "text (1024d)" in prompt
        assert "image (768d)" in prompt

    def test_build_prompt_with_hash(self) -> None:
        agent = TypeInferenceAgent()
        extraction = ExtractionResult(
            hash_value="abc123def456789",
        )

        prompt = agent._build_prompt(extraction, "json")

        # Hash is truncated to first 16 chars
        assert "abc123def456789" in prompt

    @pytest.mark.asyncio
    async def test_infer_calls_agent(self) -> None:
        """Test that infer() calls the underlying pydantic-ai agent."""
        # Create a mock result
        mock_proposal = TypeProposal(
            primary_type="wildlife_photo",
            reasoning="Test reasoning",
        )

        # Mock the agent.run method
        mock_run_result = AsyncMock()
        mock_run_result.output = mock_proposal

        agent = TypeInferenceAgent()

        # Patch the agent's run method
        with patch.object(agent._agent, "run", return_value=mock_run_result) as mock_run:
            extraction = ExtractionResult(
                extracted_text="A leopard in the wild",
            )

            result = await agent.infer(extraction, medium="image")

            # Verify run was called
            mock_run.assert_called_once()

            # Verify result
            assert result.primary_type == "wildlife_photo"
            assert result.reasoning == "Test reasoning"

    @pytest.mark.asyncio
    async def test_infer_passes_deps(self) -> None:
        """Test that dependencies are passed correctly to the agent."""
        mock_proposal = TypeProposal(
            primary_type="product_record",
            reasoning="JSON product data",
        )

        mock_run_result = AsyncMock()
        mock_run_result.output = mock_proposal

        search_fn = AsyncMock(return_value=[])

        agent = TypeInferenceAgent(search_types_fn=search_fn)

        with patch.object(agent._agent, "run", return_value=mock_run_result) as mock_run:
            extraction = ExtractionResult()

            await agent.infer(extraction, medium="json")

            # Check that deps were passed
            call_kwargs = mock_run.call_args
            deps = call_kwargs.kwargs.get("deps") or call_kwargs.args[1]
            assert deps.search_types_fn is search_fn
            assert deps.medium == "json"


@pytest.mark.integration
class TestTypeInferenceIntegration:
    """Integration tests requiring running Sparkstation gateway.

    Run with: pytest -m integration
    """

    @pytest.fixture
    def agent(self) -> TypeInferenceAgent:
        return TypeInferenceAgent()

    @pytest.mark.asyncio
    async def test_real_inference_wildlife_photo(
        self, agent: TypeInferenceAgent
    ) -> None:
        """Test real inference on a wildlife photo description."""
        extraction = ExtractionResult(
            extracted_text="A majestic leopard stalking through tall grass at sunset",
            image_embedding=[0.1] * 768,
            extra={"width": 1920, "height": 1080},
        )

        result = await agent.infer(extraction, medium="image")

        # Should propose a wildlife-related type
        assert "photo" in result.primary_type.lower() or "wildlife" in result.primary_type.lower()
        assert result.reasoning is not None
        assert len(result.reasoning) > 0

    @pytest.mark.asyncio
    async def test_real_inference_product_record(
        self, agent: TypeInferenceAgent
    ) -> None:
        """Test real inference on a product record."""
        extraction = ExtractionResult(
            extracted_text="product_id: SKU-12345, name: Safari Hat, price: 49.99, category: outdoor",
            text_embedding=[0.1] * 1024,
            extra={"is_array": False, "key_count": 4},
        )

        result = await agent.infer(extraction, medium="json")

        # Should propose a product-related type
        assert "product" in result.primary_type.lower() or "record" in result.primary_type.lower()
