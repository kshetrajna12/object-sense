"""Tests for the feature extraction module."""

from __future__ import annotations

import io
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from object_sense.extraction import (
    ExtractionOrchestrator,
    ExtractionResult,
    ImageExtractor,
    JsonExtractor,
    TextExtractor,
)
from object_sense.models.enums import Medium


def create_test_image(width: int = 100, height: int = 100, format: str = "JPEG") -> bytes:
    """Create a minimal test image."""
    img = Image.new("RGB", (width, height), color="red")
    buffer = io.BytesIO()
    img.save(buffer, format=format)
    return buffer.getvalue()


def mock_embedding_client() -> MagicMock:
    """Create a mock embedding client."""
    client = MagicMock()
    # Return appropriately sized embeddings
    client.embed_image = AsyncMock(return_value=[[0.1] * 768])
    client.embed_text = AsyncMock(return_value=[[0.2] * 1024])
    client.embed_text_clip = AsyncMock(return_value=[[0.3] * 768])
    return client


class TestExtractionResult:
    """Tests for ExtractionResult dataclass."""

    def test_default_values(self) -> None:
        result = ExtractionResult()
        assert result.text_embedding is None
        assert result.image_embedding is None
        assert result.clip_text_embedding is None
        assert result.hash_value is None
        assert result.signature_type == "extracted"
        assert result.extracted_text is None
        assert result.extra == {}

    def test_merge_overwrites_non_none(self) -> None:
        r1 = ExtractionResult(
            text_embedding=[0.1, 0.2],
            signature_type="first",
            extra={"a": 1},
        )
        r2 = ExtractionResult(
            image_embedding=[0.3, 0.4],
            extra={"b": 2},
        )

        merged = r1.merge(r2)

        assert merged.text_embedding == [0.1, 0.2]  # kept from r1
        assert merged.image_embedding == [0.3, 0.4]  # from r2
        assert merged.extra == {"a": 1, "b": 2}  # merged

    def test_merge_other_takes_precedence(self) -> None:
        r1 = ExtractionResult(text_embedding=[0.1])
        r2 = ExtractionResult(text_embedding=[0.2])

        merged = r1.merge(r2)

        assert merged.text_embedding == [0.2]  # r2 takes precedence


class TestImageExtractor:
    """Tests for ImageExtractor."""

    async def test_extract_returns_image_embedding(self) -> None:
        client = mock_embedding_client()
        extractor = ImageExtractor(embedding_client=client)

        image_bytes = create_test_image()
        result = await extractor.extract(image_bytes)

        assert result.image_embedding is not None
        assert len(result.image_embedding) == 768
        assert result.signature_type == "image"
        client.embed_image.assert_called_once()

    async def test_extract_gets_dimensions(self) -> None:
        client = mock_embedding_client()
        extractor = ImageExtractor(embedding_client=client)

        image_bytes = create_test_image(width=200, height=150)
        result = await extractor.extract(image_bytes)

        assert result.extra.get("width") == 200
        assert result.extra.get("height") == 150

    async def test_extract_invalid_image(self) -> None:
        client = mock_embedding_client()
        client.embed_image = AsyncMock(return_value=[])  # No embedding for invalid
        extractor = ImageExtractor(embedding_client=client)

        result = await extractor.extract(b"not an image")

        assert result.image_embedding is None
        assert result.extra.get("width") is None

    def test_serialize_exif_value_bytes(self) -> None:
        extractor = ImageExtractor()
        result = extractor._serialize_exif_value(b"hello")
        assert result == "hello"

    def test_serialize_exif_value_rational(self) -> None:
        extractor = ImageExtractor()
        result = extractor._serialize_exif_value((1, 4))
        assert result == 0.25

    def test_gps_to_decimal_north(self) -> None:
        extractor = ImageExtractor()
        # 40° 26' 46" N
        result = extractor._gps_to_decimal([40.0, 26.0, 46.0], "N")
        assert result == pytest.approx(40.446111, rel=1e-4)

    def test_gps_to_decimal_south(self) -> None:
        extractor = ImageExtractor()
        # 33° 51' 54" S
        result = extractor._gps_to_decimal([33.0, 51.0, 54.0], "S")
        assert result == pytest.approx(-33.865, rel=1e-4)


class TestTextExtractor:
    """Tests for TextExtractor."""

    async def test_extract_returns_both_embeddings(self) -> None:
        client = mock_embedding_client()
        extractor = TextExtractor(embedding_client=client)

        result = await extractor.extract(b"Hello world")

        assert result.text_embedding is not None
        assert len(result.text_embedding) == 1024
        assert result.clip_text_embedding is not None
        assert len(result.clip_text_embedding) == 768
        assert result.signature_type == "text"

    async def test_extract_stores_text(self) -> None:
        client = mock_embedding_client()
        extractor = TextExtractor(embedding_client=client)

        result = await extractor.extract(b"Hello world")

        assert result.extracted_text == "Hello world"
        assert result.extra.get("char_count") == 11
        assert result.extra.get("truncated") is False

    async def test_extract_truncates_long_text(self) -> None:
        client = mock_embedding_client()
        extractor = TextExtractor(embedding_client=client, max_chars=50)

        long_text = "x" * 100
        result = await extractor.extract(long_text.encode())

        assert len(result.extracted_text) == 50  # type: ignore[arg-type]
        assert result.extra.get("truncated") is True
        assert result.extra.get("char_count") == 100  # original length

    async def test_extract_empty_text(self) -> None:
        client = mock_embedding_client()
        extractor = TextExtractor(embedding_client=client)

        result = await extractor.extract(b"")

        assert result.text_embedding is None
        assert result.extracted_text is None

    async def test_decode_latin1_fallback(self) -> None:
        client = mock_embedding_client()
        extractor = TextExtractor(embedding_client=client)

        # Latin-1 encoded text with non-UTF8 byte
        latin1_text = "café".encode("latin-1")
        result = await extractor.extract(latin1_text)

        assert result.extracted_text is not None


class TestJsonExtractor:
    """Tests for JsonExtractor."""

    async def test_extract_object(self) -> None:
        client = mock_embedding_client()
        extractor = JsonExtractor(embedding_client=client)

        data = json.dumps({"name": "test", "value": 42}).encode()
        result = await extractor.extract(data)

        assert result.text_embedding is not None
        assert result.hash_value is not None
        assert result.signature_type == "json"
        assert result.extra.get("is_array") is False
        assert result.extra.get("key_count") == 2

    async def test_extract_array(self) -> None:
        client = mock_embedding_client()
        extractor = JsonExtractor(embedding_client=client)

        data = json.dumps([{"id": 1}, {"id": 2}]).encode()
        result = await extractor.extract(data)

        assert result.extra.get("is_array") is True
        assert result.extra.get("key_count") == 1  # keys from first item

    async def test_extract_invalid_json(self) -> None:
        client = mock_embedding_client()
        extractor = JsonExtractor(embedding_client=client)

        result = await extractor.extract(b"not json")

        assert result.text_embedding is None
        assert result.hash_value is None

    async def test_schema_hash_is_stable(self) -> None:
        client = mock_embedding_client()
        extractor = JsonExtractor(embedding_client=client)

        # Same structure, different values
        data1 = json.dumps({"name": "alice", "age": 30}).encode()
        data2 = json.dumps({"name": "bob", "age": 25}).encode()

        result1 = await extractor.extract(data1)
        result2 = await extractor.extract(data2)

        # Same schema → same hash
        assert result1.hash_value == result2.hash_value

    async def test_schema_hash_differs_for_different_structure(self) -> None:
        client = mock_embedding_client()
        extractor = JsonExtractor(embedding_client=client)

        data1 = json.dumps({"name": "alice"}).encode()
        data2 = json.dumps({"name": "alice", "age": 30}).encode()

        result1 = await extractor.extract(data1)
        result2 = await extractor.extract(data2)

        # Different schema → different hash
        assert result1.hash_value != result2.hash_value

    def test_infer_schema_primitives(self) -> None:
        extractor = JsonExtractor()

        assert extractor._infer_schema(None) == {"type": "null"}
        assert extractor._infer_schema(True) == {"type": "boolean"}
        assert extractor._infer_schema(42) == {"type": "integer"}
        assert extractor._infer_schema(3.14) == {"type": "number"}
        assert extractor._infer_schema("hello") == {"type": "string"}

    def test_infer_schema_object(self) -> None:
        extractor = JsonExtractor()

        schema = extractor._infer_schema({"a": 1, "b": "x"})

        assert schema["type"] == "object"
        assert schema["properties"]["a"] == {"type": "integer"}
        assert schema["properties"]["b"] == {"type": "string"}

    def test_infer_schema_array(self) -> None:
        extractor = JsonExtractor()

        schema = extractor._infer_schema([1, 2, 3])

        assert schema["type"] == "array"
        assert schema["items"] == {"type": "integer"}

    def test_to_text_flattens(self) -> None:
        extractor = JsonExtractor()

        text = extractor._to_text({"user": {"name": "alice", "age": 30}})

        assert "user.name: alice" in text
        assert "user.age: 30" in text


class TestExtractionOrchestrator:
    """Tests for ExtractionOrchestrator."""

    async def test_extract_image(self) -> None:
        client = mock_embedding_client()
        orchestrator = ExtractionOrchestrator(embedding_client=client)

        image_bytes = create_test_image()
        result = await orchestrator.extract(image_bytes)

        assert result.image_embedding is not None
        assert result.extra.get("medium") == "image"

    async def test_extract_text(self) -> None:
        client = mock_embedding_client()
        orchestrator = ExtractionOrchestrator(embedding_client=client)

        result = await orchestrator.extract(b"Hello world", medium=Medium.TEXT)

        assert result.text_embedding is not None
        assert result.clip_text_embedding is not None
        assert result.extra.get("medium") == "text"

    async def test_extract_json(self) -> None:
        client = mock_embedding_client()
        orchestrator = ExtractionOrchestrator(embedding_client=client)

        data = json.dumps({"key": "value"}).encode()
        result = await orchestrator.extract(data, medium=Medium.JSON)

        assert result.text_embedding is not None
        assert result.hash_value is not None
        assert result.extra.get("medium") == "json"

    async def test_extract_probes_medium(self) -> None:
        client = mock_embedding_client()
        orchestrator = ExtractionOrchestrator(embedding_client=client)

        # JSON content should be auto-detected
        data = json.dumps({"auto": "detected"}).encode()
        result = await orchestrator.extract(data)

        assert result.extra.get("medium") == "json"

    async def test_extract_video_placeholder(self) -> None:
        client = mock_embedding_client()
        orchestrator = ExtractionOrchestrator(embedding_client=client)

        result = await orchestrator.extract(b"fake", medium=Medium.VIDEO)

        assert result.signature_type == "video"
        assert "not implemented" in result.extra.get("note", "").lower()

    async def test_extract_batch(self) -> None:
        client = mock_embedding_client()
        orchestrator = ExtractionOrchestrator(embedding_client=client)

        items = [
            (b"text content", Medium.TEXT, None),
            (json.dumps({"x": 1}).encode(), Medium.JSON, None),
        ]

        results = await orchestrator.extract_batch(items)

        assert len(results) == 2
        assert results[0].extra.get("medium") == "text"
        assert results[1].extra.get("medium") == "json"


@pytest.mark.integration
class TestExtractionIntegration:
    """Integration tests requiring running Sparkstation gateway.

    Run with: pytest -m integration
    """

    @pytest.fixture
    def orchestrator(self) -> ExtractionOrchestrator:
        return ExtractionOrchestrator()

    async def test_real_image_extraction(self, orchestrator: ExtractionOrchestrator) -> None:
        """Test real image extraction against Sparkstation."""
        image_bytes = create_test_image()
        result = await orchestrator.extract(image_bytes)

        assert result.image_embedding is not None
        assert len(result.image_embedding) == 768
        assert result.extra.get("width") == 100
        assert result.extra.get("height") == 100

    async def test_real_text_extraction(self, orchestrator: ExtractionOrchestrator) -> None:
        """Test real text extraction against Sparkstation."""
        result = await orchestrator.extract(b"A majestic leopard in the savanna")

        assert result.text_embedding is not None
        assert len(result.text_embedding) == 1024
        assert result.clip_text_embedding is not None
        assert len(result.clip_text_embedding) == 768

    async def test_real_json_extraction(self, orchestrator: ExtractionOrchestrator) -> None:
        """Test real JSON extraction against Sparkstation."""
        data = json.dumps({
            "species": "leopard",
            "location": "Serengeti",
            "count": 2,
        }).encode()

        result = await orchestrator.extract(data)

        assert result.text_embedding is not None
        assert result.hash_value is not None
        assert "species: leopard" in (result.extracted_text or "")
