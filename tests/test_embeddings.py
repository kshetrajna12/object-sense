"""Tests for the embedding client."""

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from object_sense.clients.embeddings import EmbeddingClient


class MockEmbeddingData:
    """Mock for openai embedding response data item."""

    def __init__(self, embedding: list[float]) -> None:
        self.embedding = embedding


class MockEmbeddingResponse:
    """Mock for openai embedding response."""

    def __init__(self, embeddings: list[list[float]]) -> None:
        self.data = [MockEmbeddingData(e) for e in embeddings]


class TestEmbeddingClient:
    """Unit tests for EmbeddingClient with mocked OpenAI client."""

    async def test_embed_text_empty(self) -> None:
        client = EmbeddingClient()
        result = await client.embed_text([])
        assert result == []

    async def test_embed_image_empty(self) -> None:
        client = EmbeddingClient()
        result = await client.embed_image([])
        assert result == []

    async def test_embed_text_clip_empty(self) -> None:
        client = EmbeddingClient()
        result = await client.embed_text_clip([])
        assert result == []

    async def test_embed_text_single(self) -> None:
        mock_embedding = [0.1] * 1024
        mock_response = MockEmbeddingResponse([mock_embedding])

        with patch.object(EmbeddingClient, "__init__", lambda self, **kwargs: None):
            client = EmbeddingClient()
            client._batch_size = 32
            client._client = MagicMock()
            client._client.embeddings = MagicMock()
            client._client.embeddings.create = AsyncMock(return_value=mock_response)

            result = await client.embed_text(["hello world"])

            assert len(result) == 1
            assert len(result[0]) == 1024
            client._client.embeddings.create.assert_called_once()

    async def test_embed_text_batching(self) -> None:
        """Test that large inputs are split into batches."""
        mock_embedding = [0.1] * 1024
        mock_response = MockEmbeddingResponse([mock_embedding] * 2)

        with patch.object(EmbeddingClient, "__init__", lambda self, **kwargs: None):
            client = EmbeddingClient()
            client._batch_size = 2  # Small batch size for testing
            client._client = MagicMock()
            client._client.embeddings = MagicMock()
            client._client.embeddings.create = AsyncMock(return_value=mock_response)

            # 5 texts should result in 3 batches (2, 2, 1)
            texts = ["text" + str(i) for i in range(5)]

            # Need different responses for different batch sizes
            responses = [
                MockEmbeddingResponse([mock_embedding] * 2),
                MockEmbeddingResponse([mock_embedding] * 2),
                MockEmbeddingResponse([mock_embedding] * 1),
            ]
            client._client.embeddings.create = AsyncMock(side_effect=responses)

            result = await client.embed_text(texts)

            assert len(result) == 5
            assert client._client.embeddings.create.call_count == 3

    async def test_embed_image_bytes_to_base64(self) -> None:
        """Test that byte images are converted to base64."""
        mock_embedding = [0.2] * 768
        mock_response = MockEmbeddingResponse([mock_embedding])

        with patch.object(EmbeddingClient, "__init__", lambda self, **kwargs: None):
            client = EmbeddingClient()
            client._batch_size = 32
            client._client = MagicMock()
            client._client.embeddings = MagicMock()
            client._client.embeddings.create = AsyncMock(return_value=mock_response)

            image_bytes = b"fake image data"
            result = await client.embed_image([image_bytes])

            assert len(result) == 1
            # Verify the call used structured format with base64
            call_args = client._client.embeddings.create.call_args
            input_arg = call_args.kwargs["input"]
            assert len(input_arg) == 1
            assert "image" in input_arg[0]
            assert input_arg[0]["image"] == base64.b64encode(image_bytes).decode("utf-8")

    async def test_embed_image_url_passthrough(self) -> None:
        """Test that URL strings are passed through unchanged."""
        mock_embedding = [0.2] * 768
        mock_response = MockEmbeddingResponse([mock_embedding])

        with patch.object(EmbeddingClient, "__init__", lambda self, **kwargs: None):
            client = EmbeddingClient()
            client._batch_size = 32
            client._client = MagicMock()
            client._client.embeddings = MagicMock()
            client._client.embeddings.create = AsyncMock(return_value=mock_response)

            image_url = "https://example.com/image.jpg"
            result = await client.embed_image([image_url])

            assert len(result) == 1
            call_args = client._client.embeddings.create.call_args
            input_arg = call_args.kwargs["input"]
            assert input_arg[0]["image"] == image_url

    async def test_embed_text_clip(self) -> None:
        """Test cross-modal text embeddings."""
        mock_embedding = [0.3] * 768
        mock_response = MockEmbeddingResponse([mock_embedding])

        with patch.object(EmbeddingClient, "__init__", lambda self, **kwargs: None):
            client = EmbeddingClient()
            client._batch_size = 32
            client._client = MagicMock()
            client._client.embeddings = MagicMock()
            client._client.embeddings.create = AsyncMock(return_value=mock_response)

            result = await client.embed_text_clip(["a red car"])

            assert len(result) == 1
            assert len(result[0]) == 768

    def test_to_clip_image_input_bytes(self) -> None:
        """Test _to_clip_image_input with bytes input."""
        client = EmbeddingClient()
        image_bytes = b"test data"

        result = client._to_clip_image_input([image_bytes])

        assert len(result) == 1
        assert result[0]["image"] == base64.b64encode(image_bytes).decode("utf-8")

    def test_to_clip_image_input_string(self) -> None:
        """Test _to_clip_image_input with string input."""
        client = EmbeddingClient()
        image_url = "https://example.com/image.png"

        result = client._to_clip_image_input([image_url])

        assert len(result) == 1
        assert result[0]["image"] == image_url

    def test_batches_exact_division(self) -> None:
        """Test batching with exact division."""
        client = EmbeddingClient(batch_size=2)
        items = [1, 2, 3, 4]

        result = client._batches(items)

        assert result == [[1, 2], [3, 4]]

    def test_batches_with_remainder(self) -> None:
        """Test batching with remainder."""
        client = EmbeddingClient(batch_size=3)
        items = [1, 2, 3, 4, 5]

        result = client._batches(items)

        assert result == [[1, 2, 3], [4, 5]]

    def test_batches_single_batch(self) -> None:
        """Test batching when items fit in single batch."""
        client = EmbeddingClient(batch_size=10)
        items = [1, 2, 3]

        result = client._batches(items)

        assert result == [[1, 2, 3]]


@pytest.mark.integration
class TestEmbeddingClientIntegration:
    """Integration tests that require running Sparkstation gateway.

    Run with: pytest -m integration
    Skip with: pytest -m "not integration"
    """

    @pytest.fixture
    def client(self) -> EmbeddingClient:
        return EmbeddingClient()

    async def test_embed_text_real(self, client: EmbeddingClient) -> None:
        """Test real text embedding against Sparkstation."""
        texts = ["Hello world", "This is a test"]
        result = await client.embed_text(texts)

        assert len(result) == 2
        assert all(len(e) == 1024 for e in result)  # bge-large dimension
        assert all(isinstance(v, float) for e in result for v in e)

    async def test_embed_text_clip_real(self, client: EmbeddingClient) -> None:
        """Test real CLIP text embedding against Sparkstation."""
        texts = ["a photo of a cat"]
        result = await client.embed_text_clip(texts)

        assert len(result) == 1
        assert len(result[0]) == 768  # CLIP dimension
