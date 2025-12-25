"""Text feature extraction.

Handles:
- CAN_EMBED_TEXT: BGE text embeddings for semantic search
- CAN_CHUNK: Text chunking for long documents (basic implementation)
- Cross-modal: CLIP text embeddings for image search
"""

from __future__ import annotations

from object_sense.clients.embeddings import EmbeddingClient
from object_sense.extraction.base import ExtractionResult

# Reasonable chunk size for embedding models
DEFAULT_MAX_CHARS = 8000  # ~2000 tokens, safe for most models


class TextExtractor:
    """Extract features from text content.

    Features extracted:
    - text_embedding: BGE embedding (1024-dim) for semantic search
    - clip_text_embedding: CLIP text embedding (768-dim) for cross-modal search
    - extracted_text: The text content (possibly truncated)
    - extra.char_count: Character count
    - extra.truncated: Whether text was truncated
    """

    def __init__(
        self,
        embedding_client: EmbeddingClient | None = None,
        max_chars: int = DEFAULT_MAX_CHARS,
    ) -> None:
        self._client = embedding_client or EmbeddingClient()
        self._max_chars = max_chars

    async def extract(self, data: bytes, *, filename: str | None = None) -> ExtractionResult:
        """Extract features from text bytes.

        Args:
            data: Raw text bytes (UTF-8 or latin-1)
            filename: Optional filename (unused)

        Returns:
            ExtractionResult with text embeddings
        """
        result = ExtractionResult(signature_type="text")

        # Decode text
        text = self._decode_text(data)
        if not text:
            return result

        original_len = len(text)
        result.extra["char_count"] = original_len

        # Truncate if needed (take beginning for now - could be smarter)
        if len(text) > self._max_chars:
            text = text[: self._max_chars]
            result.extra["truncated"] = True
        else:
            result.extra["truncated"] = False

        result.extracted_text = text

        # Generate embeddings
        # BGE for rich semantic search
        text_embeddings = await self._client.embed_text([text])
        if text_embeddings:
            result.text_embedding = text_embeddings[0]

        # CLIP text for cross-modal search (find images matching this text)
        clip_embeddings = await self._client.embed_text_clip([text])
        if clip_embeddings:
            result.clip_text_embedding = clip_embeddings[0]

        return result

    def _decode_text(self, data: bytes) -> str | None:
        """Decode bytes to text, trying UTF-8 then latin-1."""
        if not data:
            return None

        # Try UTF-8 first
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            pass

        # Fall back to latin-1 (accepts any byte sequence)
        try:
            return data.decode("latin-1")
        except UnicodeDecodeError:
            return None
