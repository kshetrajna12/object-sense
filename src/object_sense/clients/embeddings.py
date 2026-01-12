"""Async client wrapper for Sparkstation embedding endpoints."""

import base64
import logging
import time
from collections.abc import Sequence
from typing import Any

from openai import AsyncOpenAI

from object_sense.config import settings

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """Async client for generating text and image embeddings via Sparkstation.

    Uses OpenAI-compatible API with local LLM gateway.

    Models:
        - bge-large: Text embeddings (1024 dimensions)
        - clip-vit: Image embeddings and cross-modal text (768 dimensions)
    """

    # Maximum items per batch request
    DEFAULT_BATCH_SIZE = 32

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        # Primary client uses configured embedding provider
        self._client = AsyncOpenAI(
            base_url=base_url or settings.embedding_base_url,
            api_key=api_key or settings.embedding_api_key,
        )

        # Image embeddings always use Sparkstation (cloud providers don't support)
        # Create separate client if text embedding provider is not Sparkstation
        if settings.embedding_provider != "sparkstation":
            self._image_client = AsyncOpenAI(
                base_url=settings.sparkstation_base_url,
                api_key=settings.sparkstation_api_key,
            )
        else:
            self._image_client = self._client  # Same client for both

        self._batch_size = batch_size

    async def embed_text(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate text embeddings using bge-large model.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (1024 dimensions each).
        """
        if not texts:
            return []

        start_time = time.time()
        embeddings: list[list[float]] = []

        for batch in self._batches(list(texts)):
            response = await self._client.embeddings.create(
                model=settings.model_text_embedding,
                input=batch,
            )
            # Results are returned in order of input
            embeddings.extend([item.embedding for item in response.data])

        if settings.log_api_calls:
            elapsed = (time.time() - start_time) * 1000  # ms
            provider = settings.embedding_provider
            model = settings.model_text_embedding
            dims = settings.dim_text_embedding
            logger.info(
                "[EMBED] %s @ %s (%d items) → %d-dim (%.0fms)",
                model, provider, len(texts), dims, elapsed
            )

        return embeddings

    async def embed_image(self, images: Sequence[bytes | str]) -> list[list[float]]:
        """Generate image embeddings using CLIP model.

        Args:
            images: List of images as bytes (raw image data) or str (URL or base64).

        Returns:
            List of embedding vectors (768 dimensions each).

        Note:
            Image embeddings always use Sparkstation CLIP (cloud providers don't support).
            CLIP requires structured array format: [{"image": "..."}]
        """
        if not images:
            return []

        start_time = time.time()
        embeddings: list[list[float]] = []

        for batch in self._batches(list(images)):
            # Convert to CLIP's structured format
            structured_input = self._to_clip_image_input(batch)

            # Always use image client (Sparkstation) for CLIP embeddings
            response = await self._image_client.embeddings.create(
                model=settings.model_image_embedding,
                input=structured_input,  # type: ignore[arg-type]
            )
            embeddings.extend([item.embedding for item in response.data])

        if settings.log_api_calls:
            elapsed = (time.time() - start_time) * 1000  # ms
            model = settings.model_image_embedding
            dims = settings.dim_image_embedding
            logger.info(
                "[EMBED] %s @ sparkstation (%d images) → %d-dim (%.0fms)",
                model, len(images), dims, elapsed
            )

        return embeddings

    async def embed_text_clip(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate text embeddings using CLIP model for cross-modal search.

        Use this when you need text embeddings in the same space as image embeddings
        (e.g., searching images by text description).

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (768 dimensions each).

        Note:
            CLIP text embeddings always use Sparkstation (for cross-modal consistency).
        """
        if not texts:
            return []

        start_time = time.time()
        embeddings: list[list[float]] = []

        for batch in self._batches(list(texts)):
            # Use image client (Sparkstation) for CLIP cross-modal embeddings
            response = await self._image_client.embeddings.create(
                model=settings.model_image_embedding,  # clip-vit for cross-modal
                input=batch,
            )
            embeddings.extend([item.embedding for item in response.data])

        if settings.log_api_calls:
            elapsed = (time.time() - start_time) * 1000  # ms
            model = settings.model_image_embedding
            logger.info(
                "[EMBED] %s @ sparkstation (%d texts, cross-modal) → 768-dim (%.0fms)",
                model, len(texts), elapsed
            )

        return embeddings

    def _to_clip_image_input(self, images: list[bytes | str]) -> list[dict[str, str]]:
        """Convert images to CLIP's required structured format."""
        result: list[dict[str, str]] = []

        for img in images:
            if isinstance(img, bytes):
                # Raw bytes → base64 encode
                b64 = base64.b64encode(img).decode("utf-8")
                result.append({"image": b64})
            else:
                # Already a string (URL or base64)
                result.append({"image": img})

        return result

    def _batches(self, items: list[Any]) -> list[list[Any]]:
        """Split items into batches."""
        return [items[i : i + self._batch_size] for i in range(0, len(items), self._batch_size)]
