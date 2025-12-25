"""Feature extraction orchestrator.

Dispatches extraction based on medium and affordances.
This is the main entry point for Step 3 of the processing loop.
"""

from __future__ import annotations

from object_sense.clients.embeddings import EmbeddingClient
from object_sense.extraction.base import ExtractionResult
from object_sense.extraction.image import ImageExtractor
from object_sense.extraction.json_extract import JsonExtractor
from object_sense.extraction.text import TextExtractor
from object_sense.models.enums import Affordance, Medium
from object_sense.utils.medium import get_affordances, probe_medium


class ExtractionOrchestrator:
    """Orchestrate feature extraction based on medium affordances.

    The orchestrator:
    1. Probes medium if not provided
    2. Gets affordances for the medium
    3. Dispatches to appropriate extractors
    4. Merges results into unified ExtractionResult

    Usage:
        orchestrator = ExtractionOrchestrator()
        result = await orchestrator.extract(image_bytes)
        # result.image_embedding, result.extra["exif"], etc.
    """

    def __init__(self, embedding_client: EmbeddingClient | None = None) -> None:
        self._client = embedding_client or EmbeddingClient()

        # Initialize extractors with shared client
        self._image_extractor = ImageExtractor(self._client)
        self._text_extractor = TextExtractor(self._client)
        self._json_extractor = JsonExtractor(self._client)

    async def extract(
        self,
        data: bytes,
        *,
        medium: Medium | None = None,
        filename: str | None = None,
    ) -> ExtractionResult:
        """Extract features from content.

        Args:
            data: Raw content bytes
            medium: Pre-detected medium (probed if not provided)
            filename: Optional filename hint

        Returns:
            ExtractionResult with all extracted features
        """
        # Probe medium if not provided
        if medium is None:
            medium = probe_medium(data, filename=filename)

        # Get affordances for this medium
        affordances = get_affordances(medium)

        # Dispatch to appropriate extractor
        result = await self._dispatch(data, medium, affordances, filename)

        # Add medium to extra for reference
        result.extra["medium"] = medium.value

        return result

    async def _dispatch(
        self,
        data: bytes,
        medium: Medium,
        affordances: frozenset[Affordance],
        filename: str | None,
    ) -> ExtractionResult:
        """Dispatch to appropriate extractor based on medium."""

        # Image extraction
        if medium == Medium.IMAGE:
            return await self._image_extractor.extract(data, filename=filename)

        # Text extraction
        if medium == Medium.TEXT:
            return await self._text_extractor.extract(data, filename=filename)

        # JSON extraction
        if medium == Medium.JSON:
            return await self._json_extractor.extract(data, filename=filename)

        # Video - not implemented in v0, return placeholder
        if medium == Medium.VIDEO:
            return ExtractionResult(
                signature_type="video",
                extra={"note": "Video extraction not implemented in v0"},
            )

        # Audio - not implemented in v0, return placeholder
        if medium == Medium.AUDIO:
            return ExtractionResult(
                signature_type="audio",
                extra={"note": "Audio extraction not implemented in v0"},
            )

        # Binary - minimal extraction
        if medium == Medium.BINARY:
            return ExtractionResult(
                signature_type="binary",
                extra={"size_bytes": len(data)},
            )

        # Unknown medium
        return ExtractionResult(signature_type="unknown")

    async def extract_batch(
        self,
        items: list[tuple[bytes, Medium | None, str | None]],
    ) -> list[ExtractionResult]:
        """Extract features from multiple items.

        Args:
            items: List of (data, medium, filename) tuples

        Returns:
            List of ExtractionResults in same order

        Note:
            Currently processes sequentially. Could be parallelized
            but embedding client already batches internally.
        """
        results = []
        for data, medium, filename in items:
            result = await self.extract(data, medium=medium, filename=filename)
            results.append(result)
        return results
