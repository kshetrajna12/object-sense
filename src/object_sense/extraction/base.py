"""Base types for feature extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ExtractionResult:
    """Result of feature extraction for an object.

    Holds all extracted features that will be stored in Signature records.
    Each field is optional — extractors populate what they can.

    Embedding strategy (late fusion):
    - text_embedding (1024-dim BGE): Rich text semantics
    - image_embedding (768-dim CLIP): Visual features
    - clip_text_embedding (768-dim CLIP): Text in visual space for cross-modal

    Entity resolution uses: hard IDs > same-modality sim > cross-modal sim
    """

    # Embeddings (late fusion)
    text_embedding: list[float] | None = None  # BGE 1024-dim
    image_embedding: list[float] | None = None  # CLIP visual 768-dim
    clip_text_embedding: list[float] | None = None  # CLIP text 768-dim

    # Hashes for identity
    hash_value: str | None = None  # Schema hash, content hash, etc.

    # Signature type identifier
    signature_type: str = "extracted"

    # Extracted text (for downstream use)
    extracted_text: str | None = None  # Caption, parsed text, JSON fields

    # Additional metadata
    extra: dict[str, Any] = field(default_factory=dict)

    def merge(self, other: ExtractionResult) -> ExtractionResult:
        """Merge another result into this one. Other's values take precedence for non-None."""
        return ExtractionResult(
            text_embedding=other.text_embedding or self.text_embedding,
            image_embedding=other.image_embedding or self.image_embedding,
            clip_text_embedding=other.clip_text_embedding or self.clip_text_embedding,
            hash_value=other.hash_value or self.hash_value,
            signature_type=(
                other.signature_type if other.signature_type != "extracted" else self.signature_type
            ),
            extracted_text=other.extracted_text or self.extracted_text,
            extra={**self.extra, **other.extra},
        )


class Extractor(Protocol):
    """Protocol for feature extractors.

    Each extractor handles specific affordances for a medium.
    Extractors are async to support IO-bound operations (embedding APIs, file parsing).
    """

    async def extract(self, data: bytes, *, filename: str | None = None) -> ExtractionResult:
        """Extract features from raw content.

        Args:
            data: Raw file/content bytes.
            filename: Optional filename hint for parsing.

        Returns:
            Extraction result with populated features.
        """
        ...
