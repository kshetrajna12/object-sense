"""Signature model for modality-specific fingerprints."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from pgvector.sqlalchemy import Vector  # type: ignore[import-untyped]
from sqlalchemy import DateTime, ForeignKey, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from object_sense.config import settings
from object_sense.models.base import Base

if TYPE_CHECKING:
    from object_sense.models.object import Object


class Signature(Base):
    """Modality-specific fingerprints for identity and similarity.

    Embedding strategy (late fusion):
    - text_embedding (1024): BGE embedding for rich text semantics
    - image_embedding (768): CLIP visual embedding for images
    - clip_text_embedding (768): CLIP text embedding for cross-modal queries

    Usage by medium:
    - Image: image_embedding (CLIP visual) + text_embedding (caption→BGE)
    - Text: text_embedding (BGE) + clip_text_embedding (CLIP text for cross-modal)
    - JSON: text_embedding (fields→BGE) + hash_value (schema hash)
    - Video: image_embedding (keyframe pool) + text_embedding (transcript→BGE)

    Entity resolution combines signals: hard IDs > same-modality sim > cross-modal sim.
    """

    __tablename__ = "signatures"

    signature_id: Mapped[UUID] = mapped_column(primary_key=True)
    object_id: Mapped[UUID] = mapped_column(ForeignKey("objects.object_id"), index=True)
    signature_type: Mapped[str] = mapped_column(String(64), index=True)
    hash_value: Mapped[str | None] = mapped_column(String(256), index=True)

    # Embedding columns (late fusion strategy)
    text_embedding: Mapped[list[Any] | None] = mapped_column(
        Vector(settings.dim_text_embedding)
    )  # BGE 1024-dim: rich text semantics
    image_embedding: Mapped[list[Any] | None] = mapped_column(
        Vector(settings.dim_image_embedding)
    )  # CLIP 768-dim: visual features
    clip_text_embedding: Mapped[list[Any] | None] = mapped_column(
        Vector(settings.dim_clip_text_embedding)
    )  # CLIP 768-dim: text for cross-modal matching

    extra: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    object: Mapped[Object] = relationship(back_populates="signatures")
