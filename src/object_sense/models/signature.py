"""Signature model for modality-specific fingerprints."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from pgvector.sqlalchemy import Vector  # type: ignore[import-untyped]
from sqlalchemy import DateTime, ForeignKey, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from object_sense.models.base import Base

if TYPE_CHECKING:
    from object_sense.models.object import Object


class Signature(Base):
    """Modality-specific fingerprints for identity and similarity.

    Different mediums produce different signatures:
    - Image: SHA256, pHash/dHash, vision embedding (CLIP), EXIF summary
    - Video: SHA256, keyframe pHash, pooled video embedding, duration
    - Text: SHA256 (normalized), simhash, text embedding
    - JSON: schema hash, content hash (canonicalized), text embedding
    """

    __tablename__ = "signatures"

    signature_id: Mapped[UUID] = mapped_column(primary_key=True)
    object_id: Mapped[UUID] = mapped_column(ForeignKey("objects.object_id"), index=True)
    signature_type: Mapped[str] = mapped_column(String(64), index=True)
    hash_value: Mapped[str | None] = mapped_column(String(256), index=True)
    embedding: Mapped[list[Any] | None] = mapped_column(Vector(1536))
    extra: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    object: Mapped[Object] = relationship(back_populates="signatures")
