"""Blob model for content-addressable storage and deduplication."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import BigInteger, DateTime, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from object_sense.models.base import Base

if TYPE_CHECKING:
    from object_sense.models.object import Object


class Blob(Base):
    """Content-addressable blob storage for deduplication.

    Blobs are identified by their SHA256 hash. Multiple Objects can
    reference the same Blob if they have identical content.
    """

    __tablename__ = "blobs"

    blob_id: Mapped[UUID] = mapped_column(primary_key=True)
    sha256: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    storage_path: Mapped[str | None] = mapped_column(String(1024))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    objects: Mapped[list[Object]] = relationship(back_populates="blob")
