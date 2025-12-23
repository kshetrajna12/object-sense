"""Object model for observations in the world model."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import DateTime, ForeignKey, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from object_sense.models.base import Base
from object_sense.models.enums import Medium, ObjectStatus

if TYPE_CHECKING:
    from object_sense.models.blob import Blob
    from object_sense.models.entity import Entity
    from object_sense.models.signature import Signature
    from object_sense.models.type import Type


class Object(Base):
    """A unit of observation in the world model.

    Objects are things we observed (files, rows, records, frames, etc.).
    They link TO entities - they are not entities themselves. An Object
    has a medium (how it's encoded) and a primary_type (what it IS).
    """

    __tablename__ = "objects"

    object_id: Mapped[UUID] = mapped_column(primary_key=True)
    medium: Mapped[Medium] = mapped_column(index=True)
    primary_type_id: Mapped[UUID | None] = mapped_column(ForeignKey("types.type_id"), index=True)
    source_id: Mapped[str] = mapped_column(String(2048), index=True)
    blob_id: Mapped[UUID | None] = mapped_column(ForeignKey("blobs.blob_id"), index=True)
    slots: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    status: Mapped[ObjectStatus] = mapped_column(default=ObjectStatus.ACTIVE)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    blob: Mapped[Blob | None] = relationship(back_populates="objects")
    primary_type: Mapped[Type | None] = relationship(back_populates="objects")
    entity_links: Mapped[list[ObjectEntityLink]] = relationship(back_populates="object")
    signatures: Mapped[list[Signature]] = relationship(back_populates="object")


class ObjectEntityLink(Base):
    """Association between Objects and Entities.

    Represents which entities an observation refers to. An Object can
    link to multiple Entities (e.g., a photo depicting multiple animals).
    """

    __tablename__ = "object_entity_links"

    object_id: Mapped[UUID] = mapped_column(ForeignKey("objects.object_id"), primary_key=True)
    entity_id: Mapped[UUID] = mapped_column(ForeignKey("entities.entity_id"), primary_key=True)
    role: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    object: Mapped[Object] = relationship(back_populates="entity_links")
    entity: Mapped[Entity] = relationship(back_populates="object_links")
