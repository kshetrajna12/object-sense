"""Entity model for persistent concepts."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import DateTime, Float, ForeignKey, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from object_sense.models.base import Base
from object_sense.models.enums import EntityStatus

if TYPE_CHECKING:
    from object_sense.models.observation import ObservationEntityLink
    from object_sense.models.type import Type


class Entity(Base):
    """A persistent concept that observations link to.

    Entities are cluster nodes representing stable concepts - either concrete
    (The Marula Leopard) or abstract (Leopard Species). Objects link TO entities;
    entities are not merges of objects into each other.
    """

    __tablename__ = "entities"

    entity_id: Mapped[UUID] = mapped_column(primary_key=True)
    type_id: Mapped[UUID | None] = mapped_column(ForeignKey("types.type_id"), index=True)
    slots: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    status: Mapped[EntityStatus] = mapped_column(default=EntityStatus.PROTO)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    type: Mapped[Type | None] = relationship(back_populates="entities")
    observation_links: Mapped[list[ObservationEntityLink]] = relationship(back_populates="entity")
