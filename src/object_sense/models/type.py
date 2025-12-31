"""Type model for semantic types."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from pgvector.sqlalchemy import Vector  # type: ignore[import-untyped]
from sqlalchemy import DateTime, ForeignKey, Integer, String, func
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship

from object_sense.config import settings
from object_sense.models.base import Base
from object_sense.models.enums import TypeCreatedVia, TypeStatus

if TYPE_CHECKING:
    from object_sense.models.entity import Entity
    from object_sense.models.observation import Observation


class Type(Base):
    """Semantic type representing what something IS.

    Types are proposed by LLMs using their priors, created immediately
    (no threshold), and validated through recurrence. They can evolve
    via alias, merge, split, or deprecation.
    """

    __tablename__ = "types"

    type_id: Mapped[UUID] = mapped_column(primary_key=True)
    canonical_name: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    aliases: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    parent_type_id: Mapped[UUID | None] = mapped_column(ForeignKey("types.type_id"), index=True)
    embedding: Mapped[list[Any] | None] = mapped_column(Vector(settings.dim_text_embedding))
    status: Mapped[TypeStatus] = mapped_column(default=TypeStatus.PROVISIONAL)
    merged_into_type_id: Mapped[UUID | None] = mapped_column(ForeignKey("types.type_id"))
    evidence_count: Mapped[int] = mapped_column(Integer, default=0)
    created_via: Mapped[TypeCreatedVia] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Self-referential relationships
    parent_type: Mapped[Type | None] = relationship(
        "Type", remote_side="Type.type_id", foreign_keys=[parent_type_id]
    )
    merged_into: Mapped[Type | None] = relationship(
        "Type", remote_side="Type.type_id", foreign_keys=[merged_into_type_id]
    )

    # Relationships to other tables
    observations: Mapped[list[Observation]] = relationship(back_populates="primary_type")
    entities: Mapped[list[Entity]] = relationship(back_populates="type")
