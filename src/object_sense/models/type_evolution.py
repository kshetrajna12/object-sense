"""TypeEvolution model for tracking type changes over time."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import DateTime, ForeignKey, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from object_sense.models.base import Base
from object_sense.models.enums import TypeEvolutionKind


class TypeEvolution(Base):
    """History of type changes (alias, merge, split, deprecate).

    Types are always provisional and subject to evolution. This table
    tracks what changed, when, and why.
    """

    __tablename__ = "type_evolution"

    evolution_id: Mapped[UUID] = mapped_column(primary_key=True)
    kind: Mapped[TypeEvolutionKind] = mapped_column(index=True)
    source_type_id: Mapped[UUID] = mapped_column(ForeignKey("types.type_id"), index=True)
    target_type_id: Mapped[UUID | None] = mapped_column(ForeignKey("types.type_id"), index=True)
    reason: Mapped[str | None] = mapped_column(String(1024))
    details: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
