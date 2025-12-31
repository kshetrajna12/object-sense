"""EntityEvolution model for tracking entity resolution decisions."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import DateTime, Float, ForeignKey, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from object_sense.models.base import Base
from object_sense.models.enums import EntityEvolutionKind

if TYPE_CHECKING:
    from object_sense.models.entity import Entity
    from object_sense.models.observation import Observation


class EntityEvolution(Base):
    """Records decisions made by the entity resolution algorithm.

    This is the audit trail for entity resolution (see design_v2_corrections.md §5).
    Every link, merge, split, or rejection is recorded here with full context.

    Used for:
    - Debugging why two observations are (or aren't) linked
    - Reprocessing if algorithm changes
    - Human review of resolution decisions
    """

    __tablename__ = "entity_evolutions"

    evolution_id: Mapped[UUID] = mapped_column(primary_key=True)

    entity_id: Mapped[UUID] = mapped_column(
        ForeignKey("entities.entity_id"), index=True
    )
    """The entity this evolution record is about."""

    kind: Mapped[EntityEvolutionKind] = mapped_column(index=True)
    """Type of evolution: link | merge | split | reject."""

    observation_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("observations.observation_id"), index=True
    )
    """For LINK operations: the observation being linked."""

    other_entity_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("entities.entity_id"), index=True
    )
    """For MERGE/SPLIT: the other entity involved."""

    algorithm_version: Mapped[str] = mapped_column(String(64))
    """Version of the resolution algorithm that made this decision."""

    match_score: Mapped[float | None] = mapped_column(Float)
    """The computed similarity/match score that drove this decision."""

    threshold_used: Mapped[float | None] = mapped_column(Float)
    """The threshold value used for comparison."""

    signals_used: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    """Breakdown of signals that contributed (e.g., {"visual": 0.8, "id_match": 1.0})."""

    reasoning: Mapped[str | None] = mapped_column(String(2048))
    """Human-readable explanation of the decision."""

    details: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    """Additional context (e.g., competing candidates, near misses)."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    entity: Mapped[Entity] = relationship(
        back_populates="evolutions",
        foreign_keys=[entity_id],
    )
    observation: Mapped[Observation | None] = relationship()
    other_entity: Mapped[Entity | None] = relationship(
        foreign_keys=[other_entity_id],
    )
