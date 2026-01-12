"""Event participant relationship model.

Links EVENT entities to their participant INDIVIDUAL entities.
This is the canonical edge for cross-modal participant inference.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import DateTime, Float, ForeignKey, Index, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from object_sense.models.base import Base

if TYPE_CHECKING:
    from object_sense.models.entity import Entity
    from object_sense.models.observation import Observation


class EventParticipant(Base):
    """Junction table linking EVENT entities to participant INDIVIDUAL entities.

    Created during cross-modal EVENT alignment when:
    - An IMAGE observation links to an EVENT (via LinkRole.CONTEXT)
    - That same observation links to a photo_subject INDIVIDUAL
    - The photo_subject is inferred to be a participant in the EVENT

    This enables querying:
    - "Who participated in this event?" → list participants by event_entity_id
    - "What events did this individual participate in?" → list events by participant_entity_id
    """

    __tablename__ = "event_participants"

    id: Mapped[UUID] = mapped_column(primary_key=True)

    event_entity_id: Mapped[UUID] = mapped_column(
        ForeignKey("entities.entity_id"), index=True
    )
    """The EVENT entity."""

    participant_entity_id: Mapped[UUID] = mapped_column(
        ForeignKey("entities.entity_id"), index=True
    )
    """The INDIVIDUAL entity participating in the event."""

    source_observation_id: Mapped[UUID] = mapped_column(
        ForeignKey("observations.observation_id"), index=True
    )
    """The observation that introduced this participant link."""

    role: Mapped[str | None] = mapped_column(String(64), nullable=True)
    """Optional role of participant: 'subject', 'victim', 'observer', etc."""

    confidence: Mapped[float] = mapped_column(Float, default=0.8)
    """Confidence in this participant inference (derived from link posteriors)."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    event_entity: Mapped[Entity] = relationship(
        foreign_keys=[event_entity_id],
        back_populates="event_participations",
    )
    participant_entity: Mapped[Entity] = relationship(
        foreign_keys=[participant_entity_id],
        back_populates="participated_events",
    )
    source_observation: Mapped[Observation] = relationship(
        back_populates="event_participant_inferences",
    )

    __table_args__ = (
        # Unique constraint: one participant per event per observation
        # (same participant can be linked from multiple observations)
        Index(
            "ix_event_participant_unique",
            "event_entity_id",
            "participant_entity_id",
            "source_observation_id",
            unique=True,
        ),
    )
