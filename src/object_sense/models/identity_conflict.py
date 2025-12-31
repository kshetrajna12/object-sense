"""IdentityConflict model for flagging ambiguous identity cases."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import DateTime, Float, ForeignKey, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from object_sense.models.base import Base
from object_sense.models.enums import IdentityConflictStatus

if TYPE_CHECKING:
    from object_sense.models.entity import Entity
    from object_sense.models.observation import Observation


class IdentityConflict(Base):
    """Records conflicting identity signals that need resolution.

    See design_v2_corrections.md §8 for the deterministic ID priority system.

    When two observations have the same deterministic ID but different entity
    assignments, or when an ID conflict is detected, we record it here for
    human review rather than making an arbitrary choice.

    Examples:
    - Same SKU but different product descriptions
    - Same GPS coords but different timestamps
    - Re-used ID in different namespaces
    """

    __tablename__ = "identity_conflicts"

    conflict_id: Mapped[UUID] = mapped_column(primary_key=True)

    id_type: Mapped[str] = mapped_column(String(64), index=True)
    """Type of the conflicting ID (e.g., "sku", "trip_id", "gps")."""

    id_value: Mapped[str] = mapped_column(String(512), index=True)
    """The actual ID value that's in conflict."""

    id_namespace: Mapped[str | None] = mapped_column(String(255))
    """Optional namespace (e.g., "safari_outfitter_1", "internal_catalog")."""

    observation_a_id: Mapped[UUID] = mapped_column(
        ForeignKey("observations.observation_id"), index=True
    )
    """First observation in the conflict."""

    observation_b_id: Mapped[UUID] = mapped_column(
        ForeignKey("observations.observation_id"), index=True
    )
    """Second observation in the conflict."""

    entity_a_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("entities.entity_id"), index=True
    )
    """Entity linked to observation A (if any)."""

    entity_b_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("entities.entity_id"), index=True
    )
    """Entity linked to observation B (if any)."""

    status: Mapped[IdentityConflictStatus] = mapped_column(
        default=IdentityConflictStatus.PENDING
    )
    """Resolution status: pending | resolved | ignored."""

    divergence_score: Mapped[float | None] = mapped_column(Float)
    """How different the conflicting observations are (0=identical, 1=completely different)."""

    resolution: Mapped[str | None] = mapped_column(String(32))
    """How it was resolved: "a_wins", "b_wins", "merge", "keep_separate", "ignore"."""

    resolved_by: Mapped[str | None] = mapped_column(String(64))
    """Who/what resolved it: "auto", "human:<user_id>", "algorithm:<version>"."""

    details: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    """Additional context (e.g., conflicting attributes, resolution reasoning)."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Relationships
    observation_a: Mapped[Observation] = relationship(foreign_keys=[observation_a_id])
    observation_b: Mapped[Observation] = relationship(foreign_keys=[observation_b_id])
    entity_a: Mapped[Entity | None] = relationship(foreign_keys=[entity_a_id])
    entity_b: Mapped[Entity | None] = relationship(foreign_keys=[entity_b_id])
