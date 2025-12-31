"""TypeCandidate model for provisional type labels."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from object_sense.models.base import Base
from object_sense.models.enums import TypeCandidateStatus

if TYPE_CHECKING:
    from object_sense.models.observation import Observation
    from object_sense.models.type import Type


class TypeCandidate(Base):
    """Provisional type label before promotion to stable Type.

    TypeCandidates are created immediately when the LLM proposes a type label.
    They go through a lifecycle: proposed → promoted | merged | rejected.

    See design_v2_corrections.md §4 and §7 for the two-stage Type lifecycle.

    Key design decisions:
    - No UNIQUE constraint on proposed_name (duplicates allowed, merge handles consolidation)
    - normalized_name index for dedup detection
    - merged_into_candidate_id for merge chains
    """

    __tablename__ = "type_candidates"

    candidate_id: Mapped[UUID] = mapped_column(primary_key=True)

    proposed_name: Mapped[str] = mapped_column(String(255))
    """The type name as proposed by the LLM (snake_case)."""

    normalized_name: Mapped[str] = mapped_column(String(255), index=True)
    """Lowercase, stripped version for dedup detection."""

    status: Mapped[TypeCandidateStatus] = mapped_column(
        default=TypeCandidateStatus.PROPOSED
    )
    """Lifecycle status: proposed | promoted | merged | rejected."""

    merged_into_candidate_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("type_candidates.candidate_id"), index=True
    )
    """If merged, points to the target candidate. Allows merge chain resolution."""

    promoted_to_type_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("types.type_id"), index=True
    )
    """If promoted, points to the stable Type created from this candidate."""

    evidence_count: Mapped[int] = mapped_column(Integer, default=1)
    """Number of observations/entities referencing this candidate."""

    coherence_score: Mapped[float | None] = mapped_column(Float)
    """Cluster tightness score, computed periodically for promotion decisions."""

    details: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    """Additional metadata (e.g., reasoning from LLM, merge history)."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    last_referenced_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    """Last time an observation used this candidate. Used for cleanup/GC."""

    # Relationships
    merged_into: Mapped[TypeCandidate | None] = relationship(
        "TypeCandidate",
        remote_side="TypeCandidate.candidate_id",
        foreign_keys=[merged_into_candidate_id],
    )
    promoted_to: Mapped[Type | None] = relationship(back_populates="source_candidate")
    observations: Mapped[list[Observation]] = relationship(
        back_populates="candidate_type"
    )
