"""Observation model for data points in the world model."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import DateTime, Float, ForeignKey, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from object_sense.models.base import Base
from object_sense.models.enums import LinkRole, LinkStatus, Medium, ObservationStatus

if TYPE_CHECKING:
    from object_sense.models.blob import Blob
    from object_sense.models.entity import Entity
    from object_sense.models.event_participant import EventParticipant
    from object_sense.models.signature import Signature
    from object_sense.models.type import Type
    from object_sense.models.type_candidate import TypeCandidate


class Observation(Base):
    """A unit of data we received (file, record, frame, note).

    Observations are what we ingest. They link TO entities - they are not
    entities themselves.

    Type reference is three-level (see design_v2_corrections.md §2):
    - observation_kind: Low-cardinality routing hint (~20 values)
    - candidate_type_id: FK to type_candidates (LLM's type proposal)
    - stable_type_id: FK to types (promoted ontological type)
    """

    __tablename__ = "observations"

    observation_id: Mapped[UUID] = mapped_column(primary_key=True)
    medium: Mapped[Medium] = mapped_column(index=True)
    source_id: Mapped[str] = mapped_column(String(2048), index=True)
    blob_id: Mapped[UUID | None] = mapped_column(ForeignKey("blobs.blob_id"), index=True)
    slots: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    status: Mapped[ObservationStatus] = mapped_column(default=ObservationStatus.ACTIVE)

    # Three-level type reference (Correction #2)
    observation_kind: Mapped[str | None] = mapped_column(String(64), index=True)
    """Low-cardinality routing hint (e.g., wildlife_photo, product_record).
    Does NOT reference type_candidates. Used ONLY for pipeline routing."""

    facets: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    """Extracted attributes from Step 4A (e.g., detected_objects, lighting)."""

    deterministic_ids: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list)
    """Array of {id_type, id_value, id_namespace} tuples (Correction #8).
    Examples: SKU, product_id, trip_id, GPS coords."""

    candidate_type_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("type_candidates.candidate_id"), index=True
    )
    """FK to type_candidates. LLM-assigned, mutable on merge. All semantic
    type labeling flows through this, NOT observation_kind."""

    stable_type_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("types.type_id"), index=True
    )
    """FK to types. Set only after TypeCandidate promotion (Correction #7).
    Can remain NULL for observations that don't fit stable types."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    blob: Mapped[Blob | None] = relationship(back_populates="observations")
    candidate_type: Mapped[TypeCandidate | None] = relationship(
        back_populates="observations"
    )
    stable_type: Mapped[Type | None] = relationship(back_populates="observations")
    entity_links: Mapped[list[ObservationEntityLink]] = relationship(
        back_populates="observation"
    )
    signatures: Mapped[list[Signature]] = relationship(back_populates="observation")
    event_participant_inferences: Mapped[list[EventParticipant]] = relationship(
        back_populates="source_observation"
    )
    """Event participant inferences created from this observation."""


class ObservationEntityLink(Base):
    """Association between Observations and Entities.

    Represents which entities an observation refers to. An Observation can
    link to multiple Entities (e.g., a photo depicting multiple animals).

    See design_v2_corrections.md §5 for link semantics:
    - posterior: Confidence score 0-1
    - status: hard (deterministic ID) | soft (similarity) | candidate (uncertain)
    - role: subject (primary) | context (supporting)
    - flags: Multi-seed conflicts, low-evidence markers, etc.
    """

    __tablename__ = "observation_entity_links"

    observation_id: Mapped[UUID] = mapped_column(
        ForeignKey("observations.observation_id"), primary_key=True
    )
    entity_id: Mapped[UUID] = mapped_column(ForeignKey("entities.entity_id"), primary_key=True)

    posterior: Mapped[float] = mapped_column(Float, default=1.0)
    """Confidence score 0-1. 1.0 for deterministic ID matches."""

    status: Mapped[LinkStatus] = mapped_column(default=LinkStatus.SOFT)
    """Link type: hard (ID match), soft (similarity), candidate (uncertain)."""

    role: Mapped[LinkRole | None] = mapped_column()
    """Role of entity in observation: subject (primary) or context (supporting)."""

    flags: Mapped[list[str]] = mapped_column(JSONB, default=list)
    """Conflict/warning flags: multi_seed_role_conflict, low_evidence, etc."""

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    observation: Mapped[Observation] = relationship(back_populates="entity_links")
    entity: Mapped[Entity] = relationship(back_populates="observation_links")
