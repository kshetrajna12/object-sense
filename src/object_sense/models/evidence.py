"""Evidence model for tracking belief provenance."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import DateTime, Float, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from object_sense.models.base import Base
from object_sense.models.enums import EvidenceSource, SubjectKind


class Evidence(Base):
    """How the system came to believe something.

    Evidence covers ALL beliefs - types, entity links, entity merges.
    The engine computes final beliefs as an aggregation over evidence.

    Examples:
    - "Object X has type wildlife_photo" (subject_kind=object, predicate=has_type)
    - "Object X refers to Entity Y" (subject_kind=object_entity_link, predicate=refers_to)
    - "Entity A is same as Entity B" (subject_kind=entity_merge, predicate=same_as)
    """

    __tablename__ = "evidence"

    evidence_id: Mapped[UUID] = mapped_column(primary_key=True)
    subject_kind: Mapped[SubjectKind] = mapped_column(index=True)
    subject_id: Mapped[UUID] = mapped_column(PG_UUID, index=True)
    predicate: Mapped[str] = mapped_column(String(255), index=True)
    target_id: Mapped[UUID | None] = mapped_column(PG_UUID, index=True)
    source: Mapped[EvidenceSource] = mapped_column()
    score: Mapped[float] = mapped_column(Float)
    details: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
