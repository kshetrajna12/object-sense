"""Entity deterministic ID model for identity anchor lookup."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import DateTime, Float, ForeignKey, Index, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from object_sense.models.base import Base

if TYPE_CHECKING:
    from object_sense.models.entity import Entity


class EntityDeterministicId(Base):
    """Junction table for entity deterministic ID lookup.

    Deterministic IDs are the strongest identity anchors (Correction #8).
    This table enables O(1)-ish lookup by (id_type, id_value, id_namespace).

    When an observation has a deterministic ID:
    1. Lookup here to find existing entity
    2. If found: hard link (posterior=1.0)
    3. If not found: CREATE entity + insert binding here
    4. If conflict (different entity): create identity_conflicts row
    """

    __tablename__ = "entity_deterministic_ids"

    id: Mapped[UUID] = mapped_column(primary_key=True)
    entity_id: Mapped[UUID] = mapped_column(
        ForeignKey("entities.entity_id"), index=True
    )

    id_type: Mapped[str] = mapped_column(String(64))
    """Type of identifier: sku, product_id, trip_id, gps, sha256, database_pk, etc."""

    id_value: Mapped[str] = mapped_column(String(512))
    """The actual identifier value."""

    id_namespace: Mapped[str] = mapped_column(String(256))
    """System/domain namespace: acme_corp, wgs84, internal, etc."""

    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    """How confident we are in this binding. 1.0 for strong IDs, lower for weak."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    entity: Mapped[Entity] = relationship(back_populates="deterministic_ids")

    __table_args__ = (
        # Unique constraint on the ID tuple for fast lookup
        # Note: We use a unique index rather than UNIQUE constraint
        # so conflicts can be detected and handled gracefully
        Index(
            "ix_entity_det_id_lookup",
            "id_type",
            "id_value",
            "id_namespace",
            unique=True,
        ),
    )
