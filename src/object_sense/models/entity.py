"""Entity model for persistent concepts."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from pgvector.sqlalchemy import Vector  # type: ignore[import-untyped]
from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from object_sense.config import settings
from object_sense.models.base import Base
from object_sense.models.enums import EntityNature, EntityStatus

if TYPE_CHECKING:
    from object_sense.models.entity_deterministic_id import EntityDeterministicId
    from object_sense.models.entity_evolution import EntityEvolution
    from object_sense.models.observation import ObservationEntityLink
    from object_sense.models.type import Type


class Entity(Base):
    """A persistent concept that observations link to.

    Entities are cluster nodes representing stable concepts - either concrete
    (The Marula Leopard) or abstract (Leopard Species). Observations link TO
    entities; entities are not merges of observations into each other.

    See design_v2_corrections.md §6 for entity_nature semantics.
    """

    __tablename__ = "entities"

    entity_id: Mapped[UUID] = mapped_column(primary_key=True)
    type_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("types.type_id"), index=True
    )

    entity_nature: Mapped[EntityNature | None] = mapped_column(index=True)
    """What kind of thing this entity represents (Correction #6).
    individual | class | group | event. Affects signal weighting in resolution."""

    canonical_entity_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("entities.entity_id"), index=True
    )
    """If merged, points to the canonical entity. Allows merge chain resolution."""

    name: Mapped[str | None] = mapped_column(String(512))
    """Human-readable name/label for the entity."""

    slots: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    status: Mapped[EntityStatus] = mapped_column(default=EntityStatus.PROTO)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # ── Prototype embeddings for ANN retrieval (entity resolution) ───────────
    # Running average embeddings used for candidate pool retrieval.
    # Updated on hard links always; on soft links only if posterior >= T_link.

    prototype_image_embedding: Mapped[list[Any] | None] = mapped_column(
        Vector(settings.dim_image_embedding)
    )
    """Running average CLIP image embedding (768-dim). Primary for individual entities."""

    prototype_text_embedding: Mapped[list[Any] | None] = mapped_column(
        Vector(settings.dim_text_embedding)
    )
    """Running average BGE text embedding (1024-dim). Primary for class/group/event entities."""

    prototype_count: Mapped[int] = mapped_column(Integer, default=0)
    """Number of observations contributing to prototype (weighted count)."""

    prototype_spread: Mapped[float | None] = mapped_column(Float)
    """Variance measure to detect multi-modality. Optional, for future use."""

    exemplar_observation_ids: Mapped[list[Any]] = mapped_column(JSONB, default=list)
    """Top 3-5 observation IDs closest to prototype. Useful for debugging/display."""

    # Relationships
    type: Mapped[Type | None] = relationship(back_populates="entities")
    canonical_entity: Mapped[Entity | None] = relationship(
        "Entity",
        remote_side="Entity.entity_id",
        foreign_keys=[canonical_entity_id],
    )
    observation_links: Mapped[list[ObservationEntityLink]] = relationship(
        back_populates="entity"
    )
    evolutions: Mapped[list[EntityEvolution]] = relationship(
        back_populates="entity",
        foreign_keys="EntityEvolution.entity_id",
    )
    deterministic_ids: Mapped[list[EntityDeterministicId]] = relationship(
        back_populates="entity"
    )
