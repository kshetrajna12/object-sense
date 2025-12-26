"""Pydantic schemas for LLM type inference.

These schemas define the structured output the LLM produces when
analyzing an observation and proposing types, slots, and entity hypotheses.

See concept_v1.md §4 Step 4 for the full specification.
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, Field


def _coerce_to_string(v: Any) -> str:
    """Coerce slot values to strings.

    LLMs may return numeric values (49.99) instead of strings ("49.99").
    This validator ensures all values become strings for consistent handling.
    """
    if v is None:
        return ""
    return str(v)


class SlotValue(BaseModel):
    """A proposed slot (property) for an observation.

    Slots are axes of variation within the same kind of thing.
    Values can be primitives or references to entities.
    """

    name: str = Field(
        description="Slot name in snake_case (e.g., 'lighting', 'species', 'location')"
    )
    value: Annotated[str, BeforeValidator(_coerce_to_string)] = Field(
        description="Slot value as string. For entity references, use 'ref:entity_name'"
    )
    is_reference: bool = Field(
        default=False,
        description="True if this slot references another entity (e.g., species, location)",
    )
    confidence: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Confidence in this slot assignment"
    )


class EntityHypothesis(BaseModel):
    """A hypothesized entity detected in the observation.

    These are SEEDS for entity resolution, not hard assignments.
    Step 5 (Entity Resolution) decides how these map to actual entities.

    Example: For a wildlife photo, the LLM might hypothesize:
    - An animal entity (species=leopard, pose=stalking)
    - A location entity (name=MalaMala)
    """

    entity_type: str = Field(
        description="Proposed type for this entity (e.g., 'animal_entity', 'location_entity')"
    )
    suggested_name: str | None = Field(
        default=None,
        description="Optional suggested name/identifier (e.g., 'Marula Leopard', 'MalaMala')",
    )
    slots: list[SlotValue] = Field(  # pyright: ignore[reportUnknownVariableType]
        default_factory=list,
        description="Proposed slots for this entity",
    )
    confidence: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Confidence in this entity hypothesis"
    )
    reasoning: str | None = Field(
        default=None,
        description="Brief explanation for why this entity was detected",
    )


class TypeProposal(BaseModel):
    """The complete type inference result from the LLM.

    This is the structured output for Step 4 of the processing loop.
    The LLM analyzes extracted features and proposes:
    - What TYPE of observation this is (primary_type)
    - What SLOTS (properties) it has
    - What ENTITIES it contains or references
    - Whether a NEW TYPE should be created
    """

    primary_type: str = Field(
        description=(
            "The semantic type of this observation (e.g., 'wildlife_photo', 'product_record'). "
            "Use snake_case, singular nouns. Reuse existing types when possible."
        )
    )
    primary_type_confidence: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Confidence in the primary type assignment"
    )
    is_existing_type: bool = Field(
        default=True,
        description="True if primary_type already exists in the type store, False if new",
    )

    slots: list[SlotValue] = Field(  # pyright: ignore[reportUnknownVariableType]
        default_factory=list,
        description="Proposed slots (properties) for this observation",
    )

    entity_hypotheses: list[EntityHypothesis] = Field(  # pyright: ignore[reportUnknownVariableType]
        default_factory=list,
        description="Hypothesized entities detected in this observation",
    )

    maybe_new_type: str | None = Field(
        default=None,
        description=(
            "If the observation warrants a new, more specific type, propose it here. "
            "Only propose if existing types don't capture the essence. "
            "Example: 'backlit_wildlife_photo' as refinement of 'wildlife_photo'"
        ),
    )
    new_type_rationale: str | None = Field(
        default=None,
        description="Explanation for why a new type should be created",
    )

    reasoning: str = Field(description="Brief explanation of the type inference reasoning")


class TypeSearchResult(BaseModel):
    """A type returned from searching the type store."""

    type_name: str
    aliases: list[str] = Field(default_factory=list)
    parent_type: str | None = None
    evidence_count: int = 0
    status: str = "provisional"


class SimilarObject(BaseModel):
    """A similar object returned from similarity search."""

    object_id: str
    primary_type: str | None
    similarity_score: float
    medium: str
    extracted_text: str | None = None
