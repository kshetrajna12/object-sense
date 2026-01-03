"""Pydantic schemas for LLM type inference.

These schemas define the structured output the LLM produces when
analyzing an observation and proposing types, slots, and entity hypotheses.

Step 4 is split into two phases (see design_v2_corrections.md §2):
- Step 4A: Semantic Probe (routing hint, facets, entity seeds)
- Step 4B: Type Proposal (TypeCandidate if new label proposed)
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, Field

from object_sense.models.enums import EntityNature


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


class DeterministicId(BaseModel):
    """A deterministic identifier extracted from an observation.

    Deterministic IDs (Correction #8) dominate identity resolution.
    If present, they anchor entity linking with posterior=1.0.

    IDs are stored as (id_type, id_value, id_namespace) tuples to handle
    cross-system uniqueness (SKU "12345" from Acme ≠ SKU "12345" from Beta).

    NAMESPACE ASSIGNMENT (bead object-sense-bk9):
    The engine controls namespace assignment, not the LLM.
    - id_namespace is OPTIONAL - omit it and let the engine assign
    - Engine uses id_type → namespace mapping (gps→geo:wgs84, upc→global:upc)
    - For other IDs, engine uses ingest context (source:<dataset>)
    - If LLM provides namespace, it's validated against allowed patterns
    """

    id_type: str = Field(
        description=(
            "Type of identifier: sku, product_id, trip_id, gps, booking_id, "
            "upc, isbn, gtin, asin, sha256, database_pk, etc."
        )
    )
    id_value: str = Field(
        description="The actual identifier value (e.g., 'PROD-12345', '-25.7461,28.1881')"
    )
    id_namespace: str | None = Field(
        default=None,
        description=(
            "OPTIONAL - Engine assigns namespace automatically based on id_type and context. "
            "If provided, must match allowed patterns: source:<dataset>, global:<authority>, "
            "geo:wgs84, user:<tenant>. Invalid namespaces are overridden."
        ),
    )
    strength: Literal["strong", "weak"] = Field(
        default="strong",
        description=(
            "strong=100% trust (SKU, database_pk, sha256). "
            "weak=high trust but verify (filename, URL, timestamp)."
        ),
    )


class EntityHypothesis(BaseModel):
    """A hypothesized entity detected in the observation.

    These are SEEDS for entity resolution, not hard assignments.
    Step 5 (Entity Resolution) decides how these map to actual entities.

    The entity_nature field (Correction #6) affects signal weighting:
    - individual: weight visual re-ID, signatures, location
    - class: weight text semantics, facet agreement
    - group: weight member overlap, temporal proximity
    - event: weight timestamp, participants, location

    Example: For a wildlife photo, the LLM might hypothesize:
    - An animal entity (nature=individual, species=leopard, pose=stalking)
    - A species entity (nature=class, name="Leopard")
    - A location entity (nature=individual, name="MalaMala")
    """

    entity_type: str = Field(
        description="Proposed type for this entity (e.g., 'animal_entity', 'location_entity')"
    )
    entity_nature: EntityNature = Field(
        description=(
            "What kind of thing this entity is (Correction #6). "
            "individual=specific instance (Marula Leopard), "
            "class=category (Leopard species), "
            "group=collection (Lion Pride), "
            "event=occurrence (Hunt on Jan 15)"
        )
    )
    suggested_name: str | None = Field(
        default=None,
        description="Optional suggested name/identifier (e.g., 'Marula Leopard', 'MalaMala')",
    )
    slots: list[SlotValue] = Field(  # pyright: ignore[reportUnknownVariableType]
        default_factory=list,
        description="Proposed slots for this entity",
    )
    deterministic_ids: list[DeterministicId] = Field(  # pyright: ignore[reportUnknownVariableType]
        default_factory=list,
        description="Any deterministic identifiers detected (SKU, GPS, trip_id, etc.)",
    )
    confidence: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Confidence in this entity hypothesis"
    )
    reasoning: str | None = Field(
        default=None,
        description="Brief explanation for why this entity was detected",
    )


class SemanticProbeResult(BaseModel):
    """Step 4A output: Semantic probe for routing and entity seeds.

    This is the FIRST phase of type inference (design_v2_corrections.md §2).
    It produces routing hints and entity hypotheses, NOT authoritative types.

    - observation_kind is a LOW-CARDINALITY routing hint (~20 values)
    - facets are extracted attributes for entity resolution
    - entity_seeds are candidates for Step 5 (Entity Resolution)
    """

    observation_kind: str = Field(
        description=(
            "Low-cardinality routing hint for pipeline routing. "
            "Examples: wildlife_photo, product_record, json_config, text_document. "
            "NOT a type reference—used only for indexing/routing. ~20 distinct values max."
        )
    )
    facets: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Extracted attribute hypotheses (JSONB). "
            "Examples: {detected_objects: ['leopard'], lighting: 'backlit', price: 49.99}"
        ),
    )
    entity_seeds: list[EntityHypothesis] = Field(  # pyright: ignore[reportUnknownVariableType]
        default_factory=list,
        description="Candidate entities for Step 5 (Entity Resolution)",
    )
    deterministic_ids: list[DeterministicId] = Field(  # pyright: ignore[reportUnknownVariableType]
        default_factory=list,
        description="Observation-level deterministic IDs (SKU, GPS, trip_id, etc.)",
    )


class TypeCandidateProposal(BaseModel):
    """Step 4B output: Proposal for a new TypeCandidate.

    Created when the LLM proposes a new type label. This becomes a
    TypeCandidate row, NOT a stable Type. Promotion happens later
    after evidence accumulates (design_v2_corrections.md §7).
    """

    proposed_name: str = Field(
        description=(
            "The proposed type name in snake_case. Will be normalized for dedup. "
            "Examples: wildlife_photo, african_leopard_sighting, product_catalog_entry"
        )
    )
    rationale: str = Field(
        description="Why this new type is needed (existing types don't fit)"
    )
    suggested_parent: str | None = Field(
        default=None,
        description="If this is a refinement of an existing type, name the parent",
    )
    confidence: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Confidence that this type proposal is valid and distinct",
    )


class TypeProposal(BaseModel):
    """Complete Step 4 output: Semantic probe + optional type candidate.

    Combines Step 4A (SemanticProbeResult) and Step 4B (TypeCandidateProposal).

    CRITICAL DISTINCTION (design_v2_corrections.md §2):
    - observation_kind: LOW-cardinality routing string (~20 values)
    - type_candidate: HIGH-cardinality type proposal (creates TypeCandidate row)

    All semantic type labeling flows through type_candidate, NOT observation_kind.
    """

    # ── Step 4A: Semantic Probe ──────────────────────────────────────────────
    observation_kind: str = Field(
        description=(
            "Low-cardinality routing hint (~20 values max). "
            "Examples: wildlife_photo, product_record, json_config. "
            "Used ONLY for pipeline routing, NOT for type tracking."
        )
    )
    facets: dict[str, Any] = Field(
        default_factory=dict,
        description="Extracted attributes (detected_objects, lighting, price, etc.)",
    )
    entity_seeds: list[EntityHypothesis] = Field(  # pyright: ignore[reportUnknownVariableType]
        default_factory=list,
        description="Entity candidates for Step 5 resolution",
    )
    deterministic_ids: list[DeterministicId] = Field(  # pyright: ignore[reportUnknownVariableType]
        default_factory=list,
        description="Observation-level deterministic identifiers",
    )

    # ── Step 4B: Type Candidate Proposal ─────────────────────────────────────
    # CRITICAL: Step 4 can ONLY propose type candidates. It cannot bind to
    # existing types. Matching/dedup is done by the engine via normalize_type_name()
    # and find_similar_candidates(). This enforces "LLM proposes, engine decides."
    type_candidate: TypeCandidateProposal | None = Field(
        default=None,
        description=(
            "If proposing a type label, include details here. "
            "Creates or matches a TypeCandidate row (engine does dedup). "
            "Leave None if no type proposal (rare - usually propose something)."
        ),
    )

    # ── Metadata ─────────────────────────────────────────────────────────────
    reasoning: str = Field(description="Brief explanation of the inference reasoning")


class TypeSearchResult(BaseModel):
    """A type returned from searching the type store."""

    type_name: str
    aliases: list[str] = Field(default_factory=list)
    parent_type: str | None = None
    evidence_count: int = 0
    status: str = "provisional"


class SimilarObservation(BaseModel):
    """A similar observation returned from similarity search."""

    observation_id: str
    primary_type: str | None
    similarity_score: float
    medium: str
    extracted_text: str | None = None
