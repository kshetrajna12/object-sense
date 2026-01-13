"""Slot reference resolution for symbolic-to-canonical upgrade.

This module provides a post-resolution pass that upgrades symbolic slot references
(ref_name) to canonical entity references (ref_entity_id).

Design principles:
1. No LLM noise in deterministic state - only register ref IDs during resolution
2. Explicit ref_nature from extractor - don't infer from slot name
3. Purely lexical normalization - no semantic mapping
4. Preserve provenance - keep ref_debug_name and record evidence
5. Create-on-miss = PROTO + low confidence - slot-created entities aren't authoritative

See plan: majestic-plotting-bubble.md for full specification.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import flag_modified

from object_sense.models.entity import Entity
from object_sense.models.entity_deterministic_id import EntityDeterministicId
from object_sense.models.enums import EntityNature, EntityStatus, EvidenceSource, SubjectKind
from object_sense.models.evidence import Evidence

if TYPE_CHECKING:
    from object_sense.resolution.candidate_pool import CandidatePoolService


logger = logging.getLogger(__name__)


# Fallback ref_nature mapping for legacy slots without explicit ref_nature.
# ONLY used if slot has ref_name but no ref_nature. Keep this list VERY short.
REF_NATURE_FALLBACK: dict[str, EntityNature] = {
    "species": EntityNature.CLASS,
    "category": EntityNature.CLASS,
}


@dataclass
class ParsedRef:
    """Parsed reference name components."""

    namespace: str
    """Namespace for the ref (default "ref", or explicit via ref:<ns>:<value>)."""

    value: str
    """Normalized value (lowercase, underscores)."""

    raw: str
    """Original ref_name for provenance."""


@dataclass
class SlotResolutionResult:
    """Result of slot reference resolution."""

    resolved_count: int = 0
    """Number of refs resolved to existing entities."""

    created_count: int = 0
    """Number of proto entities created for missing refs."""

    unresolved_count: int = 0
    """Number of refs that could not be resolved."""

    evidence: list[Evidence] = field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    """Evidence records for resolution decisions."""


def parse_ref_name(ref_name: str) -> ParsedRef | None:
    """Parse ref:<value> or ref:<namespace>:<value> into components.

    Args:
        ref_name: The reference name string from the slot.

    Returns:
        ParsedRef with namespace, value, and raw, or None if invalid format.

    Examples:
        >>> parse_ref_name("ref:leopard_species")
        ParsedRef(namespace="ref", value="leopard_species", raw="ref:leopard_species")

        >>> parse_ref_name("ref:wildlife:african_leopard")
        ParsedRef(namespace="wildlife", value="african_leopard", raw="ref:wildlife:african_leopard")

        >>> parse_ref_name("just_a_name")  # Invalid - no ref: prefix
        None
    """
    raw = ref_name

    # STRICT: must start with ref:
    if not ref_name.startswith("ref:"):
        return None

    ref_name = ref_name[4:]  # Strip "ref:" prefix

    # Check for namespace:value pattern (ref:<ns>:<value>)
    if ":" in ref_name:
        namespace, value = ref_name.split(":", 1)
        namespace = _normalize_lexical(namespace)
    else:
        # Default namespace (ref:<value>)
        namespace = "ref"
        value = ref_name

    value = _normalize_lexical(value)
    return ParsedRef(namespace=namespace, value=value, raw=raw)


def _normalize_lexical(name: str) -> str:
    """Purely lexical normalization. No semantic mapping.

    - Lowercase
    - Trim whitespace
    - Replace non-alphanumeric with underscore
    - Collapse multiple underscores

    Args:
        name: The string to normalize.

    Returns:
        Normalized string.
    """
    name = name.lower().strip()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


class SlotResolver:
    """Resolves symbolic slot references to canonical entity refs.

    This is a post-resolution pass that:
    1. Scans entity slots for symbolic refs ({type: "reference", ref_name: ...})
    2. Looks up or creates referenced entities via deterministic IDs
    3. Rewrites slots to canonical form ({ref_entity_id: ..., ref_nature: ...})
    4. Records evidence for resolution decisions

    Usage:
        async with AsyncSession(engine) as session:
            pool = CandidatePoolService(session)
            resolver = SlotResolver(session, pool)
            result = await resolver.resolve_entity_slots(entities)
    """

    def __init__(
        self,
        session: AsyncSession,
        pool: CandidatePoolService,
    ) -> None:
        """Initialize the slot resolver.

        Args:
            session: Database session.
            pool: Candidate pool service for entity lookup/creation.
        """
        self._session = session
        self._pool = pool
        # Cache: (id_type, id_namespace, id_value) -> canonical_entity_id
        self._canonical_cache: dict[tuple[str, str, str], UUID] = {}

    async def resolve_entity_slots(
        self,
        entities: list[Entity],
        *,
        create_on_miss: bool = True,
        emit_resolved_evidence: bool = False,
    ) -> SlotResolutionResult:
        """Resolve symbolic slot references to canonical entity refs.

        Args:
            entities: Entities whose slots should be resolved.
            create_on_miss: If True, create proto entities for unresolved refs.
            emit_resolved_evidence: If True, emit evidence for resolved refs (verbose).

        Returns:
            SlotResolutionResult with counts and evidence.
        """
        result = SlotResolutionResult()

        for entity in entities:
            if not entity.slots:
                continue

            slots_modified = False

            for slot_name, slot_value in list(entity.slots.items()):
                if not self._is_symbolic_ref(slot_value):
                    continue

                # Skip if already resolved (idempotency)
                if "ref_entity_id" in slot_value:
                    continue

                resolution = await self._resolve_slot_ref(
                    entity=entity,
                    slot_name=slot_name,
                    slot_value=slot_value,
                    create_on_miss=create_on_miss,
                    emit_resolved_evidence=emit_resolved_evidence,
                )

                if resolution.success:
                    slots_modified = True
                    if resolution.created:
                        result.created_count += 1
                    else:
                        result.resolved_count += 1
                else:
                    result.unresolved_count += 1

                if resolution.evidence:
                    result.evidence.append(resolution.evidence)

            # Ensure SQLAlchemy detects the JSONB mutation
            if slots_modified:
                flag_modified(entity, "slots")

        return result

    def _is_symbolic_ref(self, value: Any) -> bool:
        """Check if a slot value is a symbolic reference."""
        if not isinstance(value, dict):
            return False
        slot_dict: dict[str, Any] = value  # pyright: ignore[reportUnknownVariableType]
        return slot_dict.get("type") == "reference" and "ref_name" in slot_dict

    async def _resolve_slot_ref(
        self,
        entity: Entity,
        slot_name: str,
        slot_value: dict[str, Any],
        create_on_miss: bool,
        emit_resolved_evidence: bool,
    ) -> _SlotRefResolution:
        """Resolve a single symbolic slot reference.

        Args:
            entity: The entity containing the slot.
            slot_name: Name of the slot.
            slot_value: The slot value dict (must be symbolic ref).
            create_on_miss: If True, create proto entities for unresolved refs.
            emit_resolved_evidence: If True, emit evidence for resolved refs.

        Returns:
            _SlotRefResolution with success status and optional evidence.
        """
        ref_name = slot_value["ref_name"]

        # Parse ref_name (must start with ref:)
        parsed = parse_ref_name(ref_name)
        if parsed is None:
            evidence = self._create_evidence(
                entity_id=entity.entity_id,
                predicate="slot_ref_unresolved",
                details={
                    "slot_name": slot_name,
                    "ref_name": ref_name,
                    "reason": "invalid_ref_name",
                },
            )
            logger.warning(
                "Invalid ref_name format (must start with 'ref:'): %s in slot %s",
                ref_name,
                slot_name,
            )
            return _SlotRefResolution(success=False, evidence=evidence)

        # Get ref_nature (required, with fallbacks)
        # Priority: 1) explicit ref_nature, 2) infer from namespace, 3) slot-name fallback
        ref_nature_str = slot_value.get("ref_nature")
        ref_nature: EntityNature | None = None

        if ref_nature_str:
            # Explicit ref_nature provided
            try:
                ref_nature = EntityNature(ref_nature_str)
            except ValueError:
                evidence = self._create_evidence(
                    entity_id=entity.entity_id,
                    predicate="slot_ref_unresolved",
                    details={
                        "slot_name": slot_name,
                        "ref_name": ref_name,
                        "reason": "invalid_ref_nature",
                        "ref_nature": ref_nature_str,
                    },
                )
                return _SlotRefResolution(success=False, evidence=evidence)
        else:
            # Try to infer from namespace (ref:<nature>:<name> format)
            # If namespace is individual/class/group/event, use it as ref_nature
            try:
                ref_nature = EntityNature(parsed.namespace)
            except ValueError:
                # Namespace isn't a valid nature - try slot-name fallback
                ref_nature = REF_NATURE_FALLBACK.get(slot_name)

        if ref_nature is None:
            evidence = self._create_evidence(
                entity_id=entity.entity_id,
                predicate="slot_ref_unresolved",
                details={
                    "slot_name": slot_name,
                    "ref_name": ref_name,
                    "reason": "missing_ref_nature",
                    "namespace": parsed.namespace,
                },
            )
            logger.warning(
                "Missing ref_nature for slot %s (ref_name=%s) - no fallback available",
                slot_name,
                ref_name,
            )
            return _SlotRefResolution(success=False, evidence=evidence)

        # Lookup by deterministic ID
        cache_key = ("ref", parsed.namespace, parsed.value)
        cached_entity_id = self._canonical_cache.get(cache_key)

        if cached_entity_id:
            # Use cached result
            resolved_entity_id = cached_entity_id
            created = False
        else:
            # Lookup in database
            from object_sense.inference.schemas import DeterministicId

            det_id = DeterministicId(
                id_type="ref",
                id_namespace=parsed.namespace,
                id_value=parsed.value,
                strength="strong",
            )

            lookup = await self._pool.lookup_by_deterministic_id(det_id)

            if lookup.entity:
                # Found existing entity - resolve to canonical
                canonical_entity = await self._resolve_to_canonical(lookup.entity)

                # Nature mismatch guard: don't rewrite if target nature != ref_nature
                if canonical_entity.entity_nature != ref_nature:
                    evidence = self._create_evidence(
                        entity_id=entity.entity_id,
                        predicate="slot_ref_nature_conflict",
                        target_id=canonical_entity.entity_id,
                        details={
                            "slot_name": slot_name,
                            "ref_name": parsed.raw,
                            "expected_nature": ref_nature.value,
                            "actual_nature": canonical_entity.entity_nature.value if canonical_entity.entity_nature else None,
                            "reason": "nature_mismatch",
                        },
                    )
                    logger.warning(
                        "Nature mismatch for ref %s: expected %s, got %s",
                        parsed.raw,
                        ref_nature.value,
                        canonical_entity.entity_nature.value if canonical_entity.entity_nature else None,
                    )
                    return _SlotRefResolution(success=False, evidence=evidence)

                resolved_entity_id = canonical_entity.entity_id
                created = False
                # Cache the result
                self._canonical_cache[cache_key] = resolved_entity_id
            elif create_on_miss:
                # Create proto entity for this ref
                try:
                    new_entity = await self._create_proto_for_ref(
                        display_name=parsed.value,
                        ref_nature=ref_nature,
                        parsed_ref=parsed,
                    )
                    resolved_entity_id = new_entity.entity_id
                    created = True
                    # Cache the result
                    self._canonical_cache[cache_key] = resolved_entity_id
                except IntegrityError:
                    # Concurrent create - refetch
                    await self._session.rollback()
                    lookup = await self._pool.lookup_by_deterministic_id(det_id)
                    if lookup.entity:
                        canonical_entity = await self._resolve_to_canonical(lookup.entity)
                        resolved_entity_id = canonical_entity.entity_id
                        created = False
                        self._canonical_cache[cache_key] = resolved_entity_id
                    else:
                        evidence = self._create_evidence(
                            entity_id=entity.entity_id,
                            predicate="slot_ref_unresolved",
                            details={
                                "slot_name": slot_name,
                                "ref_name": ref_name,
                                "reason": "concurrent_create_failed",
                            },
                        )
                        return _SlotRefResolution(success=False, evidence=evidence)
            else:
                # No create-on-miss
                evidence = self._create_evidence(
                    entity_id=entity.entity_id,
                    predicate="slot_ref_unresolved",
                    details={
                        "slot_name": slot_name,
                        "ref_name": ref_name,
                        "reason": "no_match",
                    },
                )
                return _SlotRefResolution(success=False, evidence=evidence)

        # Upgrade slot to canonical form (preserve existing fields)
        slot_value["ref_entity_id"] = str(resolved_entity_id)
        slot_value["ref_raw_name"] = parsed.raw  # Original ref:... string for provenance
        slot_value["ref_debug_name"] = parsed.value  # Normalized display name
        slot_value["ref_namespace"] = parsed.namespace  # Namespace for debugging/query

        # Create evidence
        evidence: Evidence | None = None
        if created:
            evidence = self._create_evidence(
                entity_id=entity.entity_id,
                predicate="slot_ref_created",
                target_id=resolved_entity_id,
                details={
                    "slot_name": slot_name,
                    "ref_name": parsed.raw,
                    "ref_nature": ref_nature.value,
                    "ref_namespace": parsed.namespace,
                    "ref_value": parsed.value,
                },
            )
            logger.info(
                "Created proto entity %s for ref %s in slot %s",
                resolved_entity_id,
                parsed.raw,
                slot_name,
            )
        elif emit_resolved_evidence:
            evidence = self._create_evidence(
                entity_id=entity.entity_id,
                predicate="slot_ref_resolved",
                target_id=resolved_entity_id,
                details={
                    "slot_name": slot_name,
                    "from_ref_name": parsed.raw,
                    "ref_nature": ref_nature.value,
                    "ref_namespace": parsed.namespace,
                },
            )

        return _SlotRefResolution(success=True, created=created, evidence=evidence)

    async def _resolve_to_canonical(self, entity: Entity) -> Entity:
        """Follow canonical_entity_id chain to canonical entity.

        ALWAYS follow merge chain - slots must never point at merged-away entities.

        Args:
            entity: The entity to resolve.

        Returns:
            The canonical entity (may be the same if not merged).
        """
        current = entity
        while current.canonical_entity_id is not None:
            next_entity = await self._session.get(Entity, current.canonical_entity_id)
            if next_entity is None:
                # Broken chain - stop here
                logger.warning(
                    "Broken canonical chain: %s -> %s (not found)",
                    current.entity_id,
                    current.canonical_entity_id,
                )
                break
            current = next_entity
        return current

    async def _create_proto_for_ref(
        self,
        display_name: str,
        ref_nature: EntityNature,
        parsed_ref: ParsedRef,
    ) -> Entity:
        """Create a proto entity for an unresolved reference.

        IMPORTANT: This is one of the ONLY places where ref deterministic IDs
        should be registered. Don't pollute deterministic state elsewhere.

        Args:
            display_name: Normalized display name for the entity.
            ref_nature: Entity nature (CLASS, GROUP, etc.).
            parsed_ref: Parsed reference components.

        Returns:
            The newly created Entity.
        """
        entity_id = uuid4()

        entity = Entity(
            entity_id=entity_id,
            entity_nature=ref_nature,
            name=display_name,
            status=EntityStatus.PROTO,
            confidence=0.7,  # Low - slot-created, not confirmed
            slots={},
        )
        self._session.add(entity)

        # Register ref deterministic ID (only here!)
        ref_binding = EntityDeterministicId(
            id=uuid4(),
            entity_id=entity_id,
            id_type="ref",
            id_namespace=parsed_ref.namespace,
            id_value=parsed_ref.value,
            confidence=0.7,  # Low - slot-created
        )
        self._session.add(ref_binding)

        await self._session.flush()
        return entity

    def _create_evidence(
        self,
        entity_id: UUID,
        predicate: str,
        details: dict[str, Any],
        target_id: UUID | None = None,
    ) -> Evidence:
        """Create an evidence record for slot resolution."""
        evidence = Evidence(
            evidence_id=uuid4(),
            subject_kind=SubjectKind.ENTITY,
            subject_id=entity_id,
            predicate=predicate,
            target_id=target_id,
            source=EvidenceSource.SYSTEM,
            score=0.7 if predicate == "slot_ref_created" else 1.0,
            details=details,
        )
        self._session.add(evidence)
        return evidence


@dataclass
class _SlotRefResolution:
    """Internal result for resolving a single slot reference."""

    success: bool
    """Whether the resolution succeeded."""

    created: bool = False
    """Whether a new entity was created (only if success=True)."""

    evidence: Evidence | None = None
    """Evidence record for this resolution."""
