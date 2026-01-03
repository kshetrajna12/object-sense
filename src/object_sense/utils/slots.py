"""Slot validation and normalization utilities.

Implements the Slot Hygiene Contract (concept_v2.md §6.3):
- Every slot value must be a primitive wrapper or entity reference
- Primitive: {"value": ..., "type": "...", "unit": optional}
- Reference: {"ref_entity_id": "...", "ref_type": "...", "display_name": optional}

Goal for v0: normalize + warn (do not block ingest).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, TypeGuard, cast
from uuid import UUID, uuid4

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from object_sense.models import Evidence

logger = logging.getLogger(__name__)

# Type aliases for slot structures
SlotValue = dict[str, Any] | list[Any] | str | int | float | bool | None
SlotDict = dict[str, Any]

# Primitive types that can be slot values
PRIMITIVE_TYPES = (str, int, float, bool)

# Known slot value types
SLOT_TYPES = frozenset({
    "string",
    "number",
    "integer",
    "boolean",
    "currency",
    "date",
    "datetime",
    "duration",
    "percentage",
    "url",
    "email",
})


def is_slot_reference(value: Any) -> TypeGuard[SlotDict]:
    """Check if a value is a properly-formed entity reference.

    References must have:
    - ref_entity_id: str or UUID
    - ref_type: str

    Optional:
    - display_name: str

    Args:
        value: The slot value to check

    Returns:
        True if this is a valid entity reference
    """
    if not isinstance(value, dict):
        return False

    value_dict = cast(SlotDict, value)

    # Must have ref_entity_id and ref_type
    if "ref_entity_id" not in value_dict or "ref_type" not in value_dict:
        return False

    ref_id = value_dict["ref_entity_id"]
    ref_type_val = value_dict["ref_type"]

    # Validate ref_entity_id is string or UUID
    if not isinstance(ref_id, (str, UUID)):
        return False

    # Validate ref_type is string
    if not isinstance(ref_type_val, str) or not ref_type_val:
        return False

    return True


def is_slot_primitive_wrapper(value: Any) -> TypeGuard[SlotDict]:
    """Check if a value is a properly-formed primitive wrapper.

    Primitive wrappers must have:
    - value: primitive (str, int, float, bool)
    - type: str

    Optional:
    - unit: str

    Args:
        value: The slot value to check

    Returns:
        True if this is a valid primitive wrapper
    """
    if not isinstance(value, dict):
        return False

    value_dict = cast(SlotDict, value)

    if "value" not in value_dict or "type" not in value_dict:
        return False

    inner = value_dict["value"]
    type_str = value_dict["type"]

    # Inner must be primitive
    if not isinstance(inner, PRIMITIVE_TYPES):
        return False

    # Type must be non-empty string
    if not isinstance(type_str, str) or not type_str:
        return False

    return True


def _infer_primitive_type(value: Any) -> str:
    """Infer the slot type for a primitive value.

    Args:
        value: A primitive value (str, int, float, bool)

    Returns:
        The inferred type name
    """
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    return "unknown"


def normalize_slot(name: str, value: Any) -> tuple[Any, list[str]]:
    """Normalize a single slot value according to Slot Hygiene Contract.

    Transforms bare primitives into wrapped form. Validates existing
    wrappers. Returns warnings for non-compliant values but does not
    block (v0 behavior: normalize + warn).

    Supported shapes:
    - Primitive wrapper: {"value": <primitive>, "type": <string>, "unit": <string?>}
    - Reference wrapper: {"ref_entity_id": <uuid>, "ref_type": <string>, "display_name": <string?>}
    - Lists: homogeneous lists of primitives or references (each element normalized)

    Args:
        name: The slot name (for warning messages)
        value: The raw slot value

    Returns:
        Tuple of (normalized_value, list of warning messages)
    """
    warnings: list[str] = []

    # Handle None
    if value is None:
        return None, warnings

    # Handle lists: normalize each element
    if isinstance(value, list):
        normalized_list: list[Any] = []
        for i, item in enumerate(cast(list[Any], value)):
            norm_item, item_warnings = normalize_slot(f"{name}[{i}]", item)
            normalized_list.append(norm_item)
            warnings.extend(item_warnings)
        return normalized_list, warnings

    # Already a reference? Validate and pass through
    if is_slot_reference(value):
        return value, warnings

    # Already a primitive wrapper? Validate and pass through
    if is_slot_primitive_wrapper(value):
        # Warn if numeric without unit (but allow)
        slot_type = cast(str, value["type"])
        if slot_type in ("number", "integer", "currency") and "unit" not in value:
            warnings.append(
                f"Slot '{name}': numeric value without unit (consider adding unit)"
            )
        return value, warnings

    # Bare primitive: wrap it
    if isinstance(value, PRIMITIVE_TYPES):
        type_name = _infer_primitive_type(value)
        normalized: SlotDict = {"value": value, "type": type_name}

        # Warn for bare numbers without unit
        if type_name in ("number", "integer"):
            warnings.append(
                f"Slot '{name}': bare numeric '{value}' wrapped as {type_name} "
                "(consider adding unit for clarity)"
            )

        # Warn for freeform strings in potentially rich domains
        # This is a heuristic: slot names suggesting entity references
        if type_name == "string" and _looks_like_entity_slot(name):
            warnings.append(
                f"Slot '{name}': freeform string '{value}' may be better "
                "as entity reference (ref_entity_id + ref_type)"
            )

        return normalized, warnings

    # Dict but not valid wrapper/reference
    if isinstance(value, dict):
        value_dict = cast(SlotDict, value)
        # Check if it looks like a malformed reference
        if "ref_entity_id" in value_dict or "ref_type" in value_dict:
            warnings.append(
                f"Slot '{name}': malformed reference (needs both ref_entity_id and ref_type)"
            )
            return value_dict, warnings

        # Check if it looks like a malformed primitive wrapper
        if "value" in value_dict:
            warnings.append(
                f"Slot '{name}': malformed primitive wrapper (needs 'type' field)"
            )
            # Add type if we can infer it
            inner: Any = value_dict.get("value")
            if isinstance(inner, PRIMITIVE_TYPES):
                normalized_dict: SlotDict = dict(value_dict)
                normalized_dict["type"] = _infer_primitive_type(inner)
                return normalized_dict, warnings
            return value_dict, warnings

        # Unknown dict structure - pass through with warning
        warnings.append(
            f"Slot '{name}': unrecognized dict structure, not a valid wrapper or reference"
        )
        return value_dict, warnings

    # Unknown type - pass through with warning
    warnings.append(
        f"Slot '{name}': unsupported value type {type(value).__name__}, passing through"
    )
    return value, warnings


def _looks_like_entity_slot(name: str) -> bool:
    """Heuristic: check if a slot name suggests it should be an entity reference.

    Args:
        name: The slot name

    Returns:
        True if the name suggests entity reference semantics
    """
    # Common patterns for entity-like slots
    entity_patterns = (
        "species",
        "category",
        "brand",
        "manufacturer",
        "location",
        "country",
        "region",
        "city",
        "owner",
        "author",
        "creator",
        "seller",
        "buyer",
        "parent",
        "child",
        "type",  # careful with this one
    )

    name_lower = name.lower()

    # Direct match
    if name_lower in entity_patterns:
        return True

    # Suffix match (e.g., "parent_category", "primary_species")
    for pattern in entity_patterns:
        if name_lower.endswith(f"_{pattern}") or name_lower.endswith(pattern):
            return True

    return False


def normalize_slots(slots: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Normalize all slots in a dict according to Slot Hygiene Contract.

    Args:
        slots: Dict of slot name -> value

    Returns:
        Tuple of (normalized_slots, list of all warning messages)
    """
    if not slots:
        return {}, []

    normalized: dict[str, Any] = {}
    all_warnings: list[str] = []

    for name, value in slots.items():
        norm_value, warnings = normalize_slot(name, value)
        normalized[name] = norm_value
        all_warnings.extend(warnings)

    return normalized, all_warnings


def extract_slot_for_index(value: Any) -> Any:
    """Extract the minimal indexable value from a slot.

    For primitives: returns .value
    For references: returns ref_entity_id
    For lists: returns list of extracted values

    This enables consistent indexing regardless of wrapper structure.

    Args:
        value: A normalized slot value

    Returns:
        The minimal value suitable for indexing/comparison
    """
    if value is None:
        return None

    if isinstance(value, list):
        return [extract_slot_for_index(item) for item in cast(list[Any], value)]

    if isinstance(value, dict):
        value_dict = cast(SlotDict, value)
        # Reference: extract ref_entity_id
        if "ref_entity_id" in value_dict:
            return value_dict["ref_entity_id"]

        # Primitive wrapper: extract value
        if "value" in value_dict:
            return value_dict["value"]

        # Unrecognized dict - return as-is
        return value_dict

    # Already a bare primitive (shouldn't happen after normalization, but handle)
    return value


def validate_slot(name: str, value: Any, *, strict: bool = False) -> list[str]:
    """Validate a slot value and return errors/warnings.

    By default returns warnings. Set strict=True to also return errors
    that would block ingest in a stricter mode.

    Args:
        name: Slot name
        value: Slot value
        strict: If True, include blocking errors

    Returns:
        List of validation messages (empty if valid)
    """
    messages: list[str] = []

    if value is None:
        return messages

    if isinstance(value, list):
        for i, item in enumerate(cast(list[Any], value)):
            item_msgs = validate_slot(f"{name}[{i}]", item, strict=strict)
            messages.extend(item_msgs)
        return messages

    # Check reference validity
    if isinstance(value, dict) and "ref_entity_id" in cast(SlotDict, value):
        if not is_slot_reference(value):
            msg = f"Slot '{name}': invalid reference (missing ref_type)"
            messages.append(msg if strict else f"[warning] {msg}")
        return messages

    # Check primitive wrapper validity
    if isinstance(value, dict) and "value" in cast(SlotDict, value):
        value_dict = cast(SlotDict, value)
        if not is_slot_primitive_wrapper(value):
            msg = f"Slot '{name}': invalid primitive wrapper"
            messages.append(msg if strict else f"[warning] {msg}")
        elif value_dict.get("type") not in SLOT_TYPES:
            type_val: Any = value_dict.get("type")
            messages.append(
                f"[warning] Slot '{name}': unknown type '{type_val}' "
                f"(consider using one of: {', '.join(sorted(SLOT_TYPES))})"
            )
        return messages

    # Bare primitive not wrapped
    if isinstance(value, PRIMITIVE_TYPES):
        messages.append(
            f"[warning] Slot '{name}': bare primitive not wrapped "
            "(use normalize_slots before persistence)"
        )
        return messages

    # Unknown structure
    if isinstance(value, dict):
        messages.append(
            f"[warning] Slot '{name}': dict is neither reference nor primitive wrapper"
        )

    return messages


# Warning codes for structured Evidence
SLOT_WARNING_CODES = {
    "bare_numeric": "bare numeric value without unit",
    "entity_reference_candidate": "freeform string may be better as entity reference",
    "malformed_reference": "malformed reference structure",
    "malformed_primitive": "malformed primitive wrapper",
    "unrecognized_structure": "unrecognized dict structure",
    "unsupported_type": "unsupported value type",
    "missing_unit": "numeric value without unit",
}

# Max warnings per subject to avoid spam
MAX_WARNINGS_PER_SUBJECT = 20


def _parse_warning_to_structured(warning: str) -> tuple[str, str, str]:
    """Parse a warning message into structured components.

    Args:
        warning: Raw warning message from normalize_slot

    Returns:
        Tuple of (slot_name, warning_code, value_type)
    """
    # Extract slot name from "Slot 'name': ..."
    slot_name = "unknown"
    if "Slot '" in warning:
        start = warning.find("Slot '") + 6
        end = warning.find("':", start)
        if end > start:
            slot_name = warning[start:end]

    # Map warning message to code
    warning_code = "unknown"
    warning_lower = warning.lower()
    if "bare numeric" in warning_lower:
        warning_code = "bare_numeric"
    elif "entity reference" in warning_lower:
        warning_code = "entity_reference_candidate"
    elif "malformed reference" in warning_lower:
        warning_code = "malformed_reference"
    elif "malformed primitive" in warning_lower:
        warning_code = "malformed_primitive"
    elif "unrecognized dict" in warning_lower:
        warning_code = "unrecognized_structure"
    elif "unsupported value type" in warning_lower:
        warning_code = "unsupported_type"
    elif "without unit" in warning_lower:
        warning_code = "missing_unit"

    # Extract value type if mentioned (e.g., "wrapped as number")
    value_type = "unknown"
    if "wrapped as " in warning_lower:
        start = warning_lower.find("wrapped as ") + 11
        end = warning_lower.find(" ", start)
        if end == -1:
            end = len(warning_lower)
        value_type = warning_lower[start:end].rstrip(")")
    elif "type '" in warning_lower:
        start = warning_lower.find("type '") + 6
        end = warning_lower.find("'", start)
        if end > start:
            value_type = warning_lower[start:end]

    return slot_name, warning_code, value_type


def create_slot_warning_evidence(
    subject_id: UUID,
    subject_kind: str,
    warnings: list[str],
) -> list["Evidence"]:
    """Create Evidence records for slot normalization warnings.

    Implements spam control:
    - Deduplicates by (slot_name, warning_code)
    - Caps to MAX_WARNINGS_PER_SUBJECT
    - Stores types/codes only, not raw values

    Args:
        subject_id: The entity or observation ID.
        subject_kind: "entity" or "observation".
        warnings: List of warning messages from normalize_slots.

    Returns:
        List of Evidence objects to be added to the session.
    """
    from object_sense.models import Evidence, EvidenceSource, SubjectKind

    # Map string to enum
    sk = SubjectKind.ENTITY if subject_kind == "entity" else SubjectKind.OBSERVATION

    # Deduplicate by (slot_name, warning_code)
    seen: set[tuple[str, str]] = set()
    evidence_records: list[Evidence] = []

    for warning in warnings:
        if len(evidence_records) >= MAX_WARNINGS_PER_SUBJECT:
            logger.debug(
                "Slot warning cap reached for %s %s, skipping remaining",
                subject_kind,
                subject_id,
            )
            break

        slot_name, warning_code, value_type = _parse_warning_to_structured(warning)
        dedup_key = (slot_name, warning_code)

        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        evidence = Evidence(
            evidence_id=uuid4(),
            subject_kind=sk,
            subject_id=subject_id,
            predicate="slot_normalization_warning",
            target_id=None,
            source=EvidenceSource.SYSTEM,
            score=0.0,
            details={
                "slot_name": slot_name,
                "warning_code": warning_code,
                "value_type": value_type,
            },
        )
        evidence_records.append(evidence)

    return evidence_records


def normalize_and_record_slot_warnings(
    slots: dict[str, Any],
    subject_id: UUID,
    session: "AsyncSession",
    subject_kind: str = "entity",
) -> dict[str, Any]:
    """Normalize slots and record any warnings as Evidence.

    This is the preferred way to normalize slots when you have a session
    and subject_id available. Ensures warnings are persisted for debugging.

    Args:
        slots: Raw slot dict to normalize.
        subject_id: The entity or observation ID.
        session: Database session for persisting Evidence.
        subject_kind: "entity" or "observation".

    Returns:
        Normalized slots dict.
    """
    normalized, warnings = normalize_slots(slots)

    # Log warnings (compact)
    if warnings:
        logger.warning(
            "Slot hygiene [%s %s]: %d warning(s)",
            subject_kind,
            subject_id,
            len(warnings),
        )

    # Create Evidence records (deduplicated, capped)
    if warnings:
        evidence_records = create_slot_warning_evidence(
            subject_id, subject_kind, warnings
        )
        for evidence in evidence_records:
            session.add(evidence)

    return normalized
