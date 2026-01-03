"""Utility modules for ObjectSense."""

from object_sense.utils.medium import (
    MEDIUM_AFFORDANCES,
    get_affordances,
    probe_medium,
    probe_medium_from_path,
)
from object_sense.utils.slots import (
    extract_slot_for_index,
    is_slot_primitive_wrapper,
    is_slot_reference,
    normalize_slot,
    normalize_slots,
    validate_slot,
)

__all__ = [
    # Medium utilities
    "MEDIUM_AFFORDANCES",
    "get_affordances",
    "probe_medium",
    "probe_medium_from_path",
    # Slot utilities
    "extract_slot_for_index",
    "is_slot_primitive_wrapper",
    "is_slot_reference",
    "normalize_slot",
    "normalize_slots",
    "validate_slot",
]
