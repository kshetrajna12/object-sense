"""Tests for slot validation and normalization utilities.

Tests the Slot Hygiene Contract (concept_v2.md §6.3):
- Primitive wrapper: {"value": <primitive>, "type": <string>, "unit": <string?>}
- Reference wrapper: {"ref_entity_id": <uuid>, "ref_type": <string>, "display_name": <string?>}
"""

from uuid import uuid4

import pytest

from object_sense.utils.slots import (
    extract_slot_for_index,
    is_slot_primitive_wrapper,
    is_slot_reference,
    normalize_slot,
    normalize_slots,
    validate_slot,
)


class TestIsSlotReference:
    """Tests for is_slot_reference function."""

    def test_valid_reference(self) -> None:
        ref = {"ref_entity_id": str(uuid4()), "ref_type": "species_entity"}
        assert is_slot_reference(ref) is True

    def test_valid_reference_with_uuid_object(self) -> None:
        ref = {"ref_entity_id": uuid4(), "ref_type": "species_entity"}
        assert is_slot_reference(ref) is True

    def test_valid_reference_with_display_name(self) -> None:
        ref = {
            "ref_entity_id": str(uuid4()),
            "ref_type": "species_entity",
            "display_name": "African Leopard",
        }
        assert is_slot_reference(ref) is True

    def test_missing_ref_entity_id(self) -> None:
        ref = {"ref_type": "species_entity"}
        assert is_slot_reference(ref) is False

    def test_missing_ref_type(self) -> None:
        ref = {"ref_entity_id": str(uuid4())}
        assert is_slot_reference(ref) is False

    def test_empty_ref_type(self) -> None:
        ref = {"ref_entity_id": str(uuid4()), "ref_type": ""}
        assert is_slot_reference(ref) is False

    def test_non_dict(self) -> None:
        assert is_slot_reference("not a dict") is False
        assert is_slot_reference(42) is False
        assert is_slot_reference(None) is False

    def test_invalid_ref_entity_id_type(self) -> None:
        ref = {"ref_entity_id": 12345, "ref_type": "species_entity"}
        assert is_slot_reference(ref) is False


class TestIsSlotPrimitiveWrapper:
    """Tests for is_slot_primitive_wrapper function."""

    def test_valid_string_wrapper(self) -> None:
        wrapper = {"value": "hello", "type": "string"}
        assert is_slot_primitive_wrapper(wrapper) is True

    def test_valid_number_wrapper(self) -> None:
        wrapper = {"value": 42.99, "type": "currency", "unit": "USD"}
        assert is_slot_primitive_wrapper(wrapper) is True

    def test_valid_integer_wrapper(self) -> None:
        wrapper = {"value": 42, "type": "integer"}
        assert is_slot_primitive_wrapper(wrapper) is True

    def test_valid_boolean_wrapper(self) -> None:
        wrapper = {"value": True, "type": "boolean"}
        assert is_slot_primitive_wrapper(wrapper) is True

    def test_missing_value(self) -> None:
        wrapper = {"type": "string"}
        assert is_slot_primitive_wrapper(wrapper) is False

    def test_missing_type(self) -> None:
        wrapper = {"value": "hello"}
        assert is_slot_primitive_wrapper(wrapper) is False

    def test_empty_type(self) -> None:
        wrapper = {"value": "hello", "type": ""}
        assert is_slot_primitive_wrapper(wrapper) is False

    def test_non_primitive_value(self) -> None:
        wrapper = {"value": {"nested": "dict"}, "type": "object"}
        assert is_slot_primitive_wrapper(wrapper) is False

    def test_non_dict(self) -> None:
        assert is_slot_primitive_wrapper("not a dict") is False
        assert is_slot_primitive_wrapper(42) is False


class TestNormalizeSlot:
    """Tests for normalize_slot function."""

    def test_bare_string_wrapped(self) -> None:
        value, warnings = normalize_slot("name", "John Doe")
        assert value == {"value": "John Doe", "type": "string"}
        assert len(warnings) == 0

    def test_bare_integer_wrapped_with_warning(self) -> None:
        value, warnings = normalize_slot("count", 42)
        assert value == {"value": 42, "type": "integer"}
        assert len(warnings) == 1
        assert "bare numeric" in warnings[0]
        assert "unit" in warnings[0]

    def test_bare_float_wrapped_with_warning(self) -> None:
        value, warnings = normalize_slot("price", 19.99)
        assert value == {"value": 19.99, "type": "number"}
        assert len(warnings) == 1
        assert "bare numeric" in warnings[0]

    def test_bare_boolean_wrapped(self) -> None:
        value, warnings = normalize_slot("active", True)
        assert value == {"value": True, "type": "boolean"}
        assert len(warnings) == 0

    def test_reference_passthrough(self) -> None:
        ref = {"ref_entity_id": str(uuid4()), "ref_type": "species_entity"}
        value, warnings = normalize_slot("species", ref)
        assert value == ref
        assert len(warnings) == 0

    def test_primitive_wrapper_passthrough(self) -> None:
        wrapper = {"value": 42.99, "type": "currency", "unit": "USD"}
        value, warnings = normalize_slot("price", wrapper)
        assert value == wrapper
        assert len(warnings) == 0

    def test_primitive_wrapper_without_unit_warning(self) -> None:
        wrapper = {"value": 42.99, "type": "number"}
        value, warnings = normalize_slot("price", wrapper)
        assert value == wrapper
        assert len(warnings) == 1
        assert "without unit" in warnings[0]

    def test_integer_type_without_unit_warning(self) -> None:
        wrapper = {"value": 100, "type": "integer"}
        value, warnings = normalize_slot("quantity", wrapper)
        assert value == wrapper
        assert len(warnings) == 1

    def test_currency_type_without_unit_warning(self) -> None:
        wrapper = {"value": 99.99, "type": "currency"}
        value, warnings = normalize_slot("price", wrapper)
        assert value == wrapper
        assert len(warnings) == 1

    def test_none_value(self) -> None:
        value, warnings = normalize_slot("optional_field", None)
        assert value is None
        assert len(warnings) == 0

    def test_freeform_string_for_entity_slot_warning(self) -> None:
        # Slots that look like they should be entity references
        value, warnings = normalize_slot("species", "leopard")
        assert value == {"value": "leopard", "type": "string"}
        assert len(warnings) == 1
        assert "entity reference" in warnings[0]

    def test_freeform_string_for_category_warning(self) -> None:
        value, warnings = normalize_slot("category", "electronics")
        assert value == {"value": "electronics", "type": "string"}
        assert len(warnings) == 1
        assert "entity reference" in warnings[0]

    def test_freeform_string_for_brand_warning(self) -> None:
        value, warnings = normalize_slot("brand", "Nike")
        assert value == {"value": "Nike", "type": "string"}
        assert len(warnings) == 1
        assert "entity reference" in warnings[0]

    def test_freeform_string_for_location_warning(self) -> None:
        value, warnings = normalize_slot("location", "New York")
        assert value == {"value": "New York", "type": "string"}
        assert len(warnings) == 1

    def test_regular_string_no_entity_warning(self) -> None:
        # Regular slot names shouldn't trigger entity warning
        value, warnings = normalize_slot("description", "A nice product")
        assert value == {"value": "A nice product", "type": "string"}
        assert len(warnings) == 0

    def test_malformed_reference_warning(self) -> None:
        # Has ref_entity_id but missing ref_type
        value, warnings = normalize_slot("species", {"ref_entity_id": str(uuid4())})
        assert len(warnings) == 1
        assert "malformed reference" in warnings[0]

    def test_malformed_primitive_wrapper_fixed(self) -> None:
        # Has value but missing type - should be fixed
        value, warnings = normalize_slot("count", {"value": 42})
        assert value["type"] == "integer"
        assert len(warnings) == 1
        assert "malformed primitive" in warnings[0]

    def test_unknown_dict_structure_warning(self) -> None:
        value, warnings = normalize_slot("weird", {"foo": "bar", "baz": 123})
        assert len(warnings) == 1
        assert "unrecognized dict structure" in warnings[0]


class TestNormalizeSlotLists:
    """Tests for list normalization in normalize_slot."""

    def test_list_of_primitives(self) -> None:
        value, warnings = normalize_slot("tags", ["red", "blue", "green"])
        assert len(value) == 3
        assert value[0] == {"value": "red", "type": "string"}
        assert value[1] == {"value": "blue", "type": "string"}
        assert value[2] == {"value": "green", "type": "string"}
        assert len(warnings) == 0

    def test_list_of_numbers_with_warnings(self) -> None:
        value, warnings = normalize_slot("scores", [100, 85, 92])
        assert len(value) == 3
        assert all(v["type"] == "integer" for v in value)
        assert len(warnings) == 3  # One warning per bare numeric

    def test_list_of_references(self) -> None:
        refs = [
            {"ref_entity_id": str(uuid4()), "ref_type": "tag_entity"},
            {"ref_entity_id": str(uuid4()), "ref_type": "tag_entity"},
        ]
        value, warnings = normalize_slot("related_tags", refs)
        assert value == refs
        assert len(warnings) == 0

    def test_list_of_already_wrapped(self) -> None:
        wrapped = [
            {"value": "red", "type": "string"},
            {"value": "blue", "type": "string"},
        ]
        value, warnings = normalize_slot("colors", wrapped)
        assert value == wrapped
        assert len(warnings) == 0

    def test_mixed_list(self) -> None:
        # Mixed content - each element normalized independently
        items = ["foo", 42, {"value": "bar", "type": "string"}]
        value, warnings = normalize_slot("mixed", items)
        assert len(value) == 3
        assert value[0] == {"value": "foo", "type": "string"}
        assert value[1] == {"value": 42, "type": "integer"}
        assert value[2] == {"value": "bar", "type": "string"}
        # Warning for bare numeric
        assert len(warnings) == 1


class TestNormalizeSlots:
    """Tests for normalize_slots function."""

    def test_empty_dict(self) -> None:
        result, warnings = normalize_slots({})
        assert result == {}
        assert len(warnings) == 0

    def test_multiple_slots(self) -> None:
        slots = {
            "name": "Test Product",
            "price": 19.99,
            "active": True,
        }
        result, warnings = normalize_slots(slots)

        assert result["name"] == {"value": "Test Product", "type": "string"}
        assert result["price"] == {"value": 19.99, "type": "number"}
        assert result["active"] == {"value": True, "type": "boolean"}

        # Warning for price (bare numeric)
        assert len(warnings) == 1

    def test_mixed_normalized_and_raw(self) -> None:
        slots = {
            "species": {"ref_entity_id": str(uuid4()), "ref_type": "species_entity"},
            "count": 5,
            "name": {"value": "Example", "type": "string"},
        }
        result, warnings = normalize_slots(slots)

        # Reference and already-wrapped should pass through
        assert result["species"] == slots["species"]
        assert result["name"] == slots["name"]

        # Bare numeric wrapped
        assert result["count"] == {"value": 5, "type": "integer"}
        assert len(warnings) == 1


class TestExtractSlotForIndex:
    """Tests for extract_slot_for_index function."""

    def test_primitive_wrapper(self) -> None:
        wrapper = {"value": 42, "type": "integer"}
        assert extract_slot_for_index(wrapper) == 42

    def test_primitive_wrapper_with_unit(self) -> None:
        wrapper = {"value": 19.99, "type": "currency", "unit": "USD"}
        assert extract_slot_for_index(wrapper) == 19.99

    def test_reference(self) -> None:
        entity_id = str(uuid4())
        ref = {"ref_entity_id": entity_id, "ref_type": "species_entity"}
        assert extract_slot_for_index(ref) == entity_id

    def test_none(self) -> None:
        assert extract_slot_for_index(None) is None

    def test_bare_primitive(self) -> None:
        # Shouldn't happen after normalization, but handle gracefully
        assert extract_slot_for_index(42) == 42
        assert extract_slot_for_index("hello") == "hello"

    def test_list_of_wrappers(self) -> None:
        items = [
            {"value": "a", "type": "string"},
            {"value": "b", "type": "string"},
        ]
        result = extract_slot_for_index(items)
        assert result == ["a", "b"]

    def test_list_of_references(self) -> None:
        id1, id2 = str(uuid4()), str(uuid4())
        items = [
            {"ref_entity_id": id1, "ref_type": "tag_entity"},
            {"ref_entity_id": id2, "ref_type": "tag_entity"},
        ]
        result = extract_slot_for_index(items)
        assert result == [id1, id2]

    def test_unrecognized_dict(self) -> None:
        # Dict without value or ref_entity_id - return as-is
        weird = {"foo": "bar"}
        assert extract_slot_for_index(weird) == weird


class TestValidateSlot:
    """Tests for validate_slot function."""

    def test_valid_reference(self) -> None:
        ref = {"ref_entity_id": str(uuid4()), "ref_type": "species_entity"}
        messages = validate_slot("species", ref)
        assert len(messages) == 0

    def test_valid_primitive_wrapper(self) -> None:
        wrapper = {"value": 42, "type": "integer"}
        messages = validate_slot("count", wrapper)
        assert len(messages) == 0

    def test_invalid_reference(self) -> None:
        ref = {"ref_entity_id": str(uuid4())}  # Missing ref_type
        messages = validate_slot("species", ref)
        assert len(messages) == 1
        assert "invalid reference" in messages[0].lower()

    def test_bare_primitive_warning(self) -> None:
        messages = validate_slot("count", 42)
        assert len(messages) == 1
        assert "bare primitive" in messages[0].lower()

    def test_unknown_type_warning(self) -> None:
        wrapper = {"value": "test", "type": "custom_weird_type"}
        messages = validate_slot("field", wrapper)
        assert len(messages) == 1
        assert "unknown type" in messages[0].lower()

    def test_known_types_no_warning(self) -> None:
        for type_name in ["string", "number", "integer", "boolean", "currency", "url"]:
            wrapper = {"value": "test", "type": type_name}
            messages = validate_slot("field", wrapper)
            assert len(messages) == 0, f"Unexpected warning for type {type_name}"

    def test_list_validation(self) -> None:
        items = [{"value": "a", "type": "string"}, 42]  # Second is bare
        messages = validate_slot("items", items)
        assert len(messages) == 1  # Warning for bare primitive

    def test_none_valid(self) -> None:
        messages = validate_slot("optional", None)
        assert len(messages) == 0

    def test_strict_mode(self) -> None:
        ref = {"ref_entity_id": str(uuid4())}  # Invalid
        messages_default = validate_slot("species", ref)
        messages_strict = validate_slot("species", ref, strict=True)

        # Default mode prefixes with [warning]
        assert "[warning]" in messages_default[0]

        # Strict mode gives raw message
        assert "[warning]" not in messages_strict[0]


class TestSlotNormalizationEdgeCases:
    """Edge case tests for slot normalization."""

    def test_deeply_nested_list(self) -> None:
        # Lists of lists are handled element-by-element
        items = [["a", "b"], ["c", "d"]]
        value, warnings = normalize_slot("nested", items)
        # Each inner list is normalized
        assert len(value) == 2
        assert value[0][0] == {"value": "a", "type": "string"}

    def test_empty_list(self) -> None:
        value, warnings = normalize_slot("empty", [])
        assert value == []
        assert len(warnings) == 0

    def test_slot_name_with_parent_category(self) -> None:
        # Suffix pattern should trigger entity warning
        value, warnings = normalize_slot("parent_category", "electronics")
        assert len(warnings) == 1
        assert "entity reference" in warnings[0]

    def test_slot_name_with_primary_species(self) -> None:
        value, warnings = normalize_slot("primary_species", "leopard")
        assert len(warnings) == 1
        assert "entity reference" in warnings[0]

    def test_false_boolean_wrapped_correctly(self) -> None:
        value, warnings = normalize_slot("disabled", False)
        assert value == {"value": False, "type": "boolean"}
        assert len(warnings) == 0

    def test_zero_integer_wrapped_correctly(self) -> None:
        value, warnings = normalize_slot("count", 0)
        assert value == {"value": 0, "type": "integer"}
        # Zero is still a numeric, should have warning
        assert len(warnings) == 1

    def test_empty_string_wrapped(self) -> None:
        value, warnings = normalize_slot("name", "")
        assert value == {"value": "", "type": "string"}
        assert len(warnings) == 0

    def test_reference_with_extra_fields(self) -> None:
        # Extra fields on reference should be preserved
        ref = {
            "ref_entity_id": str(uuid4()),
            "ref_type": "species_entity",
            "display_name": "Leopard",
            "confidence": 0.95,
        }
        value, warnings = normalize_slot("species", ref)
        assert value == ref
        assert len(warnings) == 0
