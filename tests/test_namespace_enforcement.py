"""Tests for engine-controlled namespace enforcement (bead object-sense-bk9).

The engine controls namespace assignment for deterministic IDs:
- LLM-invented namespaces are overridden
- ID types map to canonical namespaces (gps→geo:wgs84, upc→global:upc)
- Invalid namespaces are overridden to context namespace
- Overrides are tracked for Evidence recording
"""

from __future__ import annotations

import pytest

from object_sense.cli import (
    NamespaceOverride,
    UnionResult,
    _is_valid_namespace,
    _resolve_namespace,
    _union_deterministic_ids,
)
from object_sense.extraction.base import ExtractedId
from object_sense.inference.schemas import DeterministicId


class TestValidNamespace:
    """Tests for namespace pattern validation."""

    def test_valid_source_namespace(self) -> None:
        assert _is_valid_namespace("source:product_catalog")
        assert _is_valid_namespace("source:wildlife_survey_2024")
        assert _is_valid_namespace("source:my-dataset")

    def test_valid_global_namespace(self) -> None:
        assert _is_valid_namespace("global:upc")
        assert _is_valid_namespace("global:isbn")
        assert _is_valid_namespace("global:asin")
        assert _is_valid_namespace("global:gtin")

    def test_valid_geo_namespace(self) -> None:
        assert _is_valid_namespace("geo:wgs84")

    def test_valid_user_namespace(self) -> None:
        assert _is_valid_namespace("user:tenant_123")
        assert _is_valid_namespace("user:my-tenant")

    def test_invalid_llm_invented_namespace(self) -> None:
        """LLM-invented namespaces should be rejected."""
        assert not _is_valid_namespace("catalog_sku")
        assert not _is_valid_namespace("product_sku")
        assert not _is_valid_namespace("acme_corp")
        assert not _is_valid_namespace("internal")

    def test_invalid_pattern(self) -> None:
        assert not _is_valid_namespace("random:namespace")
        assert not _is_valid_namespace("geo:something_else")
        assert not _is_valid_namespace("")
        assert not _is_valid_namespace("just_a_string")


class TestResolveNamespace:
    """Tests for namespace resolution logic."""

    def test_id_type_mapping_takes_precedence(self) -> None:
        """Engine-controlled mappings always win, even over valid proposed namespaces."""
        # GPS always maps to geo:wgs84
        ns, reason = _resolve_namespace("gps", "source:my_data", "source:default")
        assert ns == "geo:wgs84"
        assert reason == "id_type_mapped"

        # UPC always maps to global:upc
        ns, reason = _resolve_namespace("upc", "global:other", "source:default")
        assert ns == "global:upc"
        assert reason == "id_type_mapped"

    def test_id_type_mapping_no_override_if_matches(self) -> None:
        """No override recorded if proposed namespace matches the mapping."""
        ns, reason = _resolve_namespace("gps", "geo:wgs84", "source:default")
        assert ns == "geo:wgs84"
        assert reason is None  # No override needed

    def test_valid_proposed_namespace_accepted(self) -> None:
        """Valid proposed namespaces are accepted for unmapped ID types."""
        ns, reason = _resolve_namespace("sku", "source:product_catalog", "source:default")
        assert ns == "source:product_catalog"
        assert reason is None

    def test_invalid_namespace_overridden(self) -> None:
        """Invalid (LLM-invented) namespaces are overridden."""
        ns, reason = _resolve_namespace("sku", "catalog_sku", "source:product_catalog")
        assert ns == "source:product_catalog"
        assert reason == "invalid_pattern"

    def test_empty_namespace_overridden(self) -> None:
        """Empty/None namespace is overridden."""
        ns, reason = _resolve_namespace("sku", None, "source:product_catalog")
        assert ns == "source:product_catalog"
        assert reason == "empty_namespace"

        # Empty string is also treated as empty (falsy), same as None
        ns, reason = _resolve_namespace("sku", "", "source:product_catalog")
        assert ns == "source:product_catalog"
        assert reason == "empty_namespace"


class TestUnionDeterministicIds:
    """Tests for the union function with namespace enforcement."""

    def test_extracted_ids_preserved(self) -> None:
        """Extracted IDs from Step 3 are preserved."""
        extracted = [
            ExtractedId(id_type="sku", id_value="PROD-001", id_namespace="source:catalog"),
        ]
        result = _union_deterministic_ids(extracted, [], "source:default")

        assert len(result.ids) == 1
        assert result.ids[0].id_value == "PROD-001"
        assert result.ids[0].id_namespace == "source:catalog"
        assert len(result.overrides) == 0

    def test_llm_invalid_namespace_overridden(self) -> None:
        """LLM IDs with invalid namespaces are overridden."""
        llm_ids = [
            DeterministicId(
                id_type="sku",
                id_value="PROD-001",
                id_namespace="catalog_sku",  # Invalid LLM-invented namespace
            ),
        ]
        result = _union_deterministic_ids([], llm_ids, "source:product_catalog")

        assert len(result.ids) == 1
        assert result.ids[0].id_namespace == "source:product_catalog"  # Overridden
        assert len(result.overrides) == 1
        assert result.overrides[0].reason == "invalid_pattern"
        assert result.overrides[0].original_namespace == "catalog_sku"

    def test_llm_empty_namespace_overridden(self) -> None:
        """LLM IDs with no namespace use context namespace."""
        llm_ids = [
            DeterministicId(
                id_type="sku",
                id_value="PROD-001",
                id_namespace=None,
            ),
        ]
        result = _union_deterministic_ids([], llm_ids, "source:catalog")

        assert result.ids[0].id_namespace == "source:catalog"
        assert len(result.overrides) == 1
        assert result.overrides[0].reason == "empty_namespace"

    def test_id_type_mapping_applied(self) -> None:
        """ID type → namespace mappings are applied."""
        llm_ids = [
            DeterministicId(
                id_type="gps",
                id_value="-25.7461,28.1881",
                id_namespace=None,
            ),
            DeterministicId(
                id_type="upc",
                id_value="012345678901",
                id_namespace=None,
            ),
        ]
        result = _union_deterministic_ids([], llm_ids, "source:default")

        assert result.ids[0].id_namespace == "geo:wgs84"
        assert result.ids[1].id_namespace == "global:upc"

    def test_deduplication_by_tuple(self) -> None:
        """IDs are deduplicated by (type, value, namespace) tuple."""
        extracted = [
            ExtractedId(id_type="sku", id_value="PROD-001", id_namespace="source:catalog"),
        ]
        llm_ids = [
            # Same ID - should be deduplicated
            DeterministicId(
                id_type="sku",
                id_value="PROD-001",
                id_namespace="source:catalog",  # Valid and matches
            ),
        ]
        result = _union_deterministic_ids(extracted, llm_ids, "source:default")

        assert len(result.ids) == 1  # Only one ID

    def test_same_value_different_namespace_not_deduplicated(self) -> None:
        """Same ID value in different (valid) namespaces are kept separate."""
        llm_ids = [
            DeterministicId(
                id_type="sku",
                id_value="PROD-001",
                id_namespace="source:catalog_a",
            ),
            DeterministicId(
                id_type="sku",
                id_value="PROD-001",
                id_namespace="source:catalog_b",
            ),
        ]
        result = _union_deterministic_ids([], llm_ids, "source:default")

        assert len(result.ids) == 2

    def test_same_sku_different_invalid_namespaces_deduplicated(self) -> None:
        """Same SKU with different invalid namespaces → same after override → deduplicated.

        This is the core bug fix: LLM inventing 'catalog_sku' vs 'product_sku' for the
        same SKU should result in ONE entity, not two.
        """
        llm_ids = [
            DeterministicId(
                id_type="sku",
                id_value="PROD-001",
                id_namespace="catalog_sku",  # Invalid, will be overridden
            ),
            DeterministicId(
                id_type="sku",
                id_value="PROD-001",
                id_namespace="product_sku",  # Invalid, will be overridden
            ),
        ]
        result = _union_deterministic_ids([], llm_ids, "source:product_catalog")

        # Both should be overridden to same namespace, then deduplicated
        assert len(result.ids) == 1
        assert result.ids[0].id_namespace == "source:product_catalog"
        # Two overrides recorded (before dedup), but only one ID
        assert len(result.overrides) == 2


class TestCrossFileEntityResolution:
    """Integration-style tests for cross-file entity resolution.

    These tests verify that the same SKU across different JSON files
    results in the same entity after namespace enforcement.
    """

    def test_same_sku_same_namespace_same_entity(self) -> None:
        """Two files with same SKU in same namespace → same entity key."""
        # Simulate two product JSON files being ingested
        file1_ids = [
            DeterministicId(id_type="sku", id_value="SKU-12345", id_namespace=None),
        ]
        file2_ids = [
            DeterministicId(id_type="sku", id_value="SKU-12345", id_namespace=None),
        ]

        # Both ingested with same context namespace
        result1 = _union_deterministic_ids([], file1_ids, "source:product_catalog")
        result2 = _union_deterministic_ids([], file2_ids, "source:product_catalog")

        # The ID tuples should be identical → same entity
        tuple1 = (
            result1.ids[0].id_type,
            result1.ids[0].id_value,
            result1.ids[0].id_namespace,
        )
        tuple2 = (
            result2.ids[0].id_type,
            result2.ids[0].id_value,
            result2.ids[0].id_namespace,
        )

        assert tuple1 == tuple2
        assert tuple1 == ("sku", "SKU-12345", "source:product_catalog")

    def test_llm_namespace_chaos_normalized(self) -> None:
        """LLM proposing different invalid namespaces → normalized to same.

        Simulates LLM proposing 'catalog_sku' for file1 and 'product_sku' for file2,
        both for the same SKU. After enforcement, they should match.
        """
        file1_ids = [
            DeterministicId(
                id_type="sku",
                id_value="SKU-12345",
                id_namespace="catalog_sku",  # LLM invention
            ),
        ]
        file2_ids = [
            DeterministicId(
                id_type="sku",
                id_value="SKU-12345",
                id_namespace="product_sku",  # Different LLM invention
            ),
        ]

        result1 = _union_deterministic_ids([], file1_ids, "source:product_catalog")
        result2 = _union_deterministic_ids([], file2_ids, "source:product_catalog")

        # Both overridden to context namespace
        assert result1.ids[0].id_namespace == "source:product_catalog"
        assert result2.ids[0].id_namespace == "source:product_catalog"

        # Same entity key
        tuple1 = (
            result1.ids[0].id_type,
            result1.ids[0].id_value,
            result1.ids[0].id_namespace,
        )
        tuple2 = (
            result2.ids[0].id_type,
            result2.ids[0].id_value,
            result2.ids[0].id_namespace,
        )
        assert tuple1 == tuple2
