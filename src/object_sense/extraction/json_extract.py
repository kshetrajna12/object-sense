"""JSON feature extraction.

Handles:
- CAN_PARSE_KEYS: Extract and analyze JSON structure
- CAN_INFER_SCHEMA: Generate schema hash for identity
- CAN_EMBED_TEXT: Embed JSON content as text
- Deterministic ID extraction from known ID fields
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from object_sense.clients.embeddings import EmbeddingClient
from object_sense.extraction.base import ExtractedId, ExtractionResult

# Allowlist of keys that are treated as deterministic identifiers.
# These are extracted as (id_type, id_value, namespace) tuples.
# Keys are matched case-insensitively.
ID_KEY_ALLOWLIST: frozenset[str] = frozenset({
    # Product identifiers
    "sku",
    "product_id",
    "productid",
    "item_id",
    "itemid",
    "upc",
    "ean",
    "gtin",
    "asin",
    "isbn",
    # Order/transaction identifiers
    "order_id",
    "orderid",
    "booking_id",
    "bookingid",
    "trip_id",
    "tripid",
    "reservation_id",
    "reservationid",
    "transaction_id",
    "transactionid",
    "invoice_id",
    "invoiceid",
    # Generic identifiers
    "id",
    "uuid",
    "guid",
    "external_id",
    "externalid",
    "ref",
    "reference",
    "code",
})


class JsonExtractor:
    """Extract features from JSON content.

    Features extracted:
    - text_embedding: BGE embedding of JSON-as-text (1024-dim)
    - hash_value: Schema hash (structure fingerprint)
    - extracted_text: Flattened key-value pairs as text
    - deterministic_ids: IDs from allowlisted keys (sku, product_id, etc.)
    - extra.schema: Inferred schema structure
    - extra.key_count: Number of keys
    - extra.is_array: Whether root is array
    """

    def __init__(
        self,
        embedding_client: EmbeddingClient | None = None,
        *,
        id_namespace: str = "json",
    ) -> None:
        self._client = embedding_client or EmbeddingClient()
        self._id_namespace = id_namespace

    async def extract(self, data: bytes, *, filename: str | None = None) -> ExtractionResult:
        """Extract features from JSON bytes.

        Args:
            data: Raw JSON bytes
            filename: Optional filename (unused)

        Returns:
            ExtractionResult with schema hash, text embedding, and deterministic IDs
        """
        result = ExtractionResult(signature_type="json")

        # Parse JSON
        try:
            text = data.decode("utf-8")
            parsed = json.loads(text)
        except (UnicodeDecodeError, json.JSONDecodeError):
            # Invalid JSON - return empty result
            return result

        # Analyze structure
        is_array = isinstance(parsed, list)
        result.extra["is_array"] = is_array

        # Infer schema and generate hash
        schema = self._infer_schema(parsed)
        result.extra["schema"] = schema
        result.hash_value = self._hash_schema(schema)

        # Count keys
        if isinstance(parsed, dict):
            result.extra["key_count"] = len(parsed)
        elif isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            result.extra["key_count"] = len(parsed[0])

        # Extract deterministic IDs from allowlisted keys
        result.deterministic_ids = self._extract_deterministic_ids(parsed)

        # Convert to text for embedding
        text_repr = self._to_text(parsed)
        if text_repr:
            result.extracted_text = text_repr

            # Embed as text
            embeddings = await self._client.embed_text([text_repr])
            if embeddings:
                result.text_embedding = embeddings[0]

        return result

    def _extract_deterministic_ids(self, obj: Any) -> list[ExtractedId]:
        """Extract deterministic IDs from JSON using allowlisted keys.

        Scans the JSON structure for keys in ID_KEY_ALLOWLIST and extracts
        their values as deterministic identifiers.

        Args:
            obj: Parsed JSON object (dict, list, or primitive)

        Returns:
            List of ExtractedId with normalized values
        """
        ids: list[ExtractedId] = []
        seen: set[tuple[str, str, str]] = set()  # Dedup within same document

        self._scan_for_ids(obj, ids, seen)
        return ids

    def _scan_for_ids(
        self,
        obj: Any,
        ids: list[ExtractedId],
        seen: set[tuple[str, str, str]],
        max_depth: int = 10,
    ) -> None:
        """Recursively scan JSON for ID fields."""
        if max_depth <= 0:
            return

        if isinstance(obj, dict):
            for key, value in obj.items():
                key_lower = key.lower()
                # Check if this key is in our allowlist
                if key_lower in ID_KEY_ALLOWLIST:
                    # Extract the ID value
                    id_value = self._normalize_id_value(value)
                    if id_value:
                        # Normalize key to canonical form (lowercase, underscores)
                        id_type = key_lower.replace("-", "_")
                        id_tuple = (id_type, id_value, self._id_namespace)
                        if id_tuple not in seen:
                            seen.add(id_tuple)
                            ids.append(ExtractedId(
                                id_type=id_type,
                                id_value=id_value,
                                id_namespace=self._id_namespace,
                                strength="strong",
                            ))
                else:
                    # Recurse into nested structures
                    self._scan_for_ids(value, ids, seen, max_depth - 1)

        elif isinstance(obj, list):
            for item in obj[:100]:  # Limit to first 100 items
                self._scan_for_ids(item, ids, seen, max_depth - 1)

    def _normalize_id_value(self, value: Any) -> str | None:
        """Normalize an ID value to string, or None if invalid.

        Only accepts non-empty strings, integers, and UUIDs.
        Rejects floats, booleans, None, and complex objects.
        """
        if value is None:
            return None
        if isinstance(value, bool):
            return None  # Booleans are not valid IDs
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            return None  # Floats are not valid IDs (precision issues)
        if isinstance(value, str):
            normalized = value.strip()
            if normalized:
                return normalized
            return None
        # Complex objects (dict, list) are not valid ID values
        return None

    def _infer_schema(self, obj: Any, max_depth: int = 5) -> dict[str, Any]:
        """Infer a type schema from JSON object.

        Returns a structure like:
        {"type": "object", "properties": {"name": {"type": "string"}, ...}}
        """
        if max_depth <= 0:
            return {"type": "any"}

        if obj is None:
            return {"type": "null"}
        if isinstance(obj, bool):
            return {"type": "boolean"}
        if isinstance(obj, int):
            return {"type": "integer"}
        if isinstance(obj, float):
            return {"type": "number"}
        if isinstance(obj, str):
            return {"type": "string"}
        if isinstance(obj, list):
            if not obj:
                return {"type": "array", "items": {"type": "any"}}
            # Infer from first item (simplified)
            item_schema = self._infer_schema(obj[0], max_depth - 1)
            return {"type": "array", "items": item_schema}
        if isinstance(obj, dict):
            properties = {}
            for key, value in sorted(obj.items()):
                properties[key] = self._infer_schema(value, max_depth - 1)
            return {"type": "object", "properties": properties}

        return {"type": "unknown"}

    def _hash_schema(self, schema: dict[str, Any]) -> str:
        """Generate a stable hash of the schema structure."""
        # Canonical JSON representation
        canonical = json.dumps(schema, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def _to_text(self, obj: Any, max_len: int = 8000) -> str:
        """Convert JSON to text representation for embedding.

        Flattens to key-value pairs for better semantic matching.
        """
        lines: list[str] = []
        self._flatten(obj, "", lines)

        text = "\n".join(lines)
        if len(text) > max_len:
            text = text[:max_len]

        return text

    def _flatten(self, obj: Any, prefix: str, lines: list[str], max_lines: int = 200) -> None:
        """Recursively flatten JSON to key-value lines."""
        if len(lines) >= max_lines:
            return

        if isinstance(obj, dict):
            for key, value in obj.items():
                new_prefix = f"{prefix}.{key}" if prefix else key
                self._flatten(value, new_prefix, lines, max_lines)
        elif isinstance(obj, list):
            for i, item in enumerate(obj[:10]):  # Limit array items
                new_prefix = f"{prefix}[{i}]"
                self._flatten(item, new_prefix, lines, max_lines)
            if len(obj) > 10:
                lines.append(f"{prefix}: ... ({len(obj)} items)")
        else:
            # Leaf value
            value_str = str(obj) if obj is not None else "null"
            if len(value_str) > 200:
                value_str = value_str[:200] + "..."
            lines.append(f"{prefix}: {value_str}")
