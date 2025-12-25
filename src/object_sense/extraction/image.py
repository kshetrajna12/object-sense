"""Image feature extraction.

Handles:
- CAN_EMBED_IMAGE: CLIP visual embeddings
- CAN_EXTRACT_EXIF: EXIF metadata extraction
- CAN_CAPTION: Image captioning (via VLM) - deferred to LLM integration phase
"""

from __future__ import annotations

import io
from typing import Any

from object_sense.clients.embeddings import EmbeddingClient
from object_sense.extraction.base import ExtractionResult


class ImageExtractor:
    """Extract features from images.

    Features extracted:
    - image_embedding: CLIP visual embedding (768-dim)
    - extra.exif: EXIF metadata if available
    - extra.width/height: Image dimensions

    Note: Caption extraction (CAN_CAPTION) requires VLM integration,
    which will be added in the LLM integration phase. When captions
    are available, they'll be embedded via BGE → text_embedding.
    """

    def __init__(self, embedding_client: EmbeddingClient | None = None) -> None:
        self._client = embedding_client or EmbeddingClient()

    async def extract(self, data: bytes, *, filename: str | None = None) -> ExtractionResult:
        """Extract features from image bytes.

        Args:
            data: Raw image bytes (JPEG, PNG, WebP, etc.)
            filename: Optional filename (unused for images)

        Returns:
            ExtractionResult with image_embedding and EXIF metadata
        """
        result = ExtractionResult(signature_type="image")

        # Extract CLIP visual embedding
        embeddings = await self._client.embed_image([data])
        if embeddings:
            result.image_embedding = embeddings[0]

        # Extract EXIF and dimensions
        exif_data, dimensions = self._extract_metadata(data)
        if exif_data:
            result.extra["exif"] = exif_data
        if dimensions:
            result.extra["width"] = dimensions[0]
            result.extra["height"] = dimensions[1]

        return result

    def _extract_metadata(
        self, data: bytes
    ) -> tuple[dict[str, Any] | None, tuple[int, int] | None]:
        """Extract EXIF metadata and dimensions from image bytes.

        Returns:
            Tuple of (exif_dict, (width, height)) - either can be None
        """
        try:
            from PIL import Image
            from PIL.ExifTags import GPSTAGS, TAGS
        except ImportError:
            # Pillow not installed - skip metadata extraction
            return None, None

        try:
            img = Image.open(io.BytesIO(data))
            dimensions = img.size

            # Get EXIF data
            exif = img.getexif()
            if not exif:
                return None, dimensions

            exif_dict: dict[str, Any] = {}

            # Extract standard EXIF tags
            for tag_id, value in exif.items():
                tag_name = TAGS.get(tag_id, str(tag_id))
                serialized = self._serialize_exif_value(value)
                if serialized is not None:
                    exif_dict[tag_name.lower()] = serialized

            # Extract GPS data if present
            gps_ifd = exif.get_ifd(0x8825)  # GPS IFD
            if gps_ifd:
                gps_dict: dict[str, Any] = {}
                for tag_id, value in gps_ifd.items():
                    tag_name = GPSTAGS.get(tag_id, str(tag_id))
                    serialized = self._serialize_exif_value(value)
                    if serialized is not None:
                        gps_dict[tag_name.lower()] = serialized

                # Convert GPS coordinates to decimal degrees
                lat = self._gps_to_decimal(
                    gps_dict.get("gpslatitude"),
                    gps_dict.get("gpslatituderef", "N"),
                )
                if lat is not None:
                    exif_dict["latitude"] = lat

                lon = self._gps_to_decimal(
                    gps_dict.get("gpslongitude"),
                    gps_dict.get("gpslongituderef", "E"),
                )
                if lon is not None:
                    exif_dict["longitude"] = lon

                alt = gps_dict.get("gpsaltitude")
                if alt is not None:
                    exif_dict["altitude"] = alt

            return exif_dict if exif_dict else None, dimensions

        except Exception:
            # Image parsing failed - return what we can
            return None, None

    def _serialize_exif_value(self, value: Any) -> Any:
        """Convert EXIF value to JSON-serializable format."""
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8", errors="replace").strip("\x00")
            except Exception:
                return None
        if isinstance(value, tuple):
            # Handle rational numbers (common in EXIF)
            if len(value) == 2 and all(isinstance(v, (int, float)) for v in value):
                if value[1] != 0:
                    return value[0] / value[1]
                return None
            return [self._serialize_exif_value(v) for v in value]
        if hasattr(value, "numerator") and hasattr(value, "denominator"):
            # IFDRational
            if value.denominator != 0:
                return float(value)
            return None
        if isinstance(value, (int, float, str, bool, type(None))):
            return value
        return str(value)

    def _gps_to_decimal(self, coords: Any, ref: str) -> float | None:
        """Convert GPS coordinates to decimal degrees."""
        try:
            if not isinstance(coords, (list, tuple)) or len(coords) != 3:
                return None

            degrees = float(coords[0]) if coords[0] else 0
            minutes = float(coords[1]) if coords[1] else 0
            seconds = float(coords[2]) if coords[2] else 0

            decimal = degrees + minutes / 60 + seconds / 3600

            if ref in ("S", "W"):
                decimal = -decimal

            return round(decimal, 6)
        except Exception:
            return None
