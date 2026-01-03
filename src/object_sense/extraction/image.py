"""Image feature extraction.

Handles:
- CAN_EMBED_IMAGE: CLIP visual embeddings
- CAN_EXTRACT_EXIF: EXIF metadata extraction
- Deterministic ID extraction from GPS coordinates
- CAN_CAPTION: Image captioning (via VLM) - deferred to LLM integration phase

Supports standard image formats (JPEG, PNG, WebP, etc.) and RAW formats
(CR2, CR3, NEF, ARW, DNG, etc.) via rawpy.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from object_sense.clients.embeddings import EmbeddingClient
from object_sense.extraction.base import ExtractedId, ExtractionResult

# GPS coordinate precision for deterministic ID (6 decimal places = ~0.1m accuracy)
GPS_PRECISION = 6

# RAW file extensions that need rawpy processing
RAW_EXTENSIONS = frozenset({
    ".raw", ".arw", ".cr2", ".cr3", ".nef", ".nrw",
    ".orf", ".rw2", ".pef", ".srw", ".x3f", ".raf",
    ".dng", ".dcr", ".kdc", ".mrw", ".3fr", ".mef",
    ".mos", ".erf", ".rwl",
})


def _is_raw_file(filename: str | None) -> bool:
    """Check if filename indicates a RAW image format."""
    if not filename:
        return False
    return Path(filename).suffix.lower() in RAW_EXTENSIONS


class ImageExtractor:
    """Extract features from images.

    Features extracted:
    - image_embedding: CLIP visual embedding (768-dim)
    - deterministic_ids: GPS coordinates as (gps, "lat,lon", wgs84) if available
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
            data: Raw image bytes (JPEG, PNG, WebP, RAW formats, etc.)
            filename: Optional filename (used to detect RAW formats)

        Returns:
            ExtractionResult with image_embedding, EXIF metadata, and GPS deterministic ID
        """
        result = ExtractionResult(signature_type="image")

        # Check if this is a RAW file that needs special handling
        is_raw = _is_raw_file(filename)

        if is_raw:
            # Process RAW file - extract embedded JPEG thumbnail for embedding
            jpeg_data, dimensions, raw_info = self._extract_raw_thumbnail(data)
            if jpeg_data:
                # Use extracted JPEG for embedding
                embeddings = await self._client.embed_image([jpeg_data])
                if embeddings:
                    result.image_embedding = embeddings[0]
            if dimensions:
                result.extra["width"] = dimensions[0]
                result.extra["height"] = dimensions[1]
            if raw_info:
                result.extra.update(raw_info)
            result.extra["is_raw"] = True
        else:
            # Standard image processing
            embeddings = await self._client.embed_image([data])
            if embeddings:
                result.image_embedding = embeddings[0]

            # Extract EXIF and dimensions
            exif_data, dimensions = self._extract_metadata(data)
            if exif_data:
                result.extra["exif"] = exif_data

                # Extract GPS as deterministic ID if available
                gps_id = self._extract_gps_id(exif_data)
                if gps_id:
                    result.deterministic_ids.append(gps_id)

            if dimensions:
                result.extra["width"] = dimensions[0]
                result.extra["height"] = dimensions[1]

        return result

    def _extract_gps_id(self, exif_data: dict[str, Any]) -> ExtractedId | None:
        """Extract GPS coordinates as a deterministic ID.

        Args:
            exif_data: Extracted EXIF data with latitude/longitude

        Returns:
            ExtractedId with (gps, "lat,lon", wgs84) or None if no GPS
        """
        lat = exif_data.get("latitude")
        lon = exif_data.get("longitude")

        if lat is None or lon is None:
            return None

        # Format as "lat,lon" with fixed precision
        gps_value = f"{lat:.{GPS_PRECISION}f},{lon:.{GPS_PRECISION}f}"

        return ExtractedId(
            id_type="gps",
            id_value=gps_value,
            id_namespace="wgs84",
            strength="strong",
        )

    def _extract_raw_thumbnail(
        self, data: bytes
    ) -> tuple[bytes | None, tuple[int, int] | None, dict[str, Any] | None]:
        """Extract JPEG thumbnail from RAW file for embedding.

        RAW files contain an embedded JPEG preview that's fast to extract
        and suitable for generating visual embeddings.

        Returns:
            Tuple of (jpeg_bytes, (width, height), raw_info) - any can be None
        """
        try:
            import rawpy
        except ImportError:
            # rawpy not available - can't process RAW
            return None, None, {"error": "rawpy not installed"}

        try:
            # rawpy needs a file, so write to temp file
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as tmp:
                tmp.write(data)
                tmp_path = tmp.name

            try:
                with rawpy.imread(tmp_path) as raw:
                    raw_type = raw.raw_type
                    raw_type_str = raw_type.name if hasattr(raw_type, "name") else str(raw_type)
                    raw_info: dict[str, Any] = {
                        "raw_type": raw_type_str,
                        "num_colors": raw.num_colors,
                    }

                    # Get dimensions from raw sizes
                    if hasattr(raw, "sizes"):
                        raw_info["raw_width"] = raw.sizes.raw_width
                        raw_info["raw_height"] = raw.sizes.raw_height
                        dimensions = (raw.sizes.width, raw.sizes.height)
                    else:
                        dimensions = None

                    # Try to extract embedded thumbnail (fastest)
                    try:
                        thumb = raw.extract_thumb()
                        if thumb.format == rawpy.ThumbFormat.JPEG:
                            return thumb.data, dimensions, raw_info
                        # Bitmap thumbnail - convert to JPEG
                        from PIL import Image

                        img = Image.fromarray(thumb.data)
                        buf = io.BytesIO()
                        img.save(buf, format="JPEG", quality=85)
                        return buf.getvalue(), dimensions, raw_info
                    except rawpy.LibRawNoThumbnailError:
                        pass  # No thumbnail, fall through to decode

                    # No thumbnail - decode RAW at half size for speed
                    rgb = raw.postprocess(use_camera_wb=True, half_size=True)
                    from PIL import Image

                    img = Image.fromarray(rgb)
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=85)
                    return buf.getvalue(), dimensions, raw_info

            finally:
                # Clean up temp file
                import os
                os.unlink(tmp_path)

        except Exception as e:
            return None, None, {"error": f"RAW processing failed: {e}"}

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
