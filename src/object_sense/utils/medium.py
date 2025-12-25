"""Medium probing and affordance detection.

Detects the medium (image/video/text/json/audio/binary) from file content
using magic bytes, extensions, and content analysis. Maps mediums to
their affordances (what operations are possible).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

from object_sense.models.enums import Affordance, Medium

# Magic byte signatures for common formats
# Format: (magic_bytes, offset, medium)
_MAGIC_SIGNATURES: Final[list[tuple[bytes, int, Medium]]] = [
    # Images
    (b"\xff\xd8\xff", 0, Medium.IMAGE),  # JPEG
    (b"\x89PNG\r\n\x1a\n", 0, Medium.IMAGE),  # PNG
    (b"GIF87a", 0, Medium.IMAGE),  # GIF87a
    (b"GIF89a", 0, Medium.IMAGE),  # GIF89a
    (b"RIFF", 0, Medium.IMAGE),  # WebP (checked further below)
    (b"BM", 0, Medium.IMAGE),  # BMP
    (b"II*\x00", 0, Medium.IMAGE),  # TIFF little-endian
    (b"MM\x00*", 0, Medium.IMAGE),  # TIFF big-endian
    # HEIC/HEIF - ftyp box with heic/mif1 brand (checked in _check_ftyp)
    # Video
    (b"\x00\x00\x00", 0, Medium.VIDEO),  # MP4/MOV/etc ftyp box (checked further)
    (b"\x1aE\xdf\xa3", 0, Medium.VIDEO),  # WebM/MKV (EBML)
    (b"RIFF", 0, Medium.VIDEO),  # AVI (checked further below)
    # Audio
    (b"ID3", 0, Medium.AUDIO),  # MP3 with ID3 tag
    (b"\xff\xfb", 0, Medium.AUDIO),  # MP3 frame sync
    (b"\xff\xfa", 0, Medium.AUDIO),  # MP3 frame sync
    (b"\xff\xf3", 0, Medium.AUDIO),  # MP3 frame sync
    (b"\xff\xf2", 0, Medium.AUDIO),  # MP3 frame sync
    (b"RIFF", 0, Medium.AUDIO),  # WAV (checked further below)
    (b"fLaC", 0, Medium.AUDIO),  # FLAC
    (b"OggS", 0, Medium.AUDIO),  # OGG
]

# Extension to medium mapping (fallback when magic bytes inconclusive)
_EXTENSION_MAP: Final[dict[str, Medium]] = {
    # Images
    ".jpg": Medium.IMAGE,
    ".jpeg": Medium.IMAGE,
    ".png": Medium.IMAGE,
    ".gif": Medium.IMAGE,
    ".webp": Medium.IMAGE,
    ".bmp": Medium.IMAGE,
    ".tiff": Medium.IMAGE,
    ".tif": Medium.IMAGE,
    ".heic": Medium.IMAGE,
    ".heif": Medium.IMAGE,
    ".svg": Medium.IMAGE,
    ".ico": Medium.IMAGE,
    # Video
    ".mp4": Medium.VIDEO,
    ".mov": Medium.VIDEO,
    ".avi": Medium.VIDEO,
    ".webm": Medium.VIDEO,
    ".mkv": Medium.VIDEO,
    ".m4v": Medium.VIDEO,
    ".wmv": Medium.VIDEO,
    ".flv": Medium.VIDEO,
    # Audio
    ".mp3": Medium.AUDIO,
    ".wav": Medium.AUDIO,
    ".flac": Medium.AUDIO,
    ".ogg": Medium.AUDIO,
    ".m4a": Medium.AUDIO,
    ".aac": Medium.AUDIO,
    ".wma": Medium.AUDIO,
    # Text
    ".txt": Medium.TEXT,
    ".md": Medium.TEXT,
    ".rst": Medium.TEXT,
    ".csv": Medium.TEXT,
    ".tsv": Medium.TEXT,
    ".log": Medium.TEXT,
    ".xml": Medium.TEXT,
    ".html": Medium.TEXT,
    ".htm": Medium.TEXT,
    # JSON
    ".json": Medium.JSON,
    ".jsonl": Medium.JSON,
    ".geojson": Medium.JSON,
}

# Affordances for each medium
MEDIUM_AFFORDANCES: Final[dict[Medium, frozenset[Affordance]]] = {
    Medium.IMAGE: frozenset(
        {
            Affordance.CAN_EMBED_IMAGE,
            Affordance.CAN_DETECT_OBJECTS,
            Affordance.CAN_EXTRACT_EXIF,
            Affordance.CAN_CAPTION,
        }
    ),
    Medium.VIDEO: frozenset(
        {
            Affordance.CAN_SAMPLE_FRAMES,
            Affordance.CAN_SEGMENT,
            Affordance.CAN_EXTRACT_AUDIO,
        }
    ),
    Medium.TEXT: frozenset(
        {
            Affordance.CAN_CHUNK,
            Affordance.CAN_EMBED_TEXT,
            Affordance.CAN_PARSE_TEXT,
        }
    ),
    Medium.JSON: frozenset(
        {
            Affordance.CAN_PARSE_KEYS,
            Affordance.CAN_INFER_SCHEMA,
            Affordance.CAN_EMBED_TEXT,  # JSON can also be embedded as text
        }
    ),
    Medium.AUDIO: frozenset(
        {
            Affordance.CAN_TRANSCRIBE,
            Affordance.CAN_EMBED_AUDIO,
        }
    ),
    Medium.BINARY: frozenset(
        {
            Affordance.CAN_ENTROPY_ANALYZE,
            Affordance.CAN_MAGIC_SNIFF,
        }
    ),
}

# Minimum bytes needed for reliable magic byte detection
_MIN_HEADER_SIZE: Final[int] = 32


def get_affordances(medium: Medium) -> frozenset[Affordance]:
    """Get the affordances (capabilities) for a medium.

    Args:
        medium: The medium to get affordances for.

    Returns:
        Set of affordances enabled by this medium.
    """
    return MEDIUM_AFFORDANCES.get(medium, frozenset())


def probe_medium(
    data: bytes,
    *,
    filename: str | None = None,
) -> Medium:
    """Detect medium from content bytes and optional filename hint.

    Detection strategy (in priority order):
    1. Magic bytes - file header signatures
    2. Content analysis - JSON parsing, text encoding detection
    3. Extension - filename hint as fallback

    Args:
        data: Raw file content bytes.
        filename: Optional filename for extension-based fallback.

    Returns:
        Detected medium type.
    """
    if not data:
        return Medium.BINARY

    # Try magic bytes first
    medium = _detect_by_magic(data)
    if medium is not None:
        return medium

    # Try content analysis for text/JSON
    medium = _detect_text_or_json(data)
    if medium is not None:
        return medium

    # Fall back to extension
    if filename:
        medium = _detect_by_extension(filename)
        if medium is not None:
            return medium

    # Unknown format
    return Medium.BINARY


def probe_medium_from_path(path: str | Path) -> Medium:
    """Detect medium from a file path.

    Reads the file header and uses the filename as a hint.

    Args:
        path: Path to the file.

    Returns:
        Detected medium type.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        PermissionError: If the file can't be read.
    """
    path = Path(path)

    # Read just the header for efficiency
    with path.open("rb") as f:
        header = f.read(_MIN_HEADER_SIZE)

    return probe_medium(header, filename=path.name)


def _detect_by_magic(data: bytes) -> Medium | None:
    """Detect medium from magic bytes."""
    # Handle RIFF container (WebP, WAV, AVI)
    if data.startswith(b"RIFF") and len(data) >= 12:
        riff_type = data[8:12]
        if riff_type == b"WEBP":
            return Medium.IMAGE
        if riff_type == b"WAVE":
            return Medium.AUDIO
        if riff_type == b"AVI ":
            return Medium.VIDEO

    # Handle ftyp box (MP4, MOV, HEIC, etc.)
    ftyp_medium = _check_ftyp(data)
    if ftyp_medium is not None:
        return ftyp_medium

    # Check other signatures
    for magic, offset, medium in _MAGIC_SIGNATURES:
        if magic == b"RIFF" or magic == b"\x00\x00\x00":
            continue  # Already handled above
        if len(data) >= offset + len(magic) and data[offset : offset + len(magic)] == magic:
            return medium

    return None


def _check_ftyp(data: bytes) -> Medium | None:
    """Check for ISO Base Media File Format (ftyp box).

    Handles MP4, MOV, HEIC, HEIF, M4A, etc.
    """
    if len(data) < 12:
        return None

    # ftyp box: size (4 bytes) + 'ftyp' (4 bytes) + brand (4 bytes)
    # Size can be at offset 0, or box starts at offset 0 with 'ftyp' at 4
    ftyp_offset = -1

    # Check if ftyp is at position 4 (common case)
    if data[4:8] == b"ftyp":
        ftyp_offset = 4
    # Check if ftyp is at position 0 (less common)
    elif data[0:4] == b"ftyp":
        ftyp_offset = 0

    if ftyp_offset == -1:
        return None

    # Get the brand (4 bytes after 'ftyp')
    brand_offset = ftyp_offset + 4
    if len(data) < brand_offset + 4:
        return None

    brand = data[brand_offset : brand_offset + 4]

    # HEIC/HEIF brands
    heic_brands = {b"heic", b"heix", b"hevc", b"hevx", b"mif1", b"msf1"}
    if brand in heic_brands:
        return Medium.IMAGE

    # Video brands (MP4, MOV, etc.)
    video_brands = {
        b"isom",
        b"iso2",
        b"iso3",
        b"iso4",
        b"iso5",
        b"iso6",
        b"mp41",
        b"mp42",
        b"mp71",
        b"avc1",
        b"qt  ",
        b"M4V ",
        b"M4VP",
    }
    if brand in video_brands:
        return Medium.VIDEO

    # Audio brands (M4A)
    audio_brands = {b"M4A ", b"M4B "}
    if brand in audio_brands:
        return Medium.AUDIO

    # Default to video for unknown ftyp (most common case)
    return Medium.VIDEO


def _detect_text_or_json(data: bytes) -> Medium | None:
    """Detect if content is text or JSON."""
    # Try to decode as UTF-8
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        # Try latin-1 as fallback (accepts any byte sequence)
        try:
            text = data.decode("latin-1")
            # Check if it looks like text (mostly printable)
            if not _is_text_like(text):
                return None
        except UnicodeDecodeError:
            return None

    # Check if it's valid JSON
    text_stripped = text.strip()
    if text_stripped.startswith(("{", "[")):
        try:
            json.loads(text_stripped)
            return Medium.JSON
        except json.JSONDecodeError:
            pass

    # Check if it looks like text
    if _is_text_like(text):
        return Medium.TEXT

    return None


def _is_text_like(text: str) -> bool:
    """Check if string looks like human-readable text."""
    if not text:
        return False

    # Count printable vs non-printable characters
    printable = 0
    non_printable = 0

    for char in text[:1000]:  # Check first 1000 chars
        if char.isprintable() or char in "\n\r\t":
            printable += 1
        else:
            non_printable += 1

    # Text should be mostly printable (>90%)
    total = printable + non_printable
    if total == 0:
        return False

    return (printable / total) > 0.9


def _detect_by_extension(filename: str) -> Medium | None:
    """Detect medium from file extension."""
    ext = Path(filename).suffix.lower()
    return _EXTENSION_MAP.get(ext)
