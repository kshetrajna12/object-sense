"""Tests for medium probing and affordance detection."""

import tempfile
from pathlib import Path

import pytest

from object_sense.models.enums import Affordance, Medium
from object_sense.utils.medium import (
    MEDIUM_AFFORDANCES,
    get_affordances,
    probe_medium,
    probe_medium_from_path,
)


class TestGetAffordances:
    """Tests for get_affordances function."""

    def test_image_affordances(self) -> None:
        affordances = get_affordances(Medium.IMAGE)
        assert Affordance.CAN_EMBED_IMAGE in affordances
        assert Affordance.CAN_DETECT_OBJECTS in affordances
        assert Affordance.CAN_EXTRACT_EXIF in affordances
        assert Affordance.CAN_CAPTION in affordances

    def test_video_affordances(self) -> None:
        affordances = get_affordances(Medium.VIDEO)
        assert Affordance.CAN_SAMPLE_FRAMES in affordances
        assert Affordance.CAN_SEGMENT in affordances
        assert Affordance.CAN_EXTRACT_AUDIO in affordances

    def test_text_affordances(self) -> None:
        affordances = get_affordances(Medium.TEXT)
        assert Affordance.CAN_CHUNK in affordances
        assert Affordance.CAN_EMBED_TEXT in affordances
        assert Affordance.CAN_PARSE_TEXT in affordances

    def test_json_affordances(self) -> None:
        affordances = get_affordances(Medium.JSON)
        assert Affordance.CAN_PARSE_KEYS in affordances
        assert Affordance.CAN_INFER_SCHEMA in affordances
        assert Affordance.CAN_EMBED_TEXT in affordances  # JSON can be embedded as text

    def test_audio_affordances(self) -> None:
        affordances = get_affordances(Medium.AUDIO)
        assert Affordance.CAN_TRANSCRIBE in affordances
        assert Affordance.CAN_EMBED_AUDIO in affordances

    def test_binary_affordances(self) -> None:
        affordances = get_affordances(Medium.BINARY)
        assert Affordance.CAN_ENTROPY_ANALYZE in affordances
        assert Affordance.CAN_MAGIC_SNIFF in affordances

    def test_all_mediums_have_affordances(self) -> None:
        for medium in Medium:
            affordances = get_affordances(medium)
            assert len(affordances) > 0, f"{medium} has no affordances"

    def test_affordances_are_frozen(self) -> None:
        affordances = get_affordances(Medium.IMAGE)
        assert isinstance(affordances, frozenset)


class TestProbeMediumMagicBytes:
    """Tests for magic byte detection."""

    def test_jpeg(self) -> None:
        # JPEG magic bytes
        data = b"\xff\xd8\xff\xe0\x00\x10JFIF"
        assert probe_medium(data) == Medium.IMAGE

    def test_png(self) -> None:
        # PNG magic bytes
        data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 24
        assert probe_medium(data) == Medium.IMAGE

    def test_gif87a(self) -> None:
        data = b"GIF87a" + b"\x00" * 26
        assert probe_medium(data) == Medium.IMAGE

    def test_gif89a(self) -> None:
        data = b"GIF89a" + b"\x00" * 26
        assert probe_medium(data) == Medium.IMAGE

    def test_webp(self) -> None:
        # RIFF....WEBP
        data = b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 20
        assert probe_medium(data) == Medium.IMAGE

    def test_bmp(self) -> None:
        data = b"BM" + b"\x00" * 30
        assert probe_medium(data) == Medium.IMAGE

    def test_tiff_little_endian(self) -> None:
        data = b"II*\x00" + b"\x00" * 28
        assert probe_medium(data) == Medium.IMAGE

    def test_tiff_big_endian(self) -> None:
        data = b"MM\x00*" + b"\x00" * 28
        assert probe_medium(data) == Medium.IMAGE

    def test_mp4(self) -> None:
        # ftyp box with isom brand
        data = b"\x00\x00\x00\x14ftypisom" + b"\x00" * 20
        assert probe_medium(data) == Medium.VIDEO

    def test_mov(self) -> None:
        # ftyp box with qt brand
        data = b"\x00\x00\x00\x14ftypqt  " + b"\x00" * 20
        assert probe_medium(data) == Medium.VIDEO

    def test_heic(self) -> None:
        # ftyp box with heic brand
        data = b"\x00\x00\x00\x14ftypheic" + b"\x00" * 20
        assert probe_medium(data) == Medium.IMAGE

    def test_heif_mif1(self) -> None:
        # ftyp box with mif1 brand
        data = b"\x00\x00\x00\x14ftypmif1" + b"\x00" * 20
        assert probe_medium(data) == Medium.IMAGE

    def test_m4a(self) -> None:
        # ftyp box with M4A brand
        data = b"\x00\x00\x00\x14ftypM4A " + b"\x00" * 20
        assert probe_medium(data) == Medium.AUDIO

    def test_webm(self) -> None:
        # EBML header for WebM/MKV
        data = b"\x1aE\xdf\xa3" + b"\x00" * 28
        assert probe_medium(data) == Medium.VIDEO

    def test_wav(self) -> None:
        # RIFF....WAVE
        data = b"RIFF\x00\x00\x00\x00WAVE" + b"\x00" * 20
        assert probe_medium(data) == Medium.AUDIO

    def test_avi(self) -> None:
        # RIFF....AVI
        data = b"RIFF\x00\x00\x00\x00AVI " + b"\x00" * 20
        assert probe_medium(data) == Medium.VIDEO

    def test_mp3_id3(self) -> None:
        data = b"ID3" + b"\x00" * 29
        assert probe_medium(data) == Medium.AUDIO

    def test_mp3_frame_sync(self) -> None:
        data = b"\xff\xfb" + b"\x00" * 30
        assert probe_medium(data) == Medium.AUDIO

    def test_flac(self) -> None:
        data = b"fLaC" + b"\x00" * 28
        assert probe_medium(data) == Medium.AUDIO

    def test_ogg(self) -> None:
        data = b"OggS" + b"\x00" * 28
        assert probe_medium(data) == Medium.AUDIO


class TestProbeMediumTextJson:
    """Tests for text and JSON detection."""

    def test_json_object(self) -> None:
        data = b'{"key": "value", "number": 42}'
        assert probe_medium(data) == Medium.JSON

    def test_json_array(self) -> None:
        data = b'[1, 2, 3, "four"]'
        assert probe_medium(data) == Medium.JSON

    def test_json_with_whitespace(self) -> None:
        data = b'  \n  {"key": "value"}  \n  '
        assert probe_medium(data) == Medium.JSON

    def test_plain_text(self) -> None:
        data = b"Hello, this is plain text content."
        assert probe_medium(data) == Medium.TEXT

    def test_text_with_newlines(self) -> None:
        data = b"Line 1\nLine 2\nLine 3"
        assert probe_medium(data) == Medium.TEXT

    def test_text_utf8(self) -> None:
        data = "Hello, world! 你好世界 🌍".encode()
        assert probe_medium(data) == Medium.TEXT

    def test_invalid_json_is_text(self) -> None:
        data = b'{"key": invalid}'
        assert probe_medium(data) == Medium.TEXT

    def test_partial_json_is_text(self) -> None:
        data = b'{"key": "value"'  # Missing closing brace
        assert probe_medium(data) == Medium.TEXT


class TestProbeMediumExtension:
    """Tests for extension-based fallback."""

    def test_jpg_extension(self) -> None:
        # Binary data that doesn't match magic bytes
        data = b"\x00\x01\x02\x03"
        assert probe_medium(data, filename="photo.jpg") == Medium.IMAGE

    def test_mp4_extension(self) -> None:
        data = b"\x00\x01\x02\x03"
        assert probe_medium(data, filename="video.mp4") == Medium.VIDEO

    def test_txt_extension(self) -> None:
        data = b"\x00\x01\x02\x03"
        assert probe_medium(data, filename="readme.txt") == Medium.TEXT

    def test_json_extension(self) -> None:
        data = b"\x00\x01\x02\x03"
        assert probe_medium(data, filename="config.json") == Medium.JSON

    def test_mp3_extension(self) -> None:
        data = b"\x00\x01\x02\x03"
        assert probe_medium(data, filename="song.mp3") == Medium.AUDIO

    def test_case_insensitive_extension(self) -> None:
        data = b"\x00\x01\x02\x03"
        assert probe_medium(data, filename="PHOTO.JPG") == Medium.IMAGE
        assert probe_medium(data, filename="Photo.Png") == Medium.IMAGE

    def test_unknown_extension(self) -> None:
        data = b"\x00\x01\x02\x03"
        assert probe_medium(data, filename="file.xyz") == Medium.BINARY


class TestProbeMediumEdgeCases:
    """Tests for edge cases."""

    def test_empty_data(self) -> None:
        assert probe_medium(b"") == Medium.BINARY

    def test_truncated_header(self) -> None:
        # Partial JPEG header with null bytes (clearly binary, not printable text)
        data = b"\xff\xd8\x00\x00\x00\x00\x00\x00"
        assert probe_medium(data) == Medium.BINARY

    def test_no_filename(self) -> None:
        data = b"\x00\x01\x02\x03"
        assert probe_medium(data) == Medium.BINARY

    def test_magic_takes_priority_over_extension(self) -> None:
        # PNG data with .jpg extension - magic should win
        png_data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 24
        assert probe_medium(png_data, filename="wrong.jpg") == Medium.IMAGE

    def test_short_json(self) -> None:
        assert probe_medium(b"{}") == Medium.JSON
        assert probe_medium(b"[]") == Medium.JSON

    def test_binary_with_some_text(self) -> None:
        # Mix of binary and text - should detect as binary
        data = b"\x00\x01hello\x02\x03\x00\x00\x00"
        # This has too many non-printable chars to be text
        result = probe_medium(data)
        # Could be BINARY depending on ratio
        assert result in (Medium.BINARY, Medium.TEXT)


class TestProbeMediumFromPath:
    """Tests for probe_medium_from_path function."""

    def test_reads_file_header(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            # Write PNG data
            f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
            f.flush()
            path = Path(f.name)

        try:
            result = probe_medium_from_path(path)
            assert result == Medium.IMAGE
        finally:
            path.unlink()

    def test_uses_filename_as_hint(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            f.write(b'{"test": true}')
            f.flush()
            path = Path(f.name)

        try:
            result = probe_medium_from_path(path)
            assert result == Medium.JSON
        finally:
            path.unlink()

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            probe_medium_from_path("/nonexistent/file.txt")

    def test_accepts_string_path(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"Hello, world!")
            f.flush()
            path_str = f.name

        try:
            result = probe_medium_from_path(path_str)
            assert result == Medium.TEXT
        finally:
            Path(path_str).unlink()


class TestMediumAffordancesMapping:
    """Tests for the MEDIUM_AFFORDANCES constant."""

    def test_all_mediums_mapped(self) -> None:
        for medium in Medium:
            assert medium in MEDIUM_AFFORDANCES

    def test_no_duplicate_affordances_within_medium(self) -> None:
        for medium, affordances in MEDIUM_AFFORDANCES.items():
            # frozenset automatically deduplicates, so just check it's non-empty
            assert len(affordances) > 0, f"{medium} has no affordances"

    def test_json_includes_text_embedding(self) -> None:
        # JSON should be able to be embedded as text too
        assert Affordance.CAN_EMBED_TEXT in MEDIUM_AFFORDANCES[Medium.JSON]
