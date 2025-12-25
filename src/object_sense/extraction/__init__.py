"""Feature extraction module.

Provides medium-specific feature extraction for ObjectSense.

Main entry point:
    from object_sense.extraction import ExtractionOrchestrator

    orchestrator = ExtractionOrchestrator()
    result = await orchestrator.extract(image_bytes)

Individual extractors:
    from object_sense.extraction import ImageExtractor, TextExtractor, JsonExtractor
"""

from object_sense.extraction.base import ExtractionResult
from object_sense.extraction.image import ImageExtractor
from object_sense.extraction.json_extract import JsonExtractor
from object_sense.extraction.orchestrator import ExtractionOrchestrator
from object_sense.extraction.text import TextExtractor

__all__ = [
    "ExtractionOrchestrator",
    "ExtractionResult",
    "ImageExtractor",
    "TextExtractor",
    "JsonExtractor",
]
