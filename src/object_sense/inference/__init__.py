"""LLM inference module for type proposals and entity resolution."""

from object_sense.inference.schemas import (
    EntityHypothesis,
    SlotValue,
    TypeProposal,
)
from object_sense.inference.type_inference import TypeInferenceAgent

__all__ = [
    "EntityHypothesis",
    "SlotValue",
    "TypeInferenceAgent",
    "TypeProposal",
]
