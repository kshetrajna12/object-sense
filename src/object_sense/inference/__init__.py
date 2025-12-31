"""LLM inference module for type proposals and entity resolution."""

from object_sense.inference.schemas import (
    DeterministicId,
    EntityHypothesis,
    SemanticProbeResult,
    SlotValue,
    TypeCandidateProposal,
    TypeProposal,
)
from object_sense.inference.type_inference import TypeInferenceAgent

__all__ = [
    "DeterministicId",
    "EntityHypothesis",
    "SemanticProbeResult",
    "SlotValue",
    "TypeCandidateProposal",
    "TypeInferenceAgent",
    "TypeProposal",
]
