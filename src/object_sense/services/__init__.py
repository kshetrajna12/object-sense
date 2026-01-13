"""Business logic services for ObjectSense."""

from object_sense.services.type_candidate import TypeCandidateService, normalize_type_name
from object_sense.services.type_evolution import (
    AliasResult,
    DeprecateResult,
    MergeResult,
    SplitResult,
    TypeEvolutionService,
)
from object_sense.services.type_promotion import PromotionResult, TypePromotionService

__all__ = [
    "AliasResult",
    "DeprecateResult",
    "MergeResult",
    "normalize_type_name",
    "PromotionResult",
    "SplitResult",
    "TypeCandidateService",
    "TypeEvolutionService",
    "TypePromotionService",
]
