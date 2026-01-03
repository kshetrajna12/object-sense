"""Entity resolution module for ObjectSense.

This module implements Step 5 of the pipeline: Entity Resolution.

See design_v2_corrections.md §5 for the algorithm specification.

Submodules:
- candidate_pool: pgvector ANN retrieval for entity candidates
- similarity: Multi-signal similarity scoring with entity_nature weighting
- reconciliation: Multi-seed consistency pass
- resolver: Main resolution algorithm
"""

from object_sense.resolution.candidate_pool import CandidatePoolService
from object_sense.resolution.reconciliation import reconcile_multi_seed_links
from object_sense.resolution.resolver import EntityResolver
from object_sense.resolution.similarity import SimilarityScorer

__all__ = [
    "CandidatePoolService",
    "EntityResolver",
    "SimilarityScorer",
    "reconcile_multi_seed_links",
]
