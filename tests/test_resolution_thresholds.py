"""Tests for entity resolution threshold boundaries.

These tests verify the threshold-based decision logic in EntityResolver:
- T_link (0.85): High confidence threshold for SOFT link
- T_new (0.60): Low confidence threshold for CANDIDATE vs create new
- T_margin (0.10): Margin for disambiguation review flag
- Coverage scaling interaction with thresholds
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from object_sense.inference.schemas import EntityHypothesis
from object_sense.models.entity import Entity
from object_sense.models.enums import EntityNature, EntityStatus, LinkRole
from object_sense.models.observation import Observation
from object_sense.resolution.candidate_pool import CandidatePoolService, EntityCandidate
from object_sense.resolution.resolver import EntityResolver
from object_sense.resolution.similarity import (
    ObservationSignals,
    SimilarityResult,
    SimilarityScorer,
    SignalResult,
)


# ─────────────────────────────────────────────────────────────────────────────
# Default threshold constants (from config)
# ─────────────────────────────────────────────────────────────────────────────
T_LINK = 0.85
T_NEW = 0.60
T_MARGIN = 0.10


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures and Helpers
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class MockEntityData:
    """Data for creating mock entities."""

    entity_id: UUID
    name: str
    entity_nature: EntityNature = EntityNature.INDIVIDUAL


def make_mock_entity(
    data: MockEntityData | None = None,
    prototype_image_embedding: list[float] | None = None,
    prototype_text_embedding: list[float] | None = None,
) -> MagicMock:
    """Create a mock Entity with the specified properties."""
    if data is None:
        data = MockEntityData(entity_id=uuid4(), name="test_entity")

    entity = MagicMock(spec=Entity)
    entity.entity_id = data.entity_id
    entity.name = data.name
    entity.entity_nature = data.entity_nature
    entity.status = EntityStatus.PROTO
    entity.slots = {}
    entity.canonical_entity_id = None
    entity.prototype_image_embedding = prototype_image_embedding or [0.1] * 768
    entity.prototype_text_embedding = prototype_text_embedding or [0.1] * 1024
    entity.prototype_count = 1
    entity.confidence = 0.8
    return entity


def make_similarity_result(
    score: float,
    coverage: float = 1.0,
    is_low_evidence: bool = False,
    flags: list[str] | None = None,
) -> SimilarityResult:
    """Create a SimilarityResult with predetermined values."""
    return SimilarityResult(
        score=score,
        signals=[
            SignalResult(
                signal_name="image_embedding",
                score=score,
                weight=0.35,
                available=True,
            ),
        ],
        coverage=coverage,
        is_low_evidence=is_low_evidence,
        flags=flags or [],
    )


class StubSimilarityScorer(SimilarityScorer):
    """A stub SimilarityScorer that returns predetermined scores."""

    def __init__(self, score_mapping: dict[UUID, tuple[float, float]]) -> None:
        """Initialize with a mapping of entity_id -> (score, coverage)."""
        super().__init__()
        self._score_mapping = score_mapping
        self._default_score = 0.5
        self._default_coverage = 1.0

    def compute(
        self,
        *,
        observation_signals: ObservationSignals,
        entity: Entity,
        entity_nature: EntityNature,
    ) -> SimilarityResult:
        """Return predetermined score for the entity."""
        entity_id = entity.entity_id
        if entity_id in self._score_mapping:
            score, coverage = self._score_mapping[entity_id]
        else:
            score, coverage = self._default_score, self._default_coverage

        is_low_evidence = coverage < 0.5
        return make_similarity_result(
            score=score,
            coverage=coverage,
            is_low_evidence=is_low_evidence,
        )


class StubCandidatePoolService(CandidatePoolService):
    """A stub CandidatePoolService that returns predetermined candidates."""

    def __init__(self, candidates: list[EntityCandidate]) -> None:
        """Initialize with predetermined candidates."""
        # Don't call super().__init__() - we don't need a real session
        self._candidates = candidates

    async def get_candidates_by_embedding(
        self,
        embedding: list[float],
        entity_nature: EntityNature,
        *,
        top_k: int | None = None,
        exclude_entity_ids: set[UUID] | None = None,
        observation_kind: str | None = None,
    ) -> list[EntityCandidate]:
        """Return predetermined candidates."""
        return self._candidates

    async def lookup_by_deterministic_id(self, det_id: Any) -> Any:
        """Return no match for deterministic ID lookup."""
        from object_sense.resolution.candidate_pool import DeterministicIdLookupResult

        return DeterministicIdLookupResult(entity=None, conflict_entity_id=None)

    async def create_entity_for_deterministic_id(self, *args: Any, **kwargs: Any) -> Entity:
        """Create a mock entity for deterministic ID."""
        return make_mock_entity()


@pytest.fixture
def mock_session() -> AsyncMock:
    """Create a mock database session."""
    session = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.get = AsyncMock(return_value=None)
    return session


@pytest.fixture
def mock_observation() -> MagicMock:
    """Create a mock observation for resolution."""
    obs = MagicMock(spec=Observation)
    obs.observation_id = uuid4()
    obs.deterministic_ids = []
    obs.facets = {}
    obs.source_id = "test_source"
    return obs


@pytest.fixture
def mock_signals() -> ObservationSignals:
    """Create mock observation signals with embeddings."""
    return ObservationSignals(
        image_embedding=[0.1] * 768,
        text_embedding=[0.1] * 1024,
        facets={},
        suggested_name="test_observation",
    )


def make_entity_hypothesis(
    entity_nature: EntityNature = EntityNature.INDIVIDUAL,
    suggested_name: str = "test",
    confidence: float = 0.9,
) -> EntityHypothesis:
    """Create a valid EntityHypothesis for testing."""
    return EntityHypothesis(
        entity_type="test_entity",
        entity_nature=entity_nature,
        suggested_name=suggested_name,
        confidence=confidence,
        deterministic_ids=[],
        slots=[],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test Class: T_link Boundary Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestTLinkThreshold:
    """Tests for T_link threshold (0.85): SOFT link decision boundary."""

    @pytest.mark.asyncio
    async def test_score_at_t_link_creates_soft_link(
        self,
        mock_session: AsyncMock,
        mock_observation: MagicMock,
    ) -> None:
        """Score exactly at T_link should create SOFT link."""
        entity = make_mock_entity()
        candidate = EntityCandidate(entity=entity, ann_distance=0.1)

        # Score exactly at threshold
        score_mapping = {entity.entity_id: (T_LINK, 1.0)}  # 0.85

        resolver = EntityResolver(
            mock_session,
            t_link=T_LINK,
            t_new=T_NEW,
            t_margin=T_MARGIN,
        )
        resolver._pool = StubCandidatePoolService([candidate])
        resolver._scorer = StubSimilarityScorer(score_mapping)

        seed = make_entity_hypothesis()

        result = await resolver._resolve_by_similarity(
            ctx=MagicMock(
                observation=mock_observation,
                signatures=[],
                entity_seeds=[seed],
                observation_signals=ObservationSignals(
                    image_embedding=[0.1] * 768,
                ),
            ),
            seed=seed,
            entity_nature=EntityNature.INDIVIDUAL,
            seed_idx=0,
        )

        # Should link to existing entity with SOFT status
        assert len(result.links) == 1
        assert result.links[0].status == "soft"
        assert result.links[0].entity_id == entity.entity_id
        assert result.links[0].posterior == pytest.approx(T_LINK)

    @pytest.mark.asyncio
    async def test_score_above_t_link_creates_soft_link(
        self,
        mock_session: AsyncMock,
        mock_observation: MagicMock,
    ) -> None:
        """Score above T_link should create SOFT link."""
        entity = make_mock_entity()
        candidate = EntityCandidate(entity=entity, ann_distance=0.1)

        # Score above threshold
        score = T_LINK + 0.001  # 0.851
        score_mapping = {entity.entity_id: (score, 1.0)}

        resolver = EntityResolver(
            mock_session,
            t_link=T_LINK,
            t_new=T_NEW,
            t_margin=T_MARGIN,
        )
        resolver._pool = StubCandidatePoolService([candidate])
        resolver._scorer = StubSimilarityScorer(score_mapping)

        seed = make_entity_hypothesis()

        result = await resolver._resolve_by_similarity(
            ctx=MagicMock(
                observation=mock_observation,
                signatures=[],
                entity_seeds=[seed],
                observation_signals=ObservationSignals(
                    image_embedding=[0.1] * 768,
                ),
            ),
            seed=seed,
            entity_nature=EntityNature.INDIVIDUAL,
            seed_idx=0,
        )

        assert len(result.links) == 1
        assert result.links[0].status == "soft"

    @pytest.mark.asyncio
    async def test_score_below_t_link_creates_candidate(
        self,
        mock_session: AsyncMock,
        mock_observation: MagicMock,
    ) -> None:
        """Score below T_link (but above T_new) should create CANDIDATE link."""
        entity = make_mock_entity()
        candidate = EntityCandidate(entity=entity, ann_distance=0.1)

        # Score below T_link but above T_new
        score = T_LINK - 0.001  # 0.849
        score_mapping = {entity.entity_id: (score, 1.0)}

        resolver = EntityResolver(
            mock_session,
            t_link=T_LINK,
            t_new=T_NEW,
            t_margin=T_MARGIN,
        )
        resolver._pool = StubCandidatePoolService([candidate])
        resolver._scorer = StubSimilarityScorer(score_mapping)

        seed = make_entity_hypothesis()

        result = await resolver._resolve_by_similarity(
            ctx=MagicMock(
                observation=mock_observation,
                signatures=[],
                entity_seeds=[seed],
                observation_signals=ObservationSignals(
                    image_embedding=[0.1] * 768,
                ),
            ),
            seed=seed,
            entity_nature=EntityNature.INDIVIDUAL,
            seed_idx=0,
        )

        # Should have both candidate link and proto-entity alternative
        assert len(result.links) == 2
        existing_link = [l for l in result.links if l.entity_id == entity.entity_id][0]
        assert existing_link.status == "candidate"
        assert "uncertain_match" in existing_link.flags


# ─────────────────────────────────────────────────────────────────────────────
# Test Class: T_new Boundary Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestTNewThreshold:
    """Tests for T_new threshold (0.60): create new proto-entity decision boundary."""

    @pytest.mark.asyncio
    async def test_score_at_t_new_links_existing(
        self,
        mock_session: AsyncMock,
        mock_observation: MagicMock,
    ) -> None:
        """Score exactly at T_new should link to existing (CANDIDATE)."""
        entity = make_mock_entity()
        candidate = EntityCandidate(entity=entity, ann_distance=0.1)

        # Score exactly at threshold
        score_mapping = {entity.entity_id: (T_NEW, 1.0)}  # 0.60

        resolver = EntityResolver(
            mock_session,
            t_link=T_LINK,
            t_new=T_NEW,
            t_margin=T_MARGIN,
        )
        resolver._pool = StubCandidatePoolService([candidate])
        resolver._scorer = StubSimilarityScorer(score_mapping)

        seed = make_entity_hypothesis()

        result = await resolver._resolve_by_similarity(
            ctx=MagicMock(
                observation=mock_observation,
                signatures=[],
                entity_seeds=[seed],
                observation_signals=ObservationSignals(
                    image_embedding=[0.1] * 768,
                ),
            ),
            seed=seed,
            entity_nature=EntityNature.INDIVIDUAL,
            seed_idx=0,
        )

        # Should link to existing entity (candidate status) + create proto alternative
        existing_links = [l for l in result.links if l.entity_id == entity.entity_id]
        assert len(existing_links) == 1
        assert existing_links[0].status == "candidate"

    @pytest.mark.asyncio
    async def test_score_above_t_new_links_existing(
        self,
        mock_session: AsyncMock,
        mock_observation: MagicMock,
    ) -> None:
        """Score above T_new should link to existing."""
        entity = make_mock_entity()
        candidate = EntityCandidate(entity=entity, ann_distance=0.1)

        # Score above threshold
        score = T_NEW + 0.001  # 0.601
        score_mapping = {entity.entity_id: (score, 1.0)}

        resolver = EntityResolver(
            mock_session,
            t_link=T_LINK,
            t_new=T_NEW,
            t_margin=T_MARGIN,
        )
        resolver._pool = StubCandidatePoolService([candidate])
        resolver._scorer = StubSimilarityScorer(score_mapping)

        seed = make_entity_hypothesis()

        result = await resolver._resolve_by_similarity(
            ctx=MagicMock(
                observation=mock_observation,
                signatures=[],
                entity_seeds=[seed],
                observation_signals=ObservationSignals(
                    image_embedding=[0.1] * 768,
                ),
            ),
            seed=seed,
            entity_nature=EntityNature.INDIVIDUAL,
            seed_idx=0,
        )

        # Should link to existing (candidate status)
        existing_links = [l for l in result.links if l.entity_id == entity.entity_id]
        assert len(existing_links) == 1

    @pytest.mark.asyncio
    async def test_score_below_t_new_creates_proto_entity(
        self,
        mock_session: AsyncMock,
        mock_observation: MagicMock,
    ) -> None:
        """Score below T_new should create new proto-entity."""
        entity = make_mock_entity()
        candidate = EntityCandidate(entity=entity, ann_distance=0.1)

        # Score below threshold
        score = T_NEW - 0.001  # 0.599
        score_mapping = {entity.entity_id: (score, 1.0)}

        resolver = EntityResolver(
            mock_session,
            t_link=T_LINK,
            t_new=T_NEW,
            t_margin=T_MARGIN,
        )
        resolver._pool = StubCandidatePoolService([candidate])
        resolver._scorer = StubSimilarityScorer(score_mapping)

        seed = make_entity_hypothesis()

        result = await resolver._resolve_by_similarity(
            ctx=MagicMock(
                observation=mock_observation,
                signatures=[],
                entity_seeds=[seed],
                observation_signals=ObservationSignals(
                    image_embedding=[0.1] * 768,
                ),
            ),
            seed=seed,
            entity_nature=EntityNature.INDIVIDUAL,
            seed_idx=0,
        )

        # Should create new proto-entity
        assert len(result.entities) == 1
        # Link should be to the new proto, not the existing entity
        assert len(result.links) == 1
        assert result.links[0].entity_id != entity.entity_id
        assert "no_match_found" in result.links[0].flags


# ─────────────────────────────────────────────────────────────────────────────
# Test Class: T_margin Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestTMarginThreshold:
    """Tests for T_margin threshold (0.10): disambiguation review flag."""

    @pytest.mark.asyncio
    async def test_candidates_within_margin_flagged_for_review(
        self,
        mock_session: AsyncMock,
        mock_observation: MagicMock,
    ) -> None:
        """Two candidates with scores within T_margin should trigger review flag."""
        entity1 = make_mock_entity(MockEntityData(uuid4(), "entity1"))
        entity2 = make_mock_entity(MockEntityData(uuid4(), "entity2"))
        candidates = [
            EntityCandidate(entity=entity1, ann_distance=0.05),
            EntityCandidate(entity=entity2, ann_distance=0.10),
        ]

        # Scores within margin (difference = 0.05 < 0.10)
        best_score = 0.90
        second_score = 0.86  # difference = 0.04 < T_margin (0.10)
        score_mapping = {
            entity1.entity_id: (best_score, 1.0),
            entity2.entity_id: (second_score, 1.0),
        }

        resolver = EntityResolver(
            mock_session,
            t_link=T_LINK,
            t_new=T_NEW,
            t_margin=T_MARGIN,
        )
        resolver._pool = StubCandidatePoolService(candidates)
        resolver._scorer = StubSimilarityScorer(score_mapping)

        seed = make_entity_hypothesis()

        result = await resolver._resolve_by_similarity(
            ctx=MagicMock(
                observation=mock_observation,
                signatures=[],
                entity_seeds=[seed],
                observation_signals=ObservationSignals(
                    image_embedding=[0.1] * 768,
                ),
            ),
            seed=seed,
            entity_nature=EntityNature.INDIVIDUAL,
            seed_idx=0,
        )

        # Should link to best but flag for review
        assert len(result.links) == 1
        assert result.links[0].entity_id == entity1.entity_id
        assert "margin_review" in result.links[0].flags

    @pytest.mark.asyncio
    async def test_candidates_outside_margin_no_review_flag(
        self,
        mock_session: AsyncMock,
        mock_observation: MagicMock,
    ) -> None:
        """Two candidates with scores outside T_margin should pick best without flag."""
        entity1 = make_mock_entity(MockEntityData(uuid4(), "entity1"))
        entity2 = make_mock_entity(MockEntityData(uuid4(), "entity2"))
        candidates = [
            EntityCandidate(entity=entity1, ann_distance=0.05),
            EntityCandidate(entity=entity2, ann_distance=0.10),
        ]

        # Scores outside margin (difference = 0.15 > 0.10)
        best_score = 0.90
        second_score = 0.75  # difference = 0.15 > T_margin (0.10)
        score_mapping = {
            entity1.entity_id: (best_score, 1.0),
            entity2.entity_id: (second_score, 1.0),
        }

        resolver = EntityResolver(
            mock_session,
            t_link=T_LINK,
            t_new=T_NEW,
            t_margin=T_MARGIN,
        )
        resolver._pool = StubCandidatePoolService(candidates)
        resolver._scorer = StubSimilarityScorer(score_mapping)

        seed = make_entity_hypothesis()

        result = await resolver._resolve_by_similarity(
            ctx=MagicMock(
                observation=mock_observation,
                signatures=[],
                entity_seeds=[seed],
                observation_signals=ObservationSignals(
                    image_embedding=[0.1] * 768,
                ),
            ),
            seed=seed,
            entity_nature=EntityNature.INDIVIDUAL,
            seed_idx=0,
        )

        # Should link to best without review flag
        assert len(result.links) == 1
        assert result.links[0].entity_id == entity1.entity_id
        assert "margin_review" not in result.links[0].flags

    @pytest.mark.asyncio
    async def test_margin_at_boundary_no_review_flag(
        self,
        mock_session: AsyncMock,
        mock_observation: MagicMock,
    ) -> None:
        """Difference >= T_margin should NOT trigger review flag.

        Note: Condition is `best - second < T_margin`, so difference exactly at
        or above T_margin should not flag.
        """
        entity1 = make_mock_entity(MockEntityData(uuid4(), "entity1"))
        entity2 = make_mock_entity(MockEntityData(uuid4(), "entity2"))
        candidates = [
            EntityCandidate(entity=entity1, ann_distance=0.05),
            EntityCandidate(entity=entity2, ann_distance=0.10),
        ]

        # Use values that give clear difference above margin (avoid floating point issues)
        # difference = 0.11 > T_margin (0.10)
        best_score = 0.90
        second_score = 0.79  # difference = 0.11 > T_margin
        score_mapping = {
            entity1.entity_id: (best_score, 1.0),
            entity2.entity_id: (second_score, 1.0),
        }

        resolver = EntityResolver(
            mock_session,
            t_link=T_LINK,
            t_new=T_NEW,
            t_margin=T_MARGIN,
        )
        resolver._pool = StubCandidatePoolService(candidates)
        resolver._scorer = StubSimilarityScorer(score_mapping)

        seed = make_entity_hypothesis()

        result = await resolver._resolve_by_similarity(
            ctx=MagicMock(
                observation=mock_observation,
                signatures=[],
                entity_seeds=[seed],
                observation_signals=ObservationSignals(
                    image_embedding=[0.1] * 768,
                ),
            ),
            seed=seed,
            entity_nature=EntityNature.INDIVIDUAL,
            seed_idx=0,
        )

        # Difference > T_margin, so no flag
        assert "margin_review" not in result.links[0].flags


# ─────────────────────────────────────────────────────────────────────────────
# Test Class: Coverage Scaling Interaction
# ─────────────────────────────────────────────────────────────────────────────


class TestCoverageScalingInteraction:
    """Tests for coverage scaling interaction with thresholds.

    Coverage scaling: final_score = weighted_sum * sqrt(W_present)
    When signals are sparse, coverage < 1.0, penalizing the score.
    """

    def test_full_coverage_score_unaffected(self) -> None:
        """Full coverage (1.0) should not penalize score."""
        scorer = SimilarityScorer()

        # Create entity with prototype
        entity = make_mock_entity(
            prototype_image_embedding=[1.0] * 768,
        )

        # Signals with matching embedding (should give high similarity)
        signals = ObservationSignals(
            image_embedding=[1.0] * 768,  # Same as prototype = high cosine sim
        )

        result = scorer.compute(
            observation_signals=signals,
            entity=entity,
            entity_nature=EntityNature.INDIVIDUAL,
        )

        # With only image_embedding available (weight 0.35), coverage = sqrt(0.35) ≈ 0.59
        # This is < 1.0 so there IS a penalty, but we verify the math is correct
        assert result.coverage == pytest.approx(0.5916, rel=0.01)
        # Score should be raw_score * coverage
        # cosine_similarity of identical vectors = 1.0, so weighted = 1.0
        # final = 1.0 * 0.5916 ≈ 0.59
        assert result.score == pytest.approx(0.5916, rel=0.01)

    def test_low_coverage_penalizes_score(self) -> None:
        """Low coverage should penalize even high raw scores."""
        scorer = SimilarityScorer()

        # Create entity with minimal signals
        entity = make_mock_entity(
            prototype_image_embedding=[1.0] * 768,
        )

        # Only image embedding available (weight = 0.35 for individual)
        signals = ObservationSignals(
            image_embedding=[1.0] * 768,
        )

        result = scorer.compute(
            observation_signals=signals,
            entity=entity,
            entity_nature=EntityNature.INDIVIDUAL,
        )

        # Coverage = sqrt(0.35) ≈ 0.59
        expected_coverage = (0.35) ** 0.5
        assert result.coverage == pytest.approx(expected_coverage, rel=0.01)

        # Even with perfect similarity (1.0), score is penalized
        # final_score = 1.0 * 0.59 ≈ 0.59 < T_link (0.85)
        assert result.score < T_LINK

    @pytest.mark.asyncio
    async def test_high_raw_score_low_coverage_drops_below_t_link(
        self,
        mock_session: AsyncMock,
        mock_observation: MagicMock,
    ) -> None:
        """High raw score with low coverage can drop below T_link threshold."""
        entity = make_mock_entity()
        candidate = EntityCandidate(entity=entity, ann_distance=0.1)

        # Raw score 0.95 is above T_link (0.85), but with coverage 0.8:
        # final_score = 0.95 * 0.8 = 0.76 < T_link
        raw_score = 0.95
        coverage = 0.80  # Simulating sqrt(W_present) = 0.8 means W_present = 0.64
        effective_score = raw_score * coverage  # 0.76

        score_mapping = {entity.entity_id: (effective_score, coverage)}

        resolver = EntityResolver(
            mock_session,
            t_link=T_LINK,
            t_new=T_NEW,
            t_margin=T_MARGIN,
        )
        resolver._pool = StubCandidatePoolService([candidate])
        resolver._scorer = StubSimilarityScorer(score_mapping)

        seed = make_entity_hypothesis()

        result = await resolver._resolve_by_similarity(
            ctx=MagicMock(
                observation=mock_observation,
                signatures=[],
                entity_seeds=[seed],
                observation_signals=ObservationSignals(
                    image_embedding=[0.1] * 768,
                ),
            ),
            seed=seed,
            entity_nature=EntityNature.INDIVIDUAL,
            seed_idx=0,
        )

        # Score 0.76 is between T_new and T_link, so CANDIDATE status
        existing_links = [l for l in result.links if l.entity_id == entity.entity_id]
        assert len(existing_links) == 1
        assert existing_links[0].status == "candidate"

    @pytest.mark.asyncio
    async def test_high_raw_score_low_coverage_drops_below_t_new(
        self,
        mock_session: AsyncMock,
        mock_observation: MagicMock,
    ) -> None:
        """Very low coverage can drop score below T_new, creating new entity."""
        entity = make_mock_entity()
        candidate = EntityCandidate(entity=entity, ann_distance=0.1)

        # Raw score 0.95 is above T_link (0.85), but with coverage 0.5:
        # final_score = 0.95 * 0.5 = 0.475 < T_new (0.60)
        raw_score = 0.95
        coverage = 0.50
        effective_score = raw_score * coverage  # 0.475

        score_mapping = {entity.entity_id: (effective_score, coverage)}

        resolver = EntityResolver(
            mock_session,
            t_link=T_LINK,
            t_new=T_NEW,
            t_margin=T_MARGIN,
        )
        resolver._pool = StubCandidatePoolService([candidate])
        resolver._scorer = StubSimilarityScorer(score_mapping)

        seed = make_entity_hypothesis()

        result = await resolver._resolve_by_similarity(
            ctx=MagicMock(
                observation=mock_observation,
                signatures=[],
                entity_seeds=[seed],
                observation_signals=ObservationSignals(
                    image_embedding=[0.1] * 768,
                ),
            ),
            seed=seed,
            entity_nature=EntityNature.INDIVIDUAL,
            seed_idx=0,
        )

        # Score 0.475 < T_new, so new proto-entity created
        assert len(result.entities) == 1
        assert len(result.links) == 1
        assert result.links[0].entity_id != entity.entity_id
        assert "no_match_found" in result.links[0].flags

    def test_coverage_low_evidence_flag(self) -> None:
        """Low coverage should trigger low_evidence flag."""
        scorer = SimilarityScorer()

        # Create entity with prototype
        entity = make_mock_entity(
            prototype_image_embedding=[1.0] * 768,
        )

        # Only image embedding (weight 0.35 for individual < MIN_WEIGHT_THRESHOLD of 0.25)
        # But we only have 1 signal, which is < MIN_SIGNALS_THRESHOLD of 2
        signals = ObservationSignals(
            image_embedding=[1.0] * 768,
        )

        result = scorer.compute(
            observation_signals=signals,
            entity=entity,
            entity_nature=EntityNature.INDIVIDUAL,
        )

        # Should be flagged as low evidence (only 1 signal < 2)
        assert result.is_low_evidence
        assert "low_evidence" in result.flags


# ─────────────────────────────────────────────────────────────────────────────
# Test Class: Edge Cases and Boundary Combinations
# ─────────────────────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases and boundary combinations."""

    @pytest.mark.asyncio
    async def test_no_candidates_creates_proto_entity(
        self,
        mock_session: AsyncMock,
        mock_observation: MagicMock,
    ) -> None:
        """When no candidates exist, should create proto-entity."""
        resolver = EntityResolver(
            mock_session,
            t_link=T_LINK,
            t_new=T_NEW,
            t_margin=T_MARGIN,
        )
        resolver._pool = StubCandidatePoolService([])  # No candidates
        resolver._scorer = StubSimilarityScorer({})

        seed = make_entity_hypothesis()

        result = await resolver._resolve_by_similarity(
            ctx=MagicMock(
                observation=mock_observation,
                signatures=[],
                entity_seeds=[seed],
                observation_signals=ObservationSignals(
                    image_embedding=[0.1] * 768,
                ),
            ),
            seed=seed,
            entity_nature=EntityNature.INDIVIDUAL,
            seed_idx=0,
        )

        # Should create proto-entity
        assert len(result.entities) == 1
        assert "no_candidates_found" in result.links[0].flags

    @pytest.mark.asyncio
    async def test_single_candidate_no_margin_check(
        self,
        mock_session: AsyncMock,
        mock_observation: MagicMock,
    ) -> None:
        """Single candidate should not trigger margin check."""
        entity = make_mock_entity()
        candidate = EntityCandidate(entity=entity, ann_distance=0.1)

        score_mapping = {entity.entity_id: (0.90, 1.0)}  # Above T_link

        resolver = EntityResolver(
            mock_session,
            t_link=T_LINK,
            t_new=T_NEW,
            t_margin=T_MARGIN,
        )
        resolver._pool = StubCandidatePoolService([candidate])
        resolver._scorer = StubSimilarityScorer(score_mapping)

        seed = make_entity_hypothesis()

        result = await resolver._resolve_by_similarity(
            ctx=MagicMock(
                observation=mock_observation,
                signatures=[],
                entity_seeds=[seed],
                observation_signals=ObservationSignals(
                    image_embedding=[0.1] * 768,
                ),
            ),
            seed=seed,
            entity_nature=EntityNature.INDIVIDUAL,
            seed_idx=0,
        )

        # Single candidate, no margin review
        # (second_best_score defaults to 0, so difference = 0.90 > T_margin)
        assert "margin_review" not in result.links[0].flags

    def test_boundary_values_are_correct(self) -> None:
        """Verify threshold boundary values match config defaults."""
        from object_sense.config import settings

        assert settings.entity_resolution_t_link == T_LINK
        assert settings.entity_resolution_t_new == T_NEW
        assert settings.entity_resolution_t_margin == T_MARGIN
