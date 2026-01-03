"""Tests for entity resolution module."""

from __future__ import annotations

from uuid import uuid4

import pytest

from object_sense.models.enums import EntityNature, LinkRole
from object_sense.resolution.reconciliation import (
    PendingLink,
    deduplicate_links,
    filter_low_posterior_links,
    reconcile_multi_seed_links,
)
from object_sense.resolution.similarity import (
    MIN_SIGNALS_THRESHOLD,
    MIN_WEIGHT_THRESHOLD,
    SIGNAL_WEIGHTS,
    ObservationSignals,
    SimilarityResult,
    SimilarityScorer,
    cosine_similarity,
)


class TestCosineSimilarity:
    """Tests for cosine_similarity function."""

    def test_identical_vectors(self) -> None:
        a = [1.0, 2.0, 3.0]
        b = [1.0, 2.0, 3.0]
        assert cosine_similarity(a, b) == pytest.approx(1.0)

    def test_opposite_vectors(self) -> None:
        a = [1.0, 0.0, 0.0]
        b = [-1.0, 0.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_orthogonal_vectors(self) -> None:
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(0.5)

    def test_mismatched_lengths(self) -> None:
        a = [1.0, 2.0]
        b = [1.0, 2.0, 3.0]
        assert cosine_similarity(a, b) == 0.0

    def test_zero_vector(self) -> None:
        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]
        assert cosine_similarity(a, b) == 0.0


class TestSignalWeights:
    """Tests for signal weight configuration."""

    def test_all_natures_have_weights(self) -> None:
        for nature in EntityNature:
            assert nature in SIGNAL_WEIGHTS, f"Missing weights for {nature}"

    def test_weights_sum_to_one(self) -> None:
        for nature, weights in SIGNAL_WEIGHTS.items():
            total = sum(weights.values())
            assert total == pytest.approx(1.0), f"Weights for {nature} sum to {total}"

    def test_individual_weights_visual_heavy(self) -> None:
        weights = SIGNAL_WEIGHTS[EntityNature.INDIVIDUAL]
        assert weights["image_embedding"] > weights["text_embedding"] if "text_embedding" in weights else True

    def test_class_weights_text_heavy(self) -> None:
        weights = SIGNAL_WEIGHTS[EntityNature.CLASS]
        assert weights["text_embedding"] >= 0.4


class TestObservationSignals:
    """Tests for ObservationSignals extraction."""

    def test_empty_signals(self) -> None:
        signals = ObservationSignals()
        assert signals.image_embedding is None
        assert signals.text_embedding is None
        assert signals.facets is None

    def test_with_embeddings(self) -> None:
        img_emb = [0.1] * 768
        text_emb = [0.2] * 1024
        signals = ObservationSignals(
            image_embedding=img_emb,
            text_embedding=text_emb,
        )
        assert signals.image_embedding == img_emb
        assert signals.text_embedding == text_emb


class TestSimilarityScorer:
    """Tests for SimilarityScorer."""

    def test_no_signals_returns_zero(self) -> None:
        """Test that no available signals returns zero score."""
        # Create a mock entity with no prototype embeddings
        from unittest.mock import MagicMock
        entity = MagicMock()
        entity.prototype_image_embedding = None
        entity.prototype_text_embedding = None
        entity.slots = {}
        entity.name = None

        signals = ObservationSignals()
        scorer = SimilarityScorer()

        result = scorer.compute(
            observation_signals=signals,
            entity=entity,
            entity_nature=EntityNature.INDIVIDUAL,
        )

        assert result.score == 0.0
        assert result.is_low_evidence
        assert "no_signals_available" in result.flags

    def test_low_evidence_threshold(self) -> None:
        assert MIN_WEIGHT_THRESHOLD == 0.25
        assert MIN_SIGNALS_THRESHOLD == 2


class TestPendingLink:
    """Tests for PendingLink dataclass."""

    def test_default_values(self) -> None:
        link = PendingLink(
            entity_id=uuid4(),
            posterior=0.9,
            status="soft",
            role=LinkRole.SUBJECT,
        )
        assert link.flags == []
        assert link.seed_index is None
        assert link.facets is None

    def test_with_flags(self) -> None:
        link = PendingLink(
            entity_id=uuid4(),
            posterior=0.9,
            status="soft",
            role=LinkRole.SUBJECT,
            flags=["test_flag"],
        )
        assert link.flags == ["test_flag"]


class TestMultiSeedReconciliation:
    """Tests for multi-seed reconciliation logic."""

    def test_single_link_no_conflict(self) -> None:
        """Single link to an entity should pass through unchanged."""
        entity_id = uuid4()
        link = PendingLink(
            entity_id=entity_id,
            posterior=0.9,
            status="soft",
            role=LinkRole.SUBJECT,
        )
        result = reconcile_multi_seed_links([link])
        assert len(result.links) == 1
        assert result.links[0].posterior == 0.9
        assert result.conflicts_detected == 0

    def test_role_conflict_detection(self) -> None:
        """Same entity as both subject and context should be flagged."""
        entity_id = uuid4()
        links = [
            PendingLink(
                entity_id=entity_id,
                posterior=0.9,
                status="soft",
                role=LinkRole.SUBJECT,
                seed_index=0,
            ),
            PendingLink(
                entity_id=entity_id,
                posterior=0.8,
                status="soft",
                role=LinkRole.CONTEXT,
                seed_index=1,
            ),
        ]
        result = reconcile_multi_seed_links(links)
        assert result.conflicts_detected >= 1
        # Check that flags were added
        for link in result.links:
            assert "multi_seed_role_conflict" in link.flags

    def test_duplicate_link_handling(self) -> None:
        """Multiple links to same entity should keep highest posterior."""
        entity_id = uuid4()
        links = [
            PendingLink(
                entity_id=entity_id,
                posterior=0.9,
                status="soft",
                role=LinkRole.SUBJECT,
                seed_index=0,
            ),
            PendingLink(
                entity_id=entity_id,
                posterior=0.7,
                status="soft",
                role=LinkRole.SUBJECT,
                seed_index=1,
            ),
        ]
        result = reconcile_multi_seed_links(links)
        # Should have both links, but one flagged as duplicate
        assert result.conflicts_detected >= 1
        # Find the duplicate flag
        duplicate_flags = [l for l in result.links if "duplicate_entity_link" in l.flags]
        assert len(duplicate_flags) >= 1

    def test_conflict_penalty_applied(self) -> None:
        """Conflicting links should have their posterior reduced."""
        entity_id = uuid4()
        links = [
            PendingLink(
                entity_id=entity_id,
                posterior=1.0,
                status="soft",
                role=LinkRole.SUBJECT,
                seed_index=0,
            ),
            PendingLink(
                entity_id=entity_id,
                posterior=1.0,
                status="soft",
                role=LinkRole.CONTEXT,
                seed_index=1,
            ),
        ]
        result = reconcile_multi_seed_links(links, conflict_penalty=0.5)
        # All links should have posterior reduced
        for link in result.links:
            assert link.posterior <= 0.5  # 1.0 * 0.5 * 0.5 (role + duplicate)

    def test_empty_links(self) -> None:
        """Empty input should return empty result."""
        result = reconcile_multi_seed_links([])
        assert len(result.links) == 0
        assert result.conflicts_detected == 0
        assert result.flags_added == 0


class TestDeduplicateLinks:
    """Tests for deduplicate_links function."""

    def test_unique_links_unchanged(self) -> None:
        """Links to different entities should all be kept."""
        links = [
            PendingLink(
                entity_id=uuid4(),
                posterior=0.9,
                status="soft",
                role=LinkRole.SUBJECT,
            ),
            PendingLink(
                entity_id=uuid4(),
                posterior=0.8,
                status="soft",
                role=LinkRole.CONTEXT,
            ),
        ]
        result = deduplicate_links(links)
        assert len(result) == 2

    def test_duplicate_keeps_highest(self) -> None:
        """Duplicate links should keep only highest posterior."""
        entity_id = uuid4()
        links = [
            PendingLink(
                entity_id=entity_id,
                posterior=0.7,
                status="soft",
                role=LinkRole.SUBJECT,
            ),
            PendingLink(
                entity_id=entity_id,
                posterior=0.9,
                status="soft",
                role=LinkRole.CONTEXT,
            ),
        ]
        result = deduplicate_links(links)
        assert len(result) == 1
        assert result[0].posterior == 0.9


class TestFilterLowPosteriorLinks:
    """Tests for filter_low_posterior_links function."""

    def test_filters_below_threshold(self) -> None:
        links = [
            PendingLink(
                entity_id=uuid4(),
                posterior=0.9,
                status="soft",
                role=LinkRole.SUBJECT,
            ),
            PendingLink(
                entity_id=uuid4(),
                posterior=0.3,
                status="candidate",
                role=LinkRole.CONTEXT,
            ),
        ]
        result = filter_low_posterior_links(links, threshold=0.5)
        assert len(result) == 1
        assert result[0].posterior == 0.9

    def test_keeps_at_threshold(self) -> None:
        links = [
            PendingLink(
                entity_id=uuid4(),
                posterior=0.5,
                status="soft",
                role=LinkRole.SUBJECT,
            ),
        ]
        result = filter_low_posterior_links(links, threshold=0.5)
        assert len(result) == 1


class TestFacetContradiction:
    """Tests for facet contradiction detection."""

    def test_no_contradiction_same_facets(self) -> None:
        """Links with same facets should not conflict."""
        entity_id = uuid4()
        facets = {"species": {"value": "leopard"}}
        links = [
            PendingLink(
                entity_id=entity_id,
                posterior=0.9,
                status="soft",
                role=LinkRole.SUBJECT,
                facets=facets,
                seed_index=0,
            ),
            PendingLink(
                entity_id=entity_id,
                posterior=0.8,
                status="soft",
                role=LinkRole.SUBJECT,
                facets=facets,
                seed_index=1,
            ),
        ]
        result = reconcile_multi_seed_links(links)
        # Should not have facet conflict (only duplicate)
        facet_conflicts = [l for l in result.links if "multi_seed_facet_conflict" in l.flags]
        assert len(facet_conflicts) == 0

    def test_contradiction_different_facets(self) -> None:
        """Links with different facets should be flagged."""
        entity_id = uuid4()
        links = [
            PendingLink(
                entity_id=entity_id,
                posterior=0.9,
                status="soft",
                role=LinkRole.SUBJECT,
                facets={"species": {"value": "leopard"}},
                seed_index=0,
            ),
            PendingLink(
                entity_id=entity_id,
                posterior=0.8,
                status="soft",
                role=LinkRole.SUBJECT,
                facets={"species": {"value": "lion"}},
                seed_index=1,
            ),
        ]
        result = reconcile_multi_seed_links(links)
        facet_conflicts = [l for l in result.links if "multi_seed_facet_conflict" in l.flags]
        assert len(facet_conflicts) > 0
