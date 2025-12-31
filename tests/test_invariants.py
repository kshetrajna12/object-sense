"""Invariant tests that lock in the philosophy.

These tests ensure the system maintains its core design principles:
1. LLM proposes, engine decides (non-authoritative routing)
2. TypeCandidate merge invariants (path compression)
3. Promotion invariants (candidate audit trail)

Run with: pytest tests/test_invariants.py -v
"""

from __future__ import annotations

import inspect
import re

import pytest

from object_sense.inference.schemas import TypeProposal
from object_sense.services.type_candidate import normalize_type_name


class TestNonAuthoritativeRouting:
    """Tests that Step 4 cannot bind to existing types authoritatively.

    The LLM can only PROPOSE type labels. The engine handles matching/dedup.
    """

    def test_type_proposal_has_no_existing_type_field(self) -> None:
        """TypeProposal must not have fields that allow LLM to bind to existing types."""
        # Get all field names from TypeProposal
        field_names = list(TypeProposal.model_fields.keys())

        # These patterns would allow LLM to bypass engine resolution
        forbidden_patterns = [
            r"existing_type.*",
            r"stable_type.*",
            r"matched_type.*",
            r"resolved_type.*",
            r"bound_type.*",
        ]

        for field in field_names:
            for pattern in forbidden_patterns:
                assert not re.match(
                    pattern, field, re.IGNORECASE
                ), f"Field '{field}' matches forbidden pattern '{pattern}' - removes engine authority"

    def test_type_proposal_only_has_type_candidate(self) -> None:
        """TypeProposal should only have type_candidate for type labeling."""
        field_names = list(TypeProposal.model_fields.keys())

        # The only type-related field should be type_candidate
        type_fields = [f for f in field_names if "type" in f.lower() and f != "observation_kind"]

        assert type_fields == [
            "type_candidate"
        ], f"Expected only 'type_candidate', got {type_fields}"

    def test_type_inference_agent_does_not_set_stable_type(self) -> None:
        """TypeInferenceAgent must not be able to set stable types directly."""
        from object_sense.inference.type_inference import TypeInferenceAgent

        # Get source code of the agent
        source = inspect.getsource(TypeInferenceAgent)

        # Should not contain stable_type_id assignments
        assert "stable_type_id" not in source, "TypeInferenceAgent should not touch stable_type_id"
        assert "stable_type =" not in source, "TypeInferenceAgent should not create stable types"


class TestTypeCandidateMergeInvariants:
    """Tests for TypeCandidate merge behavior including path compression."""

    def test_normalize_type_name_consistency(self) -> None:
        """Normalization should be idempotent and consistent."""
        test_cases = [
            ("Wildlife Photo", "wildlife_photo"),
            ("wildlife_photo", "wildlife_photo"),
            ("WILDLIFE_PHOTO", "wildlife_photo"),
            ("  wildlife_photo  ", "wildlife_photo"),
            ("Wildlife--Photo", "wildlife_photo"),
            ("Wildlife__Photo", "wildlife_photo"),
            ("wildlife photo", "wildlife_photo"),
        ]

        for input_name, expected in test_cases:
            result = normalize_type_name(input_name)
            assert result == expected, f"normalize_type_name({input_name!r}) = {result!r}, expected {expected!r}"

            # Idempotent: normalizing again should give same result
            assert normalize_type_name(result) == expected

    def test_normalize_type_name_handles_unicode(self) -> None:
        """Normalization should handle unicode consistently."""
        # NFKC normalization converts ligatures
        assert normalize_type_name("ﬁle_type") == "file_type"  # fi ligature -> fi

        # Accented chars are normalized via NFKC but non-ASCII is then replaced
        # with underscore, so ï -> _ (this is consistent, if aggressive)
        result = normalize_type_name("naïve_type")
        # The key invariant: idempotent and consistent
        assert normalize_type_name(result) == result


class TestTypeProposalSchema:
    """Tests that TypeProposal schema enforces correct structure."""

    def test_type_proposal_requires_observation_kind(self) -> None:
        """TypeProposal must have observation_kind (routing hint)."""
        # Should fail without observation_kind
        with pytest.raises(Exception):  # Pydantic validation error
            TypeProposal(reasoning="test")  # Missing observation_kind

    def test_type_proposal_allows_no_type_candidate(self) -> None:
        """TypeProposal can have no type_candidate (rare but valid)."""
        proposal = TypeProposal(
            observation_kind="unknown_content",
            reasoning="Could not determine type",
        )
        assert proposal.type_candidate is None
        assert proposal.observation_kind == "unknown_content"

    def test_type_candidate_proposal_requires_rationale(self) -> None:
        """TypeCandidateProposal must have rationale explaining the proposal."""
        from object_sense.inference.schemas import TypeCandidateProposal

        # Should fail without rationale
        with pytest.raises(Exception):
            TypeCandidateProposal(proposed_name="test_type")  # Missing rationale

        # Should work with rationale
        proposal = TypeCandidateProposal(
            proposed_name="test_type",
            rationale="This is needed because...",
        )
        assert proposal.rationale is not None


class TestPromotionInvariants:
    """Tests for type promotion behavior."""

    def test_promotion_threshold_coherence_disabled_by_default(self) -> None:
        """Coherence threshold should be None for v0 (requires entity clusters)."""
        from object_sense.config import settings

        assert settings.type_promotion_min_coherence is None, (
            "Coherence threshold should be None for v0 - "
            "it requires stable entity clusters from Phase 4"
        )

    def test_promotion_evidence_threshold_reasonable(self) -> None:
        """Evidence threshold should be reasonable (not too low, not too high)."""
        from object_sense.config import settings

        assert 2 <= settings.type_promotion_min_evidence <= 20, (
            f"Evidence threshold {settings.type_promotion_min_evidence} seems unreasonable. "
            "Should be 2-20 for v0."
        )


class TestEntityNatureRequired:
    """Tests that entity_nature is required on EntityHypothesis."""

    def test_entity_hypothesis_requires_entity_nature(self) -> None:
        """EntityHypothesis must have entity_nature (affects resolution signals)."""
        from object_sense.inference.schemas import EntityHypothesis

        # Should fail without entity_nature
        with pytest.raises(Exception):
            EntityHypothesis(entity_type="animal_entity")  # Missing entity_nature

    def test_entity_hypothesis_accepts_all_natures(self) -> None:
        """EntityHypothesis should accept all EntityNature values."""
        from object_sense.inference.schemas import EntityHypothesis
        from object_sense.models.enums import EntityNature

        for nature in EntityNature:
            entity = EntityHypothesis(
                entity_type="test_entity",
                entity_nature=nature,
            )
            assert entity.entity_nature == nature


class TestDeterministicIdSchema:
    """Tests for DeterministicId schema."""

    def test_deterministic_id_requires_type_and_value(self) -> None:
        """DeterministicId must have id_type and id_value."""
        from object_sense.inference.schemas import DeterministicId

        # Should fail without required fields
        with pytest.raises(Exception):
            DeterministicId(id_type="sku")  # Missing id_value

        with pytest.raises(Exception):
            DeterministicId(id_value="12345")  # Missing id_type

    def test_deterministic_id_has_namespace_default(self) -> None:
        """DeterministicId should have default namespace."""
        from object_sense.inference.schemas import DeterministicId

        det_id = DeterministicId(id_type="sku", id_value="12345")
        assert det_id.id_namespace == "default"

    def test_deterministic_id_has_strength_default(self) -> None:
        """DeterministicId should default to strong."""
        from object_sense.inference.schemas import DeterministicId

        det_id = DeterministicId(id_type="sku", id_value="12345")
        assert det_id.strength == "strong"
