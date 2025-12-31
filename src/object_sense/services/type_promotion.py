"""Type promotion service for elevating TypeCandidates to stable Types.

This module implements the promotion logic (design_v2_corrections.md §7):
1. Check candidates against thresholds (evidence, coherence, time window)
2. Create stable Type row with embedding (from entity cluster)
3. Infer parent_type_id from entity graph
4. Update TypeCandidate.status = 'promoted'
5. Update TypeCandidate.promoted_to_type_id
6. Update observations.stable_type_id for relevant observations

Background jobs call this service periodically to promote eligible candidates.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from object_sense.config import settings
from object_sense.models.enums import TypeCandidateStatus, TypeCreatedVia, TypeStatus
from object_sense.models.observation import Observation
from object_sense.models.type import Type
from object_sense.models.type_candidate import TypeCandidate


@dataclass
class PromotionResult:
    """Result of a type promotion operation."""

    promoted: bool
    candidate_id: UUID
    type_id: UUID | None = None
    reason: str | None = None
    observations_updated: int = 0


class TypePromotionService:
    """Service for promoting TypeCandidates to stable Types.

    Usage:
        async with AsyncSession(engine) as session:
            service = TypePromotionService(session)
            results = await service.run_promotion_job()
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize the service with a database session."""
        self._session = session

    async def check_promotion_eligibility(
        self,
        candidate: TypeCandidate,
    ) -> tuple[bool, str]:
        """Check if a candidate is eligible for promotion.

        Promotion happens when ANY of these conditions are met:
        1. Evidence threshold: evidence_count >= min_evidence
        2. Coherence threshold: coherence_score > min_coherence
        3. Time window: survives survival_days without merge/contradiction

        Args:
            candidate: The TypeCandidate to check.

        Returns:
            Tuple of (eligible, reason).
        """
        if candidate.status != TypeCandidateStatus.PROPOSED:
            return False, f"Candidate is {candidate.status}, not proposed"

        # Check evidence threshold
        if candidate.evidence_count >= settings.type_promotion_min_evidence:
            return True, f"Evidence count {candidate.evidence_count} >= {settings.type_promotion_min_evidence}"

        # Check coherence threshold
        if (
            candidate.coherence_score is not None
            and candidate.coherence_score >= settings.type_promotion_min_coherence
        ):
            return True, f"Coherence score {candidate.coherence_score:.2f} >= {settings.type_promotion_min_coherence}"

        # Check time window survival
        survival_threshold = datetime.now() - timedelta(days=settings.type_promotion_survival_days)
        if candidate.created_at.replace(tzinfo=None) < survival_threshold:
            # Additional check: has it been merged or contradicted since creation?
            # For now, we just check age since merge tracking is in the future
            return True, f"Survived {settings.type_promotion_survival_days} days without merge"

        return False, "Does not meet any promotion threshold"

    async def promote_candidate(
        self,
        candidate: TypeCandidate,
        *,
        embedding: list[Any] | None = None,
        parent_type_id: UUID | None = None,
    ) -> PromotionResult:
        """Promote a TypeCandidate to a stable Type.

        Args:
            candidate: The candidate to promote.
            embedding: Optional embedding for the type (computed from entity cluster).
            parent_type_id: Optional parent type (inferred from entity graph).

        Returns:
            PromotionResult with details of the operation.
        """
        # Check eligibility first
        eligible, reason = await self.check_promotion_eligibility(candidate)
        if not eligible:
            return PromotionResult(
                promoted=False,
                candidate_id=candidate.candidate_id,
                reason=reason,
            )

        # Create the stable Type
        type_id = uuid4()
        type_row = Type(
            type_id=type_id,
            canonical_name=candidate.proposed_name,
            aliases=[],
            parent_type_id=parent_type_id,
            embedding=embedding,
            status=TypeStatus.PROVISIONAL,  # Newly promoted, will become STABLE with more evidence
            evidence_count=candidate.evidence_count,
            created_via=TypeCreatedVia.LLM_PROPOSED,
        )
        self._session.add(type_row)

        # Update the candidate
        candidate.status = TypeCandidateStatus.PROMOTED
        candidate.promoted_to_type_id = type_id

        # Update observations that reference this candidate
        observations_updated = await self._update_observation_stable_types(
            candidate.candidate_id, type_id
        )

        return PromotionResult(
            promoted=True,
            candidate_id=candidate.candidate_id,
            type_id=type_id,
            reason=reason,
            observations_updated=observations_updated,
        )

    async def _update_observation_stable_types(
        self,
        candidate_id: UUID,
        type_id: UUID,
    ) -> int:
        """Update observations with the promoted stable_type_id.

        Args:
            candidate_id: The candidate that was promoted.
            type_id: The new stable Type ID.

        Returns:
            Number of observations updated.
        """
        stmt = (
            update(Observation)
            .where(Observation.candidate_type_id == candidate_id)
            .where(Observation.stable_type_id.is_(None))
            .values(stable_type_id=type_id)
        )
        result = await self._session.execute(stmt)
        return result.rowcount  # type: ignore[return-value]

    async def run_promotion_job(
        self,
        *,
        limit: int = 100,
    ) -> list[PromotionResult]:
        """Run the promotion job for all eligible candidates.

        This should be called periodically (e.g., hourly) by a background worker.

        Args:
            limit: Maximum number of candidates to process.

        Returns:
            List of PromotionResults for each processed candidate.
        """
        # Find eligible candidates
        stmt = (
            select(TypeCandidate)
            .where(TypeCandidate.status == TypeCandidateStatus.PROPOSED)
            .where(TypeCandidate.evidence_count >= settings.type_promotion_min_evidence)
            .order_by(TypeCandidate.evidence_count.desc())
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        candidates = list(result.scalars().all())

        # Also check for time-based promotion (candidates that survived)
        survival_threshold = datetime.now() - timedelta(days=settings.type_promotion_survival_days)
        stmt_time = (
            select(TypeCandidate)
            .where(TypeCandidate.status == TypeCandidateStatus.PROPOSED)
            .where(TypeCandidate.created_at < survival_threshold)
            .where(TypeCandidate.evidence_count >= 2)  # At least 2 references to avoid noise
            .order_by(TypeCandidate.evidence_count.desc())
            .limit(limit)
        )
        result_time = await self._session.execute(stmt_time)
        time_candidates = list(result_time.scalars().all())

        # Combine and deduplicate
        seen_ids: set[UUID] = set()
        all_candidates: list[TypeCandidate] = []
        for c in candidates + time_candidates:
            if c.candidate_id not in seen_ids:
                seen_ids.add(c.candidate_id)
                all_candidates.append(c)

        # Process each candidate
        results: list[PromotionResult] = []
        for candidate in all_candidates[:limit]:
            result = await self.promote_candidate(candidate)
            results.append(result)

        return results

    async def run_cleanup_job(
        self,
        *,
        limit: int = 1000,
    ) -> dict[str, int]:
        """Run the cleanup job for stale candidates.

        This should be called periodically (e.g., daily) by a background worker.

        Cleanup rules:
        - Reject candidates with evidence_count == 1 after reject_after_days
        - GC (delete) rejected candidates after gc_after_days

        Args:
            limit: Maximum number of candidates to process.

        Returns:
            Dict with counts: {"rejected": N, "gc_deleted": M}
        """
        now = datetime.now()

        # Reject stale candidates
        reject_threshold = now - timedelta(days=settings.type_candidate_reject_after_days)
        stmt_reject = (
            update(TypeCandidate)
            .where(TypeCandidate.status == TypeCandidateStatus.PROPOSED)
            .where(TypeCandidate.evidence_count == 1)
            .where(TypeCandidate.created_at < reject_threshold)
            .values(status=TypeCandidateStatus.REJECTED)
        )
        result_reject = await self._session.execute(stmt_reject)
        rejected_count: int = getattr(result_reject, "rowcount", 0) or 0

        # GC deleted candidates (for now, just count - actual deletion would need more care)
        gc_threshold = now - timedelta(days=settings.type_candidate_gc_after_days)
        stmt_gc_count = (
            select(TypeCandidate)
            .where(TypeCandidate.status == TypeCandidateStatus.REJECTED)
            .where(TypeCandidate.updated_at < gc_threshold)
        )
        result_gc = await self._session.execute(stmt_gc_count)
        gc_eligible = len(list(result_gc.scalars().all()))

        # Note: Actual deletion is commented out for safety
        # In production, you'd want to archive before deleting
        # stmt_delete = (
        #     delete(TypeCandidate)
        #     .where(TypeCandidate.status == TypeCandidateStatus.REJECTED)
        #     .where(TypeCandidate.updated_at < gc_threshold)
        #     .limit(limit)
        # )
        # await self._session.execute(stmt_delete)

        _ = limit  # Reserved for future use when implementing actual GC deletion
        return {
            "rejected": rejected_count,
            "gc_eligible": gc_eligible,
        }

    async def compute_type_embedding(
        self,
        candidate_id: UUID,
    ) -> list[float] | None:
        """Compute embedding for a type based on its entity cluster.

        This is a placeholder - actual implementation would:
        1. Find all entities linked to observations with this candidate_type_id
        2. Aggregate entity embeddings (e.g., centroid)
        3. Return the aggregated embedding

        Args:
            candidate_id: The candidate to compute embedding for.

        Returns:
            The computed embedding, or None if not enough data.
        """
        # TODO: Implement entity cluster embedding computation
        # For now, return None and skip embedding for promoted types
        _ = candidate_id  # Will be used when implementing cluster embedding
        return None

    async def infer_parent_type(
        self,
        candidate: TypeCandidate,
    ) -> UUID | None:
        """Infer parent type from entity graph relationships.

        This is a placeholder - actual implementation would:
        1. Look at entities linked to observations with this candidate
        2. Find common patterns in entity types
        3. Suggest a parent based on type hierarchy

        Args:
            candidate: The candidate to infer parent for.

        Returns:
            The inferred parent type ID, or None if no parent detected.
        """
        # TODO: Implement parent type inference from entity graph
        # For now, return None (no parent)
        _ = candidate  # Will be used when implementing parent inference
        return None
