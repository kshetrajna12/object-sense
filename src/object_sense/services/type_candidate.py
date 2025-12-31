"""TypeCandidate service for managing provisional type labels.

This module implements the TypeCandidate lifecycle (design_v2_corrections.md §7):
- Proposal: Create candidates from LLM output
- Dedup: Normalize names, detect duplicates
- Merge: Consolidate duplicate candidates
- Promotion: Elevate candidates to stable Types (separate module)

TypeCandidates allow duplicates (no UNIQUE constraint on proposed_name).
Consolidation happens via merge operations, not insert rejection.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from object_sense.models.enums import TypeCandidateStatus
from object_sense.models.type_candidate import TypeCandidate

if TYPE_CHECKING:
    from object_sense.inference.schemas import TypeCandidateProposal


def normalize_type_name(name: str) -> str:
    """Normalize a type name for dedup detection.

    Normalization rules:
    - Lowercase
    - Unicode NFKC normalization
    - Strip whitespace
    - Replace non-alphanumeric with underscore
    - Collapse multiple underscores
    - Remove leading/trailing underscores

    Examples:
        "Wildlife Photo" -> "wildlife_photo"
        "wildlife_photo" -> "wildlife_photo"
        "Wildlife__Photo" -> "wildlife_photo"
        "WILDLIFE-PHOTO" -> "wildlife_photo"
        "  wildlife_photo  " -> "wildlife_photo"
    """
    # NFKC normalization for unicode equivalence
    name = unicodedata.normalize("NFKC", name)

    # Lowercase
    name = name.lower()

    # Strip whitespace
    name = name.strip()

    # Replace non-alphanumeric with underscore
    name = re.sub(r"[^a-z0-9]", "_", name)

    # Collapse multiple underscores
    name = re.sub(r"_+", "_", name)

    # Remove leading/trailing underscores
    name = name.strip("_")

    return name


class TypeCandidateService:
    """Service for managing TypeCandidate lifecycle.

    Usage:
        async with AsyncSession(engine) as session:
            service = TypeCandidateService(session)
            candidate = await service.get_or_create("wildlife_photo")
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize the service with a database session."""
        self._session = session

    async def get_or_create(
        self,
        proposed_name: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> tuple[TypeCandidate, bool]:
        """Get an existing candidate or create a new one.

        Args:
            proposed_name: The type name as proposed (will be normalized).
            details: Optional metadata (e.g., LLM reasoning).

        Returns:
            Tuple of (TypeCandidate, created_flag).
            created_flag is True if a new candidate was created.
        """
        normalized = normalize_type_name(proposed_name)

        # Check for existing candidate with same normalized name
        existing = await self.find_by_normalized_name(normalized)

        if existing:
            # Increment evidence count and update last_referenced_at
            existing.evidence_count += 1
            existing.last_referenced_at = datetime.now()
            return existing, False

        # Create new candidate
        candidate = TypeCandidate(
            candidate_id=uuid4(),
            proposed_name=proposed_name,
            normalized_name=normalized,
            status=TypeCandidateStatus.PROPOSED,
            evidence_count=1,
            details=details or {},
            last_referenced_at=datetime.now(),
        )
        self._session.add(candidate)

        return candidate, True

    async def create_from_proposal(
        self,
        proposal: TypeCandidateProposal,
    ) -> tuple[TypeCandidate, bool]:
        """Create a TypeCandidate from a TypeCandidateProposal.

        Args:
            proposal: The proposal from the LLM.

        Returns:
            Tuple of (TypeCandidate, created_flag).
        """
        details = {
            "rationale": proposal.rationale,
            "suggested_parent": proposal.suggested_parent,
            "confidence": proposal.confidence,
        }
        return await self.get_or_create(proposal.proposed_name, details=details)

    async def find_by_normalized_name(
        self,
        normalized_name: str,
        *,
        include_merged: bool = False,
    ) -> TypeCandidate | None:
        """Find an active candidate by normalized name.

        Args:
            normalized_name: The normalized type name to search for.
            include_merged: If True, also return merged candidates.

        Returns:
            The matching TypeCandidate or None.
        """
        statuses = [TypeCandidateStatus.PROPOSED, TypeCandidateStatus.PROMOTED]
        if include_merged:
            statuses.append(TypeCandidateStatus.MERGED)

        stmt = (
            select(TypeCandidate)
            .where(TypeCandidate.normalized_name == normalized_name)
            .where(TypeCandidate.status.in_(statuses))
            .order_by(TypeCandidate.evidence_count.desc())
            .limit(1)
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def find_by_id(self, candidate_id: UUID) -> TypeCandidate | None:
        """Find a candidate by ID."""
        return await self._session.get(TypeCandidate, candidate_id)

    async def resolve_canonical(
        self,
        candidate_id: UUID,
        *,
        compress_path: bool = True,
    ) -> TypeCandidate | None:
        """Resolve a candidate to its canonical form following merge chains.

        If a candidate has been merged, follows merged_into_candidate_id
        until finding the canonical (non-merged) candidate.

        Path compression (enabled by default): After resolving, updates all
        intermediate candidates to point directly to the canonical. This
        prevents O(n) chain traversal on subsequent lookups.

        Args:
            candidate_id: The candidate ID to resolve.
            compress_path: If True, update intermediate pointers to canonical.

        Returns:
            The canonical TypeCandidate, or None if not found.
        """
        current_id = candidate_id
        visited: list[UUID] = []  # Ordered list for path compression

        while True:
            if current_id in visited:
                # Circular reference - shouldn't happen, but handle gracefully
                return None
            visited.append(current_id)

            candidate = await self.find_by_id(current_id)
            if candidate is None:
                return None

            if candidate.merged_into_candidate_id is None:
                # Found canonical. Apply path compression if enabled.
                if compress_path and len(visited) > 1:
                    canonical_id = candidate.candidate_id
                    # Update all intermediate candidates to point directly to canonical
                    for intermediate_id in visited[:-1]:  # Exclude canonical itself
                        intermediate = await self.find_by_id(intermediate_id)
                        if intermediate and intermediate.merged_into_candidate_id != canonical_id:
                            intermediate.merged_into_candidate_id = canonical_id
                return candidate

            current_id = candidate.merged_into_candidate_id

    async def merge(
        self,
        source_id: UUID,
        target_id: UUID,
        *,
        reason: str | None = None,
    ) -> TypeCandidate:
        """Merge one candidate into another.

        The source candidate will be marked as MERGED and point to the target.
        Evidence count is transferred to the target.

        Args:
            source_id: The candidate being merged (will become MERGED).
            target_id: The candidate to merge into (remains PROPOSED).
            reason: Optional reason for the merge.

        Returns:
            The target candidate (canonical after merge).

        Raises:
            ValueError: If source or target not found, or if merging into self.
        """
        if source_id == target_id:
            raise ValueError("Cannot merge candidate into itself")

        source = await self.find_by_id(source_id)
        target = await self.find_by_id(target_id)

        if source is None:
            raise ValueError(f"Source candidate {source_id} not found")
        if target is None:
            raise ValueError(f"Target candidate {target_id} not found")

        # Resolve target to canonical in case it's already merged
        canonical_target = await self.resolve_canonical(target_id)
        if canonical_target is None:
            raise ValueError(f"Could not resolve canonical for {target_id}")

        # Transfer evidence count
        canonical_target.evidence_count += source.evidence_count

        # Mark source as merged
        source.status = TypeCandidateStatus.MERGED
        source.merged_into_candidate_id = canonical_target.candidate_id

        # Record merge in details
        if reason:
            merge_history = source.details.get("merge_history", [])
            merge_history.append(
                {
                    "merged_into": str(canonical_target.candidate_id),
                    "reason": reason,
                    "at": datetime.now().isoformat(),
                }
            )
            source.details = {**source.details, "merge_history": merge_history}

        return canonical_target

    async def find_similar_candidates(
        self,
        proposed_name: str,
        *,
        limit: int = 10,
    ) -> list[TypeCandidate]:
        """Find candidates with similar normalized names.

        Uses prefix matching on normalized_name to find potential duplicates.
        Useful for dedup detection before merging.

        Args:
            proposed_name: The name to find similar candidates for.
            limit: Maximum number of results.

        Returns:
            List of similar TypeCandidates ordered by evidence count.
        """
        normalized = normalize_type_name(proposed_name)

        # Find candidates with matching prefix (first 5 chars)
        prefix = normalized[:5] if len(normalized) >= 5 else normalized

        stmt = (
            select(TypeCandidate)
            .where(TypeCandidate.normalized_name.startswith(prefix))
            .where(TypeCandidate.status == TypeCandidateStatus.PROPOSED)
            .order_by(TypeCandidate.evidence_count.desc())
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def increment_evidence(self, candidate_id: UUID) -> TypeCandidate | None:
        """Increment evidence count for a candidate.

        Args:
            candidate_id: The candidate to update.

        Returns:
            The updated candidate, or None if not found.
        """
        candidate = await self.find_by_id(candidate_id)
        if candidate is None:
            return None

        # If merged, increment the canonical candidate instead
        if candidate.merged_into_candidate_id is not None:
            canonical = await self.resolve_canonical(candidate_id)
            if canonical:
                canonical.evidence_count += 1
                canonical.last_referenced_at = datetime.now()
                return canonical
            return None

        candidate.evidence_count += 1
        candidate.last_referenced_at = datetime.now()
        return candidate

    async def get_promotion_candidates(
        self,
        *,
        min_evidence_count: int = 5,
        min_coherence_score: float | None = None,
    ) -> list[TypeCandidate]:
        """Find candidates eligible for promotion to stable Type.

        Promotion thresholds (from design_v2_corrections.md §7):
        - Evidence: evidence_count >= 5 entities OR 20 observations
        - Coherence: cluster tightness score > 0.75 (optional for now)

        Args:
            min_evidence_count: Minimum evidence count for promotion.
            min_coherence_score: Minimum coherence score (if set).

        Returns:
            List of candidates eligible for promotion.
        """
        stmt = (
            select(TypeCandidate)
            .where(TypeCandidate.status == TypeCandidateStatus.PROPOSED)
            .where(TypeCandidate.evidence_count >= min_evidence_count)
        )

        if min_coherence_score is not None:
            stmt = stmt.where(TypeCandidate.coherence_score >= min_coherence_score)

        stmt = stmt.order_by(TypeCandidate.evidence_count.desc())

        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def count_by_status(self) -> dict[TypeCandidateStatus, int]:
        """Count candidates by status.

        Returns:
            Dict mapping status to count.
        """
        stmt = select(TypeCandidate.status, func.count(TypeCandidate.candidate_id)).group_by(
            TypeCandidate.status
        )
        result = await self._session.execute(stmt)

        counts: dict[TypeCandidateStatus, int] = {}
        for status, count in result.all():
            counts[status] = count

        return counts
