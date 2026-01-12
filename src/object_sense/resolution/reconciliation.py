"""Multi-seed reconciliation pass for entity resolution.

This module implements Step 5.5: Multi-seed consistency pass.

Strategy (per user spec - keep it simple for v0):
- Role conflicts: same entity as subject + context → posterior *= 0.5, flag
- Facet contradictions: same entity with incompatible facets → posterior *= 0.5, flag
- Duplicate links: same entity multiple times → keep max, downweight rest, flag
- NO smart fixing, auto-merge, or topology edits

See design_v2_corrections.md §5 for specifications.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, cast
from uuid import UUID

from object_sense.models.enums import LinkRole, LinkStatus


@dataclass
class PendingLink:
    """A link being considered during resolution.

    This is a mutable structure used during the resolution process,
    before links are persisted to the database.
    """

    entity_id: UUID
    posterior: float
    status: LinkStatus
    role: LinkRole | None
    flags: list[str] = field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    seed_index: int | None = None
    """Which entity_seed this link came from (for multi-seed tracking)."""

    facets: dict[str, Any] | None = None
    """Facets from the seed that generated this link (for contradiction detection)."""


@dataclass
class ReconciliationResult:
    """Result of multi-seed reconciliation."""

    links: list[PendingLink]
    """The reconciled links (may be modified from input)."""

    conflicts_detected: int
    """Number of conflicts detected and handled."""

    flags_added: int
    """Total number of flags added across all links."""


def reconcile_multi_seed_links(
    links: list[PendingLink],
    *,
    conflict_penalty: float = 0.5,
) -> ReconciliationResult:
    """Perform multi-seed consistency check across all links.

    This is a lightweight reconciliation pass that:
    1. Detects role conflicts (same entity as subject + context)
    2. Detects facet contradictions (same entity with incompatible facets)
    3. Detects duplicate links (same entity multiple times)

    It does NOT do heavy logic - just downweights + flags.

    Args:
        links: List of pending links from resolution.
        conflict_penalty: Multiplier for conflicting links (default 0.5).

    Returns:
        ReconciliationResult with reconciled links and conflict stats.
    """
    if not links:
        return ReconciliationResult(links=[], conflicts_detected=0, flags_added=0)

    # Group links by entity_id
    by_entity: dict[UUID, list[PendingLink]] = defaultdict(list)
    for link in links:
        by_entity[link.entity_id].append(link)

    conflicts_detected = 0
    flags_added = 0
    reconciled: list[PendingLink] = []

    for _entity_id, entity_links in by_entity.items():
        if len(entity_links) == 1:
            # Single link to this entity - no conflicts possible
            reconciled.append(entity_links[0])
            continue

        # Multiple links to same entity - check for conflicts

        # 1. Role conflicts: same entity as subject + context
        has_subject = any(l.role == LinkRole.SUBJECT for l in entity_links)
        has_context = any(l.role == LinkRole.CONTEXT for l in entity_links)

        if has_subject and has_context:
            conflicts_detected += 1
            for link in entity_links:
                link.posterior *= conflict_penalty
                if "multi_seed_role_conflict" not in link.flags:
                    link.flags.append("multi_seed_role_conflict")
                    flags_added += 1

        # 2. Facet contradictions: different seeds with incompatible facets
        if _has_facet_contradiction(entity_links):
            conflicts_detected += 1
            for link in entity_links:
                link.posterior *= conflict_penalty
                if "multi_seed_facet_conflict" not in link.flags:
                    link.flags.append("multi_seed_facet_conflict")
                    flags_added += 1

        # 3. Duplicate links: keep max-posterior, downweight rest
        if len(entity_links) > 1:
            # Sort by posterior descending
            sorted_links = sorted(entity_links, key=lambda l: l.posterior, reverse=True)
            best_link = sorted_links[0]

            # Add the best link
            reconciled.append(best_link)

            # Downweight and flag the rest
            for link in sorted_links[1:]:
                link.posterior *= conflict_penalty
                if "duplicate_entity_link" not in link.flags:
                    link.flags.append("duplicate_entity_link")
                    flags_added += 1
                reconciled.append(link)

            conflicts_detected += 1
        else:
            reconciled.extend(entity_links)

    return ReconciliationResult(
        links=reconciled,
        conflicts_detected=conflicts_detected,
        flags_added=flags_added,
    )


def _has_facet_contradiction(links: list[PendingLink]) -> bool:
    """Check if links have contradictory facets.

    A contradiction is when two seeds claim different values for the same facet key.

    Args:
        links: Links to check (all for the same entity).

    Returns:
        True if a facet contradiction is detected.
    """
    # Collect facets from each link's seed
    facet_values: dict[str, list[Any]] = defaultdict(list)

    for link in links:
        if link.facets:
            for key, value in link.facets.items():
                facet_values[key].append(value)

    # Check for contradictions (different values for same key)
    for key, values in facet_values.items():
        if len(values) > 1:
            # Check if values are actually different
            if not _all_values_compatible(values):
                return True

    return False


def _all_values_compatible(values: list[Any]) -> bool:
    """Check if all values in a list are compatible.

    For v0, this is simple equality check.
    Future: Handle structured values, references, etc.

    Args:
        values: List of facet values to compare.

    Returns:
        True if all values are compatible (same or equivalent).
    """
    if not values:
        return True

    first = values[0]
    for value in values[1:]:
        if not _values_equal(first, value):
            return False

    return True


def _values_equal(a: object, b: object) -> bool:
    """Check if two facet values are equal.

    Handles structured values like references.
    """
    if a == b:
        return True

    # Handle reference values
    if isinstance(a, dict) and isinstance(b, dict):
        a_dict = cast(dict[str, Any], a)
        b_dict = cast(dict[str, Any], b)
        # Entity references
        if "ref_entity_id" in a_dict and "ref_entity_id" in b_dict:
            return bool(a_dict["ref_entity_id"] == b_dict["ref_entity_id"])
        # Primitive values with metadata
        if "value" in a_dict and "value" in b_dict:
            return bool(a_dict["value"] == b_dict["value"])

    return False


def filter_low_posterior_links(
    links: list[PendingLink],
    threshold: float,
) -> list[PendingLink]:
    """Filter out links below a posterior threshold.

    This is useful after reconciliation to remove very low-confidence links.

    Args:
        links: Links to filter.
        threshold: Minimum posterior to keep.

    Returns:
        Links with posterior >= threshold.
    """
    return [link for link in links if link.posterior >= threshold]


def deduplicate_links(links: list[PendingLink]) -> list[PendingLink]:
    """Remove duplicate links to the same entity, keeping highest posterior.

    This is a final cleanup step to ensure each entity appears at most once.

    Args:
        links: Links to deduplicate.

    Returns:
        Deduplicated links.
    """
    by_entity: dict[UUID, PendingLink] = {}

    for link in links:
        existing = by_entity.get(link.entity_id)
        if existing is None or link.posterior > existing.posterior:
            by_entity[link.entity_id] = link

    return list(by_entity.values())
