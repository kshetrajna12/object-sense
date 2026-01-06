"""CLI for ObjectSense.

Commands:
    ingest <path>            - Ingest files/directories
    show-observation <id>    - Show observation details
    show-type <name>         - Show type details
    show-entity <id>         - Show entity details
    review-types             - List all types in the system
    search <query>           - Search observations by query
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated
from uuid import UUID, uuid4

# Configure logging from environment variable
_log_level = os.environ.get("LOG_LEVEL", "WARNING").upper()
logging.basicConfig(
    level=getattr(logging, _log_level, logging.WARNING),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from object_sense.db import async_session_factory, engine, init_db
from object_sense.extraction.base import ExtractedId
from object_sense.extraction.orchestrator import ExtractionOrchestrator
from object_sense.inference.schemas import DeterministicId, EntityHypothesis, TypeProposal
from object_sense.inference.type_inference import TypeInferenceAgent
from object_sense.models import (
    Blob,
    Entity,
    EntityNature,
    Evidence,
    EvidenceSource,
    Medium,
    Observation,
    ObservationStatus,
    Signature,
    SubjectKind,
    Type,
    TypeCreatedVia,
    TypeStatus,
)
from object_sense.resolution.resolver import EntityResolver
from object_sense.services.type_candidate import TypeCandidateService
from object_sense.utils.medium import probe_medium

app = typer.Typer(
    name="object-sense",
    help="ObjectSense — semantic substrate for persistent object identity and type awareness",
    no_args_is_help=True,
)
console = Console()

# Supported file extensions for ingestion
SUPPORTED_EXTENSIONS = {
    # Standard images
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".tif",
    # RAW image formats (camera-specific)
    ".raw", ".arw", ".cr2", ".cr3", ".nef", ".nrw",  # Sony, Canon, Nikon
    ".orf", ".rw2", ".pef", ".srw", ".x3f",  # Olympus, Panasonic, Pentax, Samsung, Sigma
    ".raf", ".dng", ".dcr", ".kdc", ".mrw",  # Fuji, Adobe DNG, Kodak, Minolta
    ".3fr", ".mef", ".mos", ".erf", ".rwl",  # Hasselblad, Mamiya, Leaf, Epson, Leica
    # Text
    ".txt", ".md", ".rst", ".csv",
    # JSON
    ".json",
}


def run_async(coro):
    """Run an async coroutine in sync context."""
    return asyncio.run(coro)


@dataclass
class NamespaceOverride:
    """Record of a namespace override for Evidence tracking."""

    id_type: str
    id_value: str
    original_namespace: str | None
    resolved_namespace: str
    reason: str  # "id_type_mapped", "invalid_pattern", "empty_namespace"


@dataclass
class NormalizedIds:
    """Result of normalize_deterministic_ids with override tracking."""

    ids: list[DeterministicId]
    overrides: list[NamespaceOverride]


# Legacy alias for backwards compatibility
UnionResult = NormalizedIds


def _is_valid_namespace(namespace: str) -> bool:
    """Check if namespace matches allowed patterns."""
    from object_sense.config import settings

    for pattern in settings.namespace_patterns:
        if re.match(pattern, namespace):
            return True
    return False


def _resolve_namespace(
    id_type: str,
    proposed_namespace: str | None,
    context_namespace: str,
) -> tuple[str, str | None]:
    """Resolve the canonical namespace for a deterministic ID.

    Returns (resolved_namespace, override_reason or None).

    Resolution order:
    1. ID type → namespace mapping (engine-controlled, always wins)
    2. Valid proposed namespace (matches allowed patterns)
    3. Fallback to context namespace
    """
    from object_sense.config import settings

    # 1. ID type mapping takes precedence (engine-controlled)
    if id_type in settings.id_type_namespace_map:
        mapped = settings.id_type_namespace_map[id_type]
        if proposed_namespace and proposed_namespace != mapped:
            return mapped, "id_type_mapped"
        return mapped, None

    # 2. Check if proposed namespace is valid
    if proposed_namespace and _is_valid_namespace(proposed_namespace):
        return proposed_namespace, None

    # 3. Override to context namespace
    reason = "invalid_pattern" if proposed_namespace else "empty_namespace"
    return context_namespace, reason


def normalize_deterministic_ids(
    ids: list[DeterministicId] | list[ExtractedId],
    context_namespace: str,
) -> NormalizedIds:
    """Normalize namespace for deterministic IDs from any source.

    This is the SINGLE function for namespace enforcement (bead object-sense-bk9).
    Apply it to:
    - extracted observation deterministic_ids
    - LLM observation deterministic_ids
    - each entity_seed.deterministic_ids

    Rules:
    1. if id_type in id_type_namespace_map → set namespace accordingly
    2. else if namespace missing/invalid → set to context_namespace
    3. if LLM provided invalid namespace → override + record NamespaceOverride

    Args:
        ids: List of DeterministicId or ExtractedId objects
        context_namespace: Fallback namespace (e.g., "source:product_catalog")

    Returns:
        NormalizedIds with normalized IDs (as DeterministicId) and override events.
    """
    result: list[DeterministicId] = []
    overrides: list[NamespaceOverride] = []

    for id_obj in ids:
        # Handle both DeterministicId and ExtractedId
        id_type = id_obj.id_type
        id_value = id_obj.id_value
        original_namespace = id_obj.id_namespace
        strength = id_obj.strength if hasattr(id_obj, "strength") else "strong"

        namespace, reason = _resolve_namespace(id_type, original_namespace, context_namespace)

        # Create normalized DeterministicId
        normalized = DeterministicId(
            id_type=id_type,
            id_value=id_value,
            id_namespace=namespace,
            strength=strength if strength in ("strong", "weak") else "strong",
        )
        result.append(normalized)

        if reason:
            overrides.append(NamespaceOverride(
                id_type=id_type,
                id_value=id_value,
                original_namespace=original_namespace,
                resolved_namespace=namespace,
                reason=reason,
            ))

    return NormalizedIds(ids=result, overrides=overrides)


def _union_deterministic_ids(
    extracted_ids: list[ExtractedId],
    llm_ids: list[DeterministicId],
    context_namespace: str,
) -> NormalizedIds:
    """Union deterministic IDs from extraction and LLM, with namespace enforcement.

    Engine controls namespaces (bead object-sense-bk9):
    - ID type mappings take precedence (gps→geo:wgs84, upc→global:upc)
    - Invalid LLM namespaces are overridden to context_namespace
    - Overrides are tracked for Evidence recording

    Args:
        extracted_ids: IDs from extraction (Step 3)
        llm_ids: IDs from LLM (Step 4)
        context_namespace: Fallback namespace (e.g., "source:product_catalog")

    Returns:
        NormalizedIds with merged, deduplicated IDs and namespace override events.
    """
    # Normalize both sources
    extracted_result = normalize_deterministic_ids(extracted_ids, context_namespace)
    llm_result = normalize_deterministic_ids(llm_ids, context_namespace)

    # Deduplicate by (id_type, id_value, id_namespace) tuple
    seen: set[tuple[str, str, str]] = set()
    merged: list[DeterministicId] = []

    # Add extracted first (baseline)
    for det_id in extracted_result.ids:
        assert det_id.id_namespace is not None, "Normalized ID must have namespace"
        key = (det_id.id_type, det_id.id_value, det_id.id_namespace)
        if key not in seen:
            seen.add(key)
            merged.append(det_id)

    # Add LLM IDs (supplementary)
    for det_id in llm_result.ids:
        assert det_id.id_namespace is not None, "Normalized ID must have namespace"
        key = (det_id.id_type, det_id.id_value, det_id.id_namespace)
        if key not in seen:
            seen.add(key)
            merged.append(det_id)

    # Combine overrides
    all_overrides = extracted_result.overrides + llm_result.overrides

    return NormalizedIds(ids=merged, overrides=all_overrides)


def _extracted_to_entity_hypothesis(extracted_ids: list[ExtractedId]) -> list[EntityHypothesis]:
    """Convert extracted deterministic IDs to minimal entity hypotheses.

    Used when Step 4 (LLM) fails but we have extracted IDs to resolve.
    Creates one hypothesis per ID with deterministic linking.
    """
    from object_sense.models.enums import EntityNature

    hypotheses: list[EntityHypothesis] = []
    for eid in extracted_ids:
        # Convert ExtractedId to DeterministicId for the hypothesis
        det_id = DeterministicId(
            id_type=eid.id_type,
            id_value=eid.id_value,
            id_namespace=eid.id_namespace,
            strength="strong" if eid.strength == "strong" else "weak",
        )
        hyp = EntityHypothesis(
            entity_type=f"{eid.id_type}_entity",
            entity_nature=EntityNature.INDIVIDUAL,
            suggested_name=f"{eid.id_type}:{eid.id_value}",
            deterministic_ids=[det_id],
            confidence=1.0,  # Deterministic IDs are high confidence
            reasoning=f"Entity anchored by extracted {eid.id_type}",
        )
        hypotheses.append(hyp)
    return hypotheses


def _normalize_entity_seed_namespaces(
    entity_seeds: list[EntityHypothesis],
    context_namespace: str,
) -> tuple[list[EntityHypothesis], list[NamespaceOverride]]:
    """Apply namespace enforcement to entity_seeds' deterministic_ids.

    LLM-produced entity_seeds may have deterministic_ids with invalid or missing
    namespaces. Uses the consolidated normalize_deterministic_ids function.

    Returns (normalized_seeds, overrides) where overrides are for Evidence.
    """
    normalized: list[EntityHypothesis] = []
    all_overrides: list[NamespaceOverride] = []

    for seed in entity_seeds:
        if not seed.deterministic_ids:
            normalized.append(seed)
            continue

        # Use consolidated normalization function
        result = normalize_deterministic_ids(seed.deterministic_ids, context_namespace)
        all_overrides.extend(result.overrides)

        # Create new EntityHypothesis with normalized IDs
        new_seed = EntityHypothesis(
            entity_type=seed.entity_type,
            entity_nature=seed.entity_nature,
            suggested_name=seed.suggested_name,
            slots=seed.slots,
            deterministic_ids=result.ids,
            confidence=seed.confidence,
            reasoning=seed.reasoning,
        )
        normalized.append(new_seed)

    return normalized, all_overrides


async def get_or_create_type(session, type_name: str, created_via: TypeCreatedVia) -> Type:
    """Get existing type or create new one."""
    stmt = select(Type).where(Type.canonical_name == type_name)
    result = await session.execute(stmt)
    existing = result.scalar_one_or_none()

    if existing:
        return existing

    new_type = Type(
        type_id=uuid4(),
        canonical_name=type_name,
        created_via=created_via,
        status=TypeStatus.PROVISIONAL,
        evidence_count=0,
    )
    session.add(new_type)
    return new_type


async def ingest_file(
    file_path: Path,
    verbose: bool = False,
    id_namespace: str | None = None,
) -> dict:
    """Ingest a single file through the full pipeline.

    Pipeline:
    1. Blob dedup check
    2. Medium probing
    3. Feature extraction (produces deterministic_ids from content)
    4. Type inference (LLM - may fail, but pipeline continues)
    5. Entity resolution (uses extracted + LLM deterministic_ids)

    All writes occur in a single transaction - rollback leaves zero side-effects.

    Args:
        file_path: Path to the file to ingest.
        verbose: If True, log detailed output.
        id_namespace: Explicit namespace for deterministic IDs (e.g., "source:product_catalog").
                      If not provided, uses "source:ingest" as default.

    Returns dict with observation_id, type, medium, and status.
    """
    import logging

    logger = logging.getLogger(__name__)

    # Derive context namespace (bead object-sense-bk9)
    # Explicit id_namespace takes precedence, otherwise use default
    context_namespace = id_namespace or "source:ingest"

    # Read file content
    content = file_path.read_bytes()
    sha256 = hashlib.sha256(content).hexdigest()

    async with async_session_factory() as session:
        # Check for duplicate blob
        stmt = select(Blob).where(Blob.sha256 == sha256)
        result = await session.execute(stmt)
        existing_blob = result.scalar_one_or_none()

        if existing_blob:
            # Check if we already have an observation for this blob + source
            stmt = select(Observation).where(
                Observation.blob_id == existing_blob.blob_id,
                Observation.source_id == str(file_path.absolute()),
            )
            result = await session.execute(stmt)
            existing_obs = result.scalar_one_or_none()
            if existing_obs:
                return {
                    "observation_id": str(existing_obs.observation_id),
                    "type": existing_obs.stable_type_id,
                    "medium": existing_obs.medium.value,
                    "status": "duplicate",
                    "message": "Observation already exists",
                }

        # Step 2: Medium probing
        medium = probe_medium(content, filename=file_path.name)

        # Step 3: Feature extraction (produces deterministic_ids from content)
        orchestrator = ExtractionOrchestrator()
        extraction_result = await orchestrator.extract(
            content, medium=medium, filename=file_path.name
        )

        # Step 4: Type inference (LLM) - may fail, but pipeline continues
        type_proposal: TypeProposal | None = None
        llm_failed = False
        try:
            inference_agent = TypeInferenceAgent()
            type_proposal = await inference_agent.infer(extraction_result, medium=medium.value)
        except Exception as e:
            llm_failed = True
            logger.warning("Step 4 (LLM type inference) failed: %s", e)

        # Step 4B: Engine resolves type candidate (LLM proposes, engine decides)
        type_candidate_service = TypeCandidateService(session)
        type_candidate = None
        type_name: str | None = None
        observation_kind = "unknown"
        facets: dict[str, object] = {}
        reasoning = ""

        if type_proposal and type_proposal.type_candidate:
            # Engine creates or matches a TypeCandidate (dedup via normalized name)
            type_candidate, _is_new = await type_candidate_service.get_or_create(
                type_proposal.type_candidate.proposed_name,
                details={
                    "rationale": type_proposal.type_candidate.rationale,
                    "suggested_parent": type_proposal.type_candidate.suggested_parent,
                    "confidence": type_proposal.type_candidate.confidence,
                },
            )
            type_name = type_candidate.proposed_name
            observation_kind = type_proposal.observation_kind
            facets = type_proposal.facets
            reasoning = type_proposal.reasoning
        elif type_proposal:
            # Type proposal exists but no type_candidate - use observation_kind
            type_name = type_proposal.observation_kind
            observation_kind = type_proposal.observation_kind
            facets = type_proposal.facets
            reasoning = type_proposal.reasoning

        # Merge GPS/datetime from extraction into facets (engine-owned, not LLM-dependent)
        # This ensures context signals are always available for entity resolution.
        if extraction_result.extra:
            extra = extraction_result.extra
            # GPS coordinates
            if "gps_latitude" in extra and "latitude" not in facets:
                facets["latitude"] = extra["gps_latitude"]
            if "gps_longitude" in extra and "longitude" not in facets:
                facets["longitude"] = extra["gps_longitude"]
            if "gps_altitude" in extra and "altitude" not in facets:
                facets["altitude"] = extra["gps_altitude"]
            # Datetime from EXIF (nested in exif dict)
            # Prefer datetimeoriginal (capture time) over datetime (modification time)
            exif = extra.get("exif", {})
            if exif and "datetime" not in facets:
                # EXIF datetime formats: 'YYYY:MM:DD HH:MM:SS' → 'YYYY-MM-DDTHH:MM:SSZ'
                raw_dt = exif.get("datetimeoriginal") or exif.get("datetime")
                if raw_dt and isinstance(raw_dt, str):
                    try:
                        # Convert EXIF datetime format to ISO8601
                        facets["datetime"] = raw_dt.replace(":", "-", 2).replace(" ", "T") + "Z"
                    except Exception:
                        pass
            # Also try direct filename extraction (backup)
            if "filename" not in facets and "filename" in extra:
                facets["filename"] = extra["filename"]

        # Union deterministic_ids from extraction + LLM with namespace enforcement
        llm_det_ids = type_proposal.deterministic_ids if type_proposal else []
        union_result = _union_deterministic_ids(
            extraction_result.deterministic_ids,
            llm_det_ids,
            context_namespace,
        )
        # Convert to dicts for JSONB storage
        deterministic_ids = [
            {
                "id_type": det_id.id_type,
                "id_value": det_id.id_value,
                "id_namespace": det_id.id_namespace,
            }
            for det_id in union_result.ids
        ]

        # Create blob if needed
        if existing_blob:
            blob = existing_blob
        else:
            blob = Blob(
                blob_id=uuid4(),
                sha256=sha256,
                size_bytes=len(content),
                storage_path=str(file_path.absolute()),
            )
            session.add(blob)

        # Create observation
        observation_id = uuid4()
        obs = Observation(
            observation_id=observation_id,
            medium=medium,
            # Link to TypeCandidate, NOT stable Type directly.
            # stable_type_id is set later when TypeCandidate is promoted.
            candidate_type_id=type_candidate.candidate_id if type_candidate else None,
            stable_type_id=None,  # Set via promotion, not LLM binding
            source_id=str(file_path.absolute()),
            blob_id=blob.blob_id,
            slots={},  # Slots are now per-entity in entity_seeds
            observation_kind=observation_kind,  # Routing hint
            facets=facets,  # Extracted attributes
            deterministic_ids=deterministic_ids,  # Union of extracted + LLM
            status=ObservationStatus.ACTIVE,
        )
        session.add(obs)

        # Store signatures and collect for resolver
        signatures_created: list[Signature] = []

        if extraction_result.hash_value:
            hash_sig = Signature(
                signature_id=uuid4(),
                observation_id=observation_id,
                signature_type=extraction_result.signature_type,
                hash_value=extraction_result.hash_value,
            )
            session.add(hash_sig)
            signatures_created.append(hash_sig)

        if extraction_result.text_embedding:
            sig = Signature(
                signature_id=uuid4(),
                observation_id=observation_id,
                signature_type="text_embedding",
                text_embedding=extraction_result.text_embedding,
            )
            session.add(sig)
            signatures_created.append(sig)

        if extraction_result.image_embedding:
            sig = Signature(
                signature_id=uuid4(),
                observation_id=observation_id,
                signature_type="image_embedding",
                image_embedding=extraction_result.image_embedding,
            )
            session.add(sig)
            signatures_created.append(sig)

        # Store evidence for type candidate assignment
        if type_candidate and type_proposal and type_proposal.type_candidate:
            evidence = Evidence(
                evidence_id=uuid4(),
                subject_kind=SubjectKind.OBSERVATION,
                subject_id=observation_id,
                predicate="has_type_candidate",
                target_id=type_candidate.candidate_id,
                source=EvidenceSource.LLM,
                score=type_proposal.type_candidate.confidence,
                details={
                    "reasoning": reasoning,
                    "proposed_name": type_candidate.proposed_name,
                    "normalized_name": type_candidate.normalized_name,
                },
            )
            session.add(evidence)

        # Record evidence for namespace overrides (bead object-sense-bk9)
        for override in union_result.overrides:
            evidence = Evidence(
                evidence_id=uuid4(),
                subject_kind=SubjectKind.OBSERVATION,
                subject_id=observation_id,
                predicate="namespace_overridden",
                target_id=None,
                source=EvidenceSource.SYSTEM,
                score=1.0,  # System decisions are definitive
                details={
                    "id_type": override.id_type,
                    "id_value": override.id_value,
                    "original_namespace": override.original_namespace,
                    "resolved_namespace": override.resolved_namespace,
                    "reason": override.reason,
                },
            )
            session.add(evidence)
            if verbose:
                logger.info(
                    "Namespace override: %s:%s from '%s' to '%s' (%s)",
                    override.id_type,
                    override.id_value,
                    override.original_namespace,
                    override.resolved_namespace,
                    override.reason,
                )

        # Flush to ensure observation and signatures have IDs before resolver
        await session.flush()

        # Step 5: Entity resolution
        # Determine entity_seeds: from LLM if available, else from extracted IDs
        entity_seeds: list[EntityHypothesis] = []
        if type_proposal and type_proposal.entity_seeds:
            entity_seeds = list(type_proposal.entity_seeds)  # Copy to allow mutation
        elif extraction_result.deterministic_ids:
            # LLM failed but we have extracted IDs - create minimal hypotheses
            entity_seeds = _extracted_to_entity_hypothesis(extraction_result.deterministic_ids)

        # ENGINE GUARDRAIL: Strip GPS from ALL deterministic IDs
        # GPS should NEVER be a deterministic ID - it causes false splits due to
        # inherent imprecision (~5-10m). GPS is a weak locality prior only.
        # See similarity.py docstring for contextual signal policy.
        for seed in entity_seeds:
            if seed.deterministic_ids:
                original_count = len(seed.deterministic_ids)
                seed.deterministic_ids = [
                    did for did in seed.deterministic_ids
                    if did.id_type.lower() != "gps"
                ]
                if len(seed.deterministic_ids) < original_count:
                    # Record that we stripped GPS
                    evidence = Evidence(
                        evidence_id=uuid4(),
                        subject_kind=SubjectKind.OBSERVATION,
                        subject_id=observation_id,
                        predicate="gps_id_stripped",
                        target_id=None,
                        source=EvidenceSource.SYSTEM,
                        score=1.0,
                        details={
                            "seed_entity_type": seed.entity_type,
                            "reason": "GPS is a weak prior, not a deterministic ID",
                        },
                    )
                    session.add(evidence)

        # ENGINE-OWNED SEED: Ensure image observations have a photo_subject entity
        # for visual re-ID. This breaks the prototype deadlock by seeding image
        # prototypes. IMPORTANT: Check for entity_type="photo_subject" specifically,
        # not just any INDIVIDUAL (the LLM may propose camera/location as individual).
        # The seed has NO deterministic IDs - pure similarity-based resolution.
        if medium == Medium.IMAGE:
            has_photo_subject = any(
                seed.entity_type == "photo_subject"
                for seed in entity_seeds
            )
            if not has_photo_subject:
                engine_subject_seed = EntityHypothesis(
                    entity_type="photo_subject",
                    entity_nature=EntityNature.INDIVIDUAL,
                    suggested_name=None,  # Let engine determine via clustering
                    slots=[],
                    deterministic_ids=[],  # CRITICAL: No deterministic IDs
                    confidence=0.6,  # Lower confidence - engine-generated
                    reasoning="Engine-generated primary subject seed for visual re-ID",
                )
                entity_seeds.append(engine_subject_seed)

        resolution_result = None
        entities_created = 0
        links_created = 0

        if entity_seeds:
            # Normalize entity_seed namespaces before resolution
            entity_seeds, seed_overrides = _normalize_entity_seed_namespaces(
                entity_seeds, context_namespace
            )

            # Record evidence for entity seed namespace overrides
            for override in seed_overrides:
                evidence = Evidence(
                    evidence_id=uuid4(),
                    subject_kind=SubjectKind.OBSERVATION,
                    subject_id=observation_id,
                    predicate="namespace_overridden",
                    target_id=None,
                    source=EvidenceSource.SYSTEM,
                    score=1.0,
                    details={
                        "id_type": override.id_type,
                        "id_value": override.id_value,
                        "original_namespace": override.original_namespace,
                        "resolved_namespace": override.resolved_namespace,
                        "reason": override.reason,
                        "source": "entity_seed",
                    },
                )
                session.add(evidence)

            resolver = EntityResolver(session)
            resolution_result = await resolver.resolve(
                observation=obs,
                signatures=signatures_created,
                entity_seeds=entity_seeds,
            )
            entities_created = len(resolution_result.entities_created)
            links_created = len(resolution_result.links)

        # Single commit for entire transaction
        await session.commit()

        result_dict = {
            "observation_id": str(observation_id),
            "type_candidate": type_name,
            "type_status": "candidate" if type_candidate else "unknown",
            "medium": medium.value,
            "status": "ingested",
            "facets": facets,
            "reasoning": reasoning,
            "entities_created": entities_created,
            "links_created": links_created,
        }

        if llm_failed:
            result_dict["llm_failed"] = True
            result_dict["fallback_resolution"] = bool(extraction_result.deterministic_ids)

        return result_dict


@app.command()
def ingest(
    path: Annotated[Path, typer.Argument(help="File or directory to ingest")],
    recursive: Annotated[
        bool, typer.Option("--recursive", "-r", help="Recursively ingest directories")
    ] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show detailed output")
    ] = False,
    id_namespace: Annotated[
        str | None,
        typer.Option(
            "--id-namespace",
            help=(
                "Namespace for deterministic IDs (e.g., 'source:product_catalog'). "
                "Controls identity resolution scope. Defaults to 'source:ingest'."
            ),
        ),
    ] = None,
):
    """Ingest files into ObjectSense.

    Runs the full pipeline: probe medium → extract features → infer type → resolve entities.
    """
    async def _ingest():
        # Initialize database
        await init_db()

        files_to_process: list[Path] = []

        if path.is_file():
            files_to_process.append(path)
        elif path.is_dir():
            pattern = "**/*" if recursive else "*"
            for f in path.glob(pattern):
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
                    files_to_process.append(f)
        else:
            console.print(f"[red]Error:[/red] Path does not exist: {path}")
            raise typer.Exit(1)

        if not files_to_process:
            console.print("[yellow]No supported files found to ingest.[/yellow]")
            raise typer.Exit(0)

        console.print(f"[blue]Ingesting {len(files_to_process)} file(s)...[/blue]\n")

        results = []
        for file_path in files_to_process:
            try:
                console.print(f"  Processing: {file_path.name}...", end=" ")
                result = await ingest_file(file_path, verbose=verbose, id_namespace=id_namespace)
                results.append(result)

                if result["status"] == "duplicate":
                    console.print("[yellow]SKIP[/yellow] (duplicate)")
                else:
                    type_display = result.get("type_candidate") or "unknown"
                    entity_info = ""
                    if result.get("entities_created", 0) > 0:
                        entity_info = f" (+{result['entities_created']} entities)"
                    console.print(f"[green]OK[/green] → {type_display}{entity_info}")

                if verbose and result["status"] != "duplicate":
                    console.print(f"    ID: {result['observation_id']}")
                    console.print(f"    Medium: {result['medium']}")
                    if result.get("slots"):
                        console.print(f"    Slots: {result['slots']}")

            except Exception as e:
                console.print(f"[red]ERROR[/red]: {e}")
                if verbose:
                    import traceback
                    console.print(traceback.format_exc())

        # Summary
        console.print()
        ingested = sum(1 for r in results if r["status"] == "ingested")
        duplicates = sum(1 for r in results if r["status"] == "duplicate")
        console.print(f"[bold]Summary:[/bold] {ingested} ingested, {duplicates} duplicates")

    run_async(_ingest())


@app.command("show-observation")
def show_observation(
    observation_id: Annotated[str, typer.Argument(help="Observation ID (full UUID or prefix)")],
):
    """Show details for a specific observation."""
    async def _show():
        await init_db()
        async with async_session_factory() as session:
            # Try exact UUID first, then prefix match
            obs = None
            try:
                oid = UUID(observation_id)
                stmt = (
                    select(Observation)
                    .options(
                        selectinload(Observation.stable_type),
                        selectinload(Observation.blob),
                        selectinload(Observation.entity_links),
                        selectinload(Observation.signatures),
                    )
                    .where(Observation.observation_id == oid)
                )
                result = await session.execute(stmt)
                obs = result.scalar_one_or_none()
            except ValueError:
                # Not a valid UUID, try prefix match
                pass

            if not obs:
                # Prefix match on UUID as text
                from sqlalchemy import String, cast
                stmt = (
                    select(Observation)
                    .options(
                        selectinload(Observation.stable_type),
                        selectinload(Observation.blob),
                        selectinload(Observation.entity_links),
                        selectinload(Observation.signatures),
                    )
                    .where(cast(Observation.observation_id, String).startswith(observation_id))
                )
                result = await session.execute(stmt)
                matches = result.scalars().all()

                if len(matches) == 0:
                    console.print(
                        f"[red]Error:[/red] No observation found matching: {observation_id}"
                    )
                    raise typer.Exit(1)
                elif len(matches) > 1:
                    console.print(
                        f"[red]Error:[/red] Ambiguous prefix '{observation_id}' "
                        f"matches {len(matches)} observations:"
                    )
                    for m in matches[:5]:
                        console.print(f"  • {m.observation_id}")
                    raise typer.Exit(1)
                obs = matches[0]

            if not obs:
                console.print(f"[red]Error:[/red] Observation not found: {observation_id}")
                raise typer.Exit(1)

            # Build display
            panel_content = []
            panel_content.append(f"[bold]ID:[/bold] {obs.observation_id}")
            panel_content.append(f"[bold]Medium:[/bold] {obs.medium.value}")
            panel_content.append(f"[bold]Status:[/bold] {obs.status.value}")
            panel_content.append(f"[bold]Source:[/bold] {obs.source_id}")

            if obs.stable_type:
                panel_content.append(f"[bold]Type:[/bold] {obs.stable_type.canonical_name}")

            if obs.blob:
                panel_content.append(f"[bold]SHA256:[/bold] {obs.blob.sha256[:16]}...")
                panel_content.append(f"[bold]Size:[/bold] {obs.blob.size_bytes:,} bytes")

            if obs.slots:
                panel_content.append("[bold]Slots:[/bold]")
                for k, v in obs.slots.items():
                    panel_content.append(f"  • {k}: {v}")

            panel_content.append(f"[bold]Created:[/bold] {obs.created_at}")

            console.print(Panel("\n".join(panel_content), title="Observation Details"))

            # Show signatures
            if obs.signatures:
                table = Table(title="Signatures")
                table.add_column("Type")
                table.add_column("Value")
                for sig in obs.signatures:
                    if sig.hash_value:
                        val = sig.hash_value
                    elif sig.text_embedding is not None:
                        val = f"text embedding ({len(sig.text_embedding)}d)"
                    elif sig.image_embedding is not None:
                        val = f"image embedding ({len(sig.image_embedding)}d)"
                    elif sig.clip_text_embedding is not None:
                        val = f"CLIP text embedding ({len(sig.clip_text_embedding)}d)"
                    else:
                        val = "-"
                    val_str = str(val)
                    display = val_str[:32] + "..." if len(val_str) > 32 else val_str
                    table.add_row(sig.signature_type, display)
                console.print(table)

            # Show entity links
            if obs.entity_links:
                # Fetch full entity details for display
                entity_ids = [link.entity_id for link in obs.entity_links]
                entities_stmt = select(Entity).where(Entity.entity_id.in_(entity_ids))
                entities_result = await session.execute(entities_stmt)
                entities_by_id = {e.entity_id: e for e in entities_result.scalars()}

                table = Table(title="Entity Links")
                table.add_column("Entity ID", style="cyan")
                table.add_column("Nature")
                table.add_column("Name")
                table.add_column("Status")
                table.add_column("Link")
                table.add_column("Posterior")
                table.add_column("Canonical")

                for link in obs.entity_links:
                    entity = entities_by_id.get(link.entity_id)
                    # Safe access to enum values
                    link_status = getattr(link.status, "value", str(link.status))
                    if entity:
                        entity_nature = getattr(entity.entity_nature, "value", str(entity.entity_nature))
                        entity_status = getattr(entity.status, "value", str(entity.status))
                        canonical = (
                            str(entity.canonical_entity_id)[:8] + "..."
                            if entity.canonical_entity_id
                            else "-"
                        )
                        table.add_row(
                            str(link.entity_id)[:8] + "...",
                            entity_nature,
                            entity.name or "-",
                            entity_status,
                            link_status,
                            f"{link.posterior:.2f}",
                            canonical,
                        )
                    else:
                        table.add_row(
                            str(link.entity_id)[:8] + "...",
                            "?",
                            "?",
                            "?",
                            link_status,
                            f"{link.posterior:.2f}",
                            "-",
                        )
                console.print(table)
            else:
                console.print("[dim]No entity links[/dim]")

    run_async(_show())


@app.command("show-type")
def show_type(
    type_name: Annotated[str, typer.Argument(help="Type canonical name")],
):
    """Show details for a specific type."""
    async def _show():
        await init_db()
        async with async_session_factory() as session:
            stmt = (
                select(Type)
                .options(selectinload(Type.observations), selectinload(Type.entities))
                .where(Type.canonical_name == type_name)
            )
            result = await session.execute(stmt)
            t = result.scalar_one_or_none()

            if not t:
                console.print(f"[red]Error:[/red] Type not found: {type_name}")
                raise typer.Exit(1)

            panel_content = []
            panel_content.append(f"[bold]ID:[/bold] {t.type_id}")
            panel_content.append(f"[bold]Name:[/bold] {t.canonical_name}")
            panel_content.append(f"[bold]Status:[/bold] {t.status.value}")
            panel_content.append(f"[bold]Created Via:[/bold] {t.created_via.value}")
            panel_content.append(f"[bold]Evidence Count:[/bold] {t.evidence_count}")

            if t.aliases:
                panel_content.append(f"[bold]Aliases:[/bold] {', '.join(t.aliases)}")

            panel_content.append(f"[bold]Observations:[/bold] {len(t.observations)}")
            panel_content.append(f"[bold]Entities:[/bold] {len(t.entities)}")
            panel_content.append(f"[bold]Created:[/bold] {t.created_at}")

            console.print(Panel("\n".join(panel_content), title=f"Type: {type_name}"))

            # Show recent observations of this type
            if t.observations:
                table = Table(title="Recent Observations")
                table.add_column("ID")
                table.add_column("Medium")
                table.add_column("Source")
                for obs in t.observations[:5]:
                    source = obs.source_id.split("/")[-1] if "/" in obs.source_id else obs.source_id
                    table.add_row(
                        str(obs.observation_id)[:8] + "...",
                        obs.medium.value,
                        source[:40] + "..." if len(source) > 40 else source,
                    )
                console.print(table)

    run_async(_show())


@app.command("show-entity")
def show_entity(
    entity_id: Annotated[str, typer.Argument(help="Entity ID (UUID)")],
):
    """Show details for a specific entity."""
    async def _show():
        await init_db()
        try:
            eid = UUID(entity_id)
        except ValueError:
            console.print(f"[red]Error:[/red] Invalid UUID: {entity_id}")
            raise typer.Exit(1) from None

        async with async_session_factory() as session:
            stmt = (
                select(Entity)
                .options(selectinload(Entity.type), selectinload(Entity.observation_links))
                .where(Entity.entity_id == eid)
            )
            result = await session.execute(stmt)
            entity = result.scalar_one_or_none()

            if not entity:
                console.print(f"[red]Error:[/red] Entity not found: {entity_id}")
                raise typer.Exit(1)

            panel_content = []
            panel_content.append(f"[bold]ID:[/bold] {entity.entity_id}")
            nature_str = entity.entity_nature.value if entity.entity_nature else "unknown"
            panel_content.append(f"[bold]Nature:[/bold] {nature_str}")
            panel_content.append(f"[bold]Status:[/bold] {entity.status.value}")
            panel_content.append(f"[bold]Confidence:[/bold] {entity.confidence:.2f}")
            if entity.name:
                panel_content.append(f"[bold]Name:[/bold] {entity.name}")

            if entity.type:
                panel_content.append(f"[bold]Type:[/bold] {entity.type.canonical_name}")

            # Prototype status - critical for debugging re-ID
            panel_content.append("")
            panel_content.append("[bold]Prototype Status:[/bold]")
            panel_content.append(f"  • prototype_count: {entity.prototype_count}")
            has_img = entity.prototype_image_embedding is not None
            has_txt = entity.prototype_text_embedding is not None
            panel_content.append(f"  • has_image_prototype: {has_img}")
            panel_content.append(f"  • has_text_prototype: {has_txt}")
            if not has_img and not has_txt:
                panel_content.append("  [yellow]⚠ No prototypes - entity cannot be found via ANN[/yellow]")

            if entity.slots:
                panel_content.append("")
                panel_content.append("[bold]Slots:[/bold]")
                for k, v in entity.slots.items():
                    panel_content.append(f"  • {k}: {v}")

            linked_count = len(entity.observation_links)
            panel_content.append(f"\n[bold]Linked Observations:[/bold] {linked_count}")
            panel_content.append(f"[bold]Created:[/bold] {entity.created_at}")

            console.print(Panel("\n".join(panel_content), title="Entity Details"))

    run_async(_show())


@app.command("review-types")
def review_types(
    status: Annotated[
        str | None, typer.Option(help="Filter by status (provisional, stable, deprecated)")
    ] = None,
    limit: Annotated[int, typer.Option(help="Maximum number of types to show")] = 20,
):
    """Review types in the system."""
    async def _review():
        await init_db()
        async with async_session_factory() as session:
            stmt = select(Type).options(selectinload(Type.observations))

            if status:
                try:
                    status_enum = TypeStatus(status)
                    stmt = stmt.where(Type.status == status_enum)
                except ValueError:
                    console.print(
                        "[red]Error:[/red] Invalid status. "
                        "Use: provisional, stable, deprecated, merged_into"
                    )
                    raise typer.Exit(1) from None

            stmt = stmt.order_by(Type.evidence_count.desc()).limit(limit)
            result = await session.execute(stmt)
            types = result.scalars().all()

            if not types:
                console.print("[yellow]No types found.[/yellow]")
                return

            table = Table(title="Types")
            table.add_column("Name", style="cyan")
            table.add_column("Status")
            table.add_column("Observations", justify="right")
            table.add_column("Evidence", justify="right")
            table.add_column("Created Via")

            for t in types:
                status_style = {
                    TypeStatus.PROVISIONAL: "yellow",
                    TypeStatus.STABLE: "green",
                    TypeStatus.DEPRECATED: "red",
                    TypeStatus.MERGED_INTO: "dim",
                }.get(t.status, "white")

                table.add_row(
                    t.canonical_name,
                    f"[{status_style}]{t.status.value}[/{status_style}]",
                    str(len(t.observations)),
                    str(t.evidence_count),
                    t.created_via.value,
                )

            console.print(table)

            # Summary
            total_stmt = select(func.count()).select_from(Type)
            total_result = await session.execute(total_stmt)
            total = total_result.scalar()
            console.print(f"\n[dim]Showing {len(types)} of {total} types[/dim]")

    run_async(_review())


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="Search query")],
    limit: Annotated[int, typer.Option(help="Maximum results")] = 10,
    full_ids: Annotated[bool, typer.Option("--full-ids", "-f", help="Show full UUIDs")] = False,
):
    """Search observations by query.

    Searches across source paths and slot values.
    """
    async def _search():
        await init_db()
        async with async_session_factory() as session:
            # Simple text search on source_id and slots
            # In production, this would use vector similarity
            search_pattern = f"%{query}%"
            stmt = (
                select(Observation)
                .options(selectinload(Observation.stable_type))
                .where(Observation.source_id.ilike(search_pattern))
                .limit(limit)
            )
            result = await session.execute(stmt)
            observations = result.scalars().all()

            if not observations:
                console.print(f"[yellow]No observations found matching '{query}'[/yellow]")
                return

            table = Table(title=f"Search Results: '{query}'")
            table.add_column("ID", no_wrap=full_ids)
            table.add_column("Type")
            table.add_column("Medium")
            table.add_column("Source")

            for obs in observations:
                source = obs.source_id.split("/")[-1] if "/" in obs.source_id else obs.source_id
                type_name = obs.stable_type.canonical_name if obs.stable_type else "-"
                if full_ids:
                    obs_id = str(obs.observation_id)
                else:
                    obs_id = str(obs.observation_id)[:8] + "..."
                table.add_row(
                    obs_id,
                    type_name,
                    obs.medium.value,
                    source[:50] + "..." if len(source) > 50 else source,
                )

            console.print(table)

    run_async(_search())


@app.command()
def stats():
    """Show system statistics."""
    async def _stats():
        await init_db()
        async with async_session_factory() as session:
            # Count observations by medium
            medium_stmt = (
                select(Observation.medium, func.count())
                .group_by(Observation.medium)
            )
            medium_result = await session.execute(medium_stmt)
            medium_counts = dict(medium_result.all())

            # Count types by status
            type_stmt = (
                select(Type.status, func.count())
                .group_by(Type.status)
            )
            type_result = await session.execute(type_stmt)
            type_counts = dict(type_result.all())

            # Count entities by status
            entity_stmt = (
                select(Entity.status, func.count())
                .group_by(Entity.status)
            )
            entity_result = await session.execute(entity_stmt)
            entity_counts = dict(entity_result.all())

            # Total counts
            total_observations = sum(medium_counts.values())
            total_types = sum(type_counts.values())
            total_entities = sum(entity_counts.values())
            total_blobs = (await session.execute(select(func.count()).select_from(Blob))).scalar()

            console.print(Panel(
                f"[bold]Observations:[/bold] {total_observations}\n"
                f"[bold]Types:[/bold] {total_types}\n"
                f"[bold]Entities:[/bold] {total_entities}\n"
                f"[bold]Blobs:[/bold] {total_blobs}",
                title="ObjectSense Statistics",
            ))

            # Observations by medium
            if medium_counts:
                table = Table(title="Observations by Medium")
                table.add_column("Medium")
                table.add_column("Count", justify="right")
                for medium, count in sorted(medium_counts.items(), key=lambda x: -x[1]):
                    table.add_row(medium.value, str(count))
                console.print(table)

            # Types by status
            if type_counts:
                table = Table(title="Types by Status")
                table.add_column("Status")
                table.add_column("Count", justify="right")
                for status, count in sorted(type_counts.items(), key=lambda x: -x[1]):
                    table.add_row(status.value, str(count))
                console.print(table)

    run_async(_stats())


@app.command("setup")
def setup():
    """Set up ObjectSense for first use.

    Starts PostgreSQL via Docker and initializes the database schema.
    Requires Docker to be installed and running.
    """
    import shutil
    import subprocess
    import time

    # Check Docker is available
    if not shutil.which("docker"):
        console.print("[red]Error:[/red] Docker not found. Please install Docker first.")
        raise typer.Exit(1)

    console.print("[blue]Starting PostgreSQL database...[/blue]")

    # Start the container
    result = subprocess.run(
        ["docker", "compose", "up", "-d", "--wait"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        console.print(f"[red]Error starting database:[/red] {result.stderr}")
        raise typer.Exit(1)

    console.print("[green]Database container started.[/green]")

    # Wait a moment for PostgreSQL to be fully ready
    console.print("[blue]Waiting for database to be ready...[/blue]")
    time.sleep(2)

    # Initialize schema
    async def _init():
        await init_db()

    try:
        run_async(_init())
        console.print("[green]Database schema initialized.[/green]")
    except Exception as e:
        console.print(f"[red]Error initializing schema:[/red] {e}")
        raise typer.Exit(1) from None

    console.print("\n[bold green]Setup complete![/bold green]")
    console.print("\nYou can now run:")
    console.print("  [cyan]object-sense ingest <path>[/cyan]  - Ingest files")
    console.print("  [cyan]object-sense stats[/cyan]          - View statistics")
    console.print("  [cyan]object-sense --help[/cyan]         - See all commands")


@app.command("init-world")
def init_world():
    """Initialize the database schema (creates tables if they don't exist)."""
    async def _init():
        await init_db()
        console.print("[green]World initialized successfully.[/green]")

    run_async(_init())


@app.command("reset-world")
def reset_world(
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Skip confirmation prompt")
    ] = False,
):
    """Reset the world - drops all tables and recreates them.

    WARNING: This destroys all data!
    """
    if not force:
        confirm = typer.confirm(
            "This will DELETE ALL DATA. Are you sure?",
            default=False,
        )
        if not confirm:
            console.print("[yellow]Aborted.[/yellow]")
            raise typer.Exit(0)

    async def _reset():
        from object_sense.models import Base

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)
        console.print("[green]World reset successfully.[/green]")

    run_async(_reset())


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
