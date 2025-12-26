"""CLI for ObjectSense.

Commands:
    ingest <path>       - Ingest files/directories
    show-object <id>    - Show object details
    show-type <name>    - Show type details
    show-entity <id>    - Show entity details
    review-types        - List all types in the system
    search <query>      - Search objects by query
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import Annotated
from uuid import UUID, uuid4

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from object_sense.db import async_session_factory, engine, init_db
from object_sense.extraction.orchestrator import ExtractionOrchestrator
from object_sense.inference.type_inference import TypeInferenceAgent
from object_sense.models import (
    Blob,
    Entity,
    Evidence,
    EvidenceSource,
    Object,
    ObjectStatus,
    Signature,
    SubjectKind,
    Type,
    TypeCreatedVia,
    TypeStatus,
)
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


async def ingest_file(file_path: Path, verbose: bool = False) -> dict:
    """Ingest a single file through the full pipeline.

    Returns dict with object_id, type, medium, and status.
    """
    # Read file content
    content = file_path.read_bytes()
    sha256 = hashlib.sha256(content).hexdigest()

    async with async_session_factory() as session:
        # Check for duplicate blob
        stmt = select(Blob).where(Blob.sha256 == sha256)
        result = await session.execute(stmt)
        existing_blob = result.scalar_one_or_none()

        if existing_blob:
            # Check if we already have an object for this blob + source
            stmt = select(Object).where(
                Object.blob_id == existing_blob.blob_id,
                Object.source_id == str(file_path.absolute()),
            )
            result = await session.execute(stmt)
            existing_obj = result.scalar_one_or_none()
            if existing_obj:
                return {
                    "object_id": str(existing_obj.object_id),
                    "type": existing_obj.primary_type_id,
                    "medium": existing_obj.medium.value,
                    "status": "duplicate",
                    "message": "Object already exists",
                }

        # Step 2: Medium probing
        medium = probe_medium(content, filename=file_path.name)

        # Step 3: Feature extraction
        orchestrator = ExtractionOrchestrator()
        extraction_result = await orchestrator.extract(
            content, medium=medium, filename=file_path.name
        )

        # Step 4: Type inference
        inference_agent = TypeInferenceAgent()
        type_proposal = await inference_agent.infer(extraction_result, medium=medium.value)

        # Create or get type
        # Always use LLM_PROPOSED since types come from inference
        primary_type = await get_or_create_type(
            session,
            type_proposal.primary_type,
            TypeCreatedVia.LLM_PROPOSED,
        )

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

        # Create object
        object_id = uuid4()
        obj = Object(
            object_id=object_id,
            medium=medium,
            primary_type_id=primary_type.type_id,
            source_id=str(file_path.absolute()),
            blob_id=blob.blob_id,
            slots=type_proposal.slots,
            status=ObjectStatus.ACTIVE,
        )
        session.add(obj)

        # Store signatures
        if extraction_result.hash_value:
            sig = Signature(
                signature_id=uuid4(),
                object_id=object_id,
                signature_type=extraction_result.signature_type,
                value=extraction_result.hash_value,
            )
            session.add(sig)

        # Store embeddings as signatures
        if extraction_result.text_embedding:
            sig = Signature(
                signature_id=uuid4(),
                object_id=object_id,
                signature_type="text_embedding",
                text_embedding=extraction_result.text_embedding,
            )
            session.add(sig)

        if extraction_result.image_embedding:
            sig = Signature(
                signature_id=uuid4(),
                object_id=object_id,
                signature_type="image_embedding",
                image_embedding=extraction_result.image_embedding,
            )
            session.add(sig)

        # Store evidence for type assignment
        evidence = Evidence(
            evidence_id=uuid4(),
            subject_kind=SubjectKind.OBJECT,
            subject_id=object_id,
            predicate="has_type",
            target_id=primary_type.type_id,
            source=EvidenceSource.LLM,
            score=0.8,  # Default confidence
            details={
                "reasoning": type_proposal.reasoning,
                "is_existing_type": type_proposal.is_existing_type,
            },
        )
        session.add(evidence)

        # Update type evidence count
        primary_type.evidence_count += 1

        await session.commit()

        return {
            "object_id": str(object_id),
            "type": type_proposal.primary_type,
            "medium": medium.value,
            "status": "ingested",
            "slots": type_proposal.slots,
            "reasoning": type_proposal.reasoning,
        }


@app.command()
def ingest(
    path: Annotated[Path, typer.Argument(help="File or directory to ingest")],
    recursive: Annotated[
        bool, typer.Option("--recursive", "-r", help="Recursively ingest directories")
    ] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show detailed output")
    ] = False,
):
    """Ingest files into ObjectSense.

    Runs the full pipeline: probe medium → extract features → infer type → store.
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
                result = await ingest_file(file_path, verbose=verbose)
                results.append(result)

                if result["status"] == "duplicate":
                    console.print("[yellow]SKIP[/yellow] (duplicate)")
                else:
                    console.print(f"[green]OK[/green] → {result['type']}")

                if verbose and result["status"] != "duplicate":
                    console.print(f"    ID: {result['object_id']}")
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


@app.command("show-object")
def show_object(
    object_id: Annotated[str, typer.Argument(help="Object ID (UUID)")],
):
    """Show details for a specific object."""
    async def _show():
        await init_db()
        try:
            oid = UUID(object_id)
        except ValueError:
            console.print(f"[red]Error:[/red] Invalid UUID: {object_id}")
            raise typer.Exit(1) from None

        async with async_session_factory() as session:
            stmt = (
                select(Object)
                .options(
                    selectinload(Object.primary_type),
                    selectinload(Object.blob),
                    selectinload(Object.entity_links),
                    selectinload(Object.signatures),
                )
                .where(Object.object_id == oid)
            )
            result = await session.execute(stmt)
            obj = result.scalar_one_or_none()

            if not obj:
                console.print(f"[red]Error:[/red] Object not found: {object_id}")
                raise typer.Exit(1)

            # Build display
            panel_content = []
            panel_content.append(f"[bold]ID:[/bold] {obj.object_id}")
            panel_content.append(f"[bold]Medium:[/bold] {obj.medium.value}")
            panel_content.append(f"[bold]Status:[/bold] {obj.status.value}")
            panel_content.append(f"[bold]Source:[/bold] {obj.source_id}")

            if obj.primary_type:
                panel_content.append(f"[bold]Type:[/bold] {obj.primary_type.canonical_name}")

            if obj.blob:
                panel_content.append(f"[bold]SHA256:[/bold] {obj.blob.sha256[:16]}...")
                panel_content.append(f"[bold]Size:[/bold] {obj.blob.size_bytes:,} bytes")

            if obj.slots:
                panel_content.append("[bold]Slots:[/bold]")
                for k, v in obj.slots.items():
                    panel_content.append(f"  • {k}: {v}")

            panel_content.append(f"[bold]Created:[/bold] {obj.created_at}")

            console.print(Panel("\n".join(panel_content), title="Object Details"))

            # Show signatures
            if obj.signatures:
                table = Table(title="Signatures")
                table.add_column("Type")
                table.add_column("Value")
                for sig in obj.signatures:
                    if sig.value:
                        val = sig.value
                    elif sig.embedding:
                        val = f"embedding ({len(sig.embedding)}d)"
                    else:
                        val = "-"
                    val_str = str(val)
                    display = val_str[:32] + "..." if len(val_str) > 32 else val_str
                    table.add_row(sig.signature_type, display)
                console.print(table)

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
                .options(selectinload(Type.objects), selectinload(Type.entities))
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

            panel_content.append(f"[bold]Objects:[/bold] {len(t.objects)}")
            panel_content.append(f"[bold]Entities:[/bold] {len(t.entities)}")
            panel_content.append(f"[bold]Created:[/bold] {t.created_at}")

            console.print(Panel("\n".join(panel_content), title=f"Type: {type_name}"))

            # Show recent objects of this type
            if t.objects:
                table = Table(title="Recent Objects")
                table.add_column("ID")
                table.add_column("Medium")
                table.add_column("Source")
                for obj in t.objects[:5]:
                    source = obj.source_id.split("/")[-1] if "/" in obj.source_id else obj.source_id
                    table.add_row(
                        str(obj.object_id)[:8] + "...",
                        obj.medium.value,
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
                .options(selectinload(Entity.type), selectinload(Entity.object_links))
                .where(Entity.entity_id == eid)
            )
            result = await session.execute(stmt)
            entity = result.scalar_one_or_none()

            if not entity:
                console.print(f"[red]Error:[/red] Entity not found: {entity_id}")
                raise typer.Exit(1)

            panel_content = []
            panel_content.append(f"[bold]ID:[/bold] {entity.entity_id}")
            panel_content.append(f"[bold]Status:[/bold] {entity.status.value}")
            panel_content.append(f"[bold]Confidence:[/bold] {entity.confidence:.2f}")

            if entity.type:
                panel_content.append(f"[bold]Type:[/bold] {entity.type.canonical_name}")

            if entity.slots:
                panel_content.append("[bold]Slots:[/bold]")
                for k, v in entity.slots.items():
                    panel_content.append(f"  • {k}: {v}")

            panel_content.append(f"[bold]Linked Objects:[/bold] {len(entity.object_links)}")
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
            stmt = select(Type).options(selectinload(Type.objects))

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
            table.add_column("Objects", justify="right")
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
                    str(len(t.objects)),
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
):
    """Search objects by query.

    Searches across source paths and slot values.
    """
    async def _search():
        await init_db()
        async with async_session_factory() as session:
            # Simple text search on source_id and slots
            # In production, this would use vector similarity
            search_pattern = f"%{query}%"
            stmt = (
                select(Object)
                .options(selectinload(Object.primary_type))
                .where(Object.source_id.ilike(search_pattern))
                .limit(limit)
            )
            result = await session.execute(stmt)
            objects = result.scalars().all()

            if not objects:
                console.print(f"[yellow]No objects found matching '{query}'[/yellow]")
                return

            table = Table(title=f"Search Results: '{query}'")
            table.add_column("ID")
            table.add_column("Type")
            table.add_column("Medium")
            table.add_column("Source")

            for obj in objects:
                source = obj.source_id.split("/")[-1] if "/" in obj.source_id else obj.source_id
                type_name = obj.primary_type.canonical_name if obj.primary_type else "-"
                table.add_row(
                    str(obj.object_id)[:8] + "...",
                    type_name,
                    obj.medium.value,
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
            # Count objects by medium
            medium_stmt = (
                select(Object.medium, func.count())
                .group_by(Object.medium)
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
            total_objects = sum(medium_counts.values())
            total_types = sum(type_counts.values())
            total_entities = sum(entity_counts.values())
            total_blobs = (await session.execute(select(func.count()).select_from(Blob))).scalar()

            console.print(Panel(
                f"[bold]Objects:[/bold] {total_objects}\n"
                f"[bold]Types:[/bold] {total_types}\n"
                f"[bold]Entities:[/bold] {total_entities}\n"
                f"[bold]Blobs:[/bold] {total_blobs}",
                title="ObjectSense Statistics",
            ))

            # Objects by medium
            if medium_counts:
                table = Table(title="Objects by Medium")
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
