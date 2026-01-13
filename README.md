# Object Sense Engine

A semantic substrate that provides persistent object identity and type awareness for AI applications and pipelines.

Object Sense answers three fundamental questions:
- **"Is this the same thing I saw before?"** — identity resolution
- **"What kind of thing is this?"** — type proposal & stabilization
- **"What things exist and how are they related?"** — world model

## Why This Exists

Modern AI systems repeatedly rediscover the same facts because they lack persistent identity. Every pipeline re-solves "is this the same thing?" — the same product matched across catalogs, the same entity extracted from different documents, the same object recognized in different photos.

Object Sense solves that once, centrally, across media and domains. It incrementally builds a coherent model of the world from evidence, without predefined schemas.

**What makes it different:**
- One identity layer shared by many apps and pipelines
- Deterministic IDs are law — create-on-miss, never overridden by similarity
- LLM proposes, engine decides
- Cross-medium by design (image, text, json, video)

## Status

Early but functional. Expect rapid iteration on APIs and heuristics; invariants (deterministic IDs, engine-decides, canonical merges) are stable.

## Quick Start

```bash
# Install
git clone https://github.com/object-sense/object-sense.git
cd object-sense
uv sync

# Start database and initialize
uv run object-sense setup

# Ingest some files
uv run object-sense ingest photo.jpg
uv run object-sense ingest ./products/*.json --id-namespace "source:catalog"

# See what was created
uv run object-sense stats
uv run object-sense search "leopard"
```

```
$ uv run object-sense stats
╭─────────────── ObjectSense Statistics ───────────────╮
│ Observations: 47                                     │
│ Types: 8                                             │
│ Entities: 12                                         │
│ Blobs: 45                                            │
╰──────────────────────────────────────────────────────╯

$ uv run object-sense search "leopard"
┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
┃ ID          ┃ Type            ┃ Medium ┃ Source               ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
│ 3f8a2b1c... │ wildlife_photo  │ image  │ leopard_rock.jpg     │
│ 7d9e4f2a... │ wildlife_photo  │ image  │ marula_leopard.cr3   │
└─────────────┴─────────────────┴────────┴──────────────────────┘
```

After ingestion, the system has:
- Created observations for each file
- Extracted embeddings, signatures, and deterministic IDs
- Proposed type candidates via LLM (engine decides stable types)
- Resolved entities (or created new ones)
- Linked observations to entities with confidence scores

## Core Concepts

### Medium ≠ Type

This is the foundational distinction.

| Concept | Definition | Examples |
|---------|------------|----------|
| **Medium** | How information is encoded | image, video, text, json, audio |
| **Type** | What the information IS | wildlife_photo, product_record, hunt_event |

A product can arrive as JSON, HTML, image, or video — it's still conceptually a "product." Separating medium from type enables cross-medium unification.

### Observation ≠ Entity

| Concept | Definition | Examples |
|---------|------------|----------|
| **Observation** | A unit of data we received | photo file, JSON record, video frame |
| **Entity** | A persistent thing in the world | The Marula Leopard, Leopard Species, Safari 2024 |

Observations are **inputs**. Entities are **outputs**. Multiple observations can link to the same entity (same leopard photographed across sessions).

## How It Works

```
1. INGEST           → Assign UUID, compute SHA256, blob dedup
2. MEDIUM PROBING   → Detect format (image/text/json/video/audio)
3. EXTRACTION       → Run medium-specific extractors (embeddings, EXIF, NER, schema)
4. TYPE INFERENCE   → LLM proposes observation_kind + type candidate + entity seeds
5. ENTITY RESOLUTION→ Resolve to existing entities or create new ones
6. STORE & INDEX    → Persist observation, entities, evidence, update indexes
```

The engine decides stable types — LLM proposals are evidence, not authority.

### Type Lifecycle

Types follow a two-stage lifecycle with evolution mechanisms:

```
LLM proposes → TypeCandidate (provisional) → [promote] → Type (stable)
                                                              ↓
                                              [alias, merge, split, deprecate]
```

**Stage 1: TypeCandidates** — LLM proposals, created immediately on first observation. Multiple names may refer to the same concept (`wildlife_photo`, `nature_image`, `animal_pic`).

**Stage 2: Stable Types** — Promoted from candidates when evidence thresholds are met. Support evolution operations:

| Operation | What it does |
|-----------|--------------|
| **Alias** | Map multiple names → one canonical type |
| **Merge** | Combine two types, migrate all observations/entities |
| **Split** | Divide overloaded type into subtypes |
| **Deprecate** | Retire type, optionally migrate to replacement |

All evolution operations create audit records in `type_evolution` table.

### Entity Resolution

Identity resolution uses a priority-based approach:

1. **Deterministic IDs** (highest priority): SKU, product_id, GTIN/UPC, file hash, canonical URL — hard identifiers always override similarity signals
2. **Similarity search**: Multi-signal scoring weighted by entity nature

| Entity Nature | Primary Signals |
|---------------|-----------------|
| `individual` | Visual embeddings, signatures, location, timestamp |
| `class` | Text embeddings, facet agreement, name matching |
| `group` | Member overlap, location, temporal proximity |
| `event` | Timestamp, participants, location, duration |

Configurable thresholds control linking decisions: when to link to existing entities, when to create new ones, and when to flag ambiguous cases for review.

## Non-goals

- **Not a classifier** — it doesn't categorize files into buckets
- **Not a taxonomy authoring tool** — types emerge from data, not configuration
- **Not a business rules engine** — it maintains identity, not policy
- **Not a replacement for domain logic** — it's infrastructure beneath your application

Object Sense is a substrate. It tells you *what things are* so your application can decide *what to do with them*.

Object Sense is not a general-purpose knowledge graph; it is an identity-first semantic engine optimized for evidence-backed resolution.

## Installation

### Prerequisites

- Python 3.12+
- Docker (for PostgreSQL with pgvector)
- [uv](https://docs.astral.sh/uv/) package manager

For standard setup, follow [Quick Start](#quick-start).

### Manual Setup

If you prefer manual control over the database:

```bash
# Start database only
docker compose up -d

# Initialize schema
uv run object-sense init-world
```

## Usage

### Ingesting Files

```bash
# Ingest a single file
uv run object-sense ingest photo.jpg

# Ingest a directory
uv run object-sense ingest ./photos/

# Ingest recursively with verbose output
uv run object-sense ingest ./data/ -r -v

# Specify namespace for deterministic IDs
uv run object-sense ingest catalog.json --id-namespace "source:acme_products"
```

Supported formats:
- **Images**: jpg, png, gif, webp, tiff, and RAW formats (arw, cr2, cr3, nef, dng, raf, etc.)
- **Text**: txt, md, rst, csv
- **Structured**: json

### Querying

```bash
# View observation details
uv run object-sense show-observation <id>

# View type details
uv run object-sense show-type wildlife_photo

# View entity details
uv run object-sense show-entity <uuid>

# Review all types
uv run object-sense review-types
uv run object-sense review-types --status provisional

# Search observations
uv run object-sense search "leopard"

# System statistics
uv run object-sense stats
```

### Type Management

```bash
# Review provisional type candidates (LLM proposals)
uv run object-sense review-candidates

# Review stable types
uv run object-sense review-types

# Manually promote a candidate to stable type
uv run object-sense type-promote wildlife_photo_observation --force

# Add an alias to a type
uv run object-sense type-alias wildlife_photo nature_image

# Merge two types (source → target)
uv run object-sense type-merge nature_image wildlife_photo --dry-run
uv run object-sense type-merge nature_image wildlife_photo

# Split an overloaded type
uv run object-sense type-split photo wildlife_photo portrait_photo product_photo

# Deprecate a type (with optional migration)
uv run object-sense type-deprecate old_type --replacement new_type

# View evolution history
uv run object-sense type-history wildlife_photo
```

### Management

```bash
# Reset database (WARNING: destroys all data)
uv run object-sense reset-world --force
```

## Architecture

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                    SEMANTIC ENGINE                          │
│  Source of truth for observations, entities, types         │
│  All identity decisions and consistency checks live here   │
└─────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   LLM LAYER     │  │  VECTOR STORE   │  │   POSTGRES      │
│                 │  │                 │  │                 │
│ • Type proposal │  │ • Embeddings    │  │ • Observations  │
│ • Entity seeds  │  │ • ANN retrieval │  │ • Entities      │
│ • Slot extract  │  │ • Similarity    │  │ • Types         │
│                 │  │                 │  │ • Evidence      │
│ Proposes, does  │  │ Perceptual      │  │ Persistent      │
│ NOT decide      │  │ cache           │  │ storage         │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

### Data Model

```
BLOB                 OBSERVATION              TYPE_CANDIDATE          TYPE
(storage)            (data point)             (provisional)           (stable)
┌──────────┐        ┌──────────────┐         ┌──────────────┐        ┌──────────┐
│ blob_id  │◀───────│observation_id│         │ candidate_id │        │ type_id  │
│ sha256   │        │ medium       │         │ proposed_name│◀───────│ name     │
│ size     │        │ obs_kind     │         │ normalized   │        │ parent   │
└──────────┘        │ facets       │         │ status       │        │ status   │
                    │ det_ids      │         │ merged_into  │        │ embedding│
                    │ candidate_   │─FK─────▶│ promoted_to  │        └────┬─────┘
                    │   type_id    │         └──────────────┘             │
                    │ stable_      │─FK───────────────────────────────────┘
                    │   type_id    │
                    └──────┬───────┘
                           │ links to
                           ▼
                    ┌──────────────┐
                    │    ENTITY    │
                    │              │
                    │ entity_id    │
                    │ entity_nature│  ← individual | class | group | event
                    │ canonical_id │  ← merge chain pointer
                    │ slots        │
                    │ confidence   │
                    └──────────────┘
```

### Key Design Decisions

1. **Empty Bootstrap**: No predefined taxonomy. Types emerge from LLM proposals validated by recurrence.

2. **Immediate Type Creation**: First proposal creates a TypeCandidate instantly. Promotion to stable Type requires evidence thresholds.

3. **Single General Resolver**: No domain-specific pipelines. Entity nature (individual/class/group/event) controls signal weighting, not separate code paths.

4. **Deterministic IDs Dominate**: Hard identifiers (SKU, product_id) always override similarity signals.

5. **Types Are Non-Authoritative**: Types improve retrieval and UX but cannot hard-gate identity decisions.

### The Emergence Order

Types crystallize from entity patterns — they are effects, not causes:

```
Observation → Entity → Type → Domain
```

## Development

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=object_sense

# Run specific test file
uv run pytest tests/test_entity_resolution.py -v
```

### Code Quality

```bash
# Format code
uv run ruff format

# Lint
uv run ruff check

# Type check
uv run pyright
```

### Project Structure

```
src/object_sense/
├── app.py                 # FastAPI application
├── cli.py                 # Typer CLI (main entry point)
├── config.py              # Pydantic settings
├── db.py                  # SQLAlchemy async setup
├── models/                # Data models
│   ├── observation.py     # Observation, Blob
│   ├── entity.py          # Entity, ObservationEntityLink
│   ├── type_.py           # Type, TypeCandidate
│   ├── evidence.py        # Evidence provenance
│   └── signature.py       # Modality-specific fingerprints
├── extraction/            # Feature extraction
│   ├── orchestrator.py    # Dispatch by medium
│   ├── image.py           # CLIP, EXIF, pHash
│   ├── text.py            # BGE embeddings, NER
│   └── json_extract.py    # Schema inference
├── inference/             # LLM integration
│   ├── type_inference.py  # pydantic-ai agent
│   └── schemas.py         # TypeProposal, EntityHypothesis
├── resolution/            # Entity resolution
│   ├── resolver.py        # Main algorithm
│   ├── similarity.py      # Multi-signal scoring
│   └── candidate_pool.py  # ANN retrieval
├── services/              # Business logic
│   ├── type_candidate.py  # TypeCandidate lifecycle
│   ├── type_promotion.py  # Candidate → Type promotion
│   └── type_evolution.py  # Type evolution (alias, merge, split, deprecate)
└── utils/                 # Utilities
    ├── medium.py          # Medium probing, affordances
    └── slots.py           # Slot validation
```

## Configuration

Environment variables (or `.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql+asyncpg://...localhost.../object_sense` | PostgreSQL connection string |
| `LLM_BASE_URL` | `http://localhost:8000/v1` | OpenAI-compatible LLM gateway |
| `MODEL_CHAT` | `qwen3-vl-4b` | Vision/chat model for type inference |
| `MODEL_TEXT_EMBEDDING` | `bge-large` | Text embedding model |
| `MODEL_IMAGE_EMBEDDING` | `clip-vit` | Image embedding model |

**Note:** Blob content is stored in PostgreSQL; the `storage_path` column records the original file path for reference.

See `src/object_sense/config.py` for all options including entity resolution thresholds and type promotion settings.

## Documentation

- [`concept_v2.md`](concept_v2.md) — Authoritative specification
- [`design_v2_corrections.md`](design_v2_corrections.md) — Design rationale
- [`design_notes.md`](design_notes.md) — Historical design discussions

## License

TBD
