# Object Sense Engine

A semantic substrate that provides persistent object identity and type awareness beneath all other AI systems.

Object Sense answers three fundamental questions:
- **"Is this the same thing I saw before?"** — identity resolution
- **"What kind of thing is this?"** — type inference
- **"What things exist and how are they related?"** — world model

This is not a file classifier, a RAG pattern, or a hand-crafted taxonomy. It is a self-organizing world model.

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

### The Emergence Order

```
Observation → Entity → Type → Domain
```

Types crystallize from entity patterns — they are effects, not causes. Domains emerge as dense regions of the type/entity graph — never explicitly defined.

## How It Works

```
1. INGEST           → Assign UUID, compute SHA256, blob dedup
2. MEDIUM PROBING   → Detect format (image/text/json/video/audio)
3. EXTRACTION       → Run medium-specific extractors (embeddings, EXIF, NER, schema)
4. TYPE INFERENCE   → LLM proposes observation_kind + type + entity_seeds
5. ENTITY RESOLUTION→ Resolve to existing entities or create new ones
6. STORE & INDEX    → Persist observation, entities, evidence, update indexes
```

### Entity Resolution

Identity resolution uses a priority-based approach:

1. **Deterministic IDs** (highest priority): SKU, product_id, GPS coords, trip_id
2. **Similarity search**: Multi-signal scoring weighted by entity nature

| Entity Nature | Primary Signals |
|---------------|-----------------|
| `individual` | Visual embeddings, signatures, location, timestamp |
| `class` | Text embeddings, facet agreement, name matching |
| `group` | Member overlap, location, temporal proximity |
| `event` | Timestamp, participants, location, duration |

Thresholds control linking decisions:
- **T_link** (0.85): High confidence → link to existing entity
- **T_new** (0.60): Below this → create new proto-entity
- **T_margin** (0.10): Ambiguous margin → flag for review

## Installation

### Prerequisites

- Python 3.12+
- Docker (for PostgreSQL with pgvector)
- [uv](https://docs.astral.sh/uv/) package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/object-sense/object-sense.git
cd object-sense

# Install dependencies
uv sync

# Start PostgreSQL and initialize schema
uv run object-sense setup
```

This starts a PostgreSQL container with pgvector and creates all tables.

### Manual Setup

If you prefer manual control:

```bash
# Start database
docker compose up -d

# Initialize schema only
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
│   └── type_promotion.py  # Candidate → Type promotion
└── utils/                 # Utilities
    ├── medium.py          # Medium probing, affordances
    └── slots.py           # Slot validation
```

## Configuration

Environment variables (or `.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql+asyncpg://...` | PostgreSQL connection |
| `LLM_BASE_URL` | `http://localhost:8000/v1` | LLM gateway endpoint |
| `MODEL_CHAT` | `qwen3-vl-4b` | Vision/chat model |
| `MODEL_REASONING` | `gpt-oss-20b` | Reasoning model |
| `ENTITY_RESOLUTION_T_LINK` | `0.85` | Link threshold |
| `ENTITY_RESOLUTION_T_NEW` | `0.60` | New entity threshold |

## Documentation

- [`concept_v2.md`](concept_v2.md) — Authoritative specification
- [`design_v2_corrections.md`](design_v2_corrections.md) — Design rationale
- [`design_notes.md`](design_notes.md) — Historical design discussions

## License

[Add license information]
