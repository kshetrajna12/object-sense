# ObjectSense Engine Architecture

> **A semantic substrate for persistent object identity and type awareness**

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        OBJECTSENSE SEMANTIC SUBSTRATE                            │
│                                                                                   │
│  "A self-organizing world model that knows what things are,                     │
│   not just what they look like"                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## High-Level Architecture

```
┌──────────────┐
│   INPUTS     │         ┌─────────────────────┐         ┌──────────────┐
│              │         │   PROCESSING        │         │  LLM         │
│  • Images    │────────▶│   PIPELINE          │◀───────▶│  SERVICES    │
│  • Text      │         │                     │         │              │
│  • JSON      │         │  6-Step Process     │         │ Sparkstation │
│  • Video     │         │                     │         │  Gateway     │
│  • Audio     │         └──────────┬──────────┘         └──────────────┘
└──────────────┘                    │
                                    ▼
                        ┌────────────────────────┐
                        │   STORAGE LAYER        │
                        │                        │
                        │  • PostgreSQL (data)   │
                        │  • pgvector (index)    │
                        │  • Blob Store (raw)    │
                        └────────────────────────┘
```

---

## Processing Pipeline (Core Loop)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  STEP 1: INGEST                                                  │  │
│  │  • Assign object_id (UUID)                                       │  │
│  │  • Store source_id, compute raw_hash                             │  │
│  │  • Check blob dedup (SHA256)                                     │  │
│  └────────────────────────────┬─────────────────────────────────────┘  │
│                               ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  STEP 2: MEDIUM PROBING                                          │  │
│  │  • File header, extension, entropy analysis                      │  │
│  │  • Sets: medium = image | video | text | json | audio | binary   │  │
│  │  • Does NOT assign type (medium ≠ type!)                         │  │
│  └────────────────────────────┬─────────────────────────────────────┘  │
│                               ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  STEP 3: FEATURE EXTRACTION (based on medium affordances)        │  │
│  │  • Visual: embeddings, detected objects, colors, EXIF            │  │
│  │  • Textual: extracted text, embeddings, named entities           │  │
│  │  • Structural: schema, keys, field patterns                      │  │
│  │  • Temporal: timestamps, sequences                               │  │
│  │  • Spatial: GPS, locations                                       │  │
│  │  • Compute signatures (pHash, simhash, etc.)                     │  │
│  └────────────────────────────┬─────────────────────────────────────┘  │
│                               ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  STEP 4: LLM TYPE INFERENCE (pydantic-ai agent)          ◀─────┐ │  │
│  │  • Input: features, signatures, medium, similar objects         │ │  │
│  │  • LLM queries existing type store (RAG for types)              │ │  │
│  │  • LLM proposes:                                                │ │  │
│  │      - primary_type (e.g., "wildlife_photo")                    │ │  │
│  │      - entity_hypotheses (seeds for next step)                  │ │  │
│  │      - slots (properties)                                       │ │  │
│  │      - maybe_new_type (if nothing fits)                         │ │  │
│  │  • Types created immediately (no threshold)                     │ │  │
│  └────────────────────────────┬─────────────────────────────────────┘  │
│                               ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  STEP 5: ENTITY RESOLUTION (soft clustering)                     │  │
│  │  • entity_hypotheses from Step 4 are SEEDS, not hard assignments│  │
│  │  • Query existing entity candidates                              │  │
│  │  • Soft match: propose links with confidence scores              │  │
│  │  • No strong match: create new proto-entity                      │  │
│  │  • Proto-entities compete, merge, and stabilize over time        │  │
│  └────────────────────────────┬─────────────────────────────────────┘  │
│                               ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  STEP 6: STORE & INDEX                                           │  │
│  │  • Bind: medium, type, slots, entity links                       │  │
│  │  • Store evidence (provenance of all beliefs)                    │  │
│  │  • Update indexes (vector, type graph, entity graph)             │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## LLM Services (Sparkstation Gateway)

```
┌────────────────────────────────────────────────────────────┐
│  Sparkstation Local LLM Gateway                            │
│  http://localhost:8000/v1 (OpenAI-compatible API)         │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  qwen3-vl-4b                                         │  │
│  │  • Vision + Chat (multimodal)                        │  │
│  │  • Used for: image captioning, visual analysis       │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  gpt-oss-20b                                         │  │
│  │  • Reasoning model (16k context)                     │  │
│  │  • Used for: type inference, entity resolution       │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  bge-large                                           │  │
│  │  • Text embeddings (1024 dimensions)                 │  │
│  │  • Used for: semantic search, type similarity        │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  clip-vit                                            │  │
│  │  • Image embeddings (768 dimensions)                 │  │
│  │  • Used for: cross-modal search, visual similarity   │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

---

## Storage Layer

```
┌────────────────────────────────────────────────────────────────────────┐
│                          STORAGE ARCHITECTURE                          │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐       │
│  │  PostgreSQL     │  │  PostgreSQL     │  │  pgvector       │       │
│  │                 │  │                 │  │                 │       │
│  │  • Objects      │  │  • Evidence     │  │  • Text embeds  │       │
│  │  • Types        │  │  • Signatures   │  │  • Image embeds │       │
│  │  • Entities     │  │  • TypeEvolution│  │  • Type embeds  │       │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘       │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │  Blob Store (Content-Addressable)                              │  │
│  │  • SHA256-keyed raw content storage                            │  │
│  │  • Automatic deduplication                                     │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Data Model (Core Primitives)

```
┌──────────────────────────────────────────────────────────────────┐
│                         DATA MODEL                                │
└──────────────────────────────────────────────────────────────────┘

BLOB                    OBJECT                   TYPE
(storage)           (observation)            (semantic)
┌──────────┐          ┌──────────┐           ┌──────────┐
│ blob_id  │◀────────│ object_id│──────────▶│ type_id  │
│ sha256   │         │ medium   │           │ name     │
│ size     │         │ source_id│           │ aliases  │
│          │         │ slots    │           │ parent   │
└──────────┘         └────┬─────┘           │ embedding│
                          │                 └────┬─────┘
                          │ links to             │
                          ▼                      │ has
                    ┌──────────┐                │
                    │  ENTITY  │◀───────────────┘
                    │          │
                    │entity_id │
                    │ type_id  │
                    │ slots    │
                    │confidence│
                    └──────────┘

Supporting Tables:
┌─────────────────────────────────────────────────┐
│ • ObjectEntityLink (M:N association)            │
│ • Evidence (provenance of all beliefs)          │
│ • Signature (modality-specific fingerprints)    │
│ • TypeEvolution (alias, merge, split history)   │
└─────────────────────────────────────────────────┘
```

---

## Identity Resolution — Three Layers

```
┌───────────────────────────────────────────────────────────────────────┐
│  Layer 0: BLOB IDENTITY (storage-level)                               │
│  ─────────────────────────────────────────────────────────────────    │
│  Goal: Don't store same bytes twice                                   │
│  Method: SHA256 hash → O(1) lookup                                    │
│  Signatures: pHash (images), simhash (text)                           │
└───────────────────────────────────────────────────────────────────────┘
                                ▼
┌───────────────────────────────────────────────────────────────────────┐
│  Layer 1: OBSERVATION IDENTITY (object-level)                         │
│  ─────────────────────────────────────────────────────────────────    │
│  Goal: Same observation, different bits?                              │
│  Example: Same photo at different resolutions                         │
│  Method: Embeddings + ANN + verification                              │
└───────────────────────────────────────────────────────────────────────┘
                                ▼
┌───────────────────────────────────────────────────────────────────────┐
│  Layer 2: ENTITY IDENTITY (the real problem)                          │
│  ─────────────────────────────────────────────────────────────────    │
│  Goal: Same real-world thing across observations?                     │
│  Signals:                                                              │
│    • Deterministic IDs (SKU, product_id, trip_id) — STRONG            │
│    • Contextual buckets (same trip, location, time) — MEDIUM          │
│    • Semantic agreement (species, title match) — MEDIUM               │
│    • Multimodal similarity (embedding proximity) — WEAK               │
│  Method: Soft clustering, new Entity nodes, observations link to them │
└───────────────────────────────────────────────────────────────────────┘
```

---

## Cross-Modal Unification Strategy

```
┌────────────────────────────────────────────────────────────────┐
│           UNIFIED MULTIMODAL FEATURE SPACE                     │
└────────────────────────────────────────────────────────────────┘

┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│  Image   │  │   Text   │  │   JSON   │  │  Video   │
│  bytes   │  │  bytes   │  │  bytes   │  │  bytes   │
└────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘
     │             │             │             │
     │  Medium-Specific Extraction (Step 3)    │
     ▼             ▼             ▼             ▼
┌────────────────────────────────────────────────────┐
│         UNIFIED FEATURE SET                        │
│                                                     │
│  visual:     [embeddings, colors, patterns]        │
│  textual:    [embeddings, entities, keywords]      │
│  structural: [schema, field types, keys]           │
│  temporal:   [timestamps, sequences, durations]    │
│  spatial:    [GPS, regions, locations]             │
│  provenance: [source, camera, user, context]       │
│  extracted_entities: [detected objects, NER]       │
└──────────────────────┬─────────────────────────────┘
                       │
                       ▼
         ┌──────────────────────────────┐
         │  ENTITY RESOLVER             │
         │  (medium-agnostic)           │
         │                              │
         │  • Query entity candidates   │
         │  • Soft clustering           │
         │  • Convergence of signals    │
         └─────────────┬────────────────┘
                       ▼
                 ┌──────────┐
                 │  ENTITY  │
                 │ (emerges) │
                 └──────────┘
```

---

## Emergence Order (Critical Concept)

```
╔═══════════════════════════════════════════════════════════════╗
║                   THE EMERGENCE ORDER                          ║
║                                                                ║
║   Observation  →  Entity  →  Type  →  Domain                  ║
║   (raw data)   (clustering) (crystallization) (dense region)  ║
║                                                                ║
║   NOT:  Domain → Type → Entity → Observation                  ║
║                                                                ║
╚═══════════════════════════════════════════════════════════════╝

Observation
    │
    │ Multiple observations cluster around shared invariants
    ▼
Entity (cluster node)
    │
    │ Patterns in entity graph stabilize
    ▼
Type (crystallized label)
    │
    │ Dense regions of interconnected types
    ▼
Domain (emergent, never explicit)
```

---

## Type Evolution Mechanisms

```
┌─────────────────────────────────────────────────────────────┐
│  Types are created immediately but evolve continuously      │
└─────────────────────────────────────────────────────────────┘

┌──────────────────┐
│  ALIAS           │  Multiple names → one canonical
│                  │  "nature_image" → "wildlife_photo"
└──────────────────┘

┌──────────────────┐
│  MERGE           │  Two types → one
│                  │  Reassign objects, deprecate other
└──────────────────┘

┌──────────────────┐
│  SPLIT           │  One type → subtypes
│                  │  "photo" → "wildlife_photo", "portrait"
└──────────────────┘

┌──────────────────┐
│  DEPRECATE       │  Soft delete, migrate objects
│                  │  Keep history in TypeEvolution table
└──────────────────┘

┌──────────────────┐
│  REFINE          │  Add subtype under existing
│                  │  "photo" → "backlit_photo" (child)
└──────────────────┘
```

---

## Key Architectural Principles

### 1. **Medium ≠ Type**
```
Medium (representation)      Type (meaning)
─────────────────────────    ─────────────────────
image                        wildlife_photo
video                        hunt_event
json                         product_record
text                         article
```

**Why this matters:** A product can arrive as JSON, HTML, image, or video — but it's still conceptually a "product". Separating medium from type allows cross-medium unification.

### 2. **LLM Priors + Emergence (The Hybrid Model)**
```
┌─────────────────────┐         ┌──────────────────────┐
│  LLM PRIORS         │         │  RECURRENCE          │
│                     │         │                      │
│ • Instant type      │    +    │ • Validation         │
│   proposals         │         │ • Stabilization      │
│ • First observation │         │ • Merge duplicates   │
│   is enough         │         │ • Prune hallucinated │
│ • Uses latent       │         │ • Build clusters     │
│   knowledge         │         │ • Crystallize domains│
└─────────────────────┘         └──────────────────────┘
```

**The Aardvark Paradox:** You recognize an aardvark on first sight because you have priors (animal, mammal, body plan). ObjectSense works the same way — LLM proposes types instantly, recurrence validates them.

### 3. **Entities Are Hubs, Modalities Are Channels**
```
Observations (any medium) → Entity ← Other observations (any medium)

Image of product  ─┐
JSON record       ─┼──→  Product Entity  ←── Cross-medium unification
HTML page         ─┘
```

### 4. **Source of Truth Hierarchy**
```
Semantic Engine  ─────────▶  Source of truth
                            (Objects, Types, Entities, Evidence)

LLM              ─────────▶  Reasoning module (proposes, doesn't dictate)

Vector Store     ─────────▶  Perceptual cache (similarity lookups)
```

---

## Architecture Roles

### Semantic Engine (Source of Truth)
- **Stores:** Objects, Types, Entities, Evidence, Signatures
- **Owns:** Identity stitching, consistency checks, type evolution
- **Interface:** FastAPI service with database backend

### LLM (Reasoning Layer)
- **Role:** Interprets features, proposes types, suggests entities
- **Interface:** 100% tool calling with structured outputs (pydantic-ai)
- **Constraint:** Consults engine, doesn't overwrite arbitrarily

### Vector Store (Perceptual Cache)
- **Role:** Fast ANN queries for similarity
- **Stores:** Object embeddings, type embeddings, entity embeddings
- **Constraint:** Tool used by engine, not where meaning lives

---

## Technology Stack (v0)

```
┌────────────────────────────────────────────────────────────┐
│  Application Layer                                         │
│  • Python 3.12+                                            │
│  • FastAPI (async web service)                             │
│  • pydantic-ai (LLM integration)                           │
│  • SQLAlchemy 2.0 (async ORM)                              │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│  Database Layer                                            │
│  • PostgreSQL (primary data store)                         │
│  • pgvector (embedding indexes)                            │
│  • asyncpg (async driver)                                  │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│  LLM Layer                                                 │
│  • Sparkstation Gateway (local)                            │
│  • OpenAI-compatible API                                   │
│  • Models: qwen3-vl-4b, gpt-oss-20b, bge-large, clip-vit  │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│  Development Tools                                         │
│  • UV (dependency management)                              │
│  • Ruff (formatting + linting)                             │
│  • Pyright (type checking)                                 │
│  • Pytest (testing)                                        │
└────────────────────────────────────────────────────────────┘
```

---

## CLI Interface

```bash
# Ingest new content
object-sense ingest <path>

# Explore the world model
object-sense show-object <id>
object-sense show-type <name>
object-sense show-entity <id>

# Query and search
object-sense search <query>
object-sense review-types

# Type evolution
object-sense merge-types <type1> <type2>
object-sense alias-type <alias> <canonical>
```

---

## v0 Scope (Current Implementation)

### Supported Mediums
- ✅ Image
- ✅ Text
- ✅ JSON
- ⏳ Video (planned)
- ⏳ Audio (planned)

### Test Domains (Emergent)
- **Wildlife:** Photo archive, GPS metadata, species entities
- **Products:** Shopify-style catalog, SKU-based linking

### End-to-End Validation Story
1. Ingest wildlife folder → see entities emerge (species, locations, individuals)
2. Ingest product catalog → see cross-medium linking (JSON + images)
3. Query engine → verify type emergence, entity clustering, slots filled

---

## Future Roadmap

### Phase 1: Shadow Mode Convergence with image_metadata_indexing
- OS reads IMI's sqlite (read-only)
- Build identity graph on real messy data
- No consumer coupling yet

### Phase 2: First Consumption
- IMI consumes ONE OS output (blob dedup or place resolution)
- Minimal coupling

### Phase 3: Earned Dependency
- Cross-session entity re-ID
- Cross-modal unification
- IMI becomes "wildlife app on ObjectSense"

---

## Summary

**ObjectSense is a semantic substrate that:**

1. **Separates medium from type** — representation vs. meaning
2. **Leverages LLM priors** — instant type proposals from first observation
3. **Validates through recurrence** — stabilization, not invention
4. **Clusters into entities** — observations link to real-world things
5. **Crystallizes types** — types are effects, not causes
6. **Emerges domains** — dense regions, never hardcoded
7. **Evolves continuously** — alias, merge, split, deprecate

**The result:** A self-organizing world model that knows what things are, not just what they look like.

---

*Generated: 2025-12-28*
*Source: concept_v1.md, design_notes.md, implementation code*
