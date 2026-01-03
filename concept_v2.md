# Object Sense Engine — Concept Document v2

> **Status:** Authoritative specification
> **Supersedes:** concept_v1.md (preserved for reference)
> **Last updated:** 2026-01

---

## 1. Purpose

ObjectSense is a **semantic substrate** — a foundational layer that provides persistent object identity and type awareness beneath all other AI systems. It answers:

- "Is this the same thing I saw before?" (identity)
- "What kind of thing is this?" (type)
- "What things exist and how are they related?" (world model)

This is not a file classifier, a RAG pattern, or a hand-crafted taxonomy. It is a self-organizing world model.

---

## 2. Core Distinctions

### 2.1 Medium ≠ Type

This is the foundational distinction.

| Concept | Definition | Examples |
|---------|------------|----------|
| **Medium** | How information is encoded | image, video, text, json, audio, binary |
| **Type** | What the information IS | wildlife_photo, product_record, hunt_event |

**Mediums are about representation. Types are about meaning.**

A product can arrive as JSON, HTML, image, or video — it's still conceptually a "product." Separating medium from type enables cross-medium unification.

### 2.2 Observation ≠ Entity

| Concept | Definition | Examples |
|---------|------------|----------|
| **Observation** | A unit of data we received | photo file, JSON record, video frame, text note |
| **Entity** | A persistent thing in the world | The Marula Leopard, Leopard Species, Safari Trip 2024-06 |

Observations are **inputs**. Entities are **outputs**.

Observations link TO entities — they are not entities themselves. Multiple observations can link to the same entity (same leopard photographed across multiple sessions).

### 2.3 The Emergence Order

```
Observation → Entity → Type → Domain
```

**NOT:**
```
Domain → Type → Entity → Observation
```

- **Types crystallize from entity patterns** — they are effects, not causes
- **Domains emerge as dense regions** of the type/entity graph — never explicitly created

---

## 3. Core Primitives

### 3.1 Observation

A unit of data we received. What we ingest.

**Properties:**
- `observation_id`: UUID (primary key)
- `medium`: image | video | text | json | audio | binary
- `source_id`: Where it came from (file path, URL, database key)
- `blob_id`: Reference to deduplicated content storage
- `slots`: JSONB (inline properties)
- `status`: active | merged | deprecated

**Type references (three-level):**

| Field | Purpose | Cardinality | Authority |
|-------|---------|-------------|-----------|
| `observation_kind` | Routing hint | Low (~20 values) | Non-authoritative |
| `candidate_type_id` | LLM's type proposal | High | Non-authoritative |
| `stable_type_id` | Promoted ontological type | Medium | Authoritative for type labeling only |

> **Note:** `stable_type_id` is authoritative for type labeling, but types remain non-authoritative for identity (see §6.2).

All semantic type labeling flows through `candidate_type_id`, not `observation_kind`. The latter is only for pipeline routing.

**Deterministic IDs:**
- `deterministic_ids`: Array of `{id_type, id_value, id_namespace}` tuples
- These anchor entity resolution with highest priority
- Examples: SKU, product_id, trip_id, GPS coordinates

### 3.2 Medium

The encoding/channel. A **property** of an observation, not a type.

Mediums determine **affordances** — what operations are possible:

| Medium | Affordances |
|--------|-------------|
| image | can_embed, can_detect_objects, can_extract_exif, can_caption |
| video | can_sample_frames, can_segment, can_extract_audio |
| text | can_chunk, can_embed, can_parse |
| json | can_parse_keys, can_infer_schema |
| audio | can_transcribe, can_embed |
| binary | can_entropy_analyze, can_magic_sniff |

### 3.3 Entity

A persistent concept that observations link to. The cluster node.

**Properties:**
- `entity_id`: UUID (primary key)
- `entity_nature`: individual | class | group | event (required)
- `canonical_entity_id`: Merge chain pointer (NULL = I am canonical)
- `name`: Human-readable label
- `slots`: JSONB (inline properties)
- `status`: proto | candidate | stable | deprecated
- `confidence`: Cluster stability score

**Entity nature determines signal weighting in resolution:**

| Nature | Primary Signals | Examples |
|--------|-----------------|----------|
| individual | Visual re-ID, signatures, location | The Marula Leopard, iPhone #ABC123 |
| class | Text embeddings, facet agreement, name match | Leopard Species, Nike Brand |
| group | Member overlap, location, temporal | Kambula Pride, Safari Trip |
| event | Timestamp, participants, location | Hunt Event, Purchase Transaction |

**Prototype embeddings:**
- Entities maintain running-average embeddings for ANN retrieval
- Updated on hard links always; on soft links only if posterior >= T_link
- Enables efficient candidate pool retrieval

### 3.4 TypeCandidate

A provisional type label proposed by the LLM and recorded by the engine (dedup/merge handled by engine).

**Properties:**
- `candidate_id`: UUID
- `proposed_name`: The label as proposed (snake_case)
- `normalized_name`: Lowercase version for dedup detection
- `status`: proposed | promoted | merged | rejected
- `merged_into_candidate_id`: Merge chain pointer
- `promoted_to_type_id`: FK to stable Type (if promoted)
- `evidence_count`: How many observations reference this

**Key design decisions:**
- No UNIQUE constraint on proposed_name (duplicates allowed)
- Merge/alias lifecycle handles consolidation
- Background job detects duplicates via normalized_name

**Merge chain resolution:**
When candidates merge via `merged_into_candidate_id`, resolution must follow the chain (with path compression) to the canonical candidate. Writes may backfill `candidate_type_id`, or reads may resolve at query time.

### 3.5 Type

A stable semantic category. Promoted from TypeCandidate.

**Properties:**
- `type_id`: UUID
- `canonical_name`: The authoritative name (snake_case, singular)
- `aliases`: Accumulated alternative names
- `parent_type_id`: Hierarchical relationship (emerges, not predefined)
- `embedding`: Computed from entity cluster
- `status`: stable | deprecated | merged_into
- `evidence_count`: How many entities support this type

**Promotion criteria (any of):**
1. Evidence count >= threshold (e.g., 5 entities or 20 observations)
2. Coherence score above threshold (cluster tightness)
3. Survives time window without merge/contradiction

Types are **effects**, not causes. They crystallize from patterns in the entity graph.

### 3.6 Slot

Properties/relations on observations and entities. Stored as JSONB.

**Slot values must be structured:**

```json
{
  "price": {"value": 42.99, "unit": "USD", "type": "currency"},
  "species": {"ref_entity_id": "uuid-here", "ref_type": "species_entity"}
}
```

**Slot types:**
- `string` — Free text (use sparingly)
- `number` — Numeric with optional unit
- `boolean` — True/false
- `enum` — Fixed set
- `reference` — Link to another entity
- `list` — Ordered collection

**Anti-patterns:**
```json
{"species": "leopard"}        // WRONG: should be entity reference
{"length": 42}                 // WRONG: 42 what? needs unit
```

### 3.7 Evidence

How the system came to believe something. Tracks provenance for all beliefs.

**Subject kinds:**
- observation (has_type, has_slot)
- entity (type_assignment, slot_value)
- observation_entity_link (refers_to)
- entity_merge (same_as)

**Evidence sources:**
- llm_v1, vision_model, heuristic, user
- Each carries a confidence score (0-1)

---

## 4. Processing Pipeline

### Step 1: Ingest

- Assign observation_id
- Store source_id, compute blob hash
- Check blob dedup (SHA256)

### Step 2: Medium Probing

- File header, extension, entropy analysis
- Sets medium (image | video | text | json | audio | binary)
- Does NOT assign type (medium ≠ type)

### Step 3: Feature Extraction

Based on medium affordances:
- **Visual:** embeddings, detected objects, EXIF
- **Textual:** text, embeddings, named entities
- **Structural:** schema, keys, fields
- **Deterministic IDs:** SKU, product_id, trip_id, GPS

Produces signatures (pHash, simhash, etc.) for identity.

### Step 4: Semantic Probe (LLM)

**4A. Routing + Facets:**
- `observation_kind` — routing hint (non-authoritative)
- `facets` — extracted attributes
- `entity_seeds` — candidate entities detected

**4B. Type Proposal:**
- Create (or reuse) a TypeCandidate for the proposed label; duplicates may exist and are consolidated by dedup/merge
- Store as evidence (non-authoritative)

### Step 5: Entity Resolution

The core algorithm. See §5 for full specification.

1. **Deterministic ID check** (HIGHEST PRIORITY)
   - IDs are `(id_type, id_value, id_namespace)` tuples
   - If match: hard link (posterior=1.0)
   - If NOT found: CREATE entity for ID, then hard link
   - If conflict: create an IdentityConflict record

2. **For each entity_seed:**
   - Get candidates via ANN (prototype embeddings)
   - Compute similarity (signal weights per entity_nature)
   - Apply thresholds (T_link, T_new, T_margin)
   - Create links or proto-entities

3. **Multi-seed consistency pass:**
   - Links have roles: `subject` (primary) and `context` (supporting)
   - Detect role conflicts, facet contradictions
   - Downweight or flag conflicting links

4. **Update entity confidence, slots**

### Step 6: Store & Index

- Persist observation, entity links, evidence
- Update type_candidates.evidence_count
- Update vector indexes

### Background: Type Promotion

Periodic job that:
- Checks type_candidates against thresholds
- Promotes eligible candidates to stable types
- Updates observations.stable_type_id
- Garbage collects rejected candidates

---

## 5. Entity Resolution Specification

### Algorithm Priority

1. **Deterministic IDs** — absolute priority, create entities if needed
2. **Similarity search** — fallback for entities without deterministic IDs (even within the same observation)

### Decision Thresholds (Configurable)

| Threshold | Default | Purpose |
|-----------|---------|---------|
| T_link | 0.85 | High confidence → soft link |
| T_new | 0.60 | Below this → create proto-entity |
| T_margin | 0.10 | If best - second_best < margin → flag for review |

### Signal Weighting by Entity Nature

**individual:**
- embedding: 0.35, signature: 0.25, location: 0.15, timestamp: 0.10, deterministic_id: 0.15

**class:**
- text_embedding: 0.40, facet_agreement: 0.30, name_match: 0.20, deterministic_id: 0.10

**group:**
- member_overlap: 0.35, location: 0.25, temporal_proximity: 0.25, deterministic_id: 0.15

**event:**
- timestamp: 0.30, participants: 0.30, location: 0.20, duration: 0.10, deterministic_id: 0.10

### Canonical Resolution

Entities can merge. When they do:
- Target entity becomes canonical
- Source entities get `canonical_entity_id` pointing to target
- Resolution follows chain with path compression
- Observation links remain valid (resolve at query time)

---

## 6. Invariants and Contracts

### 6.1 Deterministic IDs Dominate Identity

If an observation contains a deterministic identifier, entity linking is **anchored** to it.

**Resolution priority:**
1. Deterministic ID match → link with posterior=1.0, status='hard'
2. Deterministic ID not found → CREATE entity for this ID, then hard link
3. No deterministic ID → fall back to similarity clustering
4. ID conflict → flag for review, trust ID over similarity

**Namespace requirement:**
All deterministic IDs MUST include namespace. Same SKU from different systems are different entities.

```json
{"id_type": "sku", "id_value": "12345", "id_namespace": "source:acme_catalog"}
{"id_type": "sku", "id_value": "12345", "id_namespace": "source:beta_catalog"}
// These are DIFFERENT entities

{"id_type": "upc", "id_value": "012345678901", "id_namespace": "global:upc"}
// Global namespace for universal identifiers
```

### 6.1.1 Namespace Authority

Namespaces follow an allowlisted pattern scheme controlled by the engine:

| Pattern | Purpose | Examples |
|---------|---------|----------|
| `source:*` | System-of-record scope | `source:acme_catalog`, `source:shopify_store_123` |
| `global:*` | Universal identifiers | `global:upc`, `global:isbn` |
| `geo:*` | Geodetic references | `geo:wgs84` |
| `user:*` | User-defined namespaces | `user:my_project` |

**Engine assigns namespaces** from context/source mapping. The LLM proposes ID type/value but must omit namespace; the engine overrides with the correct namespace and records evidence of the assignment.

### 6.2 Type Non-Authoritativeness Contract

Types exist to improve retrieval and UX, NOT to determine identity.

**Types MAY:**
- Improve retrieval/routing as a **soft prior**
- Generate descriptions and UX groupings
- Inform feature extraction pipelines

**Types MAY NOT:**
- Hard-gate candidate search in entity resolution
- Force entity merges/splits
- Be used as a deterministic identity feature

**Enforcement:**
- Entity resolution must work with observation_kind="unknown"
- Identity must remain correct even if all types are "unknown"
- No `if observation_kind == X` in identity logic (only soft boosts)

### 6.3 Slot Hygiene Contract

Every slot value must be either:
1. **Primitive with metadata:** `{"value": 42.99, "unit": "USD"}`
2. **Typed entity reference:** `{"ref_entity_id": "uuid", "ref_type": "species_entity"}`

This enables:
- Slot → Type promotion (detecting when slot domain is rich)
- Consistent indexing and querying
- Migration between representations

---

## 7. Anti-Goals

### 7.1 No Global Domain Partitioning

ObjectSense must remain **domain-agnostic** at the core.

**Allowed:**
- Pluggable feature extractors per medium (image, text, json)
- Entity resolution strategies per entity_nature (individual, class, group, event)
- Evidence-based routing (soft priors)

**Forbidden:**
```python
# WRONG - domain-specific resolvers
if is_wildlife_domain(observation):
    wildlife_pipeline()
elif is_product_domain(observation):
    product_pipeline()
```

```yaml
# WRONG - domain config files
domains:
  wildlife:
    entity_types: [animal_entity, species_entity]
```

**The rule:**
> Only one entity resolver. It may dispatch to strategies (per entity_nature), but there is no top-level domain fork.

### 7.2 No LLM Authority Over Identity

The LLM is a reasoning module, not the source of truth.

**LLM proposes:**
- Type assignments (as TypeCandidates)
- Entity seeds (hypotheses for resolution)
- Slot extractions

**Engine decides:**
- Final type assignments
- Entity identity (via resolution algorithm)
- Merge/split operations

Similarity signals can never override deterministic IDs.

### 7.3 No Premature Type Hierarchy

Types don't have parent relationships until evidence warrants them.

- A type with one entity has no parent
- Hierarchy emerges from cluster analysis, not configuration
- Parent assignment is a promotion-time decision

---

## 8. Architecture Roles

### 8.1 Semantic Engine (Source of Truth)

Stores and maintains:
- Observations, Entities, Types
- Evidence and provenance
- Signatures and embeddings
- Evolution history

All identity decisions, consistency checks, and type lifecycle live here.

### 8.2 LLM (Reasoning Layer)

The LLM:
- Interprets features and context
- Proposes type assignments
- Suggests new types
- Proposes slot values and relationships
- Resolves ambiguous cases

Operates via **tool calling** with **structured outputs**.

The LLM **consults** the semantic engine; it does not overwrite it.

### 8.3 Vector Store (Perceptual Cache)

Stores embeddings for:
- Observations (images, text, JSON)
- Entity prototypes (for ANN retrieval)

A **tool** the engine uses, not where meaning lives.

---

## 9. Data Model Summary

```
BLOB                     OBSERVATION              TYPE_CANDIDATE          TYPE
(storage)                (data point)             (provisional)           (stable)
┌──────────┐            ┌──────────┐             ┌──────────┐            ┌──────────┐
│ blob_id  │◀───────────│observation│             │candidate │            │ type_id  │
│ sha256   │            │  _id     │             │  _id     │◀───────────│ name     │
│ size     │            │ medium   │             │ proposed │            │ parent   │
└──────────┘            │ obs_kind │             │  _name   │            │ embedding│
                        │ facets   │             │normalized│            │ status   │
                        │ det_ids  │             │  _name   │            │ evidence │
                        │candidate_│─FK──────────▶│ merged_  │            │  _count  │
                        │ type_id  │             │  into_id │            └────┬─────┘
                        │ stable_  │─FK──────────────────────────────────────▶│
                        │  type_id │             │ promoted_│                 │
                        └────┬─────┘             │  to_type │                 │
                             │                   └──────────┘                 │
                             │ links to                                       │
                             ▼                                                │
                        ┌──────────┐                                         │
                        │  ENTITY  │◀────────────────────────────────────────┘
                        │          │
                        │ entity_id│
                        │ type_id  │
                        │ entity_  │  ← individual | class | group | event
                        │  nature  │
                        │canonical_│  ← NULL = canonical; else → merged target
                        │entity_id │
                        │ slots    │
                        │confidence│
                        │prototype_│  ← Running average embeddings for ANN
                        │ *_embed  │
                        └──────────┘

ENTITY_DETERMINISTIC_IDS             IDENTITY_CONFLICTS
(lookup table)                       (ID match + attr conflict)
┌──────────────┐                     ┌──────────────┐
│ id_type      │                     │ conflict_id  │
│ id_value     │                     │observation_id│
│ id_namespace │                     │ entity_id    │
│ entity_id    │                     │ conflict_type│
│ confidence   │                     │ status       │
└──────────────┘                     └──────────────┘

ENTITY_EVOLUTION                     OBSERVATION_ENTITY_LINK
(merge/split/link history)           (observation → entity)
┌──────────────┐                     ┌──────────────┐
│ evolution_id │                     │observation_id│
│ entity_id    │                     │ entity_id    │
│ kind         │ ← link|merge|split  │ posterior    │
│ observation  │                     │ status       │ ← hard|soft|candidate
│  _id         │                     │ role         │ ← subject|context
│ algorithm    │                     │ flags        │
│  _version    │                     └──────────────┘
└──────────────┘
```

---

## 10. What This Is NOT

| NOT | Instead |
|-----|---------|
| Another LangChain-style orchestration | A persistent semantic substrate |
| A nicer RAG pattern | A world model beneath RAG |
| A specific taxonomy (products/wildlife) | A general type emergence system |
| A monolithic knowledge graph | A self-organizing ontology |
| A replacement for LLMs or VLMs | A layer that uses them as reasoning modules |
| A file classifier | A meaning layer |
| A domain-specific pipeline system | A single general-purpose identity engine |

---

## 11. Summary

**ObjectSense** is a semantic substrate that:

1. **Separates medium from type** — representation vs. meaning
2. **Separates observation from entity** — data points vs. real-world things
3. **Prioritizes deterministic IDs** — hard anchors over soft similarity
4. **Leverages LLM priors** — instant type proposals from first observation
5. **Validates through recurrence** — TypeCandidates stabilize into Types
6. **Clusters into entities** — observations link to persistent concepts
7. **Crystallizes types** — types are effects, not causes
8. **Remains domain-agnostic** — one resolver, signal weighting not domain forks
9. **Maintains non-authoritative types** — types inform but don't determine identity

The result: a self-organizing world model that knows what things are, not just what they look like.

---

*This document supersedes concept_v1.md. For design rationale, see design_v2_corrections.md. For implementation details, see the source code.*
