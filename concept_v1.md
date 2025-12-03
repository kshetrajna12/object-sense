# Object Sense Engine — Concept Document v1

> **Goal:** Build a foundational "object sense" layer — a semantic substrate that knows *what a thing is* (and what kind of thing it is), before any downstream search, RAG, or task-specific logic. This is the layer that everything else sits on.

---

## 1. The Problem

Modern AI systems are great at:
- Generating text (LLMs)
- Measuring similarity (embeddings + vector DBs)
- Classifying within a fixed label set (ML classifiers)

They are terrible at:
- **Object permanence:** "Is this the same thing I saw before?"
- **Ontological sense:** "What *kind* of thing is this, in the world?"
- **Type evolution:** "Is this a new kind of thing I should add to my universe?"
- **Structured world models:** "What things exist, how are they related, and how do they change over time?"

Everything today is either:
- Stateless (LLM calls)
- Shallow (vector search over chunks)
- Hand-crafted (human-made taxonomies, knowledge graphs)

**The missing primitive:** a persistent, self-updating **object and type layer** that all other systems consult. A machine-level answer to "what is this thing?" and "what kinds of things exist?"

---

## 2. Core Philosophy

### 2.1 Medium ≠ Type

This is the foundational distinction.

| Concept | Definition | Examples |
|---------|------------|----------|
| **Medium** | How information is encoded | image, video, text, json, audio, binary |
| **Type** | What the information actually IS | wildlife_photo, product_record, hunt_event, leopard_entity |

**Mediums are about representation. Types are about meaning.**

If you confuse them, the system becomes a taxonomy, not a world model.

A product can arrive as JSON, HTML, image, or video — but it's still conceptually a "product." Separating medium from type allows cross-medium unification.

### 2.2 LLM Priors + Emergence (The Hybrid Model)

**The Aardvark Paradox:** You can recognize an aardvark on first sight — no prior examples needed. Why? Because you have strong priors (animal, mammal, body plan, wildlife context).

ObjectSense works the same way:

| Source | Role |
|--------|------|
| **LLM priors** | Instant type proposals, even on first observation |
| **Recurrence** | Stabilization, validation, pruning hallucinations |

**The LLM already knows:**
- What an animal, species, photo, product, event is
- That aardvarks → mammals → animals → living things
- These are latent conceptual priors encoded in weights, not hardcoded domains

**Recurrence is NOT for inventing types** (the LLM does that instantly).

**Recurrence IS FOR:**
- Validating types are real (not hallucinated)
- Merging duplicate type names
- Pruning types that don't hold up
- Building durable entity clusters
- Crystallizing domains

**The emergent ontology sits ON TOP of LLM priors, not instead of them.**

### 2.3 The Emergence Order

```
Observation → Entity (clustering) → Type (crystallization) → Domain (emergent region)
```

**NOT:**
```
Domain → Type → Entity → Observation
```

Types crystallize out of the entity graph. They are an **effect**, not a cause.

Domains are never explicitly created. They emerge as dense regions of the conceptual graph.

**Important:** There is no Domain table or primitive in v0. "Domains" are just emergent neighborhoods in the type/entity graph (high intra-connectivity, shared feature patterns). They may later be materialized for UX or optimization, but they are not a first-class ontological primitive. This prevents anyone from trying to create domains as top-down configuration.

---

## 3. Core Primitives

### 3.1 Object

A unit of **observation** in the world model. Files, rows, records, frames, notes, clips, etc.

Objects are NOT the same as Entities. Entities are a separate layer built by clustering Objects. An Object is "something we observed"; an Entity is "a persistent thing in the world that observations refer to."

```
Object:
  object_id: UUID
  medium: image | video | text | json | audio | binary
  primary_type_id: FK → Type (nullable until assigned)
  source_id: string (file path, URL, database key)
  blob_id: FK → Blob (for dedup)
  entity_links: [FK → Entity] (which entities this observation refers to)
  slots: JSONB (inline properties/relations)
  status: active | merged | deprecated
  created_at, updated_at
```

**Note on primary_type_id:** This is the engine's best current guess at "what kind of observation this is" (e.g., wildlife_photo, product_record). It is not immutable — it may change as types merge/split, and all changes are backed by Evidence.

### 3.2 Medium

The encoding/channel. A **property** of an object, not a type.

Mediums determine **affordances** — what operations are possible:

| Medium | Affordances |
|--------|-------------|
| image | can_embed, can_detect_objects, can_extract_exif, can_caption |
| video | can_sample_frames, can_segment, can_extract_audio |
| text | can_chunk, can_embed, can_parse |
| json | can_parse_keys, can_infer_schema |
| audio | can_transcribe, can_embed |
| binary | can_entropy_analyze, can_magic_sniff |

### 3.3 Type

The semantic claim about what something IS.

**A Type is nothing more than a stabilized label over an entity cluster, usually first proposed by the LLM (using its priors) and later validated by recurrence.**

Types don't exist independently of entities. They emerge from patterns in the entity graph.

```
Type:
  type_id: UUID
  canonical_name: string (snake_case, singular)
  aliases: string[] (accumulates over time)
  parent_type_id: FK → Type (emerges later, not defined upfront)
  embedding: vector (computed from entity cluster, not predefined)
  status: provisional | stable | deprecated | merged_into
  merged_into_type_id: FK → Type (nullable)
  evidence_count: int (how many entities support this type)
  created_via: llm_proposed | cluster_emerged | user_confirmed
  created_at
```

**Key principle:** Types don't have parent relationships or embeddings until evidence warrants them. A type with one entity is provisional. A type with many entities has a stable embedding computed from its cluster.

Types are:
- Proposed instantly by the LLM (using its priors)
- Created immediately (no threshold)
- Always provisional until recurrence stabilizes them
- Subject to evolution (alias, merge, split, deprecate)

### 3.4 Slot

Properties/relations on objects and entities. Values can be:
- Primitives (string, number, bool)
- References to other Entities
- Lists of the above

**Storage:** Slots are stored inline on Objects/Entities as JSONB. "Slots" is a conceptual construct, not a separate table in v0. (A separate SlotAssignment table may be added later for indexing/constraints if needed.)

```
Object:
  slots: {
    species: ref(species_entity:leopard)    ← relational
    pose: "stalking"                         ← primitive
    location: ref(location_entity:malamala)  ← relational
    lighting: "backlit"                      ← primitive
  }
```

**Type vs. Slot Rule:**
- **Type** = different kind of thing, with its own schema/participants/behavior
- **Slot** = axis of variation within the same kind of thing

**The identity boundary test:** Types define identity boundaries; slots do not.

Ask: "If I flip this value, is it still the same object?"
- Flip species → different entity → species is a **type** (referenced via slot)
- Flip lighting → same object → lighting is a **slot**
- Flip product_id → different product → product is a **type**, product_id is a slot referencing it

Default to slots. Promote to type only when flipping would cross an identity boundary.

**Rich domains become entity types:** species, locations, individuals are NOT bare strings — they're Types referenced via slots.

### 3.5 Entity

A cluster node that observations link to. Entities represent stable concepts — either concrete or abstract.

**Entities can be:**
- **Concrete individuals:** The Marula Leopard, a specific Nike shoe
- **Abstract concepts:** Leopard Species, the color Blue
- **Compositional groups:** Kambula Pride, the Safari Trip on 2024-06-12
- **Hierarchical abstractions:** Felidae Family, Mammal Class

This gives full expressiveness without domain kits. "Species" is not a special domain — it's just an entity type that emerges when patterns repeat.

Entities are **new nodes** that observations link to — not merges of observations into each other.

```
Entity:
  entity_id: UUID
  type_id: FK → Type (emerges, not predefined)
  slots: { ... }
  observations: [Object IDs that link to this entity]
  status: proto | candidate | stable | deprecated
  confidence: float (how stable is this cluster?)
```

**Key insight:** Entity types like `animal_entity`, `product_entity`, `species_entity` are NOT manually seeded. They emerge as types when patterns repeat across entity candidates. The LLM may propose such types on first observation using priors, but they stabilize only through recurrence.

### 3.6 Evidence

How the system came to believe something. Evidence covers ALL beliefs — types, entity links, and entity merges.

```
Evidence:
  evidence_id: UUID
  subject_kind: "object" | "entity" | "object_entity_link" | "entity_merge"
  subject_id: UUID
  predicate: string  # "has_type", "refers_to", "same_as", etc.
  target_id: UUID | null  # for links/merges
  source: llm_v1 | vision_model | heuristic | user
  score: float (0-1)
  details: JSON
  created_at
```

Examples:
- "Object X has type wildlife_photo" → subject_kind=object, predicate=has_type
- "Object X refers to Entity Y" → subject_kind=object_entity_link, predicate=refers_to
- "Entity A is same as Entity B" → subject_kind=entity_merge, predicate=same_as

The engine computes final beliefs as an aggregation over evidence. This keeps the "beliefs are evidence-based" story consistent across all relationships.

### 3.7 Signatures

Modality-specific fingerprints for identity and similarity.

| Medium | Signatures |
|--------|------------|
| image | SHA256, pHash/dHash, vision embedding (CLIP), EXIF summary |
| video | SHA256, keyframe pHash, pooled video embedding, duration |
| text | SHA256 (normalized), simhash, text embedding |
| json | schema hash, content hash (canonicalized), text embedding |

---

## 4. The Processing Loop

When a new blob arrives:

```
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: Ingest                                              │
│   → Assign object_id                                        │
│   → Store source_id, compute raw_hash                       │
│   → Check blob dedup (SHA256)                               │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: Medium Probing                                      │
│   → File header, extension, entropy, partial decode         │
│   → Sets: medium = image (or video, text, json, etc.)       │
│   → Does NOT assign type                                    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: Feature Extraction (based on medium affordances)   │
│   → Visual: embeddings, detected objects, colors, patterns  │
│   → Textual: extracted text, embeddings, named entities     │
│   → Structural: schema, keys, field patterns                │
│   → Temporal: timestamps, sequences                         │
│   → Spatial: GPS, locations                                 │
│   → Compute signatures (pHash, simhash, etc.)               │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 4: LLM Type Inference (tool calling + structured out)  │
│   → Input: features, signatures, medium, similar objects    │
│   → LLM queries existing type store (RAG for types)         │
│   → LLM proposes:                                           │
│       primary_type = "wildlife_photo"                       │
│       sub_entities = [...]  ← these are HYPOTHESES          │
│       slots = {...}                                         │
│       maybe_new_type = "backlit_wildlife_photo"             │
│   → Types created immediately (no threshold)                │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 5: Entity Resolution (soft clustering)                 │
│   → sub_entities from Step 4 are SEEDS, not hard assignments│
│   → Step 4 says "I think there's an animal here, maybe this │
│     species, maybe this individual"                         │
│   → Step 5 decides how that maps to existing/new entities   │
│   → Query existing entity candidates                        │
│   → Soft match: propose links with confidence scores        │
│   → No strong match: create new proto-entity                │
│   → Proto-entities are cheap, ephemeral hypotheses          │
│   → They compete, merge, and stabilize over time            │
│   → "Is this the same leopard?" is a hypothesis, not fact   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 6: Store & Index                                       │
│   → Bind: medium, type, slots, entity links                 │
│   → Store evidence                                          │
│   → Update indexes (vector, type graph, entity graph)       │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Identity — Three Layers

Identity is not one problem. It's three distinct layers:

### Layer 0: Blob Identity (storage-level)

**Goal:** Don't store the same bytes twice.

- Exact duplicate: `sha256(raw_bytes)` → O(1) lookup
- Near-duplicate: pHash, simhash → flag for review

This is storage hygiene, not semantics.

### Layer 1: Observation Identity (object-level)

**Goal:** Decide if two Objects are the "same observation" even if bits differ.

Example: Same photo exported at different resolutions.

Pipeline:
1. Exact hash check
2. Candidate retrieval (ANN query with blocking)
3. Verification (hard filters + LLM judgment if ambiguous)
4. Decision: duplicate (alias) | near-duplicate (link) | distinct

**Key:** Two cameras shooting the same leopard = different observations. This layer is about observations, not entities.

### Layer 2: Entity Identity (the real problem)

**Goal:** Decide when multiple observations are about the same underlying entity.

This is where ObjectSense actually lives.

**Signals for entity unification:**

| Signal Type | Examples | Strength |
|-------------|----------|----------|
| Deterministic IDs | SKU, product_id, trip_id | Strong |
| Contextual buckets | Same trip, same shop, same time | Medium |
| Semantic agreement | Same species, same product title | Medium |
| Multimodal similarity | Embedding proximity | Weak (hint) |

**Entities are new nodes.** Observations link to them. No "merge observation A into B" problem.

---

## 6. Cross-Medium Unification

**Core insight:** You never unify modalities directly. You unify all modalities into the same entity space.

**Entities are the hub. Modalities are observation channels.**

### The General Identity Engine

No domain-specific pipelines. One engine that:

1. **Extracts modality-native features** from every observation
2. **Projects into unified feature space** (multimodal embeddings + typed features)
3. **Clusters by recurring invariants** (IDs, appearance, location-time, schema patterns)
4. **Entities emerge from clusters**
5. **Types crystallize from entity patterns**
6. **Domains emerge as dense regions**

### The Explicit Pipeline (backbone of the system)

```
┌─────────────────────────────────────────────────────────────┐
│ RAW MODALITY                                                │
│   image bytes, JSON text, video file, audio clip, etc.      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ MODALITY EXTRACTORS (medium-specific)                       │
│   image → vision embedding, detected objects, EXIF          │
│   text → text embedding, named entities, structure          │
│   json → schema, typed fields, text embedding               │
│   video → frame embeddings, audio track, duration           │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ UNIFIED FEATURE SET (common format)                         │
│   visual:     [embeddings, colors, patterns]                │
│   textual:    [embeddings, entities, keywords]              │
│   structural: [schema, field types, keys]                   │
│   temporal:   [timestamps, sequences, durations]            │
│   spatial:    [GPS, regions, locations]                     │
│   provenance: [source, camera, user, context]               │
│   extracted_entities: [detected objects, named entities]    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ ENTITY RESOLVER (generic, medium-agnostic)                  │
│   → Query existing entity candidates                        │
│   → Soft clustering by invariants                           │
│   → Proto-entities compete, merge, stabilize                │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ ENTITY (emerges from cluster)                               │
│   type emerges, slots fill in, observations link            │
└─────────────────────────────────────────────────────────────┘
```

**This is the backbone of the entire system.** All observations — regardless of medium — convert into a UnifiedFeatureSet and flow into the same entity resolver.

Cross-medium unification happens because:
- All observations propose into the same entity candidate pool
- IDs, names, patterns are extracted uniformly
- Clustering is medium-agnostic

---

## 7. Type System

### 7.1 Creation

- Types are proposed instantly by the LLM (using priors)
- Created immediately — no threshold, no waiting for N occurrences
- The ontology is always provisional, always revisable

### 7.2 Evolution Mechanisms

| Mechanism | Description |
|-----------|-------------|
| **Alias** | Multiple names → one canonical type |
| **Merge** | Two types → one (reassign objects) |
| **Split** | One overloaded type → subtypes |
| **Deprecate** | Soft delete, migrate objects |
| **Refine** | Add subtype under existing type |

### 7.3 Type vs. Slot Decision Rule

**Default:** Model new distinctions as slots.

**Promote to Type when:**
1. Different schema (different required slots/participants)
2. Different affordances/routing
3. Natural query root ("give me all hunt_events")
4. Has its own properties and relations

**Everything else stays as slots:** lighting, pose, weather, time_of_day, etc.

### 7.4 Slot → Type Promotion

When a slot's value domain becomes rich (e.g., species needs taxonomy, conservation_status):
1. Create entity type (e.g., `species_entity`)
2. Convert slot to relational reference

### 7.5 Type → Slot Demotion

When a type is always just parent_type + one attribute:
1. Demote to slot on parent type
2. Keep as alias/saved filter if useful

---

## 8. Architecture Roles

### 8.1 Semantic Engine (Source of Truth)

Stores:
- Objects
- Types
- Entities
- Evidence
- Signatures
- Type evolution history

All behavior — identity stitching, consistency checks, type proposals — lives here.

### 8.2 LLM (Reasoning Layer)

The LLM is **not** the world model. It's the reasoning module that:
- Interprets features and context
- Proposes type assignments
- Suggests new types
- Proposes slot values and relationships
- Resolves ambiguous cases

The LLM operates via **tool calling** with **structured outputs**.

The LLM **consults** the semantic engine; it does not overwrite it arbitrarily.

### 8.3 Vector Store (Perceptual Cache)

Stores embeddings for:
- Objects (images, text, JSON, video)
- Types (for similarity detection)
- Entities (for clustering)

Supports fast ANN queries. It's a **tool** the engine uses, not where meaning lives.

---

## 9. Data Model Summary

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│    Blob     │      │   Object    │      │    Type     │
│  (storage)  │◄────▶│(observation)│─────▶│ (semantic)  │
└─────────────┘      └──────┬──────┘      └─────────────┘
                            │
                            │ links to
                            ▼
                     ┌─────────────┐
                     │   Entity    │
                     │ (persistent │
                     │   concept)  │
                     └─────────────┘

Supporting tables:
- Evidence (how beliefs were formed — types, entity links, merges)
- Signatures (fingerprints per object)
- TypeEvolution (alias, merge, split history)

Note: Slots are stored inline as JSONB on Objects/Entities, not as a separate table in v0.
```

---

## 10. What This Is NOT

To keep the intent clear:

| NOT | Instead |
|-----|---------|
| Another LangChain-style orchestration | A persistent semantic substrate |
| A nicer RAG pattern | A world model beneath RAG |
| A specific taxonomy (products/wildlife) | A general type emergence system |
| A monolithic knowledge graph | A self-organizing ontology |
| A replacement for LLMs or VLMs | A layer that uses them as reasoning modules |
| A file classifier | A meaning layer |

---

## 11. v0 Deliverable Shape

A realistic v0:

- **Python/FastAPI service** ("object-sense")
- **Postgres** for objects, types, entities, evidence, signatures
- **pgvector** for embeddings
- **Local LLM + VLM** for type inference and feature extraction

**CLI for exploration:**
```
object-sense ingest <path>
object-sense show-object <id>
object-sense show-type <name>
object-sense show-entity <id>
object-sense review-types
object-sense search <query>
```

Even without UI, this lets you see your archive as a structured world of objects, types, and entities — not just files and tags.

### v0 Scope (brutal cut)

**Mediums supported:** image + text + JSON

**Domains to exercise (emergent, not configured):**
- Wildlife (your photo archive)
- Products (Shopify-ish data / store dump)

**End-to-end validation story:**

1. **Ingest a wildlife folder:**
   - Photos, text notes, GPS/JSON metadata
   - See entities emerge: species, locations, individual leopards/lions
   - Verify: types proposed, entities clustered, slots filled

2. **Ingest a product catalog:**
   - JSON records + product images
   - See product_entities form and images link cross-medium
   - Verify: SKU-based linking, type emergence

3. **Query the engine:**
   - "Show me all entities of type species_entity"
   - "Show me all animal_entities with >3 observations"
   - "Show me all wildlife_photos depicting the same individual as this photo"

No UI needed. CLI + debug queries validate the ontology and clustering loop.

---

## 12. Open Questions

| Question | Status |
|----------|--------|
| Unified feature space alignment | Open — how do visual/textual/structural features align technically? |
| Query interface for consumers | Deferred |
| v0 scope | Defined in §11 — verify feasibility after first spike |

---

## 13. Summary

**ObjectSense** is a semantic substrate that:

1. **Separates medium from type** — representation vs. meaning
2. **Leverages LLM priors** — instant type proposals from first observation
3. **Validates through recurrence** — stabilization, not invention
4. **Clusters into entities** — observations link to real-world things
5. **Crystallizes types** — types are effects, not causes
6. **Emerges domains** — dense regions, never hardcoded
7. **Evolves continuously** — alias, merge, split, deprecate

The result: a self-organizing world model that knows what things are, not just what they look like.

---

*This document supersedes `inital_idea.md`. See `design_notes.md` for detailed discussion history.*
