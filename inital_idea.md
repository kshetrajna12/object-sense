# Object Sense Engine — High-Level Concept Doc (v0)

> **Goal:** Build a foundational “object sense” layer — a semantic OS that knows *what a thing is* (and what kind of thing it is), before any downstream search, RAG, or task-specific logic. This is **not** a better vector DB, not a nicer RAG pipeline, not a taxonomy. It’s the substrate that everything else should sit on.

---

## 1. Motivation

Modern AI systems are great at:
- Generating text (LLMs)
- Measuring similarity (embeddings + vector DBs)
- Classifying within a fixed label set (ML classifiers)

They are terrible at:
- **Object permanence:** “Is this the same thing I saw before?”
- **Ontological sense:** “What *kind* of thing is this blob, in the world?”
- **Type evolution:** “Is this a new kind of thing I should add to my universe?”
- **Structured world models:** “What things exist, how are they related, and how do they change over time?”

Everything today is either:
- Stateless (LLM calls)
- Shallow (vector search over chunks)
- Hand-crafted (human-made taxonomies, knowledge graphs)

**The missing primitive:** a persistent, self-updating **object and type layer** that all other systems consult. A machine-level answer to “what is this thing?” and “what kinds of things exist?”

This is what the **Object Sense Engine** is meant to be.

---

## 2. Core Idea

Whenever a new “blob” shows up — file, record, message, image, video, JSON, log — the Object Sense Engine runs the same conceptual loop:

1. **Have I seen this exact object before?**  
   → Identity / dedup / near-dup: *object permanence*.

2. **Have I seen this kind of object before?**  
   → Type inference: *object category in the world*, not just a cluster.

3. **If I haven’t, is this a new type worth adding?**  
   → Ontological expansion: *the system grows its own type system over time.*

This is **not** “classify into 500 labels from a training set”.  
This is: **discover, stabilize, and maintain the universe of “things” the system knows about.**

Everything else (search, RAG, agents, analytics) becomes a *consumer* of this layer.

---

## 3. Roles of the Components

The architecture has three big roles:

### 3.1 Semantic Engine (World Model + Memory)
This is the **source of truth**.

It stores:

- **Objects** — specific things (files, images, JSON records, entities inside images, events, etc.)
- **Types** — kinds of things (image, video, wildlife_photo, product_record, chat_log, etc.)
- **Slots** — properties/relations (species, pose, lighting, parent, children, etc.)
- **Evidence** — how each belief was formed (which model, what score, which heuristic)
- **Affordances** — what operations make sense for that type (can_embed, can_segment, can_transcribe)
- **History & updates** — how the understanding of an object/type changed over time

All other behavior — identity stitching, consistency checks, proposals for new types — lives here.

### 3.2 LLM (Cortex / Reasoning Layer)

The LLM is **not** the world model. It’s the **reasoning module** that:

- Interprets raw signatures + context
- Proposes type assignments for objects
- Suggests new subtypes when patterns recur
- Proposes slot values (attributes) and relationships
- Helps resolve ambiguous cases

The LLM operates over **structured prompts** that include:

- Raw object metadata
- Extracted signatures
- Candidate types
- Similar known objects and their types/slots
- Existing type hierarchy
- Pending type proposals

The LLM **consults** the semantic engine; it does not overwrite it arbitrarily.

### 3.3 Vector DB / Embedding Store (Perceptual Cache)

This is **not** the world model either. It’s the **sensory layer**:

- Stores embeddings for objects (images, text, JSON structures, video, etc.)
- Supports fast nearest-neighbor queries (“what looks / reads / behaves like this?”)
- Helps the engine and LLM find similar examples when reasoning about new objects

The vector store is a **tool** the engine uses, not the place where meaning lives.

---

## 4. Core Data Model (v0)

### 4.1 Objects

A generic “thing” the system knows about.

- `object_id` (UUID)
- `source_id` (e.g. file path, URL, database key)
- `primary_type_id` (FK → Type)
- `canonical_object_id` (optional: for dedup/aliasing)
- `created_at`, `updated_at`
- `status` (active, merged, deleted, etc.)

Every “thing” at any level (file, image, entity inside an image, event, etc.) is an **Object**.

### 4.2 Types

Kinds of things in the universe.

- `type_id` (UUID)
- `name` (e.g. `"image"`, `"photo"`, `"wildlife_photo"`, `"product_record"`, `"chat_log"`)
- `parent_type_id` (for simple inheritance / hierarchy)
- `description` (human-readable)
- `embedding` (vector in “type space”, e.g. average of examples)
- `created_via` (`seed`, `inferred`, `user_confirmed`)
- `is_base_type` (bool)

Types can be very general (`image`) or more specific (`wildlife_photo`). The system can *propose* new types over time.

### 4.3 Evidence

How the system came to believe something about an object and its type.

- `evidence_id`
- `object_id`
- `type_id`
- `source` (e.g. `file_magic`, `heuristic`, `llm_v0`, `image_probe`, `json_parser`)
- `score` (0–1 confidence)
- `details` (JSON payload — logits, heuristic flags, prompt fragments, etc.)
- `created_at`

The engine computes final beliefs (e.g. primary_type) as an aggregation over evidence, not via one-shot classification.

### 4.4 Signatures

Modality- and method-specific fingerprints used for identity and similarity.

- `object_id`
- `kind` (e.g. `image_header`, `image_embedding`, `image_phash`, `json_schema`, `text_embedding`, `video_embedding`)
- `data` (hash / vector / small JSON with signature info)

These let the engine:
- detect duplicates / near-duplicates
- find similar objects for context during reasoning

### 4.5 Slots (Optional in v0, Essential Long-Term)

Objects can have **slots** (properties/relations), whose values can be:

- primitives (string, number, bool)
- references to other Objects
- lists of these

Conceptually:

```text
Object:
  id
  type_id
  slots: { slot_name → Value }

Value:
  - primitive
  - object_ref (object_id)
  - list[Value]
```

This lets you do nested structure: an image object referencing a leopard object, which references a species object, which has its own attributes, etc.

---

## 5. Processing a New Blob (High-Level Flow)

When a new blob (file/record/message) arrives, the engine doesn’t assume anything. It runs a **layered probe → infer → refine** cycle.

### Step 1 — Ingest

- Assign a new `object_id`.
- Store `source_id` and raw size.
- Compute `raw_hash` (strong hash of bytes).
- If `raw_hash` matches an existing object → treat as alias / duplicate.

### Step 2 — Universal Probes (Modality-Agnostic)

Without assuming the type, run cheap, generic tests:

- **Magic bytes / MIME sniffing** → initial hints (image, video, text, etc.)
- **UTF-8 decode test** → can be text?
- **Image header probe** → looks like a known image format?
- **Video container probe** → looks like a known container?
- **Entropy profile** → text vs media vs high-entropy binary?
- **Extension hint** (if present) → soft evidence only

Each probe emits **Evidence** rows for base types (`image`, `text`, `video`, `structured_record`, `binary`), with scores.

### Step 3 — Modality Confidence & Conditional Deep Probes

From the evidence, compute confidence per base type.

- If `image` confidence > threshold → run image-specific extractors:
  - EXIF
  - perceptual hash
  - image embedding
- If `text` → parse / embed
- If `structured_record` → JSON parse, schema signature
- If `video` → inspect headers, sample frames, frame embeddings
- Etc.

These deeper probes add more signatures + evidence.

### Step 4 — Type Inference (LLM-Assisted)

Now build a **structured prompt** for the LLM including:

- Probed metadata
- Signatures (summaries, not raw vectors)
- Candidate base types + scores
- Similar known objects & their types
- Existing type hierarchy
- Any relevant type proposals

Ask the LLM for a structured response:

- Most likely **primary type**
- Confidence
- Optional **candidate subtype** (new type proposal)
- Optional **slot candidates** (e.g. entities in an image, domain tags, etc.)

The engine then:
- stores this as Evidence
- recomputes `primary_type_id` based on all evidence so far
- optionally materializes important slot values

### Step 5 — Type Novelty & Proposals

When the LLM repeatedly suggests new subtypes, e.g. `"wildlife_photo"` under `"photo"`, the engine:

- groups suggestions by `(parent_type, proposed_name)`
- tracks support count and supporting objects
- once a threshold is hit:
  - creates a `TypeProposal`
  - waits for human or auto-accept logic
- when accepted:
  - inserts new `Type`
  - backfills Evidence for supporting objects

This is how the **type system grows over time** based on data, not manually pre-defined taxonomies.

---

## 6. Objects Within Objects (Non-Linear Structure)

The system should **not** assume a linear depth like:

> file → image → scene → entities → attributes

Instead, everything is just **Objects with types and slots**. For example:

- `O_file` (type = `file`)
  - `slots["content"] → O_image`

- `O_image` (type = `image`)
  - `slots["entities"] → [O_leopard, O_tree]`
  - `slots["time_context"] → O_time`

- `O_leopard` (type = `animal_instance`)
  - `slots["species"] → O_species_leopard`
  - `slots["pose"] → O_pose_resting`
  - `slots["perch"] → O_branch`
  - `slots["coat_pattern"] → O_coat_pattern`

- `O_coat_pattern` (type = `coat_pattern`)
  - `slots["base_color"] = "golden"`
  - `slots["pattern"] = "rosettes"`

This supports:
- arbitrary nesting
- attributes that are themselves objects
- relations represented either as slots or dedicated relation-objects

The **Object Sense Engine** doesn’t care about “levels”. It just recursively applies the same question: *“What is this thing?”* to any sub-part it’s asked to look at.

---

## 7. Roles Over Time

This v0 is deliberately modest in **surface area** but deep in **principle**.

Over time, this becomes the foundation for:

- **Wildlife world modeling**  
  Group photos into sightings, individuals, behaviors, habitats, etc.

- **Product/world modeling**  
  Understand products, stores, merchants, attributes, relationships.

- **Personal knowledge graph**  
  Notes, documents, chats, code, images — all as objects with types and links.

- **Agent memory & context**  
  Agents operate *against* this world model rather than ad-hoc RAG over blobs.

Everything that wants to “know” something should go through this layer.

---

## 8. What This is NOT

To keep the intent clear, this is **not**:

- Another LangChain-style orchestration script
- A nicer RAG pattern
- A specific taxonomy for products/wildlife/etc.
- A monolithic knowledge graph with hand-labeled edges
- A replacement for LLMs or VLMs

Instead, it is:

- **The place where objects, types, and relationships live**
- The **source of truth** for “what things exist and how they relate”
- The **interface layer** between raw data and higher-level tasks
- A **persistent memory + world model** beneath any reasoning layer

---

## 9. v0 Deliverable Shape (Practical)

A realistic v0 implementation in your stack could be:

- A small Python/FastAPI service (“object-sense”) running on Sparkstation
- A Postgres instance on TuskerBox storing:
  - `objects`
  - `types`
  - `evidence`
  - `signatures`
  - `type_proposals`
- A vector store (could be pgvector, Qdrant, etc.) for signatures
- A local LLM + VLM for:
  - type inference prompts
  - basic content understanding
- A CLI for playing with it:
  - `object-sense ingest <path>`
  - `object-sense classify <file>`
  - `object-sense review-types`
  - `object-sense show-object <id>`

Even without any UI, this is already interesting: you can start to **see your archive as a structured world of objects and types**, not just files and tags.

---

## 10. Future Directions (Beyond v0)

- **Identity Stitching v1**  
  Go beyond files → infer that multiple images belong to the same event (sighting) or individual.

- **Events and Processes**  
  Introduce Event objects (hunts, chases, feeding, plays) as first-class entities.

- **Cross-Domain Types**  
  Products, wildlife, documents, services – all in one ontology with shared primitives (`Object`, `Type`, `Slot`).

- **Constraint & Consistency Engine**  
  Detect and resolve contradictions (“this can’t be both a JPEG and a JSON record”).

- **Learning Loop**  
  Use feedback (accept/reject type proposals, corrections) to refine prompts, thresholds, and heuristics.

This is the **Great Work substrate**: the part that will still make sense 10–20 years from now, regardless of which LLM or VLM is in vogue.
