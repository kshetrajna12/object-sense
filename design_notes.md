# Design Notes — Object Sense Engine

## Discussion Log

### Topic: Bootstrap Problem

**The Question:**
How do you seed the initial type hierarchy? The system needs *some* structure to begin reasoning, but the vision is for types to emerge from data rather than be predefined.

**Tensions:**
- Too little structure → LLM has no anchors, proposals are chaotic
- Too much structure → just another hand-crafted taxonomy, defeats the purpose

**Options to explore:**

1. **Minimal base types only**
   - Start with ~5-10 fundamental modality types: `object`, `image`, `video`, `audio`, `text`, `structured_record`, `binary`
   - Everything else emerges from data
   - Risk: slow convergence, early chaos

2. **Seeded domain types**
   - For a specific use case (e.g., wildlife), seed with known useful types: `photo`, `animal`, `species`, `sighting`
   - Faster to useful, but less general
   - Risk: biases the system toward expected patterns

3. **Bootstrap from examples**
   - Feed the system a curated batch of diverse objects
   - Let it propose types, human accepts/rejects to establish initial vocabulary
   - More organic, but requires upfront curation effort

4. **Hierarchical seeding**
   - Seed only the top 1-2 levels (base types + broad categories)
   - Let specifics emerge: `image` → `photo` (seeded) → `wildlife_photo` (emerged)

**Sub-questions:**
- What's the minimal viable seed for v0?
- Should base types be immutable or can they evolve?
- How do you prevent the LLM from proposing types that duplicate seed types with different names?

---

### CORRECTION: Medium ≠ Type

**The fundamental error in the original framing:**
Treating `image`, `video`, `text`, `structured_record`, `binary` as "base types" conflates representation with meaning.

- **Medium** = how information is encoded
- **Type** = what the information actually is

| Medium | Type (examples) |
|--------|-----------------|
| image | wildlife_photo, product_image, portrait |
| video | hunt_event, tutorial, surveillance_clip |
| json | product_record, config, api_response |
| text | chat_log, article, code_file |

**Why this distinction is essential:**
A product can arrive as JSON, HTML, image, or video — but it's still conceptually a "product". If you use medium as type, you can never unify these.

---

## Corrected Ontological Primitives for v0

### 1. Medium (property, not type)
What channel/encoding does this come in?

- `image`
- `video`
- `audio`
- `text`
- `json`
- `binary`

These tell the engine what *operations* are possible. They are sensory, not conceptual.

### 2. Object
A unit in the world model. Everything is an Object — files, entities, events, attributes.

```
object_id = uuid
medium = image   ← says nothing about what it IS
```

### 3. Type
The semantic claim about what something IS:

- `photo`, `wildlife_photo`, `backlit_wildlife_photo`
- `product_record`, `product_page`
- `leopard_entity`, `tree_entity`
- `coat_pattern`, `pose`
- `hunt_event`, `feeding_event`
- `trip_record`, `sighting`

Types are *interpreted*, not derived from medium. The system proposes them.

### 4. Affordances
Functional capabilities tied to medium:

| Medium | Affordances |
|--------|-------------|
| image | can_embed, can_detect_objects, can_extract_exif |
| video | can_sample_frames, can_segment, can_extract_audio |
| json | can_parse_keys, can_infer_schema |
| text | can_chunk, can_embed, can_parse |
| binary | can_entropy_analyze, can_magic_sniff |

Affordances are NOT types. They're functional flags.

---

## Corrected Processing Flow

```
Blob arrives
    ↓
Step 1: Ingest
    → assign object_id
    → store source_id, raw_hash
    ↓
Step 2: Medium Probing
    → file header, extension, entropy, partial decode
    → Sets: medium = image
    → Does NOT assign type
    ↓
Step 3: Medium-based Extraction
    → Use affordances to pick extractors
    → Generate signatures, embeddings
    ↓
Step 4: LLM Type Inference
    → Input: signatures, embeddings, medium, similar objects
    → Output:
        primary_type = "wildlife_photo"
        sub_entities = [...]
        slots = {...}
        maybe_propose_new_type = "backlit_wildlife_photo"
    ↓
Step 5: Engine Binding
    → medium = image
    → type = wildlife_photo
    → nested objects = entities, events, attributes
    → evidence stored
```

---

## Why This Matters (Philosophy)

This is the difference between:
- ❌ A file classifier
- ✅ A world model

If medium = type, the system becomes:
- Brittle (tied to formats)
- Shallow (can't abstract)
- Incapable of unifying cross-medium representations of the same concept

Separating medium from type allows:
- A JSON product record
- A product page HTML
- A product image
- A product video

...to all converge into one conceptual "product" object.

---

## DECISION: Empty Semantic Type Space at Bootstrap

**The answer:** Start with tabula rasa for semantics, but not for perception.

### Why This Works

**1. Mediums give us perception without semantics**

The system can "see" via modality probes:
- Detect medium (image, video, text, json, audio, binary)
- Run appropriate extractors
- Compute signatures and embeddings
- Determine affordances

No semantic types needed to get perception working. This is the sensory cortex.

**2. Types emerge from data**

Given medium + signatures + embeddings + metadata + similar objects, the LLM proposes types like:
- `wildlife_photo`
- `product_record`
- `hunt_event`

The conceptual universe forms from the ground up.

**3. LLM queries type store before proposing (CRITICAL)**

This prevents chaos. On every new object:

```
(a) Engine searches type store:
    → search_types(query_embedding) → top-k candidates
    → search_similar_objects → examples with their types

(b) LLM sees only local neighborhood:
    → 5-10 most relevant existing types
    → a few similar objects
    → definitions and aliases

(c) LLM decides:
    → REUSE existing type
    → CREATE_NEW (if nothing fits)
    → REFINE existing (later)
    → SPLIT overloaded type (later)
```

"Look around before inventing something new."

**4. Types are created immediately, stabilized through evolution**

~~Early proposals are just evidence. Gates:~~
~~- Repetition threshold (e.g., 10+ occurrences)~~

CORRECTION: No threshold. Insert the first type immediately.

Why? If the system can modify/split/merge/deprecate later, waiting is just friction. The first proposal is as valid as the tenth.

**Required evolution mechanisms:**
- **Type embeddings** — detect when new proposal ≈ existing type
- **Alias table** — one canonical name, multiple surface forms
- **Merge** — combine two types, reassign objects
- **Split** — create subtypes, redistribute objects
- **Deprecate** — soft delete, migrate away

The ontology is always provisional, always revisable. This is the key insight.

Over time: types stabilize through use, not through gating.

**5. Naming chaos controlled via naming contract**

Not seeded types — constraints:
- `snake_case` names
- Singular nouns
- One core concept per name
- Reuse existing types when possible
- Only propose new when nothing fits
- All proposals go through canonicalization
- Similar proposals merge into aliases
- Type embeddings detect near-duplicates
- Type registry shown in prompts

This solves "wildlife_photo vs nature_image vs animal_picture" without premature hierarchy.

### The Pattern

```
Empty type space
    ↓
Object arrives
    ↓
Medium probing → perception
    ↓
Query type store → local neighborhood
    ↓
LLM proposes (reuse or new)
    ↓
If new: create type immediately (no threshold)
    ↓
Type evolution over time (alias, merge, split, deprecate)
    ↓
Self-organizing ontology
```

---

## Type Evolution Mechanisms (Critical for Immediate Insertion)

Since types are created immediately without threshold, the system MUST support:

### 1. Alias
Multiple names → one canonical type

```
"nature_image" → alias of "wildlife_photo"
"animal_pic" → alias of "wildlife_photo"
```

Detection: type embedding similarity above threshold
Action: auto-alias or flag for review

### 2. Merge
Two types discovered to be the same concept

```
merge("wildlife_photo", "nature_image")
→ pick canonical (or create new)
→ reassign all objects
→ deprecate the other
```

### 3. Split
One type discovered to be overloaded

```
split("photo")
→ "wildlife_photo", "portrait", "product_photo"
→ redistribute objects based on signatures/embeddings
```

### 4. Deprecate
Type no longer valid or useful

```
deprecate("old_type")
→ soft delete (keep for history)
→ migrate objects to replacement type
```

### 5. Refine / Subtype
More specific type emerges under existing type

```
"wildlife_photo" exists
LLM proposes "backlit_wildlife_photo"
→ create as subtype
→ parent relationship established
```

**Status:** Concept documented. Revisit mechanics during implementation.

---

---

## DECISION: Type vs. Slot Granularity

### The Core Rule

**Type** = a different kind of thing, with its own schema/participants/behavior

**Slot** = an axis of variation within the same kind of thing

### Examples

| Concept | Type or Slot? | Why |
|---------|---------------|-----|
| `wildlife_photo` vs `portrait` | Type | Different schema, affordances, use |
| `backlit` vs `frontlit` | Slot (`lighting`) | Same object kind, visual axis varies |
| `leopard`, `lion` | Type (`species_entity`) referenced via slot | Species have their own structure |
| `animal_entity` vs `tree_entity` | Type | Different conceptual kind, different slots |
| `hunt_event` vs `feeding_event` | Type | Different schema, participants, phases |
| `resting_pose` vs `stalking_pose` | Slot (`pose`) | Same entity, frequently varying property |

### Critical Insight: Rich Domains Are Entity Types

Species, locations, individuals — these are NOT bare string values.

They are **Types** (`species_entity`, `location_entity`) that objects reference via slots:

```
animal_entity:
  type: animal_entity
  slots:
    species: ref(species_entity:leopard)
    pose: "stalking"
    location: ref(location_entity:maasai_mara)

species_entity:leopard:
  type: species_entity
  slots:
    common_name: "Leopard"
    scientific_name: "Panthera pardus"
    conservation_status: "Vulnerable"
```

### The Decision Rule (for LLM prompts)

**Default:** Model new distinctions as **slots** on an existing Type.

**Promote to Type** when ALL or MOST of these hold:
1. Different schema (different required slots/participants)
2. Different pipelines/affordances/routing
3. Natural query root ("give me all hunt_events", "all wildlife_photos")
4. Has its own properties and relations (like species, location, individual)

**Everything else stays as slots** — especially continuous/enumerated axes like:
- lighting, pose, weather, time_of_day, mood, quality

### Canonical Representation

Wrong:
```
type: backlit_wildlife_photo
```

Correct:
```
type: wildlife_photo
slots:
  lighting: backlit
```

The engine MAY define convenience views/segments (e.g., "backlit_wildlife_photo" as a saved filter) but that's not a fundamental Type.

### Evolution: Demotion and Promotion

**Type → Slot Demotion:**

If we created `backlit_wildlife_photo` as a Type, but notice:
- It's always `wildlife_photo` + `lighting=backlit`
- No extra schema, no special routing

→ Demote: rewrite instances to `wildlife_photo` with slot
→ Keep `backlit_wildlife_photo` as alias/saved filter if useful

**Slot → Type Promotion:**

If a slot's value domain becomes rich (e.g., species needs taxonomy, conservation_status, typical_behavior):

→ Create `species_entity` Type
→ Turn `species` slot into a relational reference

### Summary

- Upfront principle: kinds = Types, variation = Slots
- Bias: prefer slots, only create Types when schema/behavior differs
- Evolution: demote pseudo-types, promote rich domains
- Ontology is provisional but biased toward clean separation

**Status:** Concept documented. Revisit mechanics during implementation.

---

---

## DECISION: Object Identity — Three Layers

Identity is NOT one problem. It's three distinct layers with different goals:

### Layer 0 — Blob / File Identity (storage-level)

**Goal:** Don't store the same bytes twice. Storage hygiene.

**Exact duplicate:**
- `sha256(raw_bytes)` → O(1) lookup in blob table
- If hit: new Object points at existing `blob_id`
- Keep separate Object rows (different provenance), reuse physical payload

**Near-duplicate (format/size changes):**
- Per-medium cheap signatures:
  - image: pHash / dHash
  - text: simhash
  - json: normalized content hash
- Flags obvious near-dups for UI hints or auto-aliasing

This layer is about storage & noise control. Don't overthink semantics here.

### Layer 1 — Observation Identity (object-level dedup)

**Goal:** Decide if two Objects are the "same observation" even if raw bits differ.

Example: Same photo exported at different resolutions or lightly cropped.

**Per-medium embedding + signature index:**
- image: CLIP/vision embeddings + pHash
- text: text embeddings + simhash
- json: semantic embedding over important fields + schema hash

**Pipeline:**

```
Step 1: Exact hash / blob reuse
    ↓
Step 2: Candidate retrieval
    → Query vector index (ANN) for top-k similar
    → Use blocking: same medium, rough time window, same source
    → Don't compare across entire universe
    ↓
Step 3: Verification
    → Hard filters (resolution, aspect ratio, EXIF, length)
    → If ambiguous: LLM/specialized model judgment
    → Ask: duplicate | near-duplicate | just similar
    ↓
Step 4: Decision
    → duplicate: alias to existing Object
    → near-duplicate: keep both, link via duplicate_of
    → similar: just related, no identity claim
```

**Key insight:** This is still about observations, not entities.
Two cameras shooting the same leopard from different angles = different observations here.

### Layer 2 — Entity Identity (the real problem)

**Goal:** Decide when multiple observations are about the same underlying entity or concept.

This is where ObjectSense actually lives.

**Two sub-problems:**

#### 2a. Same entity, same medium (e.g., "same leopard")

Specialized signals:
- Visual re-ID (spot/rosette patterns, scars, mane)
- Context: same location, similar timestamps, same guide/vehicle/trip
- Species consistency

Pipeline:
```
Group candidates via blocking:
    → same species
    → same area (MalaMala vs Kabini)
    → time window (same trip, same season)
    ↓
Within block:
    → run re-ID / embeddings
    → cluster by similarity
    ↓
Each cluster = candidate individual:
    → create/update animal_entity
    → link all observations to it
```

For v0: cluster within single trip/dataset, not across entire archive.
Leave room for fancier re-ID later.

#### 2b. Same concept, different medium (product JSON vs product image)

Rely on:
- External IDs: SKU, product_id, handle, URL
- Merchant/shop context
- Textual similarity (title/description vs ALT text/captions)
- Embeddings of text + extracted image attributes

Pipeline:
```
Object ingested
    ↓
Extract entity keys (product_id, trip_id, animal_name)
    ↓
If matching entity exists → link to it
If not → create new entity and link
```

Entity resolution here is:
- Iterative
- Schema-specific
- NOT one magical generic pass

### Signatures Per Medium (v0)

| Medium | Signatures |
|--------|------------|
| Image | SHA256 (bytes), pHash/dHash, vision embedding (CLIP), EXIF summary hash |
| Video | SHA256 (file), keyframe pHash fingerprints, pooled video embedding, duration bucket |
| Text | SHA256 (normalized), simhash, text embedding |
| JSON | schema hash (keys/structure), content hash (canonicalized), text embedding on important fields |

### Threshold Tuning

Don't guess by hand. Treat as learned/configurable:
- Start with conservative defaults
- Log: candidate pairs, decisions, confidence
- Allow human feedback ("these ARE the same", "these are NOT")
- Adjust thresholds over time / per corpus

For v0: hard-coded conservative thresholds + human review for merges.

### Canonical Object — The Right Model

**Blob-level:** First-seen is canonical; later ones are references.

**Object-level (observations):** Don't delete any observations.
- Mark group with `canonical_object_id` (best quality / most metadata)
- Others are aliases
- All point to same higher-level entity

**Entity-level:** No "merge OBS A into OBS B" problem.
- Create new node (`animal_entity`, `product_entity`)
- Link observations to it
- Sidesteps canonicalization headaches entirely

### Summary

| Layer | Goal | Method |
|-------|------|--------|
| 0 - Blob | Don't store same bytes twice | Hash, cheap signatures |
| 1 - Observation | Same observation, different bits? | Embeddings, ANN, verification |
| 2 - Entity | Same real-world thing? | Re-ID, context, clustering, linking |

Layer 0/1 = guardrails and optimization
Layer 2 = where the world model lives

**Status:** Concept documented. Revisit mechanics during implementation.

---

---

## DECISION: Cross-Medium Unification

### Core Insight

You never unify modalities directly.
You unify all modalities into the **same entity space**.

**Entities are the hub; modalities are just observation channels.**

### The Pattern

```
Observation (image, JSON, video, text)
            ↓
Domain-specific feature extraction
            ↓
Entity features (common format):
    - extracted_text
    - detected_attributes (species, product_attrs)
    - provenance (shop, trip, timestamp, location)
    - deterministic_ids (SKU, trip_id, GPS)
    - embeddings
            ↓
Generic entity resolution layer
            ↓
Entity (product_entity, animal_entity, etc.)
```

### Entity Unification Signals

Entities unify when **multiple signals converge**:

| Signal Type | Examples | Strength |
|-------------|----------|----------|
| Deterministic IDs | SKU, product_id, trip_id, GPS coords | Strong |
| Contextual buckets | Same trip, same shop, same time window | Medium |
| Semantic agreement | Same species, same product title | Medium |
| Multimodal similarity | CLIP-style embedding proximity | Weak (hint only) |

No single signal is enough. Convergence creates confidence.

### Which Observation Creates the Entity?

The **first observation with stable identity** creates the entity:

| Domain | Typical Creator | Why |
|--------|-----------------|-----|
| Products | JSON record | Has `product_id` |
| Wildlife | First photo with species+location | Establishes subject |
| Trips | Trip record / booking | Has `trip_id` |

Later observations link to the entity via entity features.

### Partial Matches

**Never force merges.** Represent uncertainty:

- Image shows a leopard, but unknown individual
  → `animal_entity: null` or `animal_entity: candidate(cluster_id)`

- JSON has product_id but no image yet
  → Entity exists, `primary_image` slot is null

- Two observations might be same entity but confidence is low
  → Keep separate, add `possible_same_as` relation

Entity linking is **incremental and revisable**.

### ~~Domain-Specific Pipelines, Common Resolution Layer~~

**CORRECTION: No domain-specific pipelines.**

Humans aren't born with domains. Neither should the system be.

---

## CORRECTION: Fully General Identity Engine

### The Deeper Insight

**Types crystallize out of the entity graph; they are an effect, not a cause.**

The order is:
```
Observation → Entity (clustering) → Type (crystallization) → Domain (emergent region)
```

NOT:
```
Domain → Type → Entity → Observation
```

### Single General Identity Engine

No "product pipeline" vs "wildlife pipeline". One engine that:

1. **Extracts modality-native features from every observation:**
   - Visual: embeddings, detected objects, colors, patterns
   - Textual: extracted text, embeddings, named entities
   - Structural: schema, keys, field patterns
   - Temporal: timestamps, sequences, durations
   - Spatial: GPS, locations, regions

2. **Merges into unified multimodal embedding + feature set:**
   - All features live in compatible spaces
   - Cross-modal alignment (CLIP-style where applicable)
   - Structured fields as typed features

3. **Clusters observations by recurring invariants:**
   - Names, IDs (deterministic)
   - Appearance patterns (visual re-ID)
   - Location-time patterns (spatiotemporal)
   - Structured field matches (schema-level)
   - Semantic similarity (embeddings)

4. **Entities emerge from clusters:**
   - When enough observations cluster around shared invariants
   - Entity = the stable core that observations point to
   - No domain logic required — just signal convergence

5. **Types crystallize from entity patterns:**
   - System notices: "these entities all have species, location, timestamp"
   - Proposes: "this is a pattern — call it wildlife_sighting"
   - Types are discovered, not predefined

6. **Domains emerge as dense regions:**
   - "Products" = dense region of entities with SKUs, prices, merchants
   - "Wildlife" = dense region with species, locations, sightings
   - Never explicitly created — naturally visible in the graph

### Cross-Medium Unification (Revised)

Handled by:
- **Shared entity candidates** — all observations propose into same candidate pool
- **Shared invariants** — IDs, names, patterns extracted uniformly
- **Shared feature spaces** — multimodal embeddings, typed features
- **Shared clustering** — one algorithm, not domain-specific matchers
- **Entity-first reasoning** — "what entity does this belong to?" not "what domain is this?"

### Summary (Corrected)

- **Don't compare modalities** — project all into unified feature space
- **Entities are the hub** — observations are channels
- **Convergence of invariants** — clustering, not matching
- **Types crystallize from entities** — effect, not cause
- **Domains emerge from density** — never explicitly defined
- **One general engine** — no domain-specific pipelines

**Status:** Concept corrected. This is the right model.

---

## EXAMPLE: First Ever Leopard Photo (The Infant Brain Model)

This walkthrough shows how the system handles its very first observation with zero prior knowledge.

### Step 1 — Raw observation arrives

```
blob_001: <image bytes>
```

No types. No domains. No species. No animal concepts. Pure tabula rasa.

### Step 2 — Medium probe

Perception only. No semantics.

```
medium: image
affordances: [can_embed, can_detect_objects, can_extract_exif, can_caption, can_crop_regions, can_compute_phash]
```

### Step 3 — Feature extraction

Vision-language model extracts:

```
features = {
  visual: [embedding],
  textual: ["leopard", "spotted big cat", "on rock"],
  detected_objects: ["leopard", "rock", "grass"],
  structural: {...},
  spatial: {gps: ...},
  temporal: {exif_time: ...},
  provenance: {camera: ..., source: ...}
}
```

Still no type. No entity. No domain. Just an observation with rich signals.

### Step 4 — Entity candidate generation

System asks: "Do these features match any existing entity clusters?"

First input = no existing clusters.

Answer: **No match → create new entity candidate.**

### Step 5 — Create proto-entity

```
entity_e001: {
  evidence: [blob_001],
  features: { species≈"leopard", ... },
  status: "proto-entity"
}
```

This is NOT yet `animal_entity` or `species_entity`. It's just "something" with no name.

Like an infant seeing a cat for the first time.

### Step 6 — LLM proposes provisional type hypotheses

```
primary_type_hypothesis: "photo"
secondary_type_hypothesis: "animal_photo"
tertiary_type_hypothesis: "big_cat_photo"
maybe_proposed_new_type: "leopard_photo"
```

These are **uncommitted**. Weakly suggested labels, not real types.

### Step 7 — Insert brand-new Types (provisional)

No types exist → system creates:

```
type_photo (first ever type!)
type_animal_photo
type_big_cat_photo
type_leopard_photo (may later be demoted to slot)
```

ALL can be merged, split, aliased, deprecated later. This is normal. This is emergence.

### Step 8 — No domains yet

Domains require many types + connectivity.

Current state:
- 1 observation
- 1 proto-entity
- 2-4 provisional types
- 0 domains

Perfect.

### Step 9 — Entity remains under-specified

```
entity_e001:
  slots: {
    species: "leopard"    ← literal string, NOT a species_entity yet
    time: <from EXIF>
    location: <gps if available>
  }
```

"leopard" is just text from captioning. Not promoted to entity yet.

**No premature entity creation.** One observation isn't enough to crystallize "species" as a concept.

### Step 10 — System waits for recurrence

When the next observation comes:
- Another leopard photo?
- A tiger photo?
- A JSON product record?
- A video of a lion?

Then:
```
Observation → compare to existing candidates → cluster → strengthen invariants → crystallize types
```

**Only recurrence hardens the ontology.**

### The Crystallization Moment (later)

After seeing:
- Multiple leopard photos
- Multiple mentions of "leopard" in text
- Multiple sightings with similar patterns

System proposes:
1. Create `species_entity` Type
2. Create `leopard_species` instance
3. Link entity candidates
4. Convert textual slot `"leopard"` → relational slot `species: ref(leopard_species)`

**The ontology improves itself through recurrence.**

### Summary: The Infant Cognitive Model

| Stage | What Happens |
|-------|--------------|
| See something | Observation with rich features |
| Represent it | Proto-entity (unnamed "thing I saw") |
| Repeated exposure | "Ah, that's a kind of thing" → Type crystallizes |
| Repeated kinds | "This region is a domain" → Domain emerges |

This is how categories, types, species, and domains form in the brain.

ObjectSense mirrors that.

### Key Principles

1. **Everything starts soft** — provisional types, proto-entities, uncommitted hypotheses
2. **No premature commitment** — "leopard" stays a string until recurrence justifies promotion
3. **Types are hypotheses first** — can be merged, split, aliased, deprecated
4. **Entities are candidates first** — proto-entities until clustering confirms
5. **Domains require density** — one observation = zero domains
6. **Crystallization through recurrence** — only repeated patterns harden into structure

---

## CORRECTION: LLM Priors + Emergence (The Aardvark Paradox)

### The Paradox

The "infant brain" model above is incomplete.

**Observation:** You (as an adult) can see an aardvark photo for the first time and immediately know what it is. No recurrence needed.

**Why?** Because you didn't start from tabula rasa. You had:
- A prior over animals
- A prior over mammals
- A prior over body plans (snouts, ears, legs)
- A prior over wildlife contexts

You weren't learning "aardvark from scratch." You were mapping a new exemplar onto existing priors.

**This is neuroscience, not philosophy.**

### The Resolution

ObjectSense cannot start with literally zero semantic priors.

It starts with:

1. **Innate priors (from the LLM)**

   The LLM already knows:
   - What an animal is
   - What a species is
   - What a photo is
   - What a product is
   - That aardvarks → mammals → animals → living things

   These aren't "domains." These are **latent conceptual priors** encoded in the LLM weights.

2. **Local emergent types (from data recurrence)**

   What we described earlier:
   - Repeated sightings
   - Repeated structured slots
   - Repeated properties
   → crystallize local types

3. **Transfer from priors → new observations**

   When the LLM sees an aardvark photo (even the first one ever):
   - It already has a rich latent manifold for animals
   - It identifies unique features → "aardvark"
   - It stamps it into a type lineage immediately:
     ```
     photo → wildlife_photo → mammal_photo → aardvark_photo
     ```

   The LLM's prior knowledge is enough to create a brand-new type on the spot.

### The Hybrid Model

**The emergent ontology does NOT replace LLM priors — it sits ON TOP of them.**

| Source | Role |
|--------|------|
| LLM priors | Instant type proposals, even on first observation |
| Recurrence | Stabilization, validation, pruning hallucinations |

### Corrected Pipeline

```
1. Observation arrives
   → Extract features and metadata

2. LLM injects prior semantic interpretation
   → Even first-ever aardvark photo:
     "this is an animal"
     "this is a mammal"
     "this is probably an aardvark"
     "aardvarks have long snouts, claws, termite-eating behavior"

3. System proposes new type: aardvark_species
   → Created on first sight (no recurrence needed)
   → LLM priors make this valid

4. Local ontology grows from experience
   → After multiple aardvark photos:
     - Confirms type is real
     - Links entities
     - Distinguishes individuals
     - Builds cluster

5. Domains emerge as type graph expands
   → Enough animal-related types cluster
   → "Wildlife" domain becomes visible
```

### What Recurrence Is Actually For

**NOT for inventing types.** The LLM can do that instantly.

**FOR:**
- Validating that a type is real (not hallucinated)
- Merging duplicate type names
- Pruning types that don't hold up
- Creating stable slots
- Building durable entity clusters
- Crystallizing domains

### Final Correct Understanding

| Capability | Source |
|------------|--------|
| Propose "aardvark" as a type instantly | LLM priors |
| Propose "wildlife_photo" immediately | LLM priors |
| Assign species and abstract types on first try | LLM priors |
| Validate these types are real | Recurrence |
| Merge duplicates | Recurrence |
| Prune hallucinations | Recurrence |
| Build durable entity clusters | Recurrence |
| Crystallize domains | Recurrence |

### TL;DR

- Humans don't need two examples to learn a new category → because we have strong priors
- ObjectSense inherits the LLM's semantic priors → so it can also instantiate new categories from a single observation
- Recurrence is only needed to **stabilize, refine, and verify** — not to **invent**

**Status:** This is the correct hybrid model.

---

---

## Implementation Decisions Log

### 2024-12-22: Core Data Model Implementation

Implemented SQLAlchemy models per concept_v1.md §3 and §9.

**Decisions made:**

1. **Vector dimensions: Sparkstation models (configurable)**
   - Using local Sparkstation LLM gateway models (not OpenAI)
   - **bge-large**: 1024-dim text embeddings → `Type.embedding`, `Signature.text_embedding`
   - **clip-vit**: 768-dim image embeddings → `Signature.image_embedding`
   - Separate columns for different modalities (can't unify without projection/padding)
   - Dimensions configured in `config.py`: `dim_text_embedding`, `dim_image_embedding`
   - Models read from `settings` at class definition time — change once, applies everywhere

2. **ObjectEntityLink as explicit association table**
   - concept_v1.md shows `entity_links: [FK → Entity]` on Object
   - Implemented as separate `ObjectEntityLink` table with:
     - Composite primary key (object_id, entity_id)
     - Optional `role` field (e.g., "subject", "background", "depicted")
   - Enables richer relationship metadata and bidirectional navigation

3. **Signature.metadata → Signature.extra**
   - Renamed to avoid collision with SQLAlchemy's `DeclarativeBase.metadata`
   - Field stores modality-specific signature data (EXIF, duration, schema info, etc.)

4. **Database initialization: simple create_all()**
   - Using `Base.metadata.create_all()` for v0
   - No Alembic migrations yet — acceptable for prototyping
   - Will need migrations before production use

5. **Async-first with asyncpg**
   - All database operations async via SQLAlchemy 2.0 async API
   - `async_sessionmaker` for session management
   - FastAPI lifespan handler calls `init_db()` on startup

**Tables created:**
- `blobs` - Content-addressable storage (SHA256 dedup)
- `types` - Semantic types with parent hierarchy, aliases, embeddings
- `entities` - Persistent concepts with slots and confidence
- `objects` - Observations with medium, primary_type, slots
- `object_entity_links` - Many-to-many Object↔Entity
- `evidence` - Belief provenance (polymorphic via subject_kind)
- `signatures` - Modality-specific fingerprints
- `type_evolution` - Type change history (alias, merge, split, deprecate)

---

## Open Questions for Next Discussion

### Unified feature space for cross-medium clustering (object-sense-cqz)

**Decided:** Store embeddings in separate columns by modality (text=1024, image=768).

**Still open:** How to unify for cross-medium entity clustering?
- Option A: Use CLIP for both text and image queries (768-dim shared space)
- Option B: Project text embeddings to image space (or vice versa)
- Option C: Late fusion — cluster within modality, then link entities across modalities
- Option D: Learn a projection layer that maps both to common space

CLIP already supports text→768 and image→768, so Option A may be simplest for v0.

(Add new questions as they arise)

---

*Notes continue below as discussion progresses...*
