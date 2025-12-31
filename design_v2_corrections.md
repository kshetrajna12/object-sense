# Design v2 Corrections — Type Authority & Entity Resolution Clarifications

> **Status:** DRAFT
> **Created:** 2025-12-28
> **Supersedes:** concept_v1.md (preserves v1 for reference)

---

## Preamble

### Why These Changes?

The v1 design (`concept_v1.md`) established the core philosophy correctly but contains several **semantic tensions** that will cause confusion during implementation:

1. **"Object" terminology overload** — sounds like "thing in world" but means "unit of observation"
2. **Type authority ambiguity** — "types crystallize from entities" (philosophy) vs. "primary_type_id assigned immediately" (implementation)
3. **Entity resolution underspecified** — no concrete algorithm, thresholds, or versioning strategy
4. **Type lifecycle unclear** — "create immediately (no threshold)" risks type explosion without safeguards
5. **Entity heterogeneity ignored** — individual leopards vs. leopard species vs. hunt events need different resolution strategies

These corrections resolve these tensions while **preserving the core philosophy**:
- Medium ≠ Type
- LLM priors + emergence
- Entities as hubs
- Engine decides, LLM proposes

---

## The 10 Corrections

### 1. Rename "Object" → "Observation"

#### Problem
**"Object" is semantically overloaded.**

When people hear "ObjectSense resolves objects," they assume:
- "Object" = thing in the world (leopard, product, location)

But the system actually means:
- "Object" = unit of observation (photo file, JSON record, video frame)

The actual "things in the world" are **Entities**.

This confusion is baked into the naming and will cause endless miscommunication.

#### Change

**Rename everywhere:**
- `Object` → `Observation`
- `ObjectEntityLink` → `ObservationEntityLink`
- `object_id` → `observation_id`
- All documentation, schemas, models

**Add explicit definition in §1:**

> **Observation**: A unit of data we received (file, record, frame, note). Observations are what we ingest.
>
> **Entity**: A persistent thing in the world (leopard, product, location). Entities are what observations refer to.

**Update the Emergence Order:**

```
Observation → Entity → Type → Domain
```

Not:
```
Object → Entity → Type → Domain
```

#### Rationale

- **Eliminates the #1 source of confusion** in explaining the system
- **Makes the architecture self-documenting** — "observation" clearly means "data point"
- **Aligns with academic terminology** — ML/AI papers use "observation" for data instances
- **Makes Emergence Order literal, not rhetorical** — observations are inputs, entities are outputs

#### Implementation Impact

**Breaking changes:**
- Rename SQLAlchemy model: `Object` → `Observation`
- Rename table: `objects` → `observations`
- Rename all foreign keys: `object_id` → `observation_id`
- Update all imports, tests, documentation
- Migration script for existing database

**Estimated effort:** 2-3 hours (mostly mechanical find-replace)

---

### 2. Fix Step 4/5 Semantics: Split "Type" into Routing Hint vs. Ontological Type

#### Problem

**The current design conflates two different concepts under "Type":**

1. **Routing hint** — "this looks like a wildlife photo, use vision extractors"
2. **Ontological type** — "wildlife_photo is a stable semantic category with N entities"

Assigning `primary_type_id` immediately in Step 4 makes the routing hint **authoritative**, which contradicts "types crystallize from entities."

If downstream systems trust `primary_type_id` for identity decisions, you get **taxonomy-led identity** (the thing you're trying to avoid).

#### Change

**Split Step 4 into two conceptually distinct outputs:**

##### 4A. Semantic Probe (routing + facets)

**Produces:**
- `observation_kind` (string, routing hint)
  - Examples: "wildlife_photo", "product_record", "json_config"
  - **Non-authoritative** — can flip freely, used for indexing/routing only
  - Stored in `observations.observation_kind` (not FK to types table)

- `facets` (JSONB, attribute hypotheses)
  - Examples: `{"detected_objects": ["leopard", "tree"], "lighting": "backlit"}`
  - Soft priors for entity resolution

- `entity_seeds` (list of EntityHypothesis)
  - Candidate entities the LLM detected
  - Input to Step 5

##### 4B. Type Proposal (non-authoritative evidence)

**Produces:**
- `TypeCandidate` rows (if LLM proposes a new type label)
  - Stored as Evidence, not as a promoted Type
  - See Correction #7 for lifecycle

**Remove:**
- `observations.primary_type_id` (foreign key)
  - Replace with `observations.observation_kind` (string)

**Add immediately (LLM-assigned, mutable):**
- `observations.candidate_type_id` (foreign key to `type_candidates`, nullable)
  - Set by Step 4B when LLM proposes a type
  - Can change as candidates merge/alias
  - Distinct from `observation_kind` (routing hint, low-cardinality string)

**Add later (after entity stabilization):**
- `observations.stable_type_id` (foreign key to `types`, nullable)
  - Set only after Type promotion (Correction #7)
  - Can remain NULL for observations that don't fit stable types

> **Terminology note:** The column is named `stable_type_id`; in prose we may also call it "promoted type". These refer to the same concept.

**The three-level type reference:**

| Field | Purpose | Cardinality | Mutability |
|-------|---------|-------------|------------|
| `observation_kind` | Routing hint | Low (~20 values) | Can flip freely |
| `candidate_type_id` | LLM's type proposal | High (all candidates) | Changes on merge/alias |
| `stable_type_id` | Promoted ontological type | Medium (stable types) | Set once on promotion |

**CRITICAL: observation_kind vs candidate_type_id**

These are NOT the same thing:

- **`observation_kind`** is a **low-cardinality routing string** (e.g., "wildlife_photo", "product_record", "json_config")
  - Fixed vocabulary (~20 values max)
  - Used ONLY for pipeline routing and soft search filtering
  - Does NOT reference type_candidates table
  - Does NOT track LLM's specific type proposals

- **`candidate_type_id`** is the **FK to type_candidates**
  - High cardinality (one per LLM-proposed type label)
  - Tracks the actual type proposal
  - Follows merge chains when candidates consolidate
  - Eventually promotes to stable_type_id

All semantic type labeling flows through `candidate_type_id`, not `observation_kind`.

#### Rationale

**This preserves compute benefits without authority leakage:**
- ✅ Step 5 can still use `observation_kind` to constrain search ("only look at animal entities")
- ✅ LLM priors still inform routing
- ❌ Types cannot hard-gate identity decisions
- ❌ Early "type" labels don't become the truth anchor

**It matches the philosophy:**
- "Types crystallize from entities" — yes, stable_type_id is only set after entity clustering
- "LLM proposes, engine decides" — yes, observation_kind is a proposal, not gospel

#### Implementation Impact

**Schema changes:**
- Add `observations.observation_kind: string` (routing hint, not FK)
- Add `observations.facets: JSONB`
- Add `observations.candidate_type_id: FK → type_candidates | null` (LLM-assigned)
- Remove `observations.primary_type_id: FK`
- Add `observations.stable_type_id: FK → types | null` (set after promotion)

**Code changes:**
- Type inference agent outputs `observation_kind` instead of `primary_type`
- Step 5 uses `observation_kind` for soft routing, ignores it for identity
- Promotion logic (Correction #7) sets `stable_type_id`

---

### 3. Add Type Non-Authoritativeness Contract

#### Problem

Without explicit constraints on what Types are **allowed to influence**, the system will drift toward "taxonomy-first identity."

Developers will be tempted to write:
```python
if observation.primary_type == "wildlife_photo":
    only_search_animal_entities()  # WRONG: hard-gating by type
```

This poisons the identity graph with taxonomy assumptions.

#### Change

**Add a new section to the spec: § Type Non-Authoritativeness Contract**

```markdown
## Type Non-Authoritativeness Contract

Types exist to improve retrieval and UX, NOT to determine identity.

### Types MAY:
- Improve retrieval/routing as a **soft prior**
  - Example: "observation_kind=wildlife_photo → boost animal entity candidates"
- Generate descriptions and UX groupings
  - Example: "Show me all observations with stable_type=wildlife_photo"
- Inform feature extraction pipelines
  - Example: "product_record → extract price, SKU fields"

### Types MAY NOT:
- **Hard-gate candidate search** in entity resolution
  - ❌ "Only search animal entities if type=wildlife_photo"
  - ✅ "Boost animal entities if observation_kind=wildlife_photo"
- **Force entity merges/splits**
  - ❌ "These must be the same entity because both are type=leopard_photo"
- **Be used as a deterministic identity feature**
  - ❌ `entity_id = hash(type, location, timestamp)`
  - ✅ `entity_id = cluster(embeddings, signatures, deterministic_ids)`

### Enforcement:
- Entity resolution tests must pass with **randomized observation_kind values**
- Identity must remain correct even if all types are "unknown"
- Code reviews explicitly check for type authority leakage
```

#### Rationale

**This is the critical safety valve.**

Without it, the system will inevitably become "types determine identity," and you'll have built a taxonomy engine.

This contract makes the philosophical claim **testable and enforceable**.

#### Implementation Impact

**Testing:**
- Add test: entity resolution with observation_kind="unknown" → should still work
- Add test: flip observation_kind mid-stream → entities should not break

**Code review checklist:**
- No `if observation.observation_kind == X` in identity logic
- All type-based routing must be soft (boost scores, not hard filters)

---

### 4. Tighten Type Definition: What a Type Actually Is

#### Problem

The v1 definition says:
> "A Type is a stabilized label over an entity cluster"

But then also:
- Types have parent relationships
- Types have embeddings
- Types are created immediately (no threshold)
- Types have status: provisional | stable | deprecated

**These are inconsistent.** If types are "stabilized labels over entity clusters," they shouldn't exist before clusters stabilize.

#### Change

**Clarify the two-stage Type lifecycle** (see Correction #7 for full details):

**Stage 1: TypeCandidate (provisional label)**
- Created immediately when LLM proposes a new label
- Status: `proposed`
- Has: name, aliases, evidence pointers
- Does NOT have: parent, embedding, stable status
- Stored in `type_candidates` table

**Stage 2: Type (stable semantic primitive)**
- Promoted from TypeCandidate after validation
- Status: `stable`, `merged`, `deprecated`
- Has: parent relationships, embedding (computed from entity cluster), evidence count
- Stored in `types` table
- Safe to expose to downstream consumers

**Update the Type definition in §3.3:**

> **Type**: A semantic category validated by entity structure and recurrence.
>
> Types are promoted from TypeCandidates when:
> - Evidence count ≥ N (e.g., 5 entities or 20 observations)
> - Coherence score above threshold (cluster tightness)
> - Survives time window W without being merged/contradicted
>
> Types are **effects**, not causes. They crystallize from patterns in the entity graph.

#### Rationale

This resolves the philosophical tension:
- ✅ "Types created immediately" → TypeCandidates created immediately
- ✅ "Types crystallize from entities" → Types promoted after entity stabilization
- ✅ "LLM priors" → TypeCandidates use LLM knowledge
- ✅ "Emergence discipline" → Types require evidence

#### Implementation Impact

**New table:** `type_candidates`
```sql
CREATE TABLE type_candidates (
  candidate_id UUID PRIMARY KEY,
  proposed_name VARCHAR NOT NULL,
  normalized_name VARCHAR NOT NULL,  -- for dedup detection
  status VARCHAR DEFAULT 'proposed',  -- proposed | promoted | merged | rejected
  merged_into_candidate_id UUID REFERENCES type_candidates(candidate_id),
  promoted_to_type_id UUID REFERENCES types(type_id),
  evidence_count INT DEFAULT 1,
  created_at TIMESTAMP,
  updated_at TIMESTAMP
);
-- No UNIQUE constraint on proposed_name (see §7 for rationale)
```

**Modified table:** `types`
```sql
-- Only contains promoted types
-- Has parent_type_id, embedding, stable status
```

**Promotion logic:**
- Background job or triggered on entity stabilization
- Checks thresholds (evidence_count, coherence, time window)
- Creates Type row, updates TypeCandidate.promoted_to_type_id

---

### 5. Make Entity Resolution (Step 5) an Explicit Algorithm

#### Problem

**Step 5 is currently vague:**
- "Soft clustering" — what algorithm?
- "Propose links with confidence scores" — what thresholds?
- "Proto-entities compete, merge, stabilize" — how?

Without concrete specification, implementation will be arbitrary and non-reproducible.

#### Change

**Add a new section: § Step 5 Entity Resolution Specification**

```markdown
## Step 5: Entity Resolution Specification

### Inputs
1. `observation_id` (the new observation being processed)
2. `observation.embeddings` (text, image, clip_text)
3. `observation.signatures` (hashes, pHash, simhash)
4. `observation.facets` (extracted attributes)
5. `observation.deterministic_ids` (SKU, product_id, trip_id, GPS coords)
6. `entity_seeds` from Step 4 (EntityHypothesis objects)
7. `observation_kind` (routing hint)

### Outputs
1. `ObservationEntityLink` rows with:
   - `entity_id` (which entity this observation links to)
   - `posterior` (confidence score 0-1)
   - `status` (soft | hard | candidate)
   - `role` (subject | background | context)

2. `Entity` updates:
   - Create new proto-entities if no match
   - Update entity.confidence
   - Update entity.slots (merge facets)

3. `MergeProposal` / `SplitProposal` events (for later review)

### Algorithm (Pseudocode)

```python
def resolve_entities(observation, entity_seeds):
    links = []

    # 1. Deterministic ID resolution (HIGHEST PRIORITY)
    for det_id in observation.deterministic_ids:
        entity = lookup_by_deterministic_id(det_id)
        if entity:
            links.append(Link(entity, posterior=1.0, status='hard'))
        else:
            # ID not found → CREATE entity anchored to this ID
            entity = create_entity_for_deterministic_id(det_id)
            links.append(Link(entity, posterior=1.0, status='hard'))
        continue  # Deterministic ID handled; skip similarity search

    # 2. For each entity seed from LLM
    # Note: Deterministic ID resolution is per-seed. Seeds without IDs
    # (e.g., species, location, event) are still resolved via similarity.
    for seed in entity_seeds:
        # Get candidate pool (routing hint used HERE as soft filter)
        candidates = get_entity_candidates(
            entity_nature=seed.entity_nature,  # See Correction #6
            observation_kind=observation.observation_kind,  # Soft boost
            limit=100
        )

        # 3. Multi-signal similarity
        scores = []
        for candidate in candidates:
            score = compute_entity_similarity(
                observation, candidate,
                signals=['embedding', 'signature', 'facet_agreement', 'locality']
            )
            scores.append((candidate, score))

        # 4. Decision with thresholds
        best_candidate, best_score = max(scores, key=lambda x: x[1])

        if best_score > T_link (e.g., 0.85):
            # High confidence link
            links.append(Link(best_candidate, posterior=best_score, status='soft'))

        elif best_score > T_new (e.g., 0.60):
            # Uncertain - create candidate link but also proto-entity
            links.append(Link(best_candidate, posterior=best_score, status='candidate'))
            proto = create_proto_entity(seed, observation)
            links.append(Link(proto, posterior=1.0-best_score, status='candidate'))

        else:
            # No match - create new proto-entity
            proto = create_proto_entity(seed, observation)
            links.append(Link(proto, posterior=0.8, status='soft'))

    # 5. Multi-seed consistency pass (lightweight reconciliation)
    links = reconcile_multi_seed_links(links, entity_seeds)

    # 6. Store all links
    return links


def reconcile_multi_seed_links(links, entity_seeds):
    """
    Lightweight consistency check across all seeds from one observation.

    Detects contradictions like:
    - Same entity linked with conflicting roles
    - Multiple seeds resolving to same entity with incompatible facets
    - Circular references (entity A member of B, B member of A)

    Does NOT do heavy logic in v0 — just flags and downweights.
    """
    # Group links by entity
    by_entity = defaultdict(list)
    for link in links:
        by_entity[link.entity_id].append(link)

    reconciled = []
    for entity_id, entity_links in by_entity.items():
        if len(entity_links) == 1:
            reconciled.append(entity_links[0])
            continue

        # Multiple seeds linking to same entity — check consistency
        if has_role_conflict(entity_links):
            # Downweight all conflicting links
            for link in entity_links:
                link.posterior *= 0.5
                link.flags.append('multi_seed_role_conflict')

        if has_facet_contradiction(entity_links, entity_seeds):
            # Flag for review but don't reject
            for link in entity_links:
                link.flags.append('multi_seed_facet_conflict')

        reconciled.extend(entity_links)

    return reconciled
```

### Decision Thresholds (Configurable)

- `T_link` = 0.85 — High confidence, link to existing entity
- `T_new` = 0.60 — Uncertain, create proto-entity
- `T_margin` = 0.10 — If `best_score - second_best_score < T_margin`, flag for review
- **"Don't know" band:** [T_new, T_link] — create candidate links to both

### Threshold Tuning Strategy

1. Start with conservative defaults (above)
2. Log all decisions with scores
3. Allow human feedback: "these ARE same" / "these are NOT same"
4. Adjust thresholds via:
   - Per-corpus calibration
   - Per-entity_nature calibration (individuals vs. classes need different thresholds)
   - Active learning on margin cases

### Entity Versioning & Canonical Resolution

**Critical: Entity IDs must be versioned with canonical resolution.**

When entities merge or split:
- Generate new `entity_id` for merged/split result
- Store merge/split event in `entity_evolution` table
- Update `canonical_entity_id` on merged source entities
- Observations keep links to old IDs (still valid for history)
- Consumers resolve via `canonical_entity_id` for current truth

**Canonical resolution mechanism:**

```sql
-- Add to entities table
ALTER TABLE entities ADD COLUMN canonical_entity_id UUID REFERENCES entities(entity_id);
-- NULL means "I am canonical"; non-NULL points to current canonical entity

-- Entity evolution tracks history
CREATE TABLE entity_evolution (
  event_id UUID PRIMARY KEY,
  event_type VARCHAR NOT NULL,  -- merge | split
  source_entity_ids UUID[] NOT NULL,
  target_entity_id UUID NOT NULL REFERENCES entities(entity_id),
  created_at TIMESTAMP NOT NULL DEFAULT now(),
  evidence JSONB
);
```

**Canonical resolution is REQUIRED but implementation is flexible:**

Queries must resolve to canonical entities. Implementation options (v0 chooses one):

1. **App-layer pointer chasing** — Python follows `canonical_entity_id` chain
2. **DB helper function** — PL/pgSQL or SQL recursive CTE
3. **Materialized view** — Pre-computed canonical mappings, refreshed on merge

Example (app-layer):
```python
def resolve_canonical(entity_id: UUID, session: Session) -> UUID:
    """Follow canonical_entity_id chain to find current canonical."""
    current = entity_id
    while True:
        entity = session.get(Entity, current)
        if entity.canonical_entity_id is None:
            return current
        current = entity.canonical_entity_id
```

The requirement is that consumers always get canonical entities. The mechanism is an implementation detail.

**Merge semantics:**
1. Create merged entity (target_entity_id)
2. Set `canonical_entity_id = target_entity_id` on all source entities
3. Record in entity_evolution
4. ObservationEntityLinks remain valid (point to source)
5. Queries use `resolve_canonical()` to get current entity

**Split semantics:**
1. Create new child entities
2. Original entity either:
   - Remains canonical (children are new observations)
   - Gets deprecated (set canonical to one of children)
3. Record in entity_evolution with evidence explaining split

This prevents the "ontology stability for consumers" problem (see v1 Open Questions).

#### Rationale

**This makes Step 5 implementable and reproducible.**

Without this spec:
- Every developer will implement a different algorithm
- Tuning will be impossible (no shared parameters)
- Debugging will be nightmare ("why did this link?")

With this spec:
- Algorithm is explicit and testable
- Thresholds are knobs you can tune
- Entity versioning prevents consumer breakage

#### Implementation Impact

**New module:** `src/object_sense/resolution/entity_resolver.py`

**New tables:**
- `entity_evolution` (for merge/split tracking)

**New config:**
```python
# config.py
entity_resolution:
  t_link: 0.85
  t_new: 0.60
  t_margin: 0.10
```

**Tests:**
- Unit tests for `compute_entity_similarity` with known pairs
- Integration tests for full resolution pipeline
- Threshold sensitivity tests

---

### 6. Add Entity Nature: Individual vs. Class vs. Group vs. Event

#### Problem

**Not all entities resolve the same way.**

The v1 design lists entity types:
- Concrete individuals (The Marula Leopard)
- Abstract concepts (Leopard Species)
- Compositional groups (Kambula Pride)
- Hierarchical abstractions (Mammal Class)

But provides **no guidance** on how to resolve them differently.

If you use one resolver:
- Individual leopards need perceptual re-ID (rosette patterns)
- Leopard species needs language agreement ("leopard" = "Panthera pardus")
- Kambula Pride needs member tracking + location clustering
- Hunt events need temporal clustering + participant linking

**One resolution strategy cannot handle all of these.**

#### Change

**Add a required field: `entity.entity_nature`**

**Allowed values:**
- `individual` — Concrete, identifiable instances (The Marula Leopard, iPhone 15 #ABC123)
- `class` — Abstract categories (Leopard Species, Blue Color, Nike Brand)
- `group` — Collections with membership (Kambula Pride, Safari Trip 2024-06-12)
- `event` — Temporal occurrences (Hunt Event, Feeding Event, Purchase Transaction)

**CRITICAL CLARIFICATION: Signal weighting, not separate pipelines**

> `entity_nature` selects **signal weighting profiles**, not separate pipelines.
>
> There is ONE entity resolver with shared machinery. The `entity_nature` determines
> which signals are weighted heavily vs. lightly for similarity computation.

This preserves the anti-domain-partitioning principle (Correction #10).

**Signal weighting profiles per entity_nature:**

```python
# In Step 5 entity resolution — ONE resolver, different weights

# Signal weight profiles (sum to 1.0 per nature)
SIGNAL_WEIGHTS = {
    'individual': {
        'embedding': 0.35,      # Visual/perceptual re-ID
        'signature': 0.25,      # pHash, rosette patterns
        'location': 0.15,       # Locality prior
        'timestamp': 0.10,      # Temporal proximity
        'deterministic_id': 0.15,  # Hard IDs if present
    },
    'class': {
        'text_embedding': 0.40,    # Language/semantic agreement
        'facet_agreement': 0.30,   # Attribute consistency
        'name_match': 0.20,        # Exact/fuzzy name matching
        'deterministic_id': 0.10,  # Taxonomic IDs if present
    },
    'group': {
        'member_overlap': 0.35,    # Shared members
        'location': 0.25,          # Spatial clustering
        'temporal_proximity': 0.25, # Time-based grouping
        'deterministic_id': 0.15,  # Group IDs if present
    },
    'event': {
        'timestamp': 0.30,         # Temporal clustering
        'participants': 0.30,      # Entity involvement
        'location': 0.20,          # Where it happened
        'duration': 0.10,          # Temporal extent
        'deterministic_id': 0.10,  # Event IDs if present
    },
}

def compute_entity_similarity(observation, candidate, entity_nature):
    """ONE resolver function, different weights per nature."""
    weights = SIGNAL_WEIGHTS[entity_nature]
    total = 0.0
    for signal, weight in weights.items():
        if weight > 0:
            score = compute_signal_similarity(observation, candidate, signal)
            total += weight * score
    return total
```

**Update EntityHypothesis schema:**

```python
class EntityHypothesis(BaseModel):
    entity_type: str  # e.g., "animal_entity", "species_entity"
    entity_nature: Literal['individual', 'class', 'group', 'event']  # NEW
    suggested_name: str | None
    slots: list[SlotValue]
    confidence: float
    reasoning: str | None
```

**Update Entity model:**

```python
class Entity(Base):
    entity_id: Mapped[UUID]
    type_id: Mapped[UUID | None]
    entity_nature: Mapped[str]  # NEW: individual | class | group | event
    slots: Mapped[dict]
    status: Mapped[EntityStatus]
    confidence: Mapped[float]
```

#### Rationale

**This prevents "one resolver, many failure modes."**

Without entity_nature:
- You'll build hidden domain logic ("if species then...")
- Resolution will be inconsistent
- Debugging will be impossible ("why did these merge?")

With entity_nature:
- Resolution strategy is explicit and testable
- Each strategy has appropriate signals
- You can tune thresholds per nature

**This also makes the LLM's job clearer:**
- "Is this an individual thing or a category?"
- Forces reasoning about the ontological type

#### Implementation Impact

**Schema changes:**
- Add `entities.entity_nature` (required field)
- Update `EntityHypothesis` schema

**Code changes:**
- Type inference agent must propose `entity_nature` for each seed
- Entity resolver dispatches to appropriate strategy
- Each strategy has separate implementation + thresholds

**Prompt changes:**
- Type inference system prompt must explain entity_nature
- Examples for each nature type

---

### 7. Type Lifecycle: TypeCandidate → Type Promotion

#### Problem

**"Types created immediately (no threshold)" will cause type explosion.**

If every LLM proposal becomes a first-class Type:
- Long-tail noise: "backlit_spotted_leopard_at_sunset_photo"
- Hallucinations: "mysterious_glowing_orb_entity"
- Inconsistency: "wildlife_photo" vs. "nature_image" vs. "animal_picture"

Your own v1 Open Questions section hints at this:
> "Evidence compaction — storing every belief forever doesn't scale"

#### Change

**Implement a two-stage Type lifecycle:**

##### Stage A: TypeCandidate (Immediate Creation)

**Created:** Immediately when LLM proposes a type label (no threshold)

**Properties:**
- `candidate_id`: UUID
- `proposed_name`: string (snake_case)
- `status`: proposed | promoted | merged | rejected
- `promoted_to_type_id`: UUID | null (FK to types table)
- `evidence_count`: int (how many observations/entities reference this)
- `coherence_score`: float (cluster tightness, computed periodically)
- `created_at`, `updated_at`

**Stored in:** `type_candidates` table (separate from `types`)

**Can be used for:**
- Type labeling via `observations.candidate_type_id` FK
- UI display as "tentative label"
- Similarity search ("find candidates near this name")

**Cannot be used for:**
- Hard identity gating
- Schema enforcement
- Downstream consumer APIs (too unstable)

**Note:** `observation_kind` is a separate low-cardinality routing field. TypeCandidates are referenced via `candidate_type_id` FK, not via observation_kind.

##### Stage B: Type (Promoted, Stable)

**Promoted when ANY of these conditions met:**

1. **Evidence threshold:**
   - `evidence_count >= N` (e.g., 5 entities OR 20 observations reference it)

2. **Coherence threshold:**
   - Cluster tightness score > C (e.g., 0.75)
   - Measured as: avg pairwise similarity of entities with this type candidate

3. **Time window:**
   - Survives W days (e.g., 7 days) without being:
     - Merged into another candidate
     - Contradicted by evidence
     - Flagged as hallucination

**Promotion process:**

```python
def promote_type_candidate(candidate):
    # Create stable Type
    type_row = Type(
        type_id=uuid4(),
        canonical_name=candidate.proposed_name,
        aliases=[],
        parent_type_id=infer_parent(),  # Based on entity cluster
        embedding=compute_type_embedding(candidate),  # From member entities
        status='stable',
        evidence_count=candidate.evidence_count,
        created_via='candidate_promotion',
    )

    # Update candidate
    candidate.status = 'promoted'
    candidate.promoted_to_type_id = type_row.type_id

    # Update observations
    update_observations_with_stable_type(candidate, type_row)
```

**Properties of promoted Type:**
- Has parent relationships (inferred from entity graph)
- Has embedding (computed from entity cluster)
- Status: stable | deprecated | merged_into
- Safe to expose in APIs, UX, downstream systems

##### Stage C: Continuous Evolution

Even promoted Types can:
- Merge (two Types → one)
- Split (one Type → subtypes)
- Deprecate (soft delete, migrate)
- Alias (multiple names → one canonical)

**Key invariant:**
> "Promoted Types are stable enough for consumers, but not immutable."

#### Rationale

**This resolves the philosophical tension:**

✅ **"LLM priors"** — TypeCandidates created instantly using LLM knowledge
✅ **"Types crystallize from entities"** — Types promoted only after entity evidence
✅ **"Immediate creation"** — Yes, but as candidates not authoritative types
✅ **"No explosion"** — Promotion thresholds prevent long-tail noise

**This matches human cognition:**
- See aardvark photo → "aardvark" (hypothesis/candidate)
- After multiple reinforcements → "aardvark" (stable concept/type)

**Concrete benefits:**
- Consumers get stable types (not churning labels)
- Evidence compaction: can GC rejected candidates after time window
- Tunable explosion control (adjust N, C, W thresholds)

#### Implementation Impact

**New table:** `type_candidates`
```sql
CREATE TABLE type_candidates (
  candidate_id UUID PRIMARY KEY,
  proposed_name VARCHAR NOT NULL,
  normalized_name VARCHAR NOT NULL,  -- lowercase, stripped, for dedup detection
  status VARCHAR DEFAULT 'proposed',  -- proposed | promoted | merged | rejected
  merged_into_candidate_id UUID REFERENCES type_candidates(candidate_id),
  promoted_to_type_id UUID REFERENCES types(type_id),
  evidence_count INT DEFAULT 1,
  coherence_score FLOAT,
  created_at TIMESTAMP,
  updated_at TIMESTAMP,
  last_referenced_at TIMESTAMP
);

-- NO global UNIQUE on proposed_name
-- Duplicates allowed; merge/alias lifecycle handles consolidation
CREATE INDEX idx_type_candidates_normalized ON type_candidates(normalized_name);
```

**Design decision: Allow duplicate candidate names**

Why no UNIQUE constraint:
- LLM may propose "wildlife_photo" and "Wildlife Photo" separately
- Different observations may propose semantically identical types
- Merge/alias lifecycle consolidates these over time
- Forcing uniqueness at insert time creates race conditions

The lifecycle handles duplicates:
1. Insert candidate with proposed_name (no constraint)
2. Background job detects duplicates via normalized_name
3. Merge lower-evidence candidates into higher-evidence ones
4. Set `merged_into_candidate_id` on merged candidates
5. Observations follow the merge chain via `candidate_type_id`

**TypeCandidate merge-chain resolution:**

When candidates merge via `merged_into_candidate_id`, observations must resolve to the canonical candidate.

Resolution strategy (v0 chooses one):
- **Write-time:** When a candidate is merged, update all observations' `candidate_type_id` to the merge target
- **Query-time:** Follow `merged_into_candidate_id` chain at read time (like entity canonical resolution)

Write-time is simpler for v0 but requires migration on each merge.
Query-time is more flexible but adds read complexity.

**Modified schema:**
- `observations.candidate_type_id` references `type_candidates.candidate_id` (FK, nullable)
- `observations.stable_type_id` references `types.type_id` (FK, nullable)

**Background jobs:**
1. **Promotion job** (runs hourly):
   - Check candidates against thresholds
   - Promote eligible candidates
   - Update observations

2. **Cleanup job** (runs daily):
   - Mark candidates as `rejected` if:
     - `evidence_count == 1` and `age > 30 days`
     - `last_referenced_at > 90 days ago`
   - Garbage collect rejected candidates after 1 year

**Code changes:**
- Type inference agent outputs candidate names
- Entity resolver uses candidate names for routing
- API endpoints filter to stable types only (unless opt-in to candidates)

---

### 8. Add Deterministic ID Overrides Everything (Core Principle)

#### Problem

**This is currently buried in v1 Open Questions:**

> "Deterministic ID priority — Noted — hard IDs (SKU, product_id, trip_id) should explicitly trump semantic similarity in entity resolution."

This is not a "noted" item. **This is a core architectural principle.**

Without it:
- LLM says "these are different products" → system ignores SKU match
- Visual similarity low → system creates duplicate entities despite same product_id
- Chaos

#### Change

**Promote to §2 Core Philosophy as Principle #5:**

```markdown
## 5. Deterministic IDs Dominate Identity

If an observation contains a **deterministic identifier** (SKU, product_id, trip_id, GPS coords, hash), entity linking is **anchored** to it.

Similarity signals (embeddings, facets) only help:
- Fill in missing attributes
- Detect ID errors/corruption
- Link observations without deterministic IDs to entities with them

### What qualifies as a deterministic ID?

- **Strong IDs** (100% trust):
  - SKU, UPC, ISBN
  - product_id, trip_id, booking_id
  - SHA256 hash (for blob-level identity)
  - GPS coordinates (for location entities)
  - Database primary keys

- **Weak IDs** (high trust, but verify):
  - Filenames (can be renamed)
  - URLs (can change)
  - Timestamps (can have duplicates)

### Resolution priority:

1. **Deterministic ID match** → link with `posterior=1.0, status='hard'`
2. **Deterministic ID not found** → **CREATE entity for this ID**, then hard link
3. **No deterministic ID** → fall back to similarity clustering
4. **ID conflict** (same ID, incompatible attributes) → flag for review, trust ID over similarity

**CRITICAL: Deterministic IDs can CREATE entities**

When an observation has a deterministic ID that doesn't exist in the system:
- The resolver MUST create an entity anchored to that ID
- This prevents duplicate entities for first-time IDs
- The ID becomes the entity's identity anchor

```python
def resolve_deterministic_id(id_type, id_value, id_namespace):
    entity = lookup_by_deterministic_id(id_type, id_value, id_namespace)
    if entity:
        return Link(entity, posterior=1.0, status='hard')
    else:
        # CREATE entity anchored to this ID
        entity = create_entity_for_id(id_type, id_value, id_namespace)
        return Link(entity, posterior=1.0, status='hard')
```

This ensures that two observations with the same SKU always link to the same entity, even if the first observation creates it.

### Anti-pattern to avoid:

❌ **Never let similarity override deterministic IDs:**

```python
# WRONG
if embedding_similarity < 0.5:
    create_new_entity()  # Ignores that SKU matches!

# RIGHT
if deterministic_id_match:
    link_to_entity(posterior=1.0)
elif embedding_similarity > T_link:
    link_to_entity(posterior=similarity)
else:
    create_proto_entity()
```

### Exception handling:

If deterministic ID says "same entity" but attributes are wildly incompatible:
- Flag as `conflict` in Evidence
- Create `ConflictReview` task
- Prefer "duplicate entry" over "ID reuse" hypothesis
```

**Update Step 5 algorithm** (see Correction #5) to check deterministic IDs **first**.

#### Rationale

**This prevents LLM-led identity poisoning.**

Deterministic IDs are **ground truth**. If you have them, use them.

Similarity is a fallback for when you don't have IDs or need to cluster unidentified observations.

**This also matches real-world systems:**
- E-commerce: SKU is law
- Booking systems: trip_id is law
- Photo archives: GPS + timestamp is strong evidence

#### Implementation Impact

**Schema changes:**

Deterministic IDs must be stored as **(id_type, id_value, id_namespace)** tuples:

- Add `observations.deterministic_ids: JSONB` to store extracted IDs
  ```json
  [
    {"id_type": "sku", "id_value": "PROD-12345", "id_namespace": "acme_corp"},
    {"id_type": "product_id", "id_value": "abc-xyz", "id_namespace": "internal"},
    {"id_type": "gps", "id_value": "-25.7461,28.1881", "id_namespace": "wgs84"}
  ]
  ```

**Why namespace matters:**
- SKU "12345" from Acme Corp ≠ SKU "12345" from Beta Inc
- product_id is only unique within a source system
- GPS coordinates need datum specification

**New table: `identity_conflicts`**

When deterministic ID matches but attributes conflict:

```sql
CREATE TABLE identity_conflicts (
  conflict_id UUID PRIMARY KEY,
  observation_id UUID NOT NULL REFERENCES observations(observation_id),
  entity_id UUID NOT NULL REFERENCES entities(entity_id),
  id_type VARCHAR NOT NULL,
  id_value VARCHAR NOT NULL,
  id_namespace VARCHAR NOT NULL,
  conflict_type VARCHAR NOT NULL,  -- attribute_mismatch | duplicate_entry | id_collision
  conflict_details JSONB NOT NULL, -- what specifically conflicts
  resolution_status VARCHAR DEFAULT 'pending',  -- pending | resolved | ignored
  resolved_by UUID,  -- user or system that resolved
  resolved_at TIMESTAMP,
  created_at TIMESTAMP NOT NULL DEFAULT now()
);

CREATE INDEX idx_identity_conflicts_pending
  ON identity_conflicts(resolution_status) WHERE resolution_status = 'pending';
```

**Code changes:**
- Feature extraction must detect and extract deterministic IDs as tuples
- Entity resolver checks IDs **before** running similarity search
- On ID match with attribute conflict → create `identity_conflicts` row
- Resolution UI/API for human review of conflicts

**Prompt changes:**
- Type inference agent must extract deterministic IDs from facets
- Mark them as high-confidence in EntityHypothesis

---

### 9. Slot Hygiene: Structured Slots to Prevent Entropy

#### Problem

**Slots stored as JSONB with no constraints will devolve into chaos:**

```python
# All of these are "valid" but incompatible:
observation.slots = {"price": 42.99}
observation.slots = {"price": "42.99"}
observation.slots = {"price": "$42.99"}
observation.slots = {"price": {"value": 42.99, "currency": "USD"}}
```

Querying, indexing, and slot→type promotion become impossible.

#### Change

**Add a "Slot Hygiene" rule in §3.4:**

```markdown
## Slot Hygiene Contract

Every slot value must be **either**:

1. **Primitive with metadata:**
   ```json
   {
     "price": {"value": 42.99, "unit": "USD", "type": "currency"}
   }
   ```

2. **Typed entity reference:**
   ```json
   {
     "species": {
       "ref_entity_id": "uuid-here",
       "ref_type": "species_entity",
       "display_name": "Leopard"
     }
   }
   ```

### Slot types:

- `string` — Free text (use sparingly)
- `number` — Numeric value with optional unit
- `boolean` — True/false
- `enum` — One of a fixed set (e.g., lighting: backlit | frontlit | sidelit)
- `reference` — Link to another entity
- `list` — Ordered collection (homogeneous types)

### Anti-patterns:

❌ Freeform strings for rich domains:
```json
{"species": "leopard"}  // WRONG: species should be an entity reference
```

✅ Entity references:
```json
{"species": {"ref_entity_id": "...", "ref_type": "species_entity"}}
```

❌ Untyped numbers:
```json
{"length": 42}  // WRONG: 42 what? mm? cm? inches?
```

✅ Numbers with units:
```json
{"length": {"value": 42, "unit": "mm"}}
```

### Enforcement:

- Slot validation functions in `src/object_sense/utils/slots.py`
- Type inference agent outputs structured slots (not raw strings)
- Promotion from primitive to reference is explicit (migration function)

### Benefits:

- **Slot → Type promotion** becomes implementable (you can detect when a slot domain is rich)
- **Indexing** is possible (extract .value fields for numeric/enum slots)
- **Querying** is consistent (no "price" vs. "$price" ambiguity)
```

#### Rationale

**This is the only way slot→type promotion can work.**

If slots are arbitrary JSON soup, you can't detect:
- "This slot has 50 unique values, maybe it should be an entity type"
- "This slot is always a reference, promote it"

With structured slots:
- You can count unique entity references
- You can measure value diversity
- You can run migrations

#### Implementation Impact

**New module:** `src/object_sense/utils/slots.py`

```python
from typing import Any, Literal

SlotType = Literal['string', 'number', 'boolean', 'enum', 'reference', 'list']

def validate_slot(name: str, value: Any) -> dict:
    """Validate and normalize a slot value."""
    # Returns structured format or raises ValidationError

def is_slot_reference(value: dict) -> bool:
    """Check if slot value is an entity reference."""
    return 'ref_entity_id' in value

def extract_slot_for_index(value: dict) -> Any:
    """Extract indexable value from structured slot."""
    # For Postgres GIN/BTREE indexes
```

**Schema enforcement:**
- Add CHECK constraints on observations.slots and entities.slots (Postgres JSONB)
- Or use application-level validation (more flexible)

**LLM prompt:**
- Type inference agent outputs slots in structured format
- Examples in system prompt

---

### 10. Anti-Goal: No Global Domain Partitioning in v0

#### Problem

**The risk of hidden domain silos.**

Even though the design says "no domain-specific pipelines," it's easy to drift into:

```python
# WRONG: implicit domain partitioning
if observation_kind == "wildlife_photo":
    use_wildlife_entity_resolver()
elif observation_kind == "product_record":
    use_product_entity_resolver()
```

This becomes "two systems in a trench coat."

#### Change

**Add to §10 (What This Is NOT) as an explicit anti-goal:**

```markdown
## Anti-Goal: No Global Domain Partitioning in v0

ObjectSense must remain **domain-agnostic** at the core.

### Allowed:

✅ **Pluggable feature extractors per medium:**
- `ImageExtractor` for medium=image
- `TextExtractor` for medium=text
- `JsonExtractor` for medium=json

✅ **Entity resolution strategies per entity_nature:**
- `IndividualResolver` for entity_nature=individual
- `ClassResolver` for entity_nature=class

✅ **Evidence-based routing:**
- "observation_kind=wildlife_photo → boost animal_entity candidates (soft prior)"

### Forbidden:

❌ **Domain-specific resolvers:**
```python
# WRONG
if is_wildlife_domain(observation):
    wildlife_pipeline()
elif is_product_domain(observation):
    product_pipeline()
```

❌ **Domain config files:**
```yaml
# WRONG
domains:
  wildlife:
    entity_types: [animal_entity, species_entity]
  products:
    entity_types: [product_entity, brand_entity]
```

❌ **Hard-coded domain assumptions:**
```python
# WRONG
if "animal" in observation_kind:
    assert entity.entity_nature == 'individual'
```

### The rule:

> **Only one entity resolver.**
> It may dispatch to strategies (per entity_nature), but there is no top-level domain fork.

### Why this matters:

- Domains are **emergent**, not config
- Cross-domain unification (product photo of animal) must work
- Prevents "we have two systems" problem
- Forces general-purpose design

### Testing enforcement:

- Integration test: wildlife observation + product observation → should both resolve
- No `if domain ==` anywhere in entity resolution code
- Code review checklist
```

#### Rationale

**This is the hardest discipline to maintain.**

It's **so much easier** to write:
```python
if observation.observation_kind.startswith("wildlife"):
    # Special wildlife logic
```

But the moment you do, you've lost the generality.

Making this an explicit anti-goal with test enforcement is the only way to prevent drift.

#### Implementation Impact

**Testing:**
- Add test: mixed corpus (wildlife + products + text notes) → all resolve correctly
- Add test: no `if observation_kind ==` in entity resolution (code scanning)

**Code review:**
- Checklist item: "No domain-specific branches in core resolution"

**Documentation:**
- Add this to contributor guide

---

## Updated Data Model

### Core Tables (Modified from v1)

```
BLOB                     OBSERVATION              TYPE_CANDIDATE          TYPE
(storage)                (data point)             (provisional)           (stable)
┌──────────┐            ┌──────────┐             ┌──────────┐            ┌──────────┐
│ blob_id  │◀──────────│observation│             │candidate │            │ type_id  │
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
                        │canonical_│  ← NULL = I am canonical; else → merged target
                        │entity_id │
                        │ slots    │
                        │confidence│
                        └──────────┘

IDENTITY_CONFLICTS                    ENTITY_EVOLUTION
(ID match + attr conflict)            (merge/split history)
┌──────────────┐                      ┌──────────────┐
│ conflict_id  │                      │ event_id     │
│observation_id│                      │ event_type   │  ← merge | split
│ entity_id    │                      │ source_ids[] │
│ id_type      │                      │ target_id    │
│ id_value     │                      │ evidence     │
│ id_namespace │                      └──────────────┘
│ conflict_type│
│ status       │
└──────────────┘
```

### New/Modified Tables

**New:**
- `type_candidates` — Provisional type labels (Stage A)
  - No UNIQUE on proposed_name (duplicates allowed, merge handles consolidation)
  - Has `normalized_name` index for dedup detection
  - Has `merged_into_candidate_id` for merge chains
- `entity_evolution` — Merge/split tracking for entity versioning
- `identity_conflicts` — Deterministic ID conflicts for human review

**Modified:**
- `objects` → `observations`
  - Add: `observation_kind` (string, routing hint, low-cardinality)
  - Add: `facets` (JSONB, extracted attributes)
  - Add: `deterministic_ids` (JSONB array of {id_type, id_value, id_namespace})
  - Add: `candidate_type_id` (FK → type_candidates, LLM-assigned, mutable)
  - Remove: `primary_type_id` (FK)
  - Add: `stable_type_id` (FK → types, nullable, set after promotion)

- `entities`
  - Add: `entity_nature` (string: individual | class | group | event)
  - Add: `canonical_entity_id` (FK → entities, NULL = canonical, else merged target)

- `types`
  - Only contains promoted types (stable)
  - Remove provisional types (move to type_candidates)

---

## Updated Processing Pipeline

### Step-by-Step (Modified from v1)

```
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: INGEST                                             │
│  • Assign observation_id (UUID)                             │
│  • Store source_id, compute blob hash                       │
│  • Check blob dedup (SHA256)                                │
└────────────────────────┬────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 2: MEDIUM PROBING                                     │
│  • File header, extension, entropy                          │
│  • Sets: medium = image | video | text | json | audio       │
│  • Does NOT assign type (medium ≠ type!)                    │
└────────────────────────┬────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 3: FEATURE EXTRACTION                                 │
│  • Visual: embeddings, detected objects, EXIF               │
│  • Textual: text, embeddings, named entities                │
│  • Structural: schema, keys, fields                         │
│  • Deterministic IDs: SKU, product_id, trip_id, GPS         │
│  • Signatures: pHash, simhash, etc.                         │
└────────────────────────┬────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 4: SEMANTIC PROBE (LLM)                               │
│                                                              │
│  4A. Routing + Facets                                       │
│  • observation_kind (routing hint, non-authoritative)       │
│  • facets (extracted attributes)                            │
│  • entity_seeds (candidate entities)                        │
│                                                              │
│  4B. Type Proposal                                          │
│  • Create TypeCandidate if new label proposed               │
│  • Store as Evidence (non-authoritative)                    │
└────────────────────────┬────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 5: ENTITY RESOLUTION                                  │
│                                                              │
│  1. Deterministic ID check (HIGHEST PRIORITY)               │
│     → IDs are (id_type, id_value, id_namespace) tuples      │
│     → If match: hard link (posterior=1.0)                   │
│     → If NOT found: CREATE entity for ID, then hard link    │
│     → If conflict: create identity_conflicts row            │
│                                                              │
│  2. For each entity_seed (per-seed ID check; non-ID seeds    │
│     still resolved via similarity):                          │
│     a. Get candidates (routing hint as soft filter)         │
│     b. Compute similarity (signal weights per entity_nature)│
│     c. Apply thresholds (T_link, T_new, T_margin)           │
│     d. Create ObservationEntityLink(s)                      │
│                                                              │
│  3. entity_nature = signal weighting (NOT separate pipes):  │
│     • individual → weight: embedding, signature, location   │
│     • class → weight: text_embedding, facet_agreement       │
│     • group → weight: member_overlap, temporal_proximity    │
│     • event → weight: timestamp, participants, location     │
│                                                              │
│  4. Multi-seed consistency pass (lightweight):              │
│     → Detect role conflicts, facet contradictions           │
│     → Downweight or flag conflicting links                  │
│                                                              │
│  5. Update entity confidence, slots                         │
│  6. Generate MergeProposal / SplitProposal if needed        │
└────────────────────────┬────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 6: STORE & INDEX                                      │
│  • Persist observation, entity links, evidence              │
│  • Update type_candidates.evidence_count                    │
│  • Update vector indexes                                    │
└─────────────────────────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  BACKGROUND: TYPE PROMOTION (periodic)                      │
│  • Check type_candidates against thresholds                 │
│  • Promote eligible candidates → types                      │
│  • Update observations.stable_type_id                       │
│  • GC rejected candidates (after time window)               │
└─────────────────────────────────────────────────────────────┘
```

---

## Migration Path from v1

### Phase 1: Non-Breaking (Design Only)
1. Create `design_v2_corrections.md` (this document)
2. Review and refine with stakeholders
3. Update beads epic with implementation tasks

### Phase 2: Schema Changes (Breaking)
1. **Rename `objects` → `observations`**
   - Migration script for table rename
   - Update all foreign keys
   - Update all code references

2. **Add new fields to `observations`:**
   - `observation_kind: string` (routing hint)
   - `facets: JSONB`
   - `deterministic_ids: JSONB` (array of {id_type, id_value, id_namespace})
   - `candidate_type_id: FK → type_candidates | null`
   - `stable_type_id: FK → types | null`
   - Remove `primary_type_id`

3. **Add fields to `entities`:**
   - `entity_nature: string` (individual | class | group | event)
   - `canonical_entity_id: FK → entities | null` (merge chain)

4. **Create `type_candidates` table**
   - No UNIQUE on proposed_name
   - Add `normalized_name` index for dedup
   - Add `merged_into_candidate_id` for merge chain

5. **Create `entity_evolution` table**

6. **Create `identity_conflicts` table**

### Phase 3: Code Refactoring
1. Update type inference agent (Step 4A/4B split)
2. Implement entity resolution algorithm (Step 5)
3. Implement resolution strategies per entity_nature
4. Implement TypeCandidate promotion logic
5. Add deterministic ID detection and priority logic
6. Add slot validation utilities

### Phase 4: Testing & Validation
1. Unit tests for all new components
2. Integration tests for full pipeline
3. Threshold tuning on test corpus
4. Performance benchmarking

### Phase 5: Documentation
1. Update `concept_v1.md` → `concept_v2.md`
2. Update API documentation
3. Update contributor guide with anti-patterns

---

## Summary of Changes

| # | Change | Impact | Priority |
|---|--------|--------|----------|
| 1 | Object → Observation | Breaking, clarifies terminology | HIGH |
| 2 | Split Type into routing vs. ontological | Schema: observation_kind + candidate_type_id + stable_type_id | HIGH |
| 3 | Type Non-Authoritativeness Contract | Testing discipline | MEDIUM |
| 4 | Tighten Type definition | Conceptual clarity | LOW |
| 5 | Explicit Entity Resolution algorithm | New impl + multi-seed consistency + canonical resolution | HIGH |
| 6 | Add entity_nature field | Signal weighting profiles, NOT separate pipelines | HIGH |
| 7 | TypeCandidate lifecycle | No UNIQUE, merge chains, normalized_name index | HIGH |
| 8 | Deterministic ID priority | (id_type, id_value, id_namespace) + identity_conflicts table | HIGH |
| 9 | Slot hygiene rules | Validation utilities | MEDIUM |
| 10 | Anti-goal: no domain partitioning | Testing discipline | MEDIUM |

---

## Next Steps

1. **Review this document** — discuss, refine, approve
2. **Create beads epic** — break down into implementable tasks
3. **Start with Phase 2** — schema changes (Object → Observation)
4. **Implement Step 5** — entity resolution algorithm
5. **Test on real corpus** — wildlife + products
6. **Iterate** — tune thresholds, refine strategies

---

*End of Design v2 Corrections*
