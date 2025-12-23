# CLAUDE.md - Project Instructions

## Session Start
- Run `bd ready` to see available work
- Run `bd list --status=in_progress` to see ongoing work
- Check `git status` for any uncommitted changes from previous sessions

## Session End
- Create beads issues for any discovered/remaining work
- Run tests if code changed
- Close completed issues: `bd close <id>`
- Sync and push: `bd sync && git push`
- Verify: `git status` should show "up to date with origin"

## Commit Preferences
- Do NOT include "Co-Authored-By: Claude" or "Generated with Claude Code" in commits

## Project Overview
Object Sense Engine — a semantic substrate that provides persistent object identity and type awareness beneath all other AI systems.

## Key Concepts

### CRITICAL DISTINCTION: Medium ≠ Type
- **Medium** = how information is encoded (image, video, text, json, audio, binary)
- **Type** = what the information actually IS (wildlife_photo, product_record, hunt_event)

Mediums are about representation. Types are about meaning. Never conflate them.

### Core Primitives
- **Object**: Any unit in the world model (files, entities, events, attributes — everything is an Object)
- **Medium**: The encoding/channel — a *property* of an object, NOT a type (image, video, text, json, audio, binary)
- **Type**: The semantic claim about what something IS — interpreted, not derived from medium
- **Affordances**: Functional capabilities tied to medium (can_embed, can_segment, can_parse) — NOT types
- **Slots**: Properties/relations on objects, values can be primitives or references to other Objects
- **Evidence**: How the system came to believe something (source, score, details)
- **Signatures**: Modality-specific fingerprints for identity/similarity (hashes, embeddings, schemas)

## Architecture Principles
1. **Semantic Engine** is the source of truth — not the LLM, not the vector store
2. **LLM** proposes but doesn't dictate — it's a reasoning module consulted by the engine
3. **Vector DB** is perceptual/sensory — similarity lookup, not meaning storage
4. Types emerge from data (TypeProposals) rather than being predefined taxonomies

## Core Loop
For every new blob:
1. Have I seen this exact object before? (identity/dedup)
2. Have I seen this kind of object before? (type inference)
3. If not, should I create a new type? (ontological expansion)

## Design Decisions

### Bootstrap: Empty Semantic Type Space
- Mediums provide perception (what extractors to run) — no seeding needed
- Semantic type space starts EMPTY
- Types emerge from data via LLM proposals
- LLM always queries type store before proposing (RAG for types)
- Naming chaos controlled via naming contract, not seeded hierarchy

### Type Creation: Immediate, No Threshold
- First proposal creates the type immediately — no waiting for N occurrences
- The ontology is always provisional, always revisable
- Required evolution mechanisms: alias, merge, split, deprecate
- Types stabilize through use, not through gating

### Type vs. Slot Granularity
- **Type** = different kind of thing, with its own schema/participants/behavior
- **Slot** = axis of variation within the same kind of thing
- **Default:** prefer slots, only create Types when schema/behavior differs
- **Rich domains become entity Types:** species, locations, individuals are NOT bare strings — they're Types referenced via slots
- **Evolution:** demote pseudo-types to slots, promote rich slot domains to Types

### Object Identity — Three Layers
Identity is NOT one problem. Three distinct layers:

| Layer | Goal | Method |
|-------|------|--------|
| 0 - Blob | Don't store same bytes twice | Hash, cheap signatures (pHash, simhash) |
| 1 - Observation | Same observation, different bits? | Embeddings, ANN, blocking, verification |
| 2 - Entity | Same real-world thing? | Re-ID, context, clustering, entity linking |

- Layer 0/1 = guardrails and optimization
- Layer 2 = where the world model lives
- Entities are NEW NODES that observations link to (not merging observations into each other)
- Threshold tuning: start conservative, log decisions, learn from feedback

### Cross-Medium Unification & General Identity Engine
- **No domain-specific pipelines** — humans aren't born with domains, neither is the system
- **One general identity engine** extracts modality-native features (visual, textual, structural, temporal, spatial)
- **Unified multimodal feature space** — all observations project into same space
- **Entities emerge from clustering** around recurring invariants (IDs, appearance, location-time, schema patterns)
- **Types crystallize from entity patterns** — types are an EFFECT, not a cause
- **Domains emerge as dense regions** of the conceptual graph — never explicitly defined

### The Emergence Order
```
Observation → Entity (clustering) → Type (crystallization) → Domain (emergent region)
```
NOT: Domain → Type → Entity → Observation

### The Hybrid Model: LLM Priors + Emergence

**The Aardvark Paradox:** You can recognize an aardvark on first sight — no recurrence needed. Why? Because you have strong priors (animal, mammal, body plan, wildlife context).

ObjectSense works the same way:

| Source | Role |
|--------|------|
| **LLM priors** | Instant type proposals, even on first observation |
| **Recurrence** | Stabilization, validation, pruning hallucinations |

**The LLM already knows:**
- What an animal/species/photo/product is
- That aardvarks → mammals → animals → living things
- These are latent conceptual priors, not hardcoded domains

**Recurrence is NOT for inventing types** (LLM does that instantly).

**Recurrence IS FOR:**
- Validating types are real (not hallucinated)
- Merging duplicate names
- Pruning types that don't hold up
- Building durable entity clusters
- Crystallizing domains

**TL;DR:** The emergent ontology sits ON TOP of LLM priors, not instead of them.

See `design_notes.md` for full walkthrough and "Aardvark Paradox" resolution.

## Design Clarifications
- **LLM interface**: 100% tool calling and structured outputs
- **Unified feature space**: Open question — how visual/textual/structural features align technically
- **Query interface**: Not a concern for now
- **v0 scope**: Defined in concept_v1.md §11 (image + text + JSON; wildlife + products)

## Tech Stack (Planned v0)
- Python/FastAPI service
- Postgres (objects, types, evidence, signatures, type_proposals)
- pgvector or Qdrant for embeddings
- Local LLM + VLM for inference

## Tooling & Code Style
- **Dependency management**: UV (`uv add`, `uv sync`, `uv run`)
- **Formatting**: ruff format
- **Linting**: ruff check
- **Type checking**: pyright or mypy
- **Testing**: pytest
- **Python version**: 3.12+

### Conventions
- snake_case for all Python code
- Type hints on all public functions
- Docstrings only where behavior isn't obvious from name/types
- Prefer explicit over clever

## Files
- `inital_idea.md` — Original high-level concept document
- `design_notes.md` — Ongoing design discussion notes
- `concept_v1.md` — **Complete concept document** (this is the canonical reference)

## Beads Workflow
- Use beads (`bd`) for tracking multi-session work, dependencies, and discovered issues
- When closing an "OPEN:" beads issue, document the decision in `design_notes.md`
- Beads tracks **work items**; docs track **knowledge and decisions**

## Skills
- Proactively suggest new skills when repetitive patterns emerge or when a reusable capability would help
- Skills live in `.claude/skills/`
