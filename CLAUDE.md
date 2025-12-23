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
- **Always add descriptions** when creating issues — titles alone lose context across sessions
- When closing an "OPEN:" beads issue, document the decision in `design_notes.md`
- Beads tracks **work items**; docs track **knowledge and decisions**

## Skills
- Proactively suggest new skills when repetitive patterns emerge or when a reusable capability would help
- Skills live in `.claude/skills/`


<!-- SPARKSTATION-START -->
# Sparkstation Local LLM Gateway

This project has access to local LLM models through Sparkstation gateway.

## Available Models

- `gpt-oss-20b` - openai/gpt-oss-20b
- `bge-large` - BAAI/bge-large-en-v1.5
- `clip-vit` - openai/clip-vit-large-patch14
- `qwen3-vl-4b` - Qwen/Qwen3-VL-4B-Instruct-FP8
- `flux-dev` - black-forest-labs/FLUX.1-dev

## API Endpoint

- **Base URL**: `http://localhost:8000/v1`
- **Protocol**: OpenAI-compatible API
- **Authentication**: Use any string as API key (e.g., `"dummy-key"`)

## Usage with OpenAI Python SDK

```python
from openai import OpenAI

# Initialize client pointing to local Sparkstation gateway
client = OpenAI(
    api_key="dummy-key",  # Any value works
    base_url="http://localhost:8000/v1"
)

# Make a request
response = client.chat.completions.create(
    model="qwen3-vl-4b",  # or "gpt-oss-20b"
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.choices[0].message.content)
```

## Usage with curl

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dummy-key" \
  -d '{
    "model": "qwen3-vl-4b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Streaming

```python
stream = client.chat.completions.create(
    model="qwen3-vl-4b",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

## Vision (Image Analysis)

The `qwen3-vl-4b` model supports vision capabilities. You can pass images via URL or base64:

### With Image URL

```python
response = client.chat.completions.create(
    model="qwen3-vl-4b",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
            ]
        }
    ]
)
```

### With Base64 Encoded Image

```python
import base64

with open("image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')

response = client.chat.completions.create(
    model="qwen3-vl-4b",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
            ]
        }
    ]
)
```

**Note**: Vision requests use more tokens (~5000+ tokens for image processing).

## Reasoning Models

The `gpt-oss-20b` model is a reasoning model that shows its thinking process. Access both the reasoning and final response:

```python
response = client.chat.completions.create(
    model="gpt-oss-20b",
    messages=[{"role": "user", "content": "What is 2+2?"}]
)

# Final answer
print(response.choices[0].message.content)
# Output: "4"

# Reasoning process (if available)
if hasattr(response.choices[0].message, 'reasoning_content'):
    print(response.choices[0].message.reasoning_content)
    # Output: "We need to add 2 and 2. That equals 4."
```

## Embeddings

Sparkstation provides both text and image embedding models for semantic search, RAG, and similarity tasks.

### Text Embeddings (bge-large)

Generate embeddings for text using the `bge-large` model:

```python
# Generate text embeddings
response = client.embeddings.create(
    model="bge-large",
    input="Hello world"
)

# Get embedding vector (1024 dimensions)
embedding = response.data[0].embedding
print(f"Embedding dimensions: {len(embedding)}")
```

### Image Embeddings (CLIP)

The `clip-vit` model generates embeddings for images using OpenAI's CLIP.

**Important**: CLIP embeddings use a structured array format (different from standard OpenAI embeddings API).

#### With Image URL
```python
response = client.embeddings.create(
    model="clip-vit",
    input=[{"image": "https://example.com/image.jpg"}]
)

embedding = response.data[0].embedding  # 768 dimensions
```

#### With Base64 Encoded Image
```python
import base64

with open("image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')

# Option 1: Raw base64 (simplest)
response = client.embeddings.create(
    model="clip-vit",
    input=[{"image": image_data}]
)

# Option 2: With data URL prefix (also works)
response = client.embeddings.create(
    model="clip-vit",
    input=[{"image": f"data:image/jpeg;base64,{image_data}"}]
)

embedding = response.data[0].embedding  # 768 dimensions
```

**Note**: The input must be an array of objects with `"image"` keys, not flat strings.

### Batch Embeddings

Generate embeddings for multiple inputs at once:

```python
response = client.embeddings.create(
    model="bge-large",
    input=["First document", "Second document", "Third document"]
)

for i, data in enumerate(response.data):
    print(f"Document {i}: {len(data.embedding)} dimensions")
```

### Cross-Modal Search with CLIP

CLIP embeddings enable searching images with text or finding similar images:

```python
# Embed text query (text uses simple string format)
text_response = client.embeddings.create(
    model="clip-vit",
    input="a red car"
)
text_embedding = text_response.data[0].embedding

# Embed image (images use structured format)
image_response = client.embeddings.create(
    model="clip-vit",
    input=[{"image": "https://example.com/car.jpg"}]
)
image_embedding = image_response.data[0].embedding

# Compare via cosine similarity (both in same 768-dim embedding space)
from numpy import dot
from numpy.linalg import norm

similarity = dot(text_embedding, image_embedding) / (norm(text_embedding) * norm(image_embedding))
print(f"Similarity: {similarity}")
```

### Use Cases

- **Semantic Search**: Embed documents and queries, find similar content via cosine similarity
- **RAG (Retrieval Augmented Generation)**: Embed knowledge base for context retrieval
- **Image Search**: Use CLIP to search images by text description or find similar images
- **Cross-Modal Retrieval**: Search images with text queries or text with image queries
- **Classification**: Use embeddings as features for downstream ML tasks

## Image Generation

Sparkstation provides FLUX.1-dev for high-quality image generation via the OpenAI-compatible `/v1/images/generations` endpoint.

### Basic Image Generation

```python
import base64

# Generate an image
response = client.images.generate(
    model="flux-dev",
    prompt="A photorealistic image of a red robot in a garden",
    n=1,
    size="512x512",
    response_format="b64_json"
)

# Save the generated image
image_data = base64.b64decode(response.data[0].b64_json)
with open("generated_image.png", "wb") as f:
    f.write(image_data)
print("Image saved to generated_image.png")
```

### With curl

```bash
curl http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dummy-key" \
  -d '{
    "model": "flux-dev",
    "prompt": "A cyberpunk city at night with neon lights",
    "n": 1,
    "size": "512x512"
  }'
```

### Using requests

```python
import requests
import base64

response = requests.post(
    "http://localhost:8000/v1/images/generations",
    headers={
        "Authorization": "Bearer dummy-key",
        "Content-Type": "application/json"
    },
    json={
        "model": "flux-dev",
        "prompt": "A watercolor painting of mountains at sunset",
        "n": 1,
        "size": "1024x1024"
    },
    timeout=120  # Image generation takes 20-60 seconds
)

if response.ok:
    data = response.json()
    image_b64 = data["data"][0]["b64_json"]
    with open("output.png", "wb") as f:
        f.write(base64.b64decode(image_b64))
    print("Image saved to output.png")
```

### Supported Parameters

| Parameter | Values | Description |
|-----------|--------|-------------|
| `model` | `flux-dev` | FLUX.1-dev image model |
| `prompt` | string | Text description of image to generate |
| `n` | 1 | Number of images (currently 1 supported) |
| `size` | `512x512`, `1024x1024` | Image dimensions |
| `response_format` | `b64_json` | Response format (base64 JSON) |

**Notes**:
- Image generation takes 20-60 seconds depending on size
- FLUX.1-dev produces high-quality photorealistic images
- First request may be slower (model warmup)

## Important Notes

- **Do not start/stop Sparkstation services** - they are managed by the system
- Models are already running and ready to use
- Use the gateway endpoint (`http://localhost:8000/v1`) for all requests
- All models support standard OpenAI APIs:
  - Chat: `/v1/chat/completions` (qwen3-vl-4b, gpt-oss-20b)
  - Embeddings: `/v1/embeddings` (bge-large, clip-vit)
  - Image Generation: `/v1/images/generations` (flux-dev)

### Model-Specific Details

- **Vision Chat** (`qwen3-vl-4b`):
  - Supports image analysis via URL or base64
  - Uses standard OpenAI vision format: `{"type": "image_url", "image_url": {"url": "..."}}`

- **Reasoning** (`gpt-oss-20b`):
  - Includes reasoning traces in `reasoning_content` field

- **Text Embeddings** (`bge-large`):
  - Generates 1024-dim embeddings for text semantic tasks
  - Standard format: `input="text"` or `input=["text1", "text2"]`

- **Image Embeddings** (`clip-vit`):
  - Generates 768-dim embeddings for images and cross-modal search
  - **Special format required**: Images must use `input=[{"image": "..."}]` (not flat strings)
  - Text queries use simple format: `input="text query"`
  - Supports URL, base64 with data URL prefix, or raw base64

- **Image Generation** (`flux-dev`):
  - Generates high-quality images from text prompts using FLUX.1-dev
  - Supports sizes: 512x512, 1024x1024
  - Takes 20-60 seconds per image
  - Returns base64-encoded PNG
<!-- SPARKSTATION-END -->