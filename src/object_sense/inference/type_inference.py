"""Type inference agent using pydantic-ai.

This module implements Step 4 of the processing loop: LLM Type Inference.

The agent:
1. Receives extracted features from an observation
2. Queries the type store for similar/existing types (RAG)
3. Proposes a TypeProposal with primary_type, slots, entity hypotheses

See concept_v1.md §4 Step 4 and §8.2 for specification.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic_ai import Agent, NativeOutput, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from object_sense.config import settings
from object_sense.inference.schemas import (
    SimilarObject,
    TypeProposal,
    TypeSearchResult,
)

if TYPE_CHECKING:
    from object_sense.extraction.base import ExtractionResult

# Type aliases for callback functions
SearchTypesFn = Callable[[str], Awaitable[list[dict[str, Any]]]]
GetTypeFn = Callable[[str], Awaitable[dict[str, Any] | None]]
FindSimilarFn = Callable[..., Awaitable[list[dict[str, Any]]]]


# System prompt for type inference
TYPE_INFERENCE_SYSTEM_PROMPT = """\
You are a type inference engine for ObjectSense, a semantic substrate that provides \
persistent object identity and type awareness.

Your task: Given extracted features from an observation, propose what TYPE of thing it is.

## Core Principles

1. **Medium ≠ Type**: Medium (image, text, json) is how data is encoded. Type is what it IS.
   - An image of a leopard → type: wildlife_photo (not "image")
   - A JSON product record → type: product_record (not "json")

2. **Reuse existing types**: Before proposing a new type, search the type store.
   Query with semantic terms from the observation. Only create new types when nothing fits.

3. **Types are semantic claims**: They represent meaning, not format.
   - Good: wildlife_photo, product_record, hunt_event
   - Bad: image_file, json_data, text_document

4. **Slots for variation**: Use slots for properties that vary within a type.
   - lighting=backlit (slot) not backlit_photo (type)
   - species=leopard (slot referencing entity) not leopard_photo (type)

5. **Entity hypotheses are seeds**: Detected entities (animals, locations, products) are \
hypotheses for the entity resolution step. Include confidence and reasoning.

## Naming Conventions

- Use snake_case for all names
- Use singular nouns (wildlife_photo not wildlife_photos)
- Be specific but not overly narrow
- Prefix entity types with the domain: animal_entity, location_entity, product_entity

## Output Requirements

Always provide:
- primary_type: Your best guess at what type this observation is
- is_existing_type: Whether this type already exists (check via search_types tool)
- slots: Key properties extracted from the observation
- entity_hypotheses: Any entities detected (animals, locations, products, etc.)
- reasoning: Brief explanation of your inference

Only set maybe_new_type if existing types truly don't capture the observation's essence.
"""


@dataclass
class TypeInferenceDeps:
    """Dependencies for the type inference agent.

    These are injected at runtime and provide access to:
    - Type store queries (search, get)
    - Similar object retrieval
    - Embedding functions
    """

    # Callback to search types by semantic query
    search_types_fn: SearchTypesFn | None = None

    # Callback to get type details by name
    get_type_fn: GetTypeFn | None = None

    # Callback to find similar objects
    find_similar_fn: FindSimilarFn | None = None

    # The extracted features being analyzed
    extraction_result: ExtractionResult | None = None

    # Medium of the observation
    medium: str = "unknown"


def create_type_inference_agent() -> Agent[TypeInferenceDeps, TypeProposal]:
    """Create the type inference agent.

    Returns an Agent configured for type inference with tools for
    querying the type store.
    """
    # Configure model with Sparkstation endpoint
    # Use reasoning model (gpt-oss-20b) for larger context window (16k vs 4k)
    # Type inference needs space for: system prompt + tools + schema + user message
    model = OpenAIChatModel(
        settings.model_reasoning,
        provider=OpenAIProvider(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
        ),
    )

    agent = Agent(
        model,
        deps_type=TypeInferenceDeps,
        # Use NativeOutput to leverage response_format instead of tool calling.
        # gpt-oss-20b doesn't reliably follow tool_choice, but response_format works.
        output_type=NativeOutput(TypeProposal),
        system_prompt=TYPE_INFERENCE_SYSTEM_PROMPT,
        # Note: tool calls are flaky with gpt-oss-20b - see object-sense-9vs
        retries=2,
    )

    @agent.tool
    async def search_types(  # pyright: ignore[reportUnusedFunction]
        ctx: RunContext[TypeInferenceDeps],
        query: str,
    ) -> list[TypeSearchResult]:
        """Search existing types by semantic query.

        Use this to find types that might match the observation before proposing new ones.
        Query with semantic terms like "wildlife photo", "product catalog", etc.

        Args:
            query: Semantic search query for finding relevant types

        Returns:
            List of matching types with their details
        """
        deps = ctx.deps
        if deps.search_types_fn is None:
            # No type store available - return empty (new system)
            return []

        results = await deps.search_types_fn(query)
        return [TypeSearchResult(**r) for r in results]

    @agent.tool
    async def get_type_details(  # pyright: ignore[reportUnusedFunction]
        ctx: RunContext[TypeInferenceDeps],
        type_name: str,
    ) -> TypeSearchResult | None:
        """Get detailed information about a specific type.

        Use this to understand a type's structure before deciding to reuse it.

        Args:
            type_name: The canonical name of the type to look up

        Returns:
            Type details or None if not found
        """
        deps = ctx.deps
        if deps.get_type_fn is None:
            return None

        result = await deps.get_type_fn(type_name)
        if result is None:
            return None
        return TypeSearchResult(**result)

    @agent.tool
    async def find_similar_objects(  # pyright: ignore[reportUnusedFunction]
        ctx: RunContext[TypeInferenceDeps],
        limit: int = 5,
    ) -> list[SimilarObject]:
        """Find objects similar to the current observation.

        Use this to see how similar observations were typed.
        Helpful for consistency with existing type assignments.

        Args:
            limit: Maximum number of similar objects to return

        Returns:
            List of similar objects with their types and scores
        """
        deps = ctx.deps
        if deps.find_similar_fn is None:
            return []

        results = await deps.find_similar_fn(limit=limit)
        return [SimilarObject(**r) for r in results]

    return agent


class TypeInferenceAgent:
    """High-level interface for type inference.

    Wraps the pydantic-ai agent with convenience methods for
    running inference on extraction results.

    Usage:
        agent = TypeInferenceAgent()
        proposal = await agent.infer(extraction_result, medium="image")
    """

    def __init__(
        self,
        *,
        search_types_fn: SearchTypesFn | None = None,
        get_type_fn: GetTypeFn | None = None,
        find_similar_fn: FindSimilarFn | None = None,
    ) -> None:
        """Initialize the type inference agent.

        Args:
            search_types_fn: Async function to search types by query
            get_type_fn: Async function to get type details by name
            find_similar_fn: Async function to find similar objects
        """
        self._agent = create_type_inference_agent()
        self._search_types_fn = search_types_fn
        self._get_type_fn = get_type_fn
        self._find_similar_fn = find_similar_fn

    async def infer(
        self,
        extraction_result: ExtractionResult,
        *,
        medium: str = "unknown",
    ) -> TypeProposal:
        """Run type inference on an extraction result.

        Args:
            extraction_result: Features extracted from the observation
            medium: The medium of the observation (image, text, json, etc.)

        Returns:
            TypeProposal with primary_type, slots, and entity hypotheses
        """
        # Build the prompt from extraction result
        prompt = self._build_prompt(extraction_result, medium)

        # Create dependencies
        deps = TypeInferenceDeps(
            search_types_fn=self._search_types_fn,
            get_type_fn=self._get_type_fn,
            find_similar_fn=self._find_similar_fn,
            extraction_result=extraction_result,
            medium=medium,
        )

        # Run the agent
        result = await self._agent.run(prompt, deps=deps)

        return result.output

    def _build_prompt(self, extraction: ExtractionResult, medium: str) -> str:
        """Build the inference prompt from extraction result."""
        parts = [
            f"Analyze this {medium} observation and propose its type.\n",
            "## Extracted Features\n",
        ]

        # Add extracted text if available
        if extraction.extracted_text:
            parts.append(f"**Extracted text/caption:**\n{extraction.extracted_text}\n")

        # Add extra metadata
        if extraction.extra:
            parts.append("\n**Additional metadata:**\n")
            for key, value in extraction.extra.items():
                if key == "medium":
                    continue  # Already shown
                parts.append(f"- {key}: {value}\n")

        # Add embedding info (just dimensions, not values)
        embeddings: list[str] = []
        if extraction.text_embedding:
            embeddings.append(f"text ({len(extraction.text_embedding)}d)")
        if extraction.image_embedding:
            embeddings.append(f"image ({len(extraction.image_embedding)}d)")
        if extraction.clip_text_embedding:
            embeddings.append(f"clip_text ({len(extraction.clip_text_embedding)}d)")
        if embeddings:
            parts.append(f"\n**Available embeddings:** {', '.join(embeddings)}\n")

        # Add hash if available
        if extraction.hash_value:
            parts.append(f"\n**Content hash:** {extraction.hash_value[:16]}...\n")

        parts.append(
            "\n## Instructions\n"
            "1. First, use search_types to find existing types that might match\n"
            "2. Use find_similar_objects to see how similar observations were typed\n"
            "3. Propose the most appropriate type, slots, and entity hypotheses\n"
        )

        return "".join(parts)
