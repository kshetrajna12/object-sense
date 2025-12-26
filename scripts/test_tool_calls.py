#!/usr/bin/env python3
"""Test tool calling: direct OpenAI client vs pydantic-ai.

This script tests whether Harmony control tokens leak through at:
1. Sparkstation/vLLM level (direct OpenAI client)
2. pydantic-ai level

Run: uv run python scripts/test_tool_calls.py
"""

from __future__ import annotations

import asyncio
import json

from openai import AsyncOpenAI
from pydantic import BaseModel
from pydantic_ai import Agent, NativeOutput
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider

# Config
BASE_URL = "http://localhost:8000/v1"
API_KEY = "dummy-key"
MODEL = "gpt-oss-20b"

# Simple tool definition for testing
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_types",
            "description": "Search for existing types by semantic query",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Semantic search query",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_similar_objects",
            "description": "Find objects similar to the current observation",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Max results to return",
                        "default": 5,
                    }
                },
                "required": [],
            },
        },
    },
]

PROMPT = """Analyze this image observation and propose its type.

## Extracted Features
**Extracted text/caption:**
A majestic leopard stalking through tall grass at sunset

**Available embeddings:** image (768d)

## Instructions
1. First, use search_types to find existing types that might match
2. Use find_similar_objects to see how similar observations were typed
3. Propose the most appropriate type
"""


async def test_direct_openai() -> dict:
    """Test direct OpenAI client call with tools."""
    print("\n" + "=" * 60)
    print("TEST 1: Direct OpenAI Client Call")
    print("=" * 60)

    client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

    try:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a type inference engine."},
                {"role": "user", "content": PROMPT},
            ],
            tools=TOOLS,
            tool_choice="auto",
        )

        print(f"\nFinish reason: {response.choices[0].finish_reason}")
        print(f"Message content: {response.choices[0].message.content[:200] if response.choices[0].message.content else 'None'}...")

        tool_calls = response.choices[0].message.tool_calls
        if tool_calls:
            print(f"\nTool calls ({len(tool_calls)}):")
            for i, tc in enumerate(tool_calls):
                print(f"  [{i}] ID: {tc.id}")
                print(f"      Name: {tc.function.name!r}")
                print(f"      Args: {tc.function.arguments}")

                # Check for control tokens
                if "<|" in tc.function.name:
                    print(f"      ⚠️  CONTROL TOKENS DETECTED IN NAME!")

            return {
                "success": True,
                "tool_calls": [
                    {"name": tc.function.name, "args": tc.function.arguments}
                    for tc in tool_calls
                ],
                "has_control_tokens": any("<|" in tc.function.name for tc in tool_calls),
            }
        else:
            print("\nNo tool calls in response")
            return {"success": True, "tool_calls": [], "has_control_tokens": False}

    except Exception as e:
        print(f"\nError: {e}")
        return {"success": False, "error": str(e)}


async def test_pydantic_ai_tool_output() -> dict:
    """Test pydantic-ai agent with tools + ToolOutput (default mode)."""
    print("\n" + "=" * 60)
    print("TEST 2a: pydantic-ai Agent (ToolOutput mode - default)")
    print("=" * 60)

    # Track tool calls
    tool_calls_made: list[dict] = []

    class TypeProposal(BaseModel):
        primary_type: str
        reasoning: str

    model = OpenAIChatModel(
        MODEL,
        provider=OpenAIProvider(base_url=BASE_URL, api_key=API_KEY),
    )

    # Default: uses tool calling for structured output
    agent = Agent(
        model,
        output_type=TypeProposal,  # Default = ToolOutput mode
        system_prompt="You are a type inference engine.",
        retries=3,
    )

    @agent.tool_plain
    def search_types(query: str) -> str:
        """Search for existing types by semantic query."""
        print(f"  -> search_types called with query={query!r}")
        tool_calls_made.append({"name": "search_types", "args": {"query": query}})
        return json.dumps([{"type_name": "wildlife_photo", "status": "stable"}])

    @agent.tool_plain
    def find_similar_objects(limit: int = 5) -> str:
        """Find objects similar to the current observation."""
        print(f"  -> find_similar_objects called with limit={limit}")
        tool_calls_made.append({"name": "find_similar_objects", "args": {"limit": limit}})
        return json.dumps([{"object_id": "abc", "primary_type": "wildlife_photo"}])

    try:
        print("\nRunning agent...")
        result = await agent.run(PROMPT)

        print(f"\nResult: {result.output}")
        print(f"\nTool calls made: {len(tool_calls_made)}")
        for tc in tool_calls_made:
            print(f"  - {tc}")

        return {
            "success": True,
            "output": result.output.model_dump(),
            "tool_calls": tool_calls_made,
        }

    except Exception as e:
        print(f"\nError: {e}")
        error_str = str(e)

        # Check if error contains control tokens
        if "<|" in error_str:
            print(f"\n⚠️  CONTROL TOKENS DETECTED IN ERROR!")
            print(f"   This suggests Sparkstation is returning raw Harmony format")

        return {
            "success": False,
            "error": error_str,
            "has_control_tokens": "<|" in error_str,
            "tool_calls": tool_calls_made,
        }


async def test_pydantic_ai_native_output() -> dict:
    """Test pydantic-ai agent with tools + NativeOutput (response_format mode)."""
    print("\n" + "=" * 60)
    print("TEST 2b: pydantic-ai Agent (NativeOutput mode - response_format)")
    print("=" * 60)

    # Track tool calls
    tool_calls_made: list[dict] = []

    class TypeProposal(BaseModel):
        primary_type: str
        reasoning: str

    model = OpenAIChatModel(
        MODEL,
        provider=OpenAIProvider(base_url=BASE_URL, api_key=API_KEY),
    )

    # NativeOutput: uses response_format JSON schema instead of tool for output
    agent = Agent(
        model,
        output_type=NativeOutput(TypeProposal),  # Uses response_format
        system_prompt="You are a type inference engine.",
        retries=3,
    )

    @agent.tool_plain
    def search_types(query: str) -> str:
        """Search for existing types by semantic query."""
        print(f"  -> search_types called with query={query!r}")
        tool_calls_made.append({"name": "search_types", "args": {"query": query}})
        return json.dumps([{"type_name": "wildlife_photo", "status": "stable"}])

    @agent.tool_plain
    def find_similar_objects(limit: int = 5) -> str:
        """Find objects similar to the current observation."""
        print(f"  -> find_similar_objects called with limit={limit}")
        tool_calls_made.append({"name": "find_similar_objects", "args": {"limit": limit}})
        return json.dumps([{"object_id": "abc", "primary_type": "wildlife_photo"}])

    try:
        print("\nRunning agent (NativeOutput)...")
        result = await agent.run(PROMPT)

        print(f"\nResult: {result.output}")
        print(f"\nTool calls made: {len(tool_calls_made)}")
        for tc in tool_calls_made:
            print(f"  - {tc}")

        return {
            "success": True,
            "output": result.output.model_dump(),
            "tool_calls": tool_calls_made,
        }

    except Exception as e:
        print(f"\nError: {e}")
        error_str = str(e)

        if "<|" in error_str:
            print(f"\n⚠️  CONTROL TOKENS DETECTED IN ERROR!")

        return {
            "success": False,
            "error": error_str,
            "has_control_tokens": "<|" in error_str,
            "tool_calls": tool_calls_made,
        }


async def test_pydantic_responses_api() -> dict:
    """Test pydantic-ai with OpenAIResponsesModel (Responses API)."""
    print("\n" + "=" * 60)
    print("TEST 2c: pydantic-ai Agent (OpenAIResponsesModel)")
    print("=" * 60)

    # Track tool calls
    tool_calls_made: list[dict] = []

    class TypeProposal(BaseModel):
        primary_type: str
        reasoning: str

    provider = OpenAIProvider(base_url=BASE_URL, api_key=API_KEY)

    try:
        model = OpenAIResponsesModel(
            MODEL,
            provider=provider,
        )
    except Exception as e:
        print(f"Failed to create OpenAIResponsesModel: {e}")
        return {"success": False, "error": str(e), "skipped": True}

    agent = Agent(
        model,
        output_type=NativeOutput(TypeProposal),
        system_prompt="You are a type inference engine.",
        retries=3,
    )

    @agent.tool_plain
    def search_types(query: str) -> str:
        """Search for existing types by semantic query."""
        print(f"  -> search_types called with query={query!r}")
        tool_calls_made.append({"name": "search_types", "args": {"query": query}})
        return json.dumps([{"type_name": "wildlife_photo", "status": "stable"}])

    @agent.tool_plain
    def find_similar_objects(limit: int = 5) -> str:
        """Find objects similar to the current observation."""
        print(f"  -> find_similar_objects called with limit={limit}")
        tool_calls_made.append({"name": "find_similar_objects", "args": {"limit": limit}})
        return json.dumps([{"object_id": "abc", "primary_type": "wildlife_photo"}])

    try:
        print("\nRunning agent (OpenAIResponsesModel)...")
        result = await agent.run(PROMPT)

        print(f"\nResult: {result.output}")
        print(f"\nTool calls made: {len(tool_calls_made)}")
        for tc in tool_calls_made:
            print(f"  - {tc}")

        return {
            "success": True,
            "output": result.output.model_dump(),
            "tool_calls": tool_calls_made,
        }

    except Exception as e:
        print(f"\nError: {e}")
        error_str = str(e)

        if "<|" in error_str:
            print(f"\n⚠️  CONTROL TOKENS DETECTED IN ERROR!")

        return {
            "success": False,
            "error": error_str,
            "has_control_tokens": "<|" in error_str,
            "tool_calls": tool_calls_made,
        }


async def test_direct_multi_turn() -> dict:
    """Test direct multi-turn tool calling (simulating pydantic-ai flow)."""
    print("\n" + "=" * 60)
    print("TEST 4: Direct OpenAI Multi-Turn Tool Calling")
    print("=" * 60)

    client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

    messages = [
        {"role": "system", "content": "You are a type inference engine. Use the provided tools."},
        {"role": "user", "content": PROMPT},
    ]

    try:
        # First call - expect tool call
        print("\n--- Turn 1: Initial request ---")
        response1 = await client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        choice1 = response1.choices[0]
        print(f"Finish reason: {choice1.finish_reason}")

        if not choice1.message.tool_calls:
            print("No tool calls - model responded directly")
            return {"success": True, "multi_turn": False}

        # Process first tool call
        tc1 = choice1.message.tool_calls[0]
        print(f"Tool call: {tc1.function.name!r}")
        print(f"Args: {tc1.function.arguments}")

        if "<|" in tc1.function.name:
            print("⚠️ CONTROL TOKENS in first tool call!")
            return {"success": False, "has_control_tokens": True, "turn": 1}

        # Add assistant message with tool call
        messages.append(choice1.message.model_dump())

        # Add tool result
        tool_result = json.dumps([{"type_name": "wildlife_photo", "status": "stable"}])
        messages.append({
            "role": "tool",
            "tool_call_id": tc1.id,
            "content": tool_result,
        })

        # Second call - continue conversation
        print("\n--- Turn 2: After tool result ---")
        response2 = await client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        choice2 = response2.choices[0]
        print(f"Finish reason: {choice2.finish_reason}")

        if choice2.message.tool_calls:
            for tc in choice2.message.tool_calls:
                print(f"Tool call: {tc.function.name!r}")
                if "<|" in tc.function.name:
                    print("⚠️ CONTROL TOKENS in second tool call!")
                    return {"success": False, "has_control_tokens": True, "turn": 2}
        else:
            print(f"Content: {choice2.message.content[:200] if choice2.message.content else 'None'}...")

        return {"success": True, "has_control_tokens": False, "multi_turn": True}

    except Exception as e:
        print(f"\nError: {e}")
        return {"success": False, "error": str(e), "has_control_tokens": "<|" in str(e)}


async def test_direct_openai_no_tools() -> dict:
    """Test direct call without tools for comparison."""
    print("\n" + "=" * 60)
    print("TEST 3: Direct OpenAI Client (NO TOOLS)")
    print("=" * 60)

    client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

    try:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a type inference engine. Respond with JSON."},
                {"role": "user", "content": "What type is this: A leopard photo. Reply as JSON with primary_type and reasoning."},
            ],
        )

        content = response.choices[0].message.content
        print(f"\nResponse:\n{content[:500]}...")

        # Check for control tokens in content
        if content and "<|" in content:
            print(f"\n⚠️  CONTROL TOKENS IN CONTENT (normal for reasoning models)")

        return {"success": True, "content": content}

    except Exception as e:
        print(f"\nError: {e}")
        return {"success": False, "error": str(e)}


async def main() -> None:
    print("Testing tool calling: Sparkstation vs pydantic-ai")
    print("Model:", MODEL)
    print("Endpoint:", BASE_URL)

    results = {}

    # Test 1: Direct OpenAI with tools
    results["direct_with_tools"] = await test_direct_openai()

    # Test 2a: pydantic-ai with ToolOutput (default)
    results["pydantic_tool_output"] = await test_pydantic_ai_tool_output()

    # Test 2b: pydantic-ai with NativeOutput (response_format)
    results["pydantic_native_output"] = await test_pydantic_ai_native_output()

    # Test 2c: pydantic-ai with OpenAIResponsesModel (Responses API)
    results["pydantic_responses_api"] = await test_pydantic_responses_api()

    # Test 3: Direct without tools (baseline)
    results["direct_no_tools"] = await test_direct_openai_no_tools()

    # Test 4: Direct multi-turn tool calling
    results["direct_multi_turn"] = await test_direct_multi_turn()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for name, result in results.items():
        status = "✓" if result.get("success") else "✗"
        tokens = "⚠️ CONTROL TOKENS" if result.get("has_control_tokens") else ""
        print(f"  {status} {name}: {tokens or 'OK'}")

    # Diagnosis
    print("\n" + "-" * 40)
    if results["direct_with_tools"].get("has_control_tokens"):
        print("→ Issue is at SPARKSTATION level (raw Harmony tokens in single-turn direct call)")
    elif results["direct_multi_turn"].get("has_control_tokens"):
        print("→ Issue is at SPARKSTATION level (raw Harmony tokens in MULTI-TURN direct call)")
        print("  Sparkstation doesn't properly handle tool result → next tool call flow")
    elif results["pydantic_tool_output"].get("has_control_tokens") and not results["pydantic_native_output"].get("has_control_tokens"):
        print("→ Issue is with pydantic-ai ToolOutput mode (final_result tool)")
        print("  NativeOutput (response_format) works - use that instead!")
    elif results["pydantic_tool_output"].get("has_control_tokens") or results["pydantic_native_output"].get("has_control_tokens"):
        print("→ Issue is at PYDANTIC-AI level (tokens appear through pydantic-ai)")
    else:
        print("→ No control tokens detected in this run (issue may be intermittent)")


if __name__ == "__main__":
    asyncio.run(main())
