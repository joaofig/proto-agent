"""
Interactive AI chat agent built with Haystack, with token streaming.

Install:
    pip install haystack-ai
    export OPENAI_API_KEY=...   # or swap the chat generator (see build_agent)

Run:
    python haystack_chat_agent.py
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from typing import Annotated
from dotenv import load_dotenv

from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage, ChatRole, StreamingChunk
from haystack.tools import tool
from haystack.utils import Secret

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL = "gpt-4o-mini"
MAX_AGENT_STEPS = 10
SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the available tools when they help you "
    "answer accurately. Be concise."
)


# ---------------------------------------------------------------------------
# Tools — plain functions; type hints + Annotated docs become the JSON schema
# ---------------------------------------------------------------------------

@tool
def get_current_time() -> str:
    """Return the current UTC date and time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


@tool
def add_numbers(
    a: Annotated[float, "First number"],
    b: Annotated[float, "Second number"],
) -> float:
    """Add two numbers and return the sum."""
    return a + b


# TODO: add your own tools here (DB lookups, HTTP calls, retrievers, ...).
TOOLS = [get_current_time, add_numbers]


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

def stream_to_console(chunk: StreamingChunk) -> None:
    """Called for every chunk the LLM streams back.

    Prints text tokens as they arrive and flags tool activity. For a quick
    default you can instead pass Haystack's built-in
    `haystack.components.generators.utils.print_streaming_chunk`.
    Replace this with an SSE/WebSocket push when wiring up a web UI.
    """
    if chunk.content:
        print(chunk.content, end="", flush=True)

    # Tool-call deltas (name arrives on the first delta of each call)
    for tc in getattr(chunk, "tool_calls", None) or []:
        if getattr(tc, "tool_name", None):
            print(f"\n[calling tool: {tc.tool_name}]", flush=True)

    # Tool results, emitted after the ToolInvoker runs
    result = getattr(chunk, "tool_call_result", None)
    if result is not None:
        print(f"[tool result: {result.result}]", flush=True)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

def build_agent() -> Agent:
    load_dotenv()

    # Swap for any Haystack ChatGenerator that supports tools + streaming,
    # e.g. AnthropicChatGenerator, OllamaChatGenerator, AzureOpenAIChatGenerator.
    model = os.getenv("OPENAI_MODEL") or "Qwen/Qwen3.8-27B"
    api_key = Secret.from_env_var("OPENAI_API_KEY")
    api_base_url = os.getenv("OPENAI_API_URL") or ""
    chat_generator = OpenAIChatGenerator(
        model=model,
        api_key=api_key,
        api_base_url=api_base_url,
    )

    agent = Agent(
        chat_generator=chat_generator,
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        streaming_callback=stream_to_console,
        exit_conditions=["text"],  # stop when the LLM answers with plain text
        max_agent_steps=MAX_AGENT_STEPS,
    )
    agent.warm_up()
    return agent


# ---------------------------------------------------------------------------
# Chat loop
# ---------------------------------------------------------------------------

def chat_loop(agent: Agent) -> None:
    history: list[ChatMessage] = []
    print("Chat started. Type 'exit' to quit, '/reset' to clear history.\n")

    while True:
        try:
            user_input = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break
        if user_input == "/reset":
            history.clear()
            print("(history cleared)\n")
            continue

        history.append(ChatMessage.from_user(user_input))
        print("agent > ", end="", flush=True)

        try:
            result = agent.run(messages=history)
        except Exception as exc:  # keep the REPL alive on API/tool errors
            history.pop()
            print(f"\n[error] {exc}\n", file=sys.stderr)
            continue

        print("\n")
        # result["messages"] holds the full turn: user msg, tool calls,
        # tool results and the final reply. Drop the system message, since
        # the Agent re-adds system_prompt on every run.
        history = [m for m in result["messages"] if not m.is_from(ChatRole.SYSTEM)]

    print("Bye!")
