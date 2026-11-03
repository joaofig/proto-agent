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
from datetime import UTC, datetime
from typing import Annotated

from dotenv import load_dotenv
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage, ChatRole, StreamingChunk
from haystack.tools import tool
from haystack.utils import Secret

from src.rag_service import RAGService

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MAX_AGENT_STEPS = 10
SYSTEM_PROMPT = (
    "You are a helpful assistant with access to a knowledge base stored in a remote PostgreSQL repository. "
    "When the user asks questions about documents, company records, technical guides, or factual details, "
    "always use 'search_knowledge_base' or 'list_knowledge_base_documents' to retrieve relevant facts from the database. "
    "Always cite the source document title, file name, and page number when answering from retrieved context. "
    "Use the available tools when they help you answer accurately. Be concise and precise."
)

# Shared RAG service instance
_rag_service = RAGService()


# ---------------------------------------------------------------------------
# Tools — plain functions; type hints + Annotated docs become the JSON schema
# ---------------------------------------------------------------------------


@tool
def get_current_time() -> str:
    """Return the current UTC date and time in ISO 8601 format."""
    return datetime.now(UTC).isoformat()


@tool
def add_numbers(
    a: Annotated[float, "First number"],
    b: Annotated[float, "Second number"],
) -> float:
    """Add two numbers and return the sum."""
    return a + b


@tool
def search_knowledge_base(
    query: Annotated[
        str,
        "The search query or keywords to look up in the remote PostgreSQL knowledge base.",
    ],
    limit: Annotated[
        int, "Maximum number of relevant document passages to retrieve (default: 5)"
    ] = 5,
) -> str:
    """Search for relevant documents and text chunks in the remote PostgreSQL database using hybrid vector and keyword search."""
    return _rag_service.search_and_format(query=query, limit=limit)


@tool
def list_knowledge_base_documents(
    limit: Annotated[
        int, "Maximum number of document entries to list (default: 20)"
    ] = 20,
) -> str:
    """List all documents currently indexed in the remote PostgreSQL database."""
    return _rag_service.list_documents_formatted(limit=limit)


@tool
def get_document_details(
    document_id: Annotated[int, "The numeric ID of the document to inspect in detail."],
) -> str:
    """Retrieve all content chunks and page details for a specific document by its ID."""
    return _rag_service.get_document_details_formatted(document_id=document_id)


TOOLS = [
    search_knowledge_base,
    list_knowledge_base_documents,
    get_document_details,
    get_current_time,
    add_numbers,
]


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


def build_agent(
    model: str | None = None,
    tools: list | None = None,
    system_prompt: str = SYSTEM_PROMPT,
) -> Agent:
    load_dotenv()

    # Swap for any Haystack ChatGenerator that supports tools + streaming,
    # e.g. AnthropicChatGenerator, OllamaChatGenerator, AzureOpenAIChatGenerator.
    model_name = (
        model
        or os.getenv("AGENT_CHAT_MODEL")
        or os.getenv("OPENAI_MODEL")
        or "Qwen/Qwen3.8-27B"
    )
    api_key = Secret.from_env_var("OPENAI_API_KEY")
    api_base_url = os.getenv("OPENAI_API_URL") or ""
    chat_generator = OpenAIChatGenerator(
        model=model_name,
        api_key=api_key,
        api_base_url=api_base_url,
    )

    agent = Agent(
        chat_generator=chat_generator,
        tools=tools or TOOLS,
        system_prompt=system_prompt,
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
        except EOFError, KeyboardInterrupt:
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
        except Exception as exc:  # noqa: BLE001  # keep the REPL alive on API/tool errors
            history.pop()
            print(f"\n[error] {exc}\n", file=sys.stderr)
            continue

        print("\n")
        # result["messages"] holds the full turn: user msg, tool calls,
        # tool results and the final reply. Drop the system message, since
        # the Agent re-adds system_prompt on every run.
        history = [m for m in result["messages"] if not m.is_from(ChatRole.SYSTEM)]

    print("Bye!")
