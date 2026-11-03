"""Embedding generation utility for RAG."""

from __future__ import annotations

import os
from typing import Sequence
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULT_EMBED_MODEL = os.getenv("AGENT_EMBED_MODEL") or os.getenv("EMBEDDING_MODEL") or "qwen3-embedding-8b"
DEFAULT_EMBED_DIM = int(os.getenv("AGENT_EMBED_DIM", "4096"))


def get_openai_client() -> OpenAI:
    """Instantiate OpenAI client using environment variables."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("OPENAI_API_URL") or None
    return OpenAI(api_key=api_key, base_url=base_url)


def embed_texts(
    texts: Sequence[str],
    model: str = DEFAULT_EMBED_MODEL,
    dimensions: int = DEFAULT_EMBED_DIM,
) -> list[list[float]]:
    """Compute embeddings for a sequence of texts.

    Args:
        texts: List of text strings to embed.
        model: Embedding model name.
        dimensions: Target embedding dimension (for models supporting Matryoshka embeddings).

    Returns:
        List of embedding vectors.
    """
    if not texts:
        return []

    client = get_openai_client()
    try:
        # First attempt with explicit dimensions
        response = client.embeddings.create(
            input=list(texts),
            model=model,
            dimensions=dimensions,
        )
    except Exception:
        # Fallback without explicit dimensions if model doesn't support dimensions param
        response = client.embeddings.create(
            input=list(texts),
            model=model,
        )

    ordered = sorted(response.data, key=lambda item: item.index)
    return [item.embedding for item in ordered]


def embed_query(query: str, model: str = DEFAULT_EMBED_MODEL, dimensions: int = DEFAULT_EMBED_DIM) -> list[float]:
    """Compute embedding for a single query text."""
    embeddings = embed_texts([query], model=model, dimensions=dimensions)
    return embeddings[0] if embeddings else []
