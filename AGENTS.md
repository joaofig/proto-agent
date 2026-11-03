# Repository Guidelines

## Project Overview
`proto-agent` is a small, synchronous CLI RAG chatbot. A Haystack `Agent` backed by an OpenAI-compatible chat endpoint answers questions using tools. The main tools query a PostgreSQL + pgvector knowledge base with hybrid search (vector + full-text/`ILIKE`, fused with Reciprocal Rank Fusion). No README, tests, CI, lint config, or DB schema/migrations exist in the repo.

## Architecture & Data Flow
Layers, top to bottom: `main.py` → `src/haystack_chat_agent.py` (agent + REPL) → `src/rag_service.py` (orchestration + formatting) → `src/embeddings.py` and `src/postgres_repository.py` (all external I/O).

Query path:
1. `main.py` → `chat_loop(build_agent())` → `Agent.run(messages=history)`.
2. The LLM picks a `@tool` (e.g. `search_knowledge_base`) → `_rag_service.search_and_format` → `RAGService.search`.
3. `embed_query` (OpenAI-compatible embeddings API). On failure: warning logged, embedding = `None`, keyword-only search.
4. `PostgresRepository.search_hybrid` runs `search_keyword` (`ts_rank_cd` on `tsv` + `ILIKE` on content/title) and `search_vector` (`embedding <=> %s::vector`, score `1 - distance`) over a pool of `max(limit*3, 20)`, then fuses with RRF (`sum(1/(rrf_k+rank))`, `rrf_k=60`). If only one list has hits it is returned as-is.
5. If `search_hybrid` raises, `RAGService` logs an error and falls back to `search_keyword`.
6. `SearchHit`s are formatted into a cited string (doc id, title, file, page, score, type) and returned to the LLM as the tool result.

Ingestion: `RAGService.index_text_document` → `insert_document` → character-window chunking (`chunk_size=500`, `chunk_overlap=50`) → `embed_texts` → `insert_chunks`. It is a programmatic helper only; no CLI/tool calls it.

## Key Directories
- `src/` – the whole application (package; always import as `from src.x import ...`).
- Root – `main.py` entry point, `pyproject.toml`, `uv.lock`.
- `.idea/`, `.venv/` – local tooling, gitignored; ignore.

## Development Commands
Run everything from the repo root (imports are `src.`-prefixed).
```bash
uv sync              # install deps (needs Python >=3.15; uv can fetch it)
uv run main.py       # start the chat REPL
uv run pytest        # run tests (none exist yet)
```
No build, lint, or format commands are configured. REPL commands: `exit`/`quit` (or Ctrl-D/Ctrl-C), `/reset` clears history.

Required environment (via `.env`, gitignored; no `.env.example`):
| Var | Default |
|---|---|
| `OPENAI_API_KEY` | required by the agent (`Secret.from_env_var`) |
| `OPENAI_API_URL` | none (custom/self-hosted base URL) |
| `AGENT_CHAT_MODEL` → `OPENAI_MODEL` | `Qwen/Qwen3.8-27B` |
| `OPENAI_EMBEDDING_MODEL` → `EMBEDDING_MODEL` | `qwen3-embedding-8b` |
| `EMBEDDING_DIM` | `4096` (must match the pgvector column) |
| `POSTGRES_HOST/PORT/USER/PASSWORD/DB` | `localhost`/`5432`/`postgres`/``(empty)``/`postgres` |

External services: an OpenAI-compatible LLM+embeddings endpoint, and Postgres with the `vector` extension and pre-created tables `user_document` and `document_chunk` (inferred from SQL; `document_chunk.tsv` is queried but never written, so it must be a generated column/trigger).

## Code Conventions & Common Patterns
- Every module: docstring, then `from __future__ import annotations`. Modern typing (`X | None`, `list[...]`); annotate all returns. `Annotated[...]` carries tool-parameter descriptions.
- Naming: `snake_case` functions, `PascalCase` classes, `UPPER_CASE` constants, `_leading_underscore` private module state (`_rag_service`).
- Imports: stdlib, third-party, `src.*` (absolute). Section banners (`# ----- Config -----`) in the agent module.
- Docstrings: one-liners; Google-style `Args:/Returns:` only where needed.
- **Sync only** – no `async`. Repository opens a new psycopg connection per call (`autocommit=True`, no pooling, no transactions); `get_connection()` closes in `finally`.
- **SQL**: raw parameterized SQL (`%s`) in triple-quoted strings, in `postgres_repository.py` only. Never interpolate values. Vectors are passed as `list[float]` via `register_vector`.
- **Data objects**: dataclasses `DocumentMetadata`, `DocumentChunk`, `SearchHit` (use `field(default_factory=...)` for mutables). Defaults `user_id=1`, `company_id=1`, `scope="company"`, `lang="en"` are hard-coded throughout.
- **Dependency injection**: light constructor DI only – `RAGService(repository=None)` defaults to `PostgresRepository()`. Embedding functions are module-level (not injected). Tools use the module-global `_rag_service` created at import time.
- **Agent tools**: plain functions decorated `@tool`; type hints + `Annotated` descriptions become the JSON schema. Register new tools in the `TOOLS` list in `src/haystack_chat_agent.py`. Tools return strings formatted for the LLM (`RAGService.*_formatted`), not raw objects.
- **State**: chat history is a local list in `chat_loop`, rebuilt from `result["messages"]` minus system messages (Agent re-adds the system prompt each run).
- **Error handling**: broad `except Exception`, no custom exceptions. Degrade gracefully (embedding failure → keyword search; hybrid failure → keyword; indexing embed failure → `embedding=None`). `chat_loop` pops the last user message and prints `[error]` to stderr on failure. Logging: `logging.getLogger(__name__)` with f-strings (logging is not configured; the agent module uses `print`).
- Import-time side effects exist: `load_dotenv()` in `embeddings`, `postgres_repository`; `int(EMBEDDING_DIM)` parsed at import; `RAGService()`/`PostgresRepository()` constructed on import (constructor reads env only, does not connect).

## Important Files
- `main.py` – entry point (still has PyCharm template comments).
- `src/haystack_chat_agent.py` – `build_agent(model, tools, system_prompt)`, `chat_loop`, `TOOLS`, `SYSTEM_PROMPT`, `MAX_AGENT_STEPS=10`, `stream_to_console`. Its module docstring (`python haystack_chat_agent.py`) is stale; use `main.py`.
- `src/rag_service.py` – `RAGService`: `search`, `*_formatted`, `index_text_document`.
- `src/postgres_repository.py` – `PostgresRepository`: `insert_document/chunks`, `list_documents`, `get_document_chunks`, `search_keyword/vector/hybrid`.
- `src/embeddings.py` – `get_openai_client`, `embed_texts` (tries `dimensions=`, retries without on any exception; sorts by `index`), `embed_query`.
- `pyproject.toml`, `uv.lock` (committed) – dependencies; no `[build-system]`, `[tool.*]`, or scripts entry.

## Runtime/Tooling Preferences
- Package manager: **uv** (`uv sync`, `uv run`, `uv add`); commit `uv.lock`.
- Python `>=3.15` (venv is `lib/python3.15`). Not a packaged project – run from the repo root.
- Key deps: `haystack-ai`, `openai`, `psycopg[binary]` (v3), `pgvector`, `pgvector-haystack`, `python-dotenv`. `pytest` is currently a runtime dependency, not a dev group.
- Never commit `.env` or secrets.

## Testing & QA
- Framework: pytest (declared), but **no tests, `conftest.py`, pytest config, CI, or linters/type-checkers exist**. Put new tests in `tests/` and run `uv run pytest` from the root.
- Testable without live services:
  - Inject a fake repository: `RAGService(repository=fake)`.
  - Patch names where they are used: `src.rag_service.embed_query` / `embed_texts`.
  - Patch `src.haystack_chat_agent._rag_service` for tool tests; drive `chat_loop` with a fake agent and patched `input`.
  - Pure logic: dataclasses, early returns on empty input, RRF fusion (monkeypatch `search_keyword`/`search_vector`), `index_text_document` chunking.
- SQL (tsvector, `ILIKE`, `<=>`) is only verifiable against real Postgres+pgvector with the tables above; no schema file is provided.
- `build_agent` needs `OPENAI_API_KEY` and calls `agent.warm_up()`; mock it in tests.
