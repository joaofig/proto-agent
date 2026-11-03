# proto-agent

A lightweight, synchronous CLI Retrieval-Augmented Generation (RAG) chatbot built with [Haystack 2.x](https://github.com/deepset-ai/haystack) and PostgreSQL ([pgvector](https://github.com/pgvector/pgvector)).

`proto-agent` provides an interactive terminal REPL where an AI agent powered by an OpenAI-compatible endpoint answers questions using custom tools to query a PostgreSQL-backed knowledge base with hybrid search (vector cosine similarity + full-text `tsvector`/`ILIKE` fused via Reciprocal Rank Fusion).

---

## Features

- **Interactive CLI REPL**: Token streaming to console, tool call inspection, and session history management (`/reset` support).
- **Hybrid Search & RRF Fusion**: Combines pgvector cosine similarity search (`<=>`) with PostgreSQL full-text search (`ts_rank_cd`, `tsv`, `ILIKE`), ranked using Reciprocal Rank Fusion (RRF).
- **Graceful Degradation**: Automatically falls back to keyword-only search if embedding generation or hybrid search fails.
- **Agent Tool Calling**: Integrates Haystack `@tool` definitions (`search_knowledge_base`, `list_knowledge_base_documents`, `get_document_details`, `get_current_time`, `add_numbers`).
- **Document Ingestion Helper**: Programmatic chunking and embedding generation for indexing raw text into PostgreSQL.

---

## Architecture & Data Flow

```
main.py
  │
  ▼
src/haystack_chat_agent.py   ──► REPL loop & Haystack Agent with streaming callback
  │
  ▼ (Tools)
src/rag_service.py           ──► RAG orchestration, formatting & document indexing
  │
  ├──► src/embeddings.py           ──► OpenAI-compatible embeddings API
  └──► src/postgres_repository.py  ──► psycopg3 + pgvector database repository
```

### Query Flow
1. **User Prompt**: Submitted via `main.py` → `chat_loop()`.
2. **Agent Reasoning**: Haystack `Agent` selects appropriate tool (e.g., `search_knowledge_base`).
3. **Query Embedding**: `embed_query` generates embedding vector (falls back to keyword-only on error).
4. **Hybrid Retrieval**: `PostgresRepository.search_hybrid` queries vector and keyword candidates, then merges and ranks hits using RRF.
5. **Context Formatting**: `RAGService.search_and_format` formats citations (document ID, title, file, page, score) for the LLM.
6. **Streaming Response**: The LLM synthesizes the answer and streams tokens directly to the terminal.

---

## Prerequisites

- **Python**: `>= 3.15`
- **Package Manager**: [`uv`](https://docs.astral.sh/uv/)
- **PostgreSQL**: Version 14+ with the [`pgvector`](https://github.com/pgvector/pgvector) extension installed.
- **LLM / Embedding Endpoint**: OpenAI or any OpenAI-compatible API (e.g., vLLM, Ollama, LocalAI, TEI).

---

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd proto-agent
   ```

2. **Install dependencies**:
   ```bash
   uv sync
   ```

3. **Configure environment variables**:
   Create a `.env` file in the project root:
   ```bash
   # LLM & Embeddings (OpenAI or OpenAI-compatible endpoint)
   OPENAI_API_KEY=your-api-key
   OPENAI_API_URL=https://api.openai.com/v1         # Optional: custom base URL
   AGENT_CHAT_MODEL=Qwen/Qwen3.8-27B                # Or OPENAI_MODEL
   OPENAI_EMBEDDING_MODEL=qwen3-embedding-8b        # Or EMBEDDING_MODEL
   EMBEDDING_DIM=4096                               # Embedding dimension

   # PostgreSQL Database
   POSTGRES_HOST=localhost
   POSTGRES_PORT=5432
   POSTGRES_USER=postgres
   POSTGRES_PASSWORD=your_password
   POSTGRES_DB=postgres
   ```

---

## Database Schema

Ensure the `vector` extension and the following tables are set up in your PostgreSQL database:

```sql
CREATE EXTENSION IF NOT EXISTS vector;

-- Documents metadata
CREATE TABLE IF NOT EXISTS user_document (
    document_id SERIAL PRIMARY KEY,
    user_id INT DEFAULT 1,
    company_id INT DEFAULT 1,
    scope VARCHAR(64) DEFAULT 'company',
    folder VARCHAR(512) DEFAULT '/',
    title VARCHAR(512) NOT NULL,
    file_name VARCHAR(512) NOT NULL,
    content_type VARCHAR(128) DEFAULT 'text/plain',
    size_bytes BIGINT DEFAULT 0,
    sha256 CHAR(64) DEFAULT '',
    pages INT DEFAULT 1,
    chunks INT DEFAULT 0,
    status VARCHAR(64) DEFAULT 'ready',
    lang VARCHAR(16) DEFAULT 'en',
    is_latest BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Document chunks and vector embeddings
CREATE TABLE IF NOT EXISTS document_chunk (
    chunk_id SERIAL PRIMARY KEY,
    document_id INT REFERENCES user_document(document_id) ON DELETE CASCADE,
    user_id INT DEFAULT 1,
    company_id INT DEFAULT 1,
    scope VARCHAR(64) DEFAULT 'company',
    page INT DEFAULT 1,
    ordinal INT DEFAULT 0,
    content TEXT NOT NULL,
    lang VARCHAR(16) DEFAULT 'en',
    embedding vector(4096), -- Set dimension matching EMBEDDING_DIM
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for fast vector and full-text searches
CREATE INDEX IF NOT EXISTS idx_document_chunk_tsv ON document_chunk USING gin(tsv);
CREATE INDEX IF NOT EXISTS idx_document_chunk_embedding ON document_chunk USING hnsw (embedding vector_cosine_ops);
```

---

## Usage

### Running the Chatbot

Start the interactive CLI REPL:
```bash
uv run main.py
```

### REPL Commands
- `exit` or `quit` (or `Ctrl+C` / `Ctrl+D`): Exit the chat session.
- `/reset`: Clear conversation history and start a fresh turn.

---

## Programmatic Document Indexing

You can index documents programmatically using `RAGService`:

```python
from src.rag_service import RAGService

rag = RAGService()
doc_id = rag.index_text_document(
    title="Sample Guide",
    file_name="guide.txt",
    content="This is the full text content of the document to be indexed.",
    chunk_size=500,
    chunk_overlap=50,
)
print(f"Indexed document ID: {doc_id}")
```

---

## Project Structure

```
proto-agent/
├── main.py                     # CLI entry point
├── pyproject.toml              # Project dependencies and configuration
├── uv.lock                     # Lockfile for reproducible dependencies
└── src/
    ├── haystack_chat_agent.py  # Haystack Agent definition, tools, and REPL loop
    ├── rag_service.py          # RAG orchestration, formatting, and indexing helper
    ├── postgres_repository.py  # PostgreSQL & pgvector query operations
    └── embeddings.py           # OpenAI-compatible text embedding utilities
```

---

## Testing

Run tests with `pytest`:
```bash
uv run pytest
```

---

## License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.
