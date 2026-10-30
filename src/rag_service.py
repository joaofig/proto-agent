"""RAG Service coordinating embedding generation, database retrieval, and document formatting."""

from __future__ import annotations

import logging
from typing import Sequence
from src.embeddings import embed_query, embed_texts
from src.postgres_repository import (
    DocumentChunk,
    DocumentMetadata,
    PostgresRepository,
    SearchHit,
)

logger = logging.getLogger(__name__)


class RAGService:
    """High-level service providing RAG operations against PostgreSQL knowledge base."""

    def __init__(self, repository: PostgresRepository | None = None) -> None:
        self.repository = repository or PostgresRepository()

    def search(self, query: str, limit: int = 5) -> list[SearchHit]:
        """Perform hybrid search with fallback to keyword search."""
        if not query or not query.strip():
            return []

        cleaned_query = query.strip()
        query_embedding: list[float] | None = None

        try:
            query_embedding = embed_query(cleaned_query)
        except Exception as exc:
            logger.warning(f"Failed to generate embedding for query '{cleaned_query}': {exc}. Falling back to keyword search.")

        try:
            return self.repository.search_hybrid(
                query=cleaned_query,
                query_embedding=query_embedding,
                limit=limit,
            )
        except Exception as exc:
            logger.error(f"Hybrid search failed: {exc}. Attempting keyword-only search.")
            return self.repository.search_keyword(query=cleaned_query, limit=limit)

    def search_and_format(self, query: str, limit: int = 5) -> str:
        """Search knowledge base and format findings into structured context for the LLM."""
        hits = self.search(query, limit=limit)
        if not hits:
            return f"No relevant documents found in the knowledge base for query: '{query}'."

        formatted_results = [f"Found {len(hits)} relevant passage(s) in PostgreSQL Knowledge Base:\n"]
        for i, hit in enumerate(hits, start=1):
            source_info = f"Document #{hit.document_id}: '{hit.title or hit.file_name}' (File: {hit.file_name}, Page: {hit.page})"
            formatted_results.append(
                f"[{i}] {source_info}\n"
                f"Relevance Score: {hit.score:.4f} (Type: {hit.search_type})\n"
                f"Content:\n{hit.content.strip()}\n"
            )

        return "\n---\n".join(formatted_results)

    def list_documents_formatted(self, limit: int = 50) -> str:
        """Format the list of indexed documents in the database."""
        docs = self.repository.list_documents(limit=limit)
        if not docs:
            return "No documents are currently indexed in the PostgreSQL knowledge base."

        lines = [f"Indexed Documents ({len(docs)}):"]
        for d in docs:
            lines.append(
                f"- ID: {d.document_id} | Title: '{d.title}' | File: '{d.file_name}' | "
                f"Pages: {d.pages} | Chunks: {d.chunks} | Status: {d.status}"
            )
        return "\n".join(lines)

    def get_document_details_formatted(self, document_id: int) -> str:
        """Retrieve and format all chunks of a document."""
        chunks = self.repository.get_document_chunks(document_id)
        if not chunks:
            return f"No chunks found for Document ID {document_id}."

        lines = [f"Document ID {document_id} Chunks ({len(chunks)}):"]
        for c in chunks:
            lines.append(f"\n[Page {c.page}, Chunk #{c.ordinal}]:\n{c.content.strip()}")
        return "\n".join(lines)

    def index_text_document(
        self,
        title: str,
        file_name: str,
        content: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        user_id: int = 1,
        company_id: int = 1,
    ) -> int:
        """Helper to index a raw text document with embeddings into PostgreSQL."""
        import hashlib
        content_bytes = content.encode("utf-8")
        sha256_hash = hashlib.sha256(content_bytes).hexdigest()

        doc = DocumentMetadata(
            document_id=0,
            title=title,
            file_name=file_name,
            size_bytes=len(content_bytes),
            sha256=sha256_hash,
            pages=1,
            user_id=user_id,
            company_id=company_id,
        )
        doc_id = self.repository.insert_document(doc)

        # Chunk text
        chunks_text: list[str] = []
        start = 0
        while start < len(content):
            end = start + chunk_size
            chunk_slice = content[start:end].strip()
            if chunk_slice:
                chunks_text.append(chunk_slice)
            start += max(1, chunk_size - chunk_overlap)

        if not chunks_text:
            chunks_text = [content.strip()]

        # Generate embeddings
        try:
            embeddings = embed_texts(chunks_text)
        except Exception as exc:
            logger.warning(f"Failed to generate embeddings for document '{title}': {exc}")
            embeddings = [None] * len(chunks_text)  # type: ignore

        chunk_objs = [
            DocumentChunk(
                document_id=doc_id,
                page=1,
                ordinal=idx,
                content=chunk_str,
                embedding=emb if emb is not None else None,
                user_id=user_id,
                company_id=company_id,
            )
            for idx, (chunk_str, emb) in enumerate(zip(chunks_text, embeddings))
        ]

        self.repository.insert_chunks(chunk_objs)
        return doc_id
