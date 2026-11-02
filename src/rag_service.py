"""RAG Service coordinating embedding generation, database retrieval, and document formatting."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Sequence

from src.document_loader import DocumentLoader
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

    def __init__(
        self,
        repository: PostgresRepository | None = None,
        document_loader: DocumentLoader | None = None,
    ) -> None:
        self.repository = repository or PostgresRepository()
        self.document_loader = document_loader or DocumentLoader()

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
        content_type: str = "text/plain",
    ) -> int:
        """Helper to index a raw text document with embeddings into PostgreSQL.

        Args:
            title: Document title.
            file_name: Source file name.
            content: Raw text content to index.
            chunk_size: Maximum characters per chunk (default: 500).
            chunk_overlap: Overlapping characters between consecutive chunks (default: 50).
            user_id: Owner user ID (default: 1).
            company_id: Owner company ID (default: 1).
            content_type: MIME content type (default: "text/plain").

        Returns:
            The inserted document_id in PostgreSQL.

        Raises:
            ValueError: If content is empty or contains no non-whitespace characters.
        """
        doc_id, _ = self._index_content(
            title=title,
            file_name=file_name,
            content=content,
            content_type=content_type,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            user_id=user_id,
            company_id=company_id,
        )
        return doc_id

    def index_file(
        self,
        file_path: str | Path,
        user_id: int = 1,
        company_id: int = 1,
    ) -> tuple[int, int]:
        """Loads and indexes a supported file (.txt, .md, .pdf), returning (doc_id, chunk_count).

        Args:
            file_path: Path to the document file.
            user_id: Owner user ID (default: 1).
            company_id: Owner company ID (default: 1).

        Returns:
            A tuple of (document_id, chunk_count).

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is invalid, unsupported, empty, or unreadable.
        """
        loaded = self.document_loader.load(file_path)
        return self._index_content(
            title=loaded.title,
            file_name=loaded.file_name,
            content=loaded.content,
            content_type=loaded.content_type,
            size_bytes=loaded.size_bytes,
            chunk_size=500,
            chunk_overlap=50,
            user_id=user_id,
            company_id=company_id,
        )

    def _index_content(
        self,
        title: str,
        file_name: str,
        content: str,
        content_type: str = "text/plain",
        size_bytes: int | None = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        user_id: int = 1,
        company_id: int = 1,
    ) -> tuple[int, int]:
        """Internal helper to validate, chunk, embed, and persist document and chunks.

        Note:
            Due to repository architecture utilizing separate autocommit connections,
            if `insert_chunks` fails after `insert_document` succeeds, an orphaned
            `user_document` record with chunks=0 may remain in the database.
        """
        if not content or not content.strip():
            raise ValueError(f"Cannot index document '{title}': content is empty.")

        # 1. Prepare and validate chunks prior to any database write
        chunks_text: list[str] = []
        start = 0
        step = max(1, chunk_size - chunk_overlap)
        while start < len(content):
            end = start + chunk_size
            chunk_slice = content[start:end].strip()
            if chunk_slice:
                chunks_text.append(chunk_slice)
            start += step

        if not chunks_text:
            chunks_text = [content.strip()]

        # 2. Generate embeddings prior to any database write
        try:
            embeddings: list[list[float] | None] = embed_texts(chunks_text)  # type: ignore[assignment]
            if len(embeddings) != len(chunks_text):
                logger.warning(
                    f"Generated embeddings count ({len(embeddings)}) mismatch with chunks count "
                    f"({len(chunks_text)}) for document '{title}'. Falling back to None embeddings."
                )
                embeddings = [None] * len(chunks_text)
        except Exception as exc:
            logger.warning(f"Failed to generate embeddings for document '{title}': {exc}")
            embeddings = [None] * len(chunks_text)

        # 3. Insert document record
        content_bytes = content.encode("utf-8")
        sha256_hash = hashlib.sha256(content_bytes).hexdigest()
        actual_size = size_bytes if size_bytes is not None else len(content_bytes)

        doc = DocumentMetadata(
            document_id=0,
            title=title,
            file_name=file_name,
            content_type=content_type,
            size_bytes=actual_size,
            sha256=sha256_hash,
            pages=1,
            user_id=user_id,
            company_id=company_id,
        )
        doc_id = self.repository.insert_document(doc)
        if doc_id <= 0:
            raise RuntimeError(
                f"Failed to insert document '{title}': invalid document_id {doc_id} returned from repository."
            )

        # 4. Insert chunks
        chunk_objs = [
            DocumentChunk(
                document_id=doc_id,
                page=1,
                ordinal=idx,
                content=chunk_str,
                embedding=emb,
                user_id=user_id,
                company_id=company_id,
            )
            for idx, (chunk_str, emb) in enumerate(zip(chunks_text, embeddings))
        ]

        inserted_ids = self.repository.insert_chunks(chunk_objs)
        if len(inserted_ids) != len(chunk_objs):
            raise RuntimeError(
                f"Inserted chunk count mismatch for document '{title}' (ID {doc_id}): "
                f"expected {len(chunk_objs)}, got {len(inserted_ids)}."
            )

        return doc_id, len(inserted_ids)
