"""PostgreSQL document and vector repository for RAG."""

from __future__ import annotations

import logging
import os
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

load_dotenv()
logger = logging.getLogger(__name__)


@dataclass
class DocumentMetadata:
    """Metadata representing a document in user_document."""

    document_id: int
    title: str
    file_name: str
    folder: str = "/"
    content_type: str = "text/plain"
    size_bytes: int = 0
    sha256: str = "0" * 64
    pages: int = 1
    chunks: int = 0
    status: str = "ready"
    lang: str = "en"
    is_latest: bool = True
    created_at: datetime | None = None
    user_id: int = 1
    company_id: int = 1
    scope: str = "company"


@dataclass
class DocumentChunk:
    """A single chunk of a document stored in document_chunk."""

    chunk_id: int | None = None
    document_id: int = 0
    page: int = 1
    ordinal: int = 0
    content: str = ""
    lang: str = "en"
    user_id: int = 1
    company_id: int = 1
    scope: str = "company"
    embedding: list[float] | None = None
    created_at: datetime | None = None


@dataclass
class SearchHit:
    """Result of a search in the document repository."""

    chunk_id: int
    document_id: int
    file_name: str
    title: str
    page: int
    ordinal: int
    content: str
    score: float = 0.0
    search_type: str = "hybrid"  # "vector", "keyword", or "hybrid"
    metadata: dict[str, Any] = field(default_factory=dict)


class PostgresRepository:
    """Manages connections and RAG search queries against remote PostgreSQL."""

    def __init__(
        self,
        host: str | None = None,
        port: int | str | None = None,
        user: str | None = None,
        password: str | None = None,
        dbname: str | None = None,
    ) -> None:
        self.host = host or os.getenv("POSTGRES_HOST", "localhost")
        self.port = int(port or os.getenv("POSTGRES_PORT", "5432"))
        self.user = user or os.getenv("POSTGRES_USER", "postgres")
        self.password = password or os.getenv("POSTGRES_PASSWORD", "")
        self.dbname = dbname or os.getenv("POSTGRES_DB", "postgres")

    @contextmanager
    def get_connection(self) -> Generator[psycopg.Connection]:
        """Create and yield a PostgreSQL connection with pgvector registered."""
        conn = psycopg.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            dbname=self.dbname,
            autocommit=True,
        )
        try:
            register_vector(conn)
            yield conn
        finally:
            conn.close()

    def insert_document(self, doc: DocumentMetadata) -> int:
        """Insert a document record into user_document and return document_id."""
        with self.get_connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO user_document (
                        user_id, company_id, scope, folder, title, file_name,
                        content_type, size_bytes, sha256, pages, chunks, status, lang, is_latest
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING document_id;
                    """,
                    (
                        doc.user_id,
                        doc.company_id,
                        doc.scope,
                        doc.folder,
                        doc.title,
                        doc.file_name,
                        doc.content_type,
                        doc.size_bytes,
                        doc.sha256,
                        doc.pages,
                        doc.chunks,
                        doc.status,
                        doc.lang,
                        doc.is_latest,
                    ),
                )
                res = cur.fetchone()
                return int(res[0]) if res else 0

    def insert_chunks(self, chunks: Sequence[DocumentChunk]) -> list[int]:
        """Insert a batch of chunks into document_chunk."""
        if not chunks:
            return []

        inserted_ids: list[int] = []
        with self.get_connection() as conn, conn.cursor() as cur:
            for chunk in chunks:
                cur.execute(
                        """
                        INSERT INTO document_chunk (
                            document_id, user_id, company_id, scope,
                            page, ordinal, content, lang,
                            embedding
                        )
                        VALUES (
                            %s, %s, %s, %s,
                            %s, %s, %s, %s,
                            %s
                        )
                        RETURNING chunk_id;
                        """,
                        (
                            chunk.document_id,
                            chunk.user_id,
                            chunk.company_id,
                            chunk.scope,
                            chunk.page,
                            chunk.ordinal,
                            chunk.content,
                            chunk.lang,
                            chunk.embedding,
                        ),
                    )
                    res = cur.fetchone()
                    if res:
                        inserted_ids.append(int(res[0]))

                # Update chunk count on document
                if chunks:
                    doc_id = chunks[0].document_id
                    cur.execute(
                        """
                        UPDATE user_document
                        SET chunks = (SELECT COUNT(*) FROM document_chunk WHERE document_id = %s)
                        WHERE document_id = %s;
                        """,
                        (doc_id, doc_id),
                    )

        return inserted_ids

    def list_documents(self, limit: int = 50) -> list[DocumentMetadata]:
        """List documents stored in the database."""
        with self.get_connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                    SELECT document_id, title, file_name, folder, content_type,
                           size_bytes, pages, chunks, status, lang, is_latest,
                           created_at, user_id, company_id, scope
                    FROM user_document
                    ORDER BY created_at DESC
                    LIMIT %s;
                    """,
                (limit,),
            )
            rows = cur.fetchall()
            return [
                DocumentMetadata(
                    document_id=row[0],
                    title=row[1] or "",
                    file_name=row[2] or "",
                    folder=row[3] or "/",
                    content_type=row[4] or "text/plain",
                    size_bytes=row[5] or 0,
                    pages=row[6] or 1,
                    chunks=row[7] or 0,
                    status=row[8] or "",
                    lang=row[9] or "en",
                    is_latest=row[10] if row[10] is not None else True,
                    created_at=row[11],
                    user_id=row[12] or 1,
                    company_id=row[13] or 1,
                    scope=row[14] or "company",
                )
                for row in rows
            ]

    def get_document_chunks(
        self, document_id: int, limit: int = 100
    ) -> list[DocumentChunk]:
        """Retrieve chunks belonging to a specific document."""
        with self.get_connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                    SELECT chunk_id, document_id, page, ordinal, content, lang,
                           user_id, company_id, scope, created_at
                    FROM document_chunk
                    WHERE document_id = %s
                    ORDER BY page ASC, ordinal ASC
                    LIMIT %s;
                    """,
                (document_id, limit),
            )
            rows = cur.fetchall()
            return [
                DocumentChunk(
                    chunk_id=row[0],
                    document_id=row[1],
                    page=row[2],
                    ordinal=row[3],
                    content=row[4],
                    lang=row[5],
                    user_id=row[6],
                    company_id=row[7],
                    scope=row[8],
                    created_at=row[9],
                )
                for row in rows
            ]

    def search_keyword(self, query: str, limit: int = 5) -> list[SearchHit]:
        """Full-text and keyword search using PostgreSQL tsvector and ILIKE."""
        cleaned_query = query.strip()
        if not cleaned_query:
            return []

        with self.get_connection() as conn, conn.cursor() as cur:
            # We use plainto_tsquery combined with ILIKE for robust matching
            sql = """
                SELECT 
                    c.chunk_id,
                    c.document_id,
                    COALESCE(d.file_name, '') AS file_name,
                    COALESCE(d.title, '') AS title,
                    c.page,
                    c.ordinal,
                    c.content,
                    ts_rank_cd(c.tsv, plainto_tsquery('english', %s)) AS rank_score
                FROM document_chunk c
                JOIN user_document d ON c.document_id = d.document_id
                WHERE 
                    c.tsv @@ plainto_tsquery('english', %s)
                    OR c.content ILIKE %s
                    OR d.title ILIKE %s
                ORDER BY rank_score DESC, c.chunk_id ASC
                LIMIT %s;
                """
            like_pattern = f"%{cleaned_query}%"
            cur.execute(
                sql,
                (cleaned_query, cleaned_query, like_pattern, like_pattern, limit),
            )
            rows = cur.fetchall()

            hits: list[SearchHit] = []
            for row in rows:
                hits.append(
                    SearchHit(
                        chunk_id=row[0],
                        document_id=row[1],
                        file_name=row[2],
                        title=row[3],
                        page=row[4],
                        ordinal=row[5],
                        content=row[6],
                        score=float(row[7]) if row[7] is not None else 0.0,
                        search_type="keyword",
                    )
                )
            return hits

    def search_vector(
        self, query_embedding: Sequence[float], limit: int = 5
    ) -> list[SearchHit]:
        """Vector similarity search using cosine distance with pgvector."""
        if not query_embedding:
            return []

        with self.get_connection() as conn, conn.cursor() as cur:
            # In pgvector: <=> is cosine distance (0 is identical, 2 is opposite)
            # Similarity score = 1.0 - cosine_distance
            sql = """
                SELECT 
                    c.chunk_id,
                    c.document_id,
                    COALESCE(d.file_name, '') AS file_name,
                    COALESCE(d.title, '') AS title,
                    c.page,
                    c.ordinal,
                    c.content,
                    (1 - (c.embedding <=> %s::vector)) AS similarity
                FROM document_chunk c
                JOIN user_document d ON c.document_id = d.document_id
                WHERE c.embedding IS NOT NULL
                ORDER BY c.embedding <=> %s::vector ASC
                LIMIT %s;
                """
            cur.execute(sql, (list(query_embedding), list(query_embedding), limit))
            rows = cur.fetchall()

            hits: list[SearchHit] = []
            for row in rows:
                hits.append(
                    SearchHit(
                        chunk_id=row[0],
                        document_id=row[1],
                        file_name=row[2],
                        title=row[3],
                        page=row[4],
                        ordinal=row[5],
                        content=row[6],
                        score=float(row[7]) if row[7] is not None else 0.0,
                        search_type="vector",
                    )
                )
            return hits

    def search_hybrid(
        self,
        query: str,
        query_embedding: Sequence[float] | None = None,
        limit: int = 5,
        rrf_k: int = 60,
    ) -> list[SearchHit]:
        """Hybrid search combining keyword search and semantic vector search via RRF.

        Reciprocal Rank Fusion (RRF) formula:
            score = sum(1.0 / (rrf_k + rank_i))
        """
        # Retrieve candidate pools for both methods
        candidate_k = max(limit * 3, 20)
        kw_hits = self.search_keyword(query, limit=candidate_k)
        vec_hits = (
            self.search_vector(query_embedding, limit=candidate_k)
            if query_embedding
            else []
        )

        if not vec_hits:
            return kw_hits[:limit]
        if not kw_hits:
            return vec_hits[:limit]

        # Compute RRF scores
        rrf_scores: dict[int, float] = {}
        hit_map: dict[int, SearchHit] = {}

        for rank, hit in enumerate(kw_hits, start=1):
            rrf_scores[hit.chunk_id] = rrf_scores.get(hit.chunk_id, 0.0) + (
                1.0 / (rrf_k + rank)
            )
            hit_map[hit.chunk_id] = hit

        for rank, hit in enumerate(vec_hits, start=1):
            rrf_scores[hit.chunk_id] = rrf_scores.get(hit.chunk_id, 0.0) + (
                1.0 / (rrf_k + rank)
            )
            if hit.chunk_id not in hit_map:
                hit_map[hit.chunk_id] = hit

        # Sort by final fused RRF score
        sorted_chunk_ids = sorted(
            rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True
        )

        results: list[SearchHit] = []
        for cid in sorted_chunk_ids[:limit]:
            item = hit_map[cid]
            results.append(
                SearchHit(
                    chunk_id=item.chunk_id,
                    document_id=item.document_id,
                    file_name=item.file_name,
                    title=item.title,
                    page=item.page,
                    ordinal=item.ordinal,
                    content=item.content,
                    score=rrf_scores[cid],
                    search_type="hybrid",
                    metadata={
                        "raw_score": item.score,
                        "rrf_score": rrf_scores[cid],
                    },
                )
            )

        return results
