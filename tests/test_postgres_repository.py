"""Unit and integration tests for PostgresRepository."""

from unittest.mock import MagicMock, patch
import pytest
from src.postgres_repository import (
    DocumentChunk,
    DocumentMetadata,
    PostgresRepository,
    SearchHit,
)


def test_search_hit_structure():
    hit = SearchHit(
        chunk_id=1,
        document_id=2,
        file_name="guide.txt",
        title="Guide",
        page=1,
        ordinal=0,
        content="Testing content",
        score=0.95,
        search_type="hybrid",
    )
    assert hit.chunk_id == 1
    assert hit.title == "Guide"
    assert hit.score == 0.95


@patch("src.postgres_repository.psycopg.connect")
@patch("src.postgres_repository.register_vector")
def test_insert_document(mock_reg_vector, mock_connect):
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_connect.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_cursor.fetchone.return_value = (42,)

    repo = PostgresRepository()
    doc = DocumentMetadata(
        document_id=0,
        title="Sample Document",
        file_name="sample.txt",
    )
    doc_id = repo.insert_document(doc)
    assert doc_id == 42
    assert mock_cursor.execute.called


@patch("src.postgres_repository.psycopg.connect")
@patch("src.postgres_repository.register_vector")
def test_list_documents(mock_reg_vector, mock_connect):
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_connect.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_cursor.fetchall.return_value = [
        (1, "Doc Title", "doc.txt", "/", "text/plain", 100, 1, 2, "indexed", "en", True, None, 1, 1, "company")
    ]

    repo = PostgresRepository()
    docs = repo.list_documents()
    assert len(docs) == 1
    assert docs[0].document_id == 1
    assert docs[0].title == "Doc Title"


def test_hybrid_search_fusion():
    repo = PostgresRepository()
    hit1 = SearchHit(chunk_id=10, document_id=1, file_name="f1.txt", title="F1", page=1, ordinal=0, content="c1", score=0.8)
    hit2 = SearchHit(chunk_id=20, document_id=2, file_name="f2.txt", title="F2", page=1, ordinal=0, content="c2", score=0.6)
    hit3 = SearchHit(chunk_id=30, document_id=3, file_name="f3.txt", title="F3", page=1, ordinal=0, content="c3", score=0.5)

    with patch.object(repo, "search_keyword", return_value=[hit1, hit2]):
        with patch.object(repo, "search_vector", return_value=[hit2, hit3]):
            results = repo.search_hybrid(query="test", query_embedding=[0.1] * 1024, limit=3, rrf_k=60)
            assert len(results) == 3
            # hit2 appears in both keyword and vector, so it should rank highest in RRF
            assert results[0].chunk_id == 20
