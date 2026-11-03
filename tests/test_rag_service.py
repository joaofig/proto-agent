"""Tests for RAGService."""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.document_loader import DocumentLoader, LoadedDocument
from src.postgres_repository import DocumentChunk, DocumentMetadata, SearchHit
from src.rag_service import RAGService


def test_search_and_format_empty() -> None:
    mock_repo = MagicMock()
    mock_repo.search_hybrid.return_value = []
    service = RAGService(repository=mock_repo)

    result = service.search_and_format("nonexistent")
    assert "No relevant documents found" in result


def test_search_and_format_with_hits() -> None:
    mock_repo = MagicMock()
    mock_repo.search_hybrid.return_value = [
        SearchHit(
            chunk_id=1,
            document_id=10,
            file_name="policies.pdf",
            title="Company Policies",
            page=3,
            ordinal=0,
            content="Employees get 25 days annual leave.",
            score=0.98,
            search_type="hybrid",
        )
    ]
    service = RAGService(repository=mock_repo)

    with patch("src.rag_service.embed_query", return_value=[0.1] * 4096):
        result = service.search_and_format("vacation policy")
        assert "Found 1 relevant passage(s)" in result
        assert "Company Policies" in result
        assert "Page: 3" in result
        assert "Employees get 25 days annual leave." in result


def test_list_documents_formatted() -> None:
    mock_repo = MagicMock()
    mock_repo.list_documents.return_value = [
        DocumentMetadata(
            document_id=1,
            title="Overview",
            file_name="overview.md",
            pages=1,
            chunks=3,
            status="indexed",
        )
    ]
    service = RAGService(repository=mock_repo)
    result = service.list_documents_formatted()
    assert "Indexed Documents (1):" in result
    assert "Overview" in result
    assert "Chunks: 3" in result


def test_get_document_details_formatted_empty() -> None:
    mock_repo = MagicMock()
    mock_repo.get_document_chunks.return_value = []
    service = RAGService(repository=mock_repo)
    result = service.get_document_details_formatted(99)
    assert "No chunks found for Document ID 99" in result


def test_get_document_details_formatted_with_chunks() -> None:
    mock_repo = MagicMock()
    mock_repo.get_document_chunks.return_value = [
        DocumentChunk(
            chunk_id=1,
            document_id=10,
            page=1,
            ordinal=0,
            content="First chunk text.",
        ),
        DocumentChunk(
            chunk_id=2,
            document_id=10,
            page=1,
            ordinal=1,
            content="Second chunk text.",
        ),
    ]
    service = RAGService(repository=mock_repo)
    result = service.get_document_details_formatted(10)
    assert "Document ID 10 Chunks (2):" in result
    assert "[Page 1, Chunk #0]:\nFirst chunk text." in result
    assert "[Page 1, Chunk #1]:\nSecond chunk text." in result


def test_index_text_document_success() -> None:
    mock_repo = MagicMock()
    mock_repo.insert_document.return_value = 42
    mock_repo.insert_chunks.return_value = [101]
    service = RAGService(repository=mock_repo)

    content = "This is a simple text document for testing."
    with patch("src.rag_service.embed_texts", return_value=[[0.1, 0.2]]) as mock_embed:
        doc_id = service.index_text_document(
            title="Test Doc",
            file_name="test.txt",
            content=content,
        )

    assert doc_id == 42
    mock_repo.insert_document.assert_called_once()
    doc_meta: DocumentMetadata = mock_repo.insert_document.call_args[0][0]
    assert doc_meta.title == "Test Doc"
    assert doc_meta.file_name == "test.txt"
    assert doc_meta.content_type == "text/plain"
    assert doc_meta.size_bytes == len(content.encode("utf-8"))
    assert doc_meta.sha256 == hashlib.sha256(content.encode("utf-8")).hexdigest()
    assert doc_meta.user_id == 1
    assert doc_meta.company_id == 1
    assert doc_meta.pages == 1

    mock_embed.assert_called_once_with([content])
    mock_repo.insert_chunks.assert_called_once()
    chunks: list[DocumentChunk] = mock_repo.insert_chunks.call_args[0][0]
    assert len(chunks) == 1
    assert chunks[0].document_id == 42
    assert chunks[0].page == 1
    assert chunks[0].ordinal == 0
    assert chunks[0].content == content
    assert chunks[0].embedding == [0.1, 0.2]
    assert chunks[0].user_id == 1
    assert chunks[0].company_id == 1


def test_index_text_document_custom_parameters() -> None:
    mock_repo = MagicMock()
    mock_repo.insert_document.return_value = 88
    mock_repo.insert_chunks.return_value = [201]
    service = RAGService(repository=mock_repo)

    content = "# Title\n\nCustom markdown content"
    with patch("src.rag_service.embed_texts", return_value=[[0.5] * 10]):
        doc_id = service.index_text_document(
            title="Custom Doc",
            file_name="custom.md",
            content=content,
            content_type="text/markdown",
            user_id=42,
            company_id=99,
        )

    assert doc_id == 88
    doc_meta: DocumentMetadata = mock_repo.insert_document.call_args[0][0]
    assert doc_meta.content_type == "text/markdown"
    assert doc_meta.user_id == 42
    assert doc_meta.company_id == 99

    chunks: list[DocumentChunk] = mock_repo.insert_chunks.call_args[0][0]
    assert chunks[0].user_id == 42
    assert chunks[0].company_id == 99


def test_index_text_document_empty_raises() -> None:
    mock_repo = MagicMock()
    service = RAGService(repository=mock_repo)

    with pytest.raises(ValueError, match="content is empty"):
        service.index_text_document(title="Empty", file_name="empty.txt", content="")

    with pytest.raises(ValueError, match="content is empty"):
        service.index_text_document(
            title="Spaces", file_name="spaces.txt", content="   \n\t  "
        )

    mock_repo.insert_document.assert_not_called()
    mock_repo.insert_chunks.assert_not_called()


def test_index_text_document_positional_arguments_compatibility() -> None:
    mock_repo = MagicMock()
    mock_repo.insert_document.return_value = 55
    mock_repo.insert_chunks.return_value = [551, 552]
    service = RAGService(repository=mock_repo)

    # 100-char text with chunk_size=70, chunk_overlap=20 -> 2 chunks (step=50)
    content = "A" * 50 + "B" * 50
    with patch(
        "src.rag_service.embed_texts", return_value=[[0.1], [0.2]]
    ) as mock_embed:
        # Call with 5 positional arguments (title, file_name, content, chunk_size, chunk_overlap)
        doc_id = service.index_text_document(
            "Positional Doc",
            "pos.txt",
            content,
            70,
            20,
        )

    assert doc_id == 55
    doc_meta: DocumentMetadata = mock_repo.insert_document.call_args[0][0]
    assert doc_meta.title == "Positional Doc"
    assert doc_meta.file_name == "pos.txt"
    assert doc_meta.content_type == "text/plain"
    assert doc_meta.user_id == 1
    assert doc_meta.company_id == 1

    chunks: list[DocumentChunk] = mock_repo.insert_chunks.call_args[0][0]
    assert len(chunks) == 2
    assert chunks[0].content == "A" * 50 + "B" * 20
    assert chunks[1].content == "B" * 50
    assert mock_embed.call_args[0][0] == [chunks[0].content, chunks[1].content]

    # Test with 7 positional arguments (title, file_name, content, chunk_size, chunk_overlap, user_id, company_id)
    with patch("src.rag_service.embed_texts", return_value=[[0.1], [0.2]]):
        service.index_text_document(
            "Positional Doc 2",
            "pos2.txt",
            content,
            70,
            20,
            7,
            8,
        )

    doc_meta_2: DocumentMetadata = mock_repo.insert_document.call_args[0][0]
    assert doc_meta_2.user_id == 7
    assert doc_meta_2.company_id == 8
    assert doc_meta_2.content_type == "text/plain"


def test_index_text_document_chunking_overlap() -> None:
    mock_repo = MagicMock()
    mock_repo.insert_document.return_value = 10
    mock_repo.insert_chunks.return_value = [101, 102, 103]
    service = RAGService(repository=mock_repo)

    part1 = "0123456789" * 50  # 500 characters
    part2 = "abcdefghij" * 50  # 500 characters
    long_content = part1 + part2  # 1000 characters

    with patch(
        "src.rag_service.embed_texts",
        side_effect=lambda texts: [[float(i + 1)] for i in range(len(texts))],
    ) as mock_embed:
        doc_id = service.index_text_document(
            title="Long Doc",
            file_name="long.txt",
            content=long_content,
            chunk_size=500,
            chunk_overlap=50,
        )

    assert doc_id == 10
    chunks: list[DocumentChunk] = mock_repo.insert_chunks.call_args[0][0]
    assert len(chunks) == 3

    # Verify chunk contents and boundary positions
    # Chunk 0: index 0 to 500
    assert chunks[0].ordinal == 0
    assert chunks[0].document_id == 10
    assert chunks[0].page == 1
    assert chunks[0].content == part1
    assert chunks[0].embedding == [1.0]

    # Chunk 1: index 450 to 950 (starts at 500 - 50 = 450)
    assert chunks[1].ordinal == 1
    assert chunks[1].document_id == 10
    assert chunks[1].page == 1
    assert chunks[1].content == part1[450:] + part2[:450]
    assert chunks[1].embedding == [2.0]

    # Overlap between chunk 0 and chunk 1 is exactly 50 chars
    assert chunks[0].content[-50:] == chunks[1].content[:50]
    assert chunks[1].content[:50] == part1[450:500]

    # Chunk 2: index 900 to 1000 (starts at 950 - 50 = 900)
    assert chunks[2].ordinal == 2
    assert chunks[2].document_id == 10
    assert chunks[2].page == 1
    assert chunks[2].content == part2[400:]
    assert chunks[2].embedding == [3.0]

    # Overlap between chunk 1 and chunk 2 is exactly 50 chars
    assert chunks[1].content[-50:] == chunks[2].content[:50]
    assert chunks[2].content[:50] == part2[400:450]

    # Verify embed_texts was called with the exact list of chunk texts
    mock_embed.assert_called_once_with(
        [chunks[0].content, chunks[1].content, chunks[2].content]
    )


def test_index_content_executes_chunking_and_embedding_before_db_insert() -> None:
    mock_repo = MagicMock()
    mock_repo.insert_document.return_value = 20
    mock_repo.insert_chunks.return_value = [201]
    service = RAGService(repository=mock_repo)

    call_order: list[str] = []

    def mock_embed_fn(texts: list[str]) -> list[list[float]]:
        call_order.append("embed_texts")
        return [[0.1]]

    def mock_insert_doc(doc: DocumentMetadata) -> int:
        call_order.append("insert_document")
        return 20

    def mock_insert_chunks(chunks: list[DocumentChunk]) -> list[int]:
        call_order.append("insert_chunks")
        return [201]

    mock_repo.insert_document.side_effect = mock_insert_doc
    mock_repo.insert_chunks.side_effect = mock_insert_chunks

    with patch("src.rag_service.embed_texts", side_effect=mock_embed_fn):
        doc_id = service.index_text_document(
            title="Order Test",
            file_name="order.txt",
            content="Simple text content",
        )

    assert doc_id == 20
    assert call_order == ["embed_texts", "insert_document", "insert_chunks"]


def test_index_content_invalid_doc_id_raises() -> None:
    mock_repo = MagicMock()
    mock_repo.insert_document.return_value = 0
    service = RAGService(repository=mock_repo)

    with patch("src.rag_service.embed_texts", return_value=[[0.1]]):
        with pytest.raises(RuntimeError, match="invalid document_id 0"):
            service.index_text_document(
                title="Bad Doc ID",
                file_name="bad.txt",
                content="Valid content string",
            )

    mock_repo.insert_document.assert_called_once()
    mock_repo.insert_chunks.assert_not_called()


def test_index_content_chunk_count_mismatch_raises() -> None:
    mock_repo = MagicMock()
    mock_repo.insert_document.return_value = 30
    # Expected 2 chunks, but repository returns 0 or 1
    mock_repo.insert_chunks.return_value = [301]
    service = RAGService(repository=mock_repo)

    content = "A" * 600
    with patch("src.rag_service.embed_texts", return_value=[[0.1], [0.2]]):
        with pytest.raises(
            RuntimeError,
            match=r"Inserted chunk count mismatch for document 'Mismatch' \(ID 30\): expected 2, got 1\.",
        ):
            service.index_text_document(
                title="Mismatch",
                file_name="mismatch.txt",
                content=content,
                chunk_size=400,
                chunk_overlap=50,
            )


def test_index_content_insert_chunks_exception_propagates() -> None:
    mock_repo = MagicMock()
    mock_repo.insert_document.return_value = 40
    mock_repo.insert_chunks.side_effect = RuntimeError(
        "Database connection lost during chunk insertion"
    )
    service = RAGService(repository=mock_repo)

    with patch("src.rag_service.embed_texts", return_value=[[0.1]]):
        with pytest.raises(RuntimeError, match="Database connection lost"):
            service.index_text_document(
                title="Exception Doc",
                file_name="exc.txt",
                content="Valid content string",
            )


def test_index_text_document_embedding_failure_fallback() -> None:
    mock_repo = MagicMock()
    mock_repo.insert_document.return_value = 1
    mock_repo.insert_chunks.return_value = [10]
    service = RAGService(repository=mock_repo)

    with patch(
        "src.rag_service.embed_texts", side_effect=Exception("Embedding API error")
    ):
        doc_id = service.index_text_document(
            title="Fallback Doc",
            file_name="fallback.txt",
            content="Fallback content that should still be indexed without embeddings.",
        )

    assert doc_id == 1
    chunks: list[DocumentChunk] = mock_repo.insert_chunks.call_args[0][0]
    assert len(chunks) == 1
    assert chunks[0].embedding is None


def test_index_text_document_embedding_mismatch_fallback() -> None:
    mock_repo = MagicMock()
    mock_repo.insert_document.return_value = 1
    mock_repo.insert_chunks.return_value = [10, 11]
    service = RAGService(repository=mock_repo)

    content = "X" * 600
    with patch("src.rag_service.embed_texts", return_value=[[0.1]]):
        doc_id = service.index_text_document(
            title="Mismatch Doc",
            file_name="mismatch.txt",
            content=content,
            chunk_size=400,
            chunk_overlap=50,
        )

    assert doc_id == 1
    chunks: list[DocumentChunk] = mock_repo.insert_chunks.call_args[0][0]
    assert len(chunks) == 2
    assert all(c.embedding is None for c in chunks)


def test_index_file_txt_success(tmp_path: Path) -> None:
    mock_repo = MagicMock()
    mock_repo.insert_document.return_value = 7
    mock_repo.insert_chunks.return_value = [71]
    service = RAGService(repository=mock_repo)

    txt_file = tmp_path / "sample_doc.txt"
    txt_file.write_text("Hello from file ingestion!", encoding="utf-8")

    with patch("src.rag_service.embed_texts", return_value=[[0.1, 0.2, 0.3]]):
        doc_id, chunk_count = service.index_file(txt_file)

    assert doc_id == 7
    assert chunk_count == 1
    doc_meta: DocumentMetadata = mock_repo.insert_document.call_args[0][0]
    assert doc_meta.title == "sample_doc"
    assert doc_meta.file_name == "sample_doc.txt"
    assert doc_meta.content_type == "text/plain"
    assert doc_meta.size_bytes == txt_file.stat().st_size
    assert doc_meta.user_id == 1
    assert doc_meta.company_id == 1


def test_index_file_markdown_success(tmp_path: Path) -> None:
    mock_repo = MagicMock()
    mock_repo.insert_document.return_value = 12
    mock_repo.insert_chunks.return_value = [121]
    service = RAGService(repository=mock_repo)

    md_file = tmp_path / "architecture.md"
    md_file.write_text(
        "# Architecture Overview\n\nDetails about system design.", encoding="utf-8"
    )

    with patch("src.rag_service.embed_texts", return_value=[[0.4, 0.5]]):
        doc_id, chunk_count = service.index_file(md_file, user_id=5, company_id=10)

    assert doc_id == 12
    assert chunk_count == 1
    doc_meta: DocumentMetadata = mock_repo.insert_document.call_args[0][0]
    assert doc_meta.title == "architecture"
    assert doc_meta.file_name == "architecture.md"
    assert doc_meta.content_type == "text/markdown"
    assert doc_meta.user_id == 5
    assert doc_meta.company_id == 10


def test_index_file_pdf_success() -> None:
    mock_repo = MagicMock()
    mock_repo.insert_document.return_value = 50
    mock_repo.insert_chunks.return_value = [501]
    mock_loader = MagicMock(spec=DocumentLoader)
    mock_loader.load.return_value = LoadedDocument(
        file_name="annual_report.pdf",
        title="annual_report",
        content="First page content\n\nSecond page content",
        content_type="application/pdf",
        size_bytes=10240,
    )

    service = RAGService(repository=mock_repo, document_loader=mock_loader)

    with patch("src.rag_service.embed_texts", return_value=[[0.1, 0.2]]):
        doc_id, chunk_count = service.index_file("annual_report.pdf")

    assert doc_id == 50
    assert chunk_count == 1
    mock_loader.load.assert_called_once_with("annual_report.pdf")
    doc_meta: DocumentMetadata = mock_repo.insert_document.call_args[0][0]
    assert doc_meta.title == "annual_report"
    assert doc_meta.file_name == "annual_report.pdf"
    assert doc_meta.content_type == "application/pdf"
    assert doc_meta.size_bytes == 10240


def test_index_file_multi_chunk_persisted_count(tmp_path: Path) -> None:
    mock_repo = MagicMock()
    mock_repo.insert_document.return_value = 60
    # Two chunks persisted in database
    mock_repo.insert_chunks.return_value = [601, 602]
    service = RAGService(repository=mock_repo)

    doc_file = tmp_path / "long_document.txt"
    # Content of 800 characters split into 2 chunks with size=500, overlap=50
    doc_file.write_text("A" * 400 + "B" * 400, encoding="utf-8")

    with patch("src.rag_service.embed_texts", return_value=[[0.1], [0.2]]):
        doc_id, chunk_count = service.index_file(doc_file)

    assert doc_id == 60
    assert chunk_count == 2
    assert len(mock_repo.insert_chunks.return_value) == chunk_count


def test_index_file_nonexistent_raises() -> None:
    mock_repo = MagicMock()
    service = RAGService(repository=mock_repo)

    with pytest.raises(FileNotFoundError):
        service.index_file("nonexistent_path_xyz.txt")

    mock_repo.insert_document.assert_not_called()
    mock_repo.insert_chunks.assert_not_called()


def test_index_file_unsupported_or_empty_raises(tmp_path: Path) -> None:
    mock_repo = MagicMock()
    service = RAGService(repository=mock_repo)

    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("")
    with pytest.raises(ValueError, match="is empty"):
        service.index_file(empty_file)

    bad_ext = tmp_path / "file.xyz"
    bad_ext.write_text("content")
    with pytest.raises(ValueError, match="Unsupported file format"):
        service.index_file(bad_ext)

    mock_repo.insert_document.assert_not_called()
    mock_repo.insert_chunks.assert_not_called()
