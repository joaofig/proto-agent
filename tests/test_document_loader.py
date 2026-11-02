"""Tests for src.document_loader."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from haystack.dataclasses import Document
from pypdf import PdfWriter

from src.document_loader import DocumentLoader, LoadedDocument


def _create_minimal_pdf(text: str = "Hello PDF World") -> bytes:
    """Generate raw bytes for a valid minimal single-page PDF containing text."""
    stream_data = f"BT\n/F1 24 Tf\n100 700 Td\n({text}) Tj\nET\n".encode("latin-1")
    header = b"%PDF-1.4\n"
    obj1 = b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    obj2 = b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
    obj3 = (
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
    )
    obj4 = (
        f"4 0 obj\n<< /Length {len(stream_data)} >>\nstream\n".encode("latin-1")
        + stream_data
        + b"endstream\nendobj\n"
    )
    obj5 = b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"

    offsets = [0]
    curr = len(header)
    for obj in [obj1, obj2, obj3, obj4, obj5]:
        offsets.append(curr)
        curr += len(obj)

    xref_offset = curr
    xref = b"xref\n0 6\n0000000000 65535 f \n"
    for off in offsets[1:]:
        xref += f"{off:010d} 00000 n \n".encode("latin-1")

    trailer = (
        b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n"
        + str(xref_offset).encode("latin-1")
        + b"\n%%EOF"
    )
    return header + obj1 + obj2 + obj3 + obj4 + obj5 + xref + trailer


@pytest.fixture
def loader() -> DocumentLoader:
    return DocumentLoader()


def test_supported_extensions_mapping(loader: DocumentLoader) -> None:
    assert loader.SUPPORTED_EXTENSIONS[".txt"] == "text/plain"
    assert loader.SUPPORTED_EXTENSIONS[".text"] == "text/plain"
    assert loader.SUPPORTED_EXTENSIONS[".md"] == "text/markdown"
    assert loader.SUPPORTED_EXTENSIONS[".markdown"] == "text/markdown"
    assert loader.SUPPORTED_EXTENSIONS[".pdf"] == "application/pdf"
    assert (
        loader.SUPPORTED_EXTENSIONS[".docx"]
        == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )


def test_missing_file_raises(loader: DocumentLoader) -> None:
    with pytest.raises(FileNotFoundError, match="File not found"):
        loader.load("non_existent_file.txt")


def test_directory_path_raises(loader: DocumentLoader, tmp_path: Path) -> None:
    dir_path = tmp_path / "subfolder"
    dir_path.mkdir()
    with pytest.raises(ValueError, match="not a regular file"):
        loader.load(dir_path)


def test_unsupported_format_raises(loader: DocumentLoader, tmp_path: Path) -> None:
    pptx_file = tmp_path / "sample.pptx"
    pptx_file.write_text("dummy")
    with pytest.raises(ValueError, match="Unsupported file format '.pptx'"):
        loader.load(pptx_file)


def test_legacy_doc_raises_with_conversion_hint(loader: DocumentLoader, tmp_path: Path) -> None:
    doc_file = tmp_path / "legacy.doc"
    doc_file.write_bytes(b"dummy legacy binary word content")
    with pytest.raises(ValueError, match=r"Convert 'legacy\.doc' to '\.docx'"):
        loader.load(doc_file)


def test_empty_file_raises(loader: DocumentLoader, tmp_path: Path) -> None:
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("")
    with pytest.raises(ValueError, match="is empty"):
        loader.load(empty_file)


def test_whitespace_only_file_raises(loader: DocumentLoader, tmp_path: Path) -> None:
    ws_file = tmp_path / "spaces.txt"
    ws_file.write_text("   \n\t  \n ")
    with pytest.raises(ValueError, match="contains no extractable text"):
        loader.load(ws_file)


def test_load_plain_text(loader: DocumentLoader, tmp_path: Path) -> None:
    txt_file = tmp_path / "guide.txt"
    txt_file.write_text("Hello World RAG Guide", encoding="utf-8")

    doc = loader.load(txt_file)
    assert isinstance(doc, LoadedDocument)
    assert doc.file_name == "guide.txt"
    assert doc.title == "guide"
    assert doc.content == "Hello World RAG Guide"
    assert doc.content_type == "text/plain"
    assert doc.size_bytes == len("Hello World RAG Guide".encode("utf-8"))


def test_load_text_extension(loader: DocumentLoader, tmp_path: Path) -> None:
    text_file = tmp_path / "manual.text"
    text_file.write_text("System Manual Content", encoding="utf-8")

    doc = loader.load(str(text_file))
    assert doc.file_name == "manual.text"
    assert doc.title == "manual"
    assert doc.content == "System Manual Content"
    assert doc.content_type == "text/plain"


def test_load_markdown_rendered_semantics(loader: DocumentLoader, tmp_path: Path) -> None:
    md_file = tmp_path / "readme.md"
    md_file.write_text(
        "# Project Readme\n\nThis is **bold** text and [a link](https://example.com).",
        encoding="utf-8",
    )

    doc = loader.load(md_file)
    assert doc.file_name == "readme.md"
    assert doc.title == "readme"
    assert doc.content_type == "text/markdown"
    assert doc.content == "Project Readme\n\nThis is bold text and a link."


def test_load_markdown_extension(loader: DocumentLoader, tmp_path: Path) -> None:
    md_file = tmp_path / "notes.markdown"
    md_file.write_text("## Notes\n\n- Item 1\n- Item 2", encoding="utf-8")

    doc = loader.load(md_file)
    assert doc.file_name == "notes.markdown"
    assert doc.title == "notes"
    assert "Notes" in doc.content
    assert "Item 1" in doc.content
    assert doc.content_type == "text/markdown"


def test_load_text_cp1252_encoding(loader: DocumentLoader, tmp_path: Path) -> None:
    cp_file = tmp_path / "windows1252.txt"
    raw_bytes = "smart “quotes” and €100 price".encode("cp1252")
    cp_file.write_bytes(raw_bytes)

    doc = loader.load(cp_file)
    assert doc.file_name == "windows1252.txt"
    assert doc.content == "smart “quotes” and €100 price"
    assert doc.content_type == "text/plain"


def test_load_text_latin1_encoding(loader: DocumentLoader, tmp_path: Path) -> None:
    latin_file = tmp_path / "latin.txt"
    latin_file.write_bytes("Café résumé".encode("latin-1"))

    doc = loader.load(latin_file)
    assert doc.file_name == "latin.txt"
    assert doc.content == "Café résumé"
    assert doc.content_type == "text/plain"


def test_load_real_pdf_with_text(loader: DocumentLoader, tmp_path: Path) -> None:
    pdf_path = tmp_path / "real_document.pdf"
    pdf_path.write_bytes(_create_minimal_pdf("Extracted text from real PDF"))

    doc = loader.load(pdf_path)
    assert doc.file_name == "real_document.pdf"
    assert doc.title == "real_document"
    assert doc.content == "Extracted text from real PDF"
    assert doc.content_type == "application/pdf"
    assert doc.size_bytes == len(pdf_path.read_bytes())


def test_load_empty_real_pdf_raises(loader: DocumentLoader, tmp_path: Path) -> None:
    pdf_path = tmp_path / "blank.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with open(pdf_path, "wb") as f:
        writer.write(f)

    with pytest.raises(ValueError, match="No extractable text found"):
        loader.load(pdf_path)


def test_load_pdf_mocked_converter(loader: DocumentLoader, tmp_path: Path) -> None:
    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 dummy content")

    with patch("src.document_loader.PyPDFToDocument") as mock_converter_cls:
        mock_converter = MagicMock()
        mock_converter.run.return_value = {
            "documents": [
                Document(
                    content="Extracted PDF page 1 text",
                    meta={"file_path": "document.pdf"},
                )
            ]
        }
        mock_converter_cls.return_value = mock_converter

        doc = loader.load(pdf_path)
        assert doc.file_name == "document.pdf"
        assert doc.title == "document"
        assert doc.content == "Extracted PDF page 1 text"
        assert doc.content_type == "application/pdf"
        assert doc.size_bytes == len(b"%PDF-1.4 dummy content")


def test_load_real_docx(loader: DocumentLoader, tmp_path: Path) -> None:
    from docx import Document as DocxDocument

    docx_path = tmp_path / "report.docx"
    doc = DocxDocument()
    doc.add_paragraph("First paragraph content")
    doc.add_paragraph("Second paragraph content")
    doc.save(docx_path)

    loaded = loader.load(docx_path)
    assert loaded.file_name == "report.docx"
    assert loaded.title == "report"
    assert loaded.content_type == (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    assert "First paragraph content" in loaded.content
    assert "Second paragraph content" in loaded.content
    assert loaded.size_bytes == len(docx_path.read_bytes())


def test_load_corrupt_docx_raises(loader: DocumentLoader, tmp_path: Path) -> None:
    docx_path = tmp_path / "corrupt.docx"
    docx_path.write_bytes(b"Not a real docx file")

    with pytest.raises(ValueError, match="Failed to extract text from Word document"):
        loader.load(docx_path)


def test_load_corrupt_pdf_raises(loader: DocumentLoader, tmp_path: Path) -> None:
    pdf_path = tmp_path / "corrupt.pdf"
    pdf_path.write_bytes(b"Not a real PDF header or content")

    with pytest.raises(
        ValueError,
        match="Failed to extract text from PDF|No extractable text found",
    ):
        loader.load(pdf_path)
