"""Document loading, format validation, and text extraction using Haystack converters."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from haystack.components.converters import (
    DOCXToDocument,
    MarkdownToDocument,
    PyPDFToDocument,
    TextFileToDocument,
)

logger = logging.getLogger(__name__)


@dataclass
class LoadedDocument:
    """Container for loaded and extracted document data."""

    file_name: str
    title: str
    content: str
    content_type: str
    size_bytes: int


class DocumentLoader:
    """Loads local documents and extracts text content using Haystack converters."""

    SUPPORTED_EXTENSIONS: ClassVar[dict[str, str]] = {
        ".txt": "text/plain",
        ".text": "text/plain",
        ".md": "text/markdown",
        ".markdown": "text/markdown",
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }
    FALLBACK_ENCODINGS: ClassVar[tuple[str, ...]] = (
        "utf-8-sig",
        "cp1252",
        "latin-1",
    )

    def load(self, file_path: str | Path) -> LoadedDocument:
        """Validates file existence and type, extracts text, and returns LoadedDocument.

        Args:
            file_path: Path to the target document file.

        Returns:
            LoadedDocument instance containing extracted text and metadata.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If the file is invalid, unsupported, empty, or unreadable.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not path.is_file():
            raise ValueError(f"Path is not a regular file: {file_path}")

        suffix = path.suffix.lower()
        if suffix == ".doc":
            raise ValueError(
                f"Legacy Word format '.doc' is not supported. "
                f"Convert '{path.name}' to '.docx' and try again."
            )
        if suffix not in self.SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(self.SUPPORTED_EXTENSIONS.keys()))
            raise ValueError(
                f"Unsupported file format '{path.suffix}'. Supported formats: {supported}"
            )

        content_type = self.SUPPORTED_EXTENSIONS[suffix]

        try:
            size_bytes = path.stat().st_size
        except OSError as exc:
            raise ValueError(f"Cannot read file '{path.name}': {exc}") from exc

        if size_bytes == 0:
            raise ValueError(f"File '{path.name}' is empty (0 bytes).")

        content: str
        if suffix in {".txt", ".text"}:
            content = self._extract_text(path)
        elif suffix in {".md", ".markdown"}:
            content = self._extract_markdown(path)
        elif suffix == ".pdf":
            content = self._extract_pdf(path).replace(chr(0), "")
        elif suffix == ".docx":
            content = self._extract_docx(path)
        else:
            raise ValueError(f"Unsupported format '{suffix}'.")

        if not content or not content.strip():
            raise ValueError(f"Document '{path.name}' contains no extractable text.")

        return LoadedDocument(
            file_name=path.name,
            title=path.stem,
            content=content.strip(),
            content_type=content_type,
            size_bytes=size_bytes,
        )

    def _extract_text(self, path: Path) -> str:
        """Extract plain text from a text file with encoding fallback."""
        for enc in self.FALLBACK_ENCODINGS:
            try:
                converter = TextFileToDocument(encoding=enc)
                result = converter.run(sources=[path])
                docs = result.get("documents", [])
                if docs and docs[0].content:
                    return docs[0].content
            except Exception as exc:
                logger.debug(f"Failed to decode '{path.name}' with encoding '{enc}': {exc}")

        raise ValueError(f"Failed to extract text from '{path.name}'.")

    def _extract_markdown(self, path: Path) -> str:
        """Extract text from a Markdown file using MarkdownToDocument with encoding fallback."""
        for enc in self.FALLBACK_ENCODINGS:
            try:
                converter = MarkdownToDocument(progress_bar=False, encoding=enc)
                result = converter.run(sources=[path])
                docs = result.get("documents", [])
                if docs and docs[0].content:
                    return docs[0].content
            except Exception as exc:
                logger.debug(f"Failed to extract markdown from '{path.name}' with encoding '{enc}': {exc}")

        raise ValueError(f"Failed to extract text from markdown '{path.name}'.")

    def _extract_pdf(self, path: Path) -> str:
        """Extract text from a PDF file using PyPDFToDocument."""
        try:
            converter = PyPDFToDocument()
            result = converter.run(sources=[path])
            docs = result.get("documents", [])
            if not docs:
                raise ValueError(f"Failed to extract text from PDF '{path.name}'.")

            contents = [doc.content for doc in docs if doc.content and doc.content.strip()]
            if not contents:
                raise ValueError(f"No extractable text found in PDF '{path.name}'.")
            return "\n\n".join(contents)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"Failed to extract text from PDF '{path.name}': {exc}") from exc

    def _extract_docx(self, path: Path) -> str:
        """Extract text from a Word .docx file using DOCXToDocument."""
        try:
            converter = DOCXToDocument()
            result = converter.run(sources=[path])
            docs = result.get("documents", [])
            if not docs:
                raise ValueError(f"Failed to extract text from Word document '{path.name}'.")

            contents = [doc.content for doc in docs if doc.content and doc.content.strip()]
            if not contents:
                raise ValueError(f"No extractable text found in Word document '{path.name}'.")
            return "\n\n".join(contents)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"Failed to extract text from Word document '{path.name}': {exc}") from exc
