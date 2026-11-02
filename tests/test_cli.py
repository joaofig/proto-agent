"""Tests for CLI argument parsing and execution flow in main.py."""

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from main import build_arg_parser, main


def test_arg_parser_no_args():
    parser = build_arg_parser()
    args = parser.parse_args([])
    assert args.doc is None


def test_arg_parser_doc_arg():
    parser = build_arg_parser()
    args = parser.parse_args(["--doc", "documents/sample.pdf"])
    assert args.doc == "documents/sample.pdf"


@patch("main.RAGService")
def test_cli_doc_success(mock_rag_service_cls, capsys):
    mock_rag_instance = MagicMock()
    mock_rag_instance.index_file.return_value = (42, 5)
    mock_rag_service_cls.return_value = mock_rag_instance

    with pytest.raises(SystemExit) as exc_info:
        main(["--doc", "/path/to/my_document.pdf"])

    assert exc_info.value.code == 0
    mock_rag_instance.index_file.assert_called_once_with(
        file_path="/path/to/my_document.pdf",
        user_id=1,
        company_id=1,
    )

    captured = capsys.readouterr()
    assert captured.out == "Successfully indexed 'my_document.pdf' (ID: 42, Chunks: 5)\n"
    assert captured.err == ""


@patch("main.RAGService")
def test_cli_doc_file_not_found(mock_rag_service_cls, capsys):
    mock_rag_instance = MagicMock()
    mock_rag_instance.index_file.side_effect = FileNotFoundError("File not found: missing.txt")
    mock_rag_service_cls.return_value = mock_rag_instance

    with pytest.raises(SystemExit) as exc_info:
        main(["--doc", "missing.txt"])

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "Error indexing document: File not found: missing.txt" in captured.err
    assert captured.out == ""


@patch("main.RAGService")
def test_cli_doc_unsupported_format(mock_rag_service_cls, capsys):
    mock_rag_instance = MagicMock()
    mock_rag_instance.index_file.side_effect = ValueError(
        "Unsupported file format '.docx'. Supported formats: .markdown, .md, .pdf, .text, .txt"
    )
    mock_rag_service_cls.return_value = mock_rag_instance

    with pytest.raises(SystemExit) as exc_info:
        main(["--doc", "document.docx"])

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "Error indexing document: Unsupported file format '.docx'" in captured.err
    assert captured.out == ""


@patch("main.RAGService")
def test_cli_doc_runtime_error(mock_rag_service_cls, capsys):
    mock_rag_instance = MagicMock()
    mock_rag_instance.index_file.side_effect = RuntimeError(
        "Inserted chunk count mismatch for document 'doc' (ID 10): expected 3, got 2."
    )
    mock_rag_service_cls.return_value = mock_rag_instance

    with pytest.raises(SystemExit) as exc_info:
        main(["--doc", "doc.txt"])

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "Error indexing document: Inserted chunk count mismatch" in captured.err
    assert captured.out == ""


@patch("main.RAGService")
def test_cli_doc_unexpected_exception(mock_rag_service_cls, capsys):
    mock_rag_instance = MagicMock()
    mock_rag_instance.index_file.side_effect = Exception("Database connection failure")
    mock_rag_service_cls.return_value = mock_rag_instance

    with pytest.raises(SystemExit) as exc_info:
        main(["--doc", "data.txt"])

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "Error indexing document: Database connection failure" in captured.err
    assert captured.out == ""


@patch("main.chat_loop")
@patch("main.build_agent")
def test_cli_no_args_starts_chat_repl(mock_build_agent, mock_chat_loop):
    mock_agent = MagicMock()
    mock_build_agent.return_value = mock_agent

    main([])

    mock_build_agent.assert_called_once_with()
    mock_chat_loop.assert_called_once_with(mock_agent)


def test_cli_doc_flag_requires_value():
    parser = build_arg_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--doc"])
    assert exc_info.value.code == 2


@patch("src.rag_service.embed_texts")
@patch("src.rag_service.PostgresRepository")
def test_cli_e2e_text_file_ingestion(mock_repo_cls, mock_embed, tmp_path, capsys):
    test_file = tmp_path / "overview.txt"
    test_file.write_text("Proto-agent RAG system architecture overview.", encoding="utf-8")

    mock_repo = MagicMock()
    mock_repo.insert_document.return_value = 101
    mock_repo.insert_chunks.return_value = [1]
    mock_repo_cls.return_value = mock_repo
    mock_embed.return_value = [[0.1, 0.2, 0.3]]

    with pytest.raises(SystemExit) as exc_info:
        main(["--doc", str(test_file)])

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert captured.out == "Successfully indexed 'overview.txt' (ID: 101, Chunks: 1)\n"
    assert captured.err == ""


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_cli_subprocess_help():
    proc = subprocess.run(
        [sys.executable, "main.py", "--help"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    assert proc.returncode == 0
    assert "--doc FILE_PATH" in proc.stdout
    assert "Interactive chat REPL" in proc.stdout


def test_cli_subprocess_doc_missing_value():
    proc = subprocess.run(
        [sys.executable, "main.py", "--doc"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    assert proc.returncode == 2
    assert "error: argument --doc: expected one argument" in proc.stderr


def test_cli_subprocess_missing_file():
    proc = subprocess.run(
        [sys.executable, "main.py", "--doc", "nonexistent_file_abc.txt"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    assert proc.returncode == 1
    assert "Error indexing document: File not found: nonexistent_file_abc.txt" in proc.stderr


def test_cli_subprocess_unsupported_format(tmp_path):
    bad_file = tmp_path / "test.csv"
    bad_file.write_text("a,b,c", encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "main.py", "--doc", str(bad_file)],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    assert proc.returncode == 1
    assert "Error indexing document: Unsupported file format '.csv'" in proc.stderr
