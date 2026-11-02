"""Entry point for the proto-agent application."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from src.haystack_chat_agent import build_agent, chat_loop
from src.rag_service import RAGService


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Proto-agent CLI: Interactive chat REPL and document indexing tool."
    )
    parser.add_argument(
        "--doc",
        type=str,
        metavar="FILE_PATH",
        help="Path to a document (.txt, .md, .pdf, .docx) to index into the knowledge base.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run CLI application."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.doc:
        doc_path = Path(args.doc)
        try:
            rag_service = RAGService()
            doc_id, chunk_count = rag_service.index_file(
                file_path=args.doc,
                user_id=1,
                company_id=1,
            )
            print(
                f"Successfully indexed '{doc_path.name}' (ID: {doc_id}, Chunks: {chunk_count})"
            )
            sys.exit(0)
        except Exception as exc:
            print(f"Error indexing document: {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        chat_loop(build_agent())


if __name__ == "__main__":
    main()
