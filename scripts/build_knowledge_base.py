"""Build the historical-failure vector store used for RAG.

Reads every ``.txt`` report in ``data/historical_failures/``, splits it into ~500-token
chunks, embeds the chunks locally with ``nomic-embed-text`` via Ollama and stores them in
ChromaDB under ``data/vector_store/``. Re-running rebuilds the collection from scratch.

Usage::

    ollama pull nomic-embed-text          # once
    python scripts/build_knowledge_base.py [--docs DIR] [--store DIR] [--query "text"]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

from config.settings import configure_logging, get_settings  # noqa: E402
from src.memory.historical_rag import HistoricalRAG, OllamaEmbedder  # noqa: E402

console = Console()


def main() -> int:
    """Build the store and optionally run a test query. Returns a process exit code."""
    settings = get_settings()
    rag_cfg = settings.load_agents_config().get("historical_rag", {}) or {}
    parser = argparse.ArgumentParser(description="Build the historical-failure RAG vector store.")
    parser.add_argument("--docs", type=Path, default=settings.historical_docs_dir, help="Folder of .txt failure reports.")
    parser.add_argument("--store", type=Path, default=settings.historical_store_dir, help="ChromaDB folder.")
    parser.add_argument("--model", default=settings.embedding_model, help="Ollama embedding model.")
    parser.add_argument("--query", default="LOX tank pressure drop followed by turbopump vibration rise",
                        help="Test query to run after building.")
    args = parser.parse_args()
    configure_logging("WARNING")

    embedder = OllamaEmbedder(settings.ollama_base_url, args.model)
    if not embedder.available():
        console.print(f"[red]Embedding model '{args.model}' is not available in Ollama at {settings.ollama_base_url}.[/red]")
        console.print(f"Install it with:  ollama pull {args.model}")
        return 1

    rag = HistoricalRAG(
        args.store, embedder,
        chunk_tokens=int(rag_cfg.get("chunk_tokens", 500)),
        overlap_tokens=int(rag_cfg.get("overlap_tokens", 50)),
    )
    console.rule("[bold]Building historical failure knowledge base")
    try:
        stats = rag.build(args.docs, progress=lambda line: console.print(f"  • {line}"))
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        return 1
    console.print(f"[green]Indexed {stats['documents']} reports as {stats['chunks']} chunks "
                  f"with {embedder.name} into {args.store}[/green]")

    if args.query:
        table = Table(title=f"Test query: {args.query}", header_style="bold")
        for column in ("#", "Case", "Similarity", "File"):
            table.add_column(column)
        for i, case in enumerate(rag.search(args.query, k=3), 1):
            table.add_row(str(i), case.title, f"{case.score:.3f}", case.source_file)
        console.print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
