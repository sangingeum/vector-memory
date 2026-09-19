"""vector-memory CLI: thin typer app over vector_memory.core (ruling §3).

Every subcommand maps 1:1 to a core op / MCP tool of the same name. One-shot
process: no daemon, no server RPC. All diagnostics go to stderr only when
--verbose is set; stdout carries results.
"""

from __future__ import annotations

from typing import Any

import typer

from .core import (
    delete_memory,
    list_collections,
    save_memories,
    save_memory,
    search_memory,
    update_memory,
)

app = typer.Typer(
    name="vector-memory",
    help="Ollama embeddings + Qdrant vector memory (one-shot CLI; no daemon).",
    no_args_is_help=True,
)


@app.command()
def save(
    text: str = typer.Argument(..., help="Text to embed and store."),
    metadata: str = typer.Option("{}", help="Metadata as JSON string, e.g. '{\"tags\": [\"a\"]}'."),
    collection: str = typer.Option("", help="Target collection (created if missing)."),
) -> None:
    """Save a document/scenario outcome into the vector DB."""
    typer.echo(save_memory(text, metadata, collection))


@app.command()
def save_many(
    texts: list[str] = typer.Argument(..., help="Texts to embed and store in one batch."),
    metadata: str = typer.Option("{}", help="Metadata applied to every document (JSON)."),
    collection: str = typer.Option("", help="Target collection (created if missing)."),
) -> None:
    """Save multiple documents in one batch (single embed + upsert)."""
    typer.echo(save_memories(list(texts), metadata, collection))


@app.command()
def search(
    query: str = typer.Argument(..., help="Query text."),
    limit: int = typer.Option(3, help="Max hits."),
    filter: str = typer.Option("", help="Payload filter as JSON, e.g. '{\"tags\": [\"x\"]}'."),
    collection: str = typer.Option("", help="Collection to search (default: configured)."),
    project: str = typer.Option("", help="Project-scoped memory (payload project= match)."),
) -> None:
    """Search stored documents/scenarios semantically similar to a query."""
    typer.echo(search_memory(query, limit, filter, collection, project))


@app.command()
def update(
    point_id: str = typer.Argument(..., help="ID of the memory to update."),
    text: str = typer.Option(None, help="New text (re-embeds). Omit to keep."),
    metadata: str = typer.Option("", help="New metadata (replaces entirely). Omit to keep."),
    collection: str = typer.Option("", help="Collection holding the point."),
) -> None:
    """Update an existing memory in place under the same ID."""
    typer.echo(update_memory(point_id, text, metadata, collection))


@app.command()
def delete(
    point_id: str = typer.Argument(..., help="ID of the memory to delete."),
    collection: str = typer.Option("", help="Collection holding the point."),
) -> None:
    """Delete a stored memory (point) by ID."""
    typer.echo(delete_memory(point_id, collection))


@app.command(name="list-collections")
def list_collections_cmd() -> None:
    """List all collections currently present in Qdrant."""
    typer.echo(list_collections())


@app.callback()
def _main(
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Verbose diagnostics (INFO) on stderr."),
) -> None:
    """Global options for the vector-memory CLI."""
    from .logsetup import configure_logging

    configure_logging(verbose=verbose)


def main() -> None:
    """Console-script entry point (``vector-memory``)."""
    app()


if __name__ == "__main__":
    main()
