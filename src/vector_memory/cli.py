"""vector-memory CLI: thin typer app over vector_memory.core (ruling §3).

Every subcommand maps 1:1 to a core op / MCP tool of the same name. One-shot
process: no daemon, no server RPC. All diagnostics go to stderr only when
--verbose is set; stdout carries results.
"""

from __future__ import annotations

import typer

from .core import (
    delete_memory,
    list_collections,
    patch_metadata,
    save_memories,
    save_memory,
    search_memory,
    set_status,
    update_memory,
)
from .embedding import EMBED_MODEL
from .store import COLLECTION_NAME

app = typer.Typer(
    name="vector-memory",
    help="Ollama embeddings + Qdrant vector memory (one-shot CLI; no daemon).",
    no_args_is_help=True,
)


def _guard(result_or_exc: str | BaseException) -> str:
    """Normalize a core-op outcome: exceptions become ``ErrorType: ...`` text."""
    from .errors import format_error

    if isinstance(result_or_exc, BaseException):
        return format_error(result_or_exc)
    return result_or_exc


def _emit(result: str) -> None:
    """Print a core-op result, honoring the stdout/stderr contract.

    ``ErrorType: ...`` lines go to stderr with exit code 1; warnings stay on
    the result string (stderr in text mode is reserved for diagnostics).
    """
    if result.startswith(("ArgumentError:", "ConfigError:", "NotFoundError:",
                          "ConflictError:", "SensitiveContentError:",
                          "BackendError:", "InternalError:")):
        typer.echo(result, err=True)
        raise typer.Exit(1)
    typer.echo(result)


@app.command()
def save(
    text: str = typer.Argument(..., help="Text to embed and store."),
    project: str = typer.Option("", help="Project scope, stored as metadata key 'project'."),
    type: str = typer.Option("", help="Entry kind, stored as metadata key 'type'."),
    tags: list[str] = typer.Option([], help="Repeatable; each becomes one entry of metadata 'tags'."),
    metadata: str = typer.Option("{}", help="Metadata as JSON string, e.g. '{\"tags\": [\"a\"]}'."),
    collection: str = typer.Option("", help="Target collection (created if missing)."),
    supersedes: str = typer.Option(
        "", help="Comma-separated point IDs this memory replaces "
        "(they are marked superseded and hidden from default search)."),
) -> None:
    """Save a document/scenario outcome into the vector DB."""
    supersedes_ids = [s.strip() for s in supersedes.split(",") if s.strip()] if supersedes else None
    typer.echo(_guard(save_memory(text, metadata, collection, project=project, type=type,
                                  tags=list(tags), supersedes=supersedes_ids)))


@app.command()
def save_many(
    texts: list[str] = typer.Argument(..., help="Texts to embed and store in one batch."),
    project: str = typer.Option("", help="Project scope, stored as metadata key 'project'."),
    type: str = typer.Option("", help="Entry kind, stored as metadata key 'type'."),
    tags: list[str] = typer.Option([], help="Repeatable; each becomes one entry of metadata 'tags'."),
    metadata: str = typer.Option("{}", help="Metadata applied to every document (JSON)."),
    collection: str = typer.Option("", help="Target collection (created if missing)."),
) -> None:
    """Save multiple documents in one batch (single embed + upsert)."""
    _emit(
        _guard(
            save_memories(
                list(texts), metadata, collection,
                project=project, type=type, tags=list(tags),
            )
        )
    )


@app.command()
def search(
    query: str = typer.Argument(..., help="Query text."),
    limit: int = typer.Option(3, help="Max hits."),
    filter: str = typer.Option("", help="Payload filter as JSON, e.g. '{\"tags\": [\"x\"]}'."),
    collection: str = typer.Option("", help="Collection to search (default: configured)."),
    project: str = typer.Option("", help="Project-scoped memory (payload project= match)."),
    include_inactive: bool = typer.Option(
        False, "--include-inactive",
        help="Include superseded/archived memories (annotated with status)."),
) -> None:
    """Search stored documents/scenarios semantically similar to a query."""
    _emit(_guard(search_memory(query, limit, filter, collection, project,
                               include_inactive=include_inactive)))


@app.command()
def update(
    point_id: str = typer.Argument(..., help="ID of the memory to update."),
    text: str = typer.Option(None, help="New text (re-embeds). Omit to keep."),
    metadata: str = typer.Option("", help="New metadata (replaces entirely). Omit to keep."),
    merge_metadata: bool = typer.Option(
        False, "--merge-metadata",
        help="Merge the given metadata keys into the existing payload instead of replacing."),
    collection: str = typer.Option("", help="Collection holding the point."),
) -> None:
    """Update an existing memory in place under the same ID."""
    _emit(_guard(update_memory(point_id, text, metadata, collection,
                               merge_metadata=merge_metadata)))


@app.command()
def patch(
    point_id: str = typer.Argument(..., help="ID of the memory to patch."),
    set: str = typer.Option("{}", "--set", help="Metadata keys to merge (JSON)."),
    unset: list[str] = typer.Option([], "--unset", help="Metadata key to remove (repeatable)."),
    collection: str = typer.Option("", help="Collection holding the point."),
) -> None:
    """Patch metadata without re-embedding (no embedding call)."""
    _emit(_guard(patch_metadata(point_id, set, list(unset), collection)))


@app.command()
def delete(
    point_id: str = typer.Argument(..., help="ID of the memory to delete."),
    collection: str = typer.Option("", help="Collection holding the point."),
) -> None:
    """Delete a stored memory (point) by ID."""
    _emit(_guard(delete_memory(point_id, collection)))


@app.command()
def migrate(
    collection: str = typer.Option("", help="Collection to migrate (default: configured)."),
    assume_model: str = typer.Option(
        "", help="Record this embedding model on migrated points (e.g. qwen3-embedding:8b). "
        "Only set it when you are certain of the model that produced the vectors."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Report what would change without writing."
    ),
) -> None:
    """Backfill system fields on legacy points (non-destructive, resumable)."""
    from .migrate import migrate_collection

    name = collection if collection.strip() else COLLECTION_NAME
    if dry_run:
        # Read-only probe: how many points lack system fields?
        from . import store as _store
        from .payload import CREATED_TS

        total, legacy = 0, 0
        offset = None
        while True:
            batch, offset = _store.qdrant.scroll(
                collection_name=name, limit=256, offset=offset,
                with_payload=True, with_vectors=False,
            )
            if not batch:
                break
            total += len(batch)
            legacy += sum(1 for p in batch if CREATED_TS not in (p.payload or {}))
            if offset is None:
                break
        typer.echo(f"dry-run: {legacy}/{total} points in {name!r} would be migrated (nothing written)")
        return
    _emit(_guard(_fmt_summary(migrate_collection(name, assume_model=assume_model))))


def _fmt_summary(summary: dict) -> str:
    return (
        f"Migrated {summary['migrated']}/{summary['scanned']} points in "
        f"{summary['collection']!r}; embed_model: {summary['embed_model']}"
    )


@app.command()
def archive(
    point_ids: str = typer.Argument(..., help="Comma-separated point IDs to archive."),
    collection: str = typer.Option("", help="Collection holding the points."),
) -> None:
    """Archive memories (hidden from default search; kept for audit)."""
    ids = [s.strip() for s in point_ids.split(",") if s.strip()]
    _emit(_guard(set_status(ids, "archived", collection)))


@app.command(name="unarchive")
def unarchive_cmd(
    point_ids: str = typer.Argument(..., help="Comma-separated point IDs to unarchive."),
    collection: str = typer.Option("", help="Collection holding the points."),
) -> None:
    """Unarchive memories (return them to active)."""
    ids = [s.strip() for s in point_ids.split(",") if s.strip()]
    _emit(_guard(set_status(ids, "active", collection)))


@app.command()
def reembed(
    collection: str = typer.Option(..., "--collection", help="Source collection (never mutated)."),
    to: str = typer.Option(..., "--to", help="Target collection (same IDs and payloads; re-embedded)."),
    batch: int = typer.Option(32, help="Points per embed batch."),
    resume: bool = typer.Option(False, "--resume", help="Skip IDs already present in the target."),
) -> None:
    """Re-embed a collection into a new one with the current model (VM model change)."""
    from .reembed import reembed_collection

    _emit(_guard(_fmt_reembed(reembed_collection(collection, to, EMBED_MODEL,
                                                 batch_size=batch, resume=resume))))


def _fmt_reembed(s: dict) -> str:
    return (
        f"Re-embedded {s['copied']}/{s['source_count']} points from {s['source']!r} "
        f"into {s['target']!r} (skipped {s['skipped']}); target now has "
        f"{s['target_count']} points, model {s['model']}"
    )


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
