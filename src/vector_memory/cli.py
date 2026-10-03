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


def _run_cli(op, *args, json_mode: bool = False, **kwargs) -> None:
    """Run a core op for a CLI command: catch, format, emit (VM-20 contract)."""
    try:
        result = op(*args, **kwargs)
    except Exception as exc:  # boundary error contract
        result = _guard(exc)
    _emit(result, json_mode=json_mode)


def _emit(result: str, *, json_mode: bool = False) -> None:
    """Print a core-op result, honoring the stdout/stderr contract (VM-20).

    ``ErrorType: ...`` lines go to stderr with exit code 1. In ``--json`` mode
    the envelope carries ``schema: 1`` and, on failure, ``error``/``error_type``.
    """
    error_prefixes = ("ArgumentError:", "ConfigError:", "NotFoundError:",
                      "ConflictError:", "SensitiveContentError:",
                      "BackendError:", "InternalError:")
    is_error = result.startswith(error_prefixes)
    if json_mode:
        import json

        payload: dict[str, object] = {"schema": 1, "ok": not is_error}
        if is_error:
            payload["error"] = result
            payload["error_type"] = result.split(":", 1)[0]
        else:
            payload["result"] = result
        typer.echo(json.dumps(payload, ensure_ascii=False))
    else:
        typer.echo(result, err=is_error)
    if is_error:
        raise typer.Exit(1)


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
    allow_duplicate: bool = typer.Option(
        False, "--allow-duplicate",
        help="Force a new point even if identical normalized text already exists."),
    on_similar: str = typer.Option(
        "", help="Near-duplicate behavior: warn (default) | skip | error."),
    allow_sensitive: bool = typer.Option(
        False, "--allow-sensitive",
        help="Override the secret-detection guard (use only for false positives)."),
    json_output: bool = typer.Option(False, "--json", help="JSON envelope output (schema 1)."),
) -> None:
    """Save a document/scenario outcome into the vector DB."""
    supersedes_ids = [s.strip() for s in supersedes.split(",") if s.strip()] if supersedes else None

    _set_override = "VM_ALLOW_SENSITIVE"
    import os as _os
    if allow_sensitive:
        _os.environ[_set_override] = "1"
    try:
        _run_cli(save_memory, text, metadata, collection, project=project, type=type,
                 tags=list(tags), supersedes=supersedes_ids,
                 allow_duplicate=allow_duplicate, on_similar=on_similar,
                 json_mode=json_output)
    finally:
        if allow_sensitive:
            _os.environ.pop(_set_override, None)

@app.command()
def save_many(
    texts: list[str] = typer.Argument(..., help="Texts to embed and store in one batch."),
    project: str = typer.Option("", help="Project scope, stored as metadata key 'project'."),
    type: str = typer.Option("", help="Entry kind, stored as metadata key 'type'."),
    tags: list[str] = typer.Option([], help="Repeatable; each becomes one entry of metadata 'tags'."),
    metadata: str = typer.Option("{}", help="Metadata applied to every document (JSON)."),
    collection: str = typer.Option("", help="Target collection (created if missing)."),
    json_output: bool = typer.Option(False, "--json", help="JSON envelope output (schema 1)."),
) -> None:
    """Save multiple documents in one batch (single embed + upsert)."""
    _run_cli(
        save_memories,
        list(texts), metadata, collection,
        project=project, type=type, tags=list(tags),
        json_mode=json_output,
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
    json_output: bool = typer.Option(False, "--json", help="JSON envelope output (schema 1)."),
) -> None:
    """Search stored documents/scenarios semantically similar to a query."""
    _run_cli(search_memory, query, limit, filter, collection, project,
             include_inactive=include_inactive, json_mode=json_output)


@app.command()
def update(
    point_id: str = typer.Argument(..., help="ID of the memory to update."),
    text: str = typer.Option(None, help="New text (re-embeds). Omit to keep."),
    metadata: str = typer.Option("", help="New metadata (replaces entirely). Omit to keep."),
    merge_metadata: bool = typer.Option(
        False, "--merge-metadata",
        help="Merge the given metadata keys into the existing payload instead of replacing."),
    collection: str = typer.Option("", help="Collection holding the point."),
    json_output: bool = typer.Option(False, "--json", help="JSON envelope output (schema 1)."),
) -> None:
    """Update an existing memory in place under the same ID."""
    _run_cli(update_memory, point_id, text, metadata, collection,
             merge_metadata=merge_metadata, json_mode=json_output)


@app.command()
def patch(
    point_id: str = typer.Argument(..., help="ID of the memory to patch."),
    set: str = typer.Option("{}", "--set", help="Metadata keys to merge (JSON)."),
    unset: list[str] = typer.Option([], "--unset", help="Metadata key to remove (repeatable)."),
    collection: str = typer.Option("", help="Collection holding the point."),
    json_output: bool = typer.Option(False, "--json", help="JSON envelope output (schema 1)."),
) -> None:
    """Patch metadata without re-embedding (no embedding call)."""
    _run_cli(patch_metadata, point_id, set, list(unset), collection,
             json_mode=json_output)


@app.command()
def delete(
    point_id: str = typer.Argument(..., help="ID of the memory to delete."),
    collection: str = typer.Option("", help="Collection holding the point."),
    json_output: bool = typer.Option(False, "--json", help="JSON envelope output (schema 1)."),
) -> None:
    """Delete a stored memory (point) by ID."""
    _run_cli(delete_memory, point_id, collection, json_mode=json_output)


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
    _run_cli(_run_safe, _fmt_summary, migrate_collection, name,
             assume_model=assume_model)


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
    _run_cli(set_status, ids, "archived", collection)


@app.command(name="unarchive")
def unarchive_cmd(
    point_ids: str = typer.Argument(..., help="Comma-separated point IDs to unarchive."),
    collection: str = typer.Option("", help="Collection holding the points."),
) -> None:
    """Unarchive memories (return them to active)."""
    ids = [s.strip() for s in point_ids.split(",") if s.strip()]
    _run_cli(set_status, ids, "active", collection)


@app.command()
def reembed(
    collection: str = typer.Option(..., "--collection", help="Source collection (never mutated)."),
    to: str = typer.Option(..., "--to", help="Target collection (same IDs and payloads; re-embedded)."),
    batch: int = typer.Option(32, help="Points per embed batch."),
    resume: bool = typer.Option(False, "--resume", help="Skip IDs already present in the target."),
) -> None:
    """Re-embed a collection into a new one with the current model (VM model change)."""
    from .reembed import reembed_collection

    _run_cli(_run_safe, _fmt_reembed, reembed_collection, collection, to,
             EMBED_MODEL, batch_size=batch, resume=resume)


def _fmt_reembed(s: dict) -> str:
    return (
        f"Re-embedded {s['copied']}/{s['source_count']} points from {s['source']!r} "
        f"into {s['target']!r} (skipped {s['skipped']}); target now has "
        f"{s['target_count']} points, model {s['model']}"
    )


def _run_safe(fmt, op, *args, **kwargs) -> str:
    """Format the summary of an op that must not raise past the CLI boundary."""
    return fmt(op(*args, **kwargs))


@app.command(name="list-collections")
def list_collections_cmd() -> None:
    """List all collections currently present in Qdrant."""
    _run_cli(list_collections)


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
