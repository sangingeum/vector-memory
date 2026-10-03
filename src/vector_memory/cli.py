"""vector-memory CLI: thin typer app over vector_memory.core (ruling §3).

Every subcommand maps 1:1 to a core op / MCP tool of the same name. One-shot
process: no daemon, no server RPC. All diagnostics go to stderr only when
--verbose is set; stdout carries results.
"""

from __future__ import annotations

import typer

from . import backup, browse, bulk
from .browse import get_memory
from .core import (
    delete_many,
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
    since: str = typer.Option("", "--since", help="Only memories created after this (7d, 24h, or ISO date)."),
    before: str = typer.Option("", "--before", help="Only memories created before this (7d, 24h, or ISO date)."),
    tag: list[str] = typer.Option([], "--tag", help="Tag filter (repeatable, any-of)."),
    type_filter: str = typer.Option("", "--type", help="Exact type filter."),
    source: str = typer.Option("", "--source", help="Exact source filter."),
    min_score: float = typer.Option(None, "--min-score", help="Drop hits below this score."),
    recency_weight: float = typer.Option(
        0.0, "--recency-weight", help="Re-rank by recency (0-1; 0 = relevance only)."),
    mmr: float = typer.Option(0.0, "--mmr", help="Maximal marginal relevance for diversity (0-1)."),
    brief: bool = typer.Option(False, "--brief", help="Truncate hit text (prefers summary)."),
    max_chars: int = typer.Option(0, "--max-chars", help="Per-hit char cap (0 = full text)."),
    output_format: str = typer.Option("text", "--format", help="text | compact"),
    agent: str = typer.Option("", "--agent", help="Filter by author agent id (_agent)."),
    json_output: bool = typer.Option(False, "--json", help="JSON envelope output (schema 1)."),
) -> None:
    """Search stored documents/scenarios semantically similar to a query."""
    _run_cli(search_memory, query, limit, filter, collection, project,
             include_inactive=include_inactive, since=since, before=before,
             tag=list(tag), type=type_filter, source=source,
             min_score=min_score, recency_weight=recency_weight, mmr=mmr,
             brief=brief, max_chars=max_chars, output_format=output_format,
             agent=agent, json_mode=json_output)


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
    point_id: str = typer.Argument(..., help="ID(s) of the memories to delete (comma-separated)."),
    collection: str = typer.Option("", help="Collection holding the point."),
    json_output: bool = typer.Option(False, "--json", help="JSON envelope output (schema 1)."),
) -> None:
    """Delete stored memories (points) by ID."""
    ids = [s.strip() for s in point_id.split(",") if s.strip()]
    _run_cli(delete_many, ids, collection, json_mode=json_output)


@app.command(name="delete-by-filter")
def delete_by_filter_cmd(
    filter: str = typer.Option("", help="Payload filter as JSON (required; empty = refused)."),
    collection: str = typer.Option("", help="Collection to delete from."),
    project: str = typer.Option("", help="Project filter."),
    no_dry_run: bool = typer.Option(False, "--no-dry-run", help="Really delete (needs --yes too)."),
    yes: bool = typer.Option(False, "--yes", help="Confirm real deletion."),
    json_output: bool = typer.Option(False, "--json", help="JSON envelope output (schema 1)."),
) -> None:
    """Bulk-delete points matching a filter (DRY-RUN by default)."""
    _run_cli(bulk.delete_by_filter, filter, collection, project,
             no_dry_run=no_dry_run, yes=yes, json_mode=json_output)


@app.command(name="delete-collection")
def delete_collection_cmd(
    name: str = typer.Argument(..., help="Collection to delete."),
    confirm: str = typer.Option("", "--confirm", help="Repeat the collection name to confirm."),
    i_know_this_is_default: bool = typer.Option(
        False, "--i-know-this-is-default",
        help="Required to delete the configured default collection."),
) -> None:
    """Delete a whole collection (double-confirmed)."""
    _run_cli(bulk.delete_collection, name, confirm,
             default_collection="" if not i_know_this_is_default else "__force_default__")


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


@app.command()
def get(
    point_ids: str = typer.Argument(..., help="Comma-separated point IDs."),
    collection: str = typer.Option("", help="Collection holding the points."),
    system: bool = typer.Option(False, "--system", help="Include _-prefixed system fields."),
    json_output: bool = typer.Option(False, "--json", help="JSON envelope output (schema 1)."),
) -> None:
    """Fetch full text + metadata for the given memories (no embedding call)."""
    ids = [s.strip() for s in point_ids.split(",") if s.strip()]
    _run_cli(get_memory, ids, collection, with_system=system, json_mode=json_output)


@app.command()
def list_memories(
    collection: str = typer.Option("", help="Collection to browse."),
    filter: str = typer.Option("", help="Payload filter as JSON."),
    limit: int = typer.Option(25, help="Page size."),
    order_by: str = typer.Option("", "--order-by", help="Sort by created|updated (newest first)."),
    project: str = typer.Option("", help="Project filter."),
    include_inactive: bool = typer.Option(False, "--include-inactive"),
    cursor: str = typer.Option("", help="Offset from the previous page's next-cursor."),
) -> None:
    """Browse memories (deterministic newest-first order, with next-cursor)."""
    _run_cli(browse.list_memories, collection, filter, limit, order_by, project,
             include_inactive, cursor)


@app.command()
def count(
    collection: str = typer.Option("", help="Collection to count."),
    filter: str = typer.Option("", help="Payload filter as JSON."),
    project: str = typer.Option("", help="Project filter."),
    include_inactive: bool = typer.Option(False, "--include-inactive"),
) -> None:
    """Count memories matching the (optional) filter."""
    _run_cli(browse.count_memories, collection, filter, project, include_inactive)


@app.command()
def stats(
    collection: str = typer.Option("", help="Collection to summarize."),
    max_scan: int = typer.Option(5000, help="Cap on points scanned."),
) -> None:
    """Per-collection summary: statuses, projects, types, created range, model."""
    _run_cli(browse.collection_stats, collection, max_scan)


@app.command()
def values(
    field: str = typer.Argument(..., help="project | type | tags | source"),
    collection: str = typer.Option("", help="Collection to inspect."),
    filter: str = typer.Option("", help="Payload filter as JSON."),
    limit: int = typer.Option(50, help="Max distinct values."),
    project: str = typer.Option("", help="Project filter."),
) -> None:
    """Discover distinct values of a metadata field (schema in use)."""
    _run_cli(browse.field_values, field, collection, filter, limit, project)


@app.command()
def export(
    collection: str = typer.Option("", help="Collection to export."),
    with_vectors: bool = typer.Option(False, "--with-vectors", help="Include vectors (large)."),
) -> None:
    """Export memories as JSONL to stdout (pipe to a file for backup)."""
    _run_cli(backup.export_memories, collection, with_vectors)


@app.command()
def import_memories(
    path: str = typer.Argument(..., help="JSONL file produced by `export` ('-' = stdin)."),
    collection: str = typer.Option("", help="Target collection (default: configured)."),
    reembed: bool = typer.Option(False, "--reembed", help="Re-embed all texts with the current model."),
    on_conflict: str = typer.Option("skip", help="skip | overwrite existing IDs."),
) -> None:
    """Import a JSONL memory export (full fidelity with vectors)."""
    import pathlib

    if path == "-":
        import sys

        data = sys.stdin.read()
    else:
        data = pathlib.Path(path).read_text(encoding="utf-8")
    _run_cli(backup.import_memories, data, collection, reembed, on_conflict)


@app.command()
def doctor(
    json_output: bool = typer.Option(False, "--json", help="Checks as a JSON array."),
) -> None:
    """Health-check Ollama, Qdrant, and the default collection (read-only)."""
    from .doctor import format_text, has_failures, run_checks

    checks = run_checks()
    if json_output:
        import json as _json

        typer.echo(_json.dumps({"schema": 1, "checks": checks}, ensure_ascii=False))
    else:
        typer.echo(format_text(checks))
    if has_failures(checks):
        raise typer.Exit(1)


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
