# vector-memory

Persistent **vector memory** for AI agents backed by **Ollama** (embeddings,
tested with `qwen3-embedding:8b`) and **Qdrant** (vector store). Two entry
points over the same core (`vector_memory.core`):

- `vector-memory-mcp` — MCP stdio server (1:1 with core ops)
- `vector-memory` — one-shot CLI (typer, 1:1 with the same core ops)

It exposes these operations:

| Tool | Description |
|---|---|
| `save_memory(text, metadata, collection, project="", type="", tags=None, supersedes=None, allow_duplicate=False, on_similar="")` | Embeds `text` via Ollama and upserts it into Qdrant. Identical normalized text is **idempotent** (the same point is refreshed) unless `allow_duplicate`. Near-duplicates above the calibrated threshold (default 0.985) are **reported, never merged** (`similar <id> <score> <text>` lines; `on_similar` = warn|skip|error). `supersedes` marks the listed IDs superseded (hidden from default search). Text matching a high-confidence secret rule is rejected. |
| `save_memories(texts, metadata, collection, ...)` | Batch version: embeds in `EMBED_BATCH`-sized batches (default 32) and upserts together. |
| `search_memory(query, limit, filter, collection, project, include_inactive, since, before, tag, type, source, min_score, recency_weight, mmr, brief, max_chars, output_format, agent)` | Semantic search. Superseded/archived memories are hidden by default (`include_inactive=True` shows them with status annotations). Enhancements: `min_score`, `recency_weight` (0–1 re-rank, 90-day half-life), `mmr` (diversity), `brief`/`max_chars` (prefers `summary`), `--format compact`. Convenience flags compile into the same filter. |
| `update_memory(point_id, text, metadata, collection, merge_metadata=False)` | Re-embeds `text` and overwrites the point in place (same ID). Omit `text` to update metadata without re-embedding; `merge_metadata=True` merges instead of replacing. |
| `patch_metadata(point_id, set, unset, collection)` | Metadata-only patch: merge `set` keys and/or remove `unset` keys. **No embedding call** — the vector is untouched. |
| `delete_memory(point_id, collection)` / `delete_many(ids, collection)` | Hard delete by ID (multi-ID supported; missing IDs reported). |
| `set_status(point_ids, status, collection)` | Archive/unarchive (CLI: `archive`/`unarchive`). |
| `get_memory(ids, collection, with_system)` / `list_memories` / `count` / `collection_stats` / `field_values` | Read/browse operations — no embedding call. `list` paginates with a `next-cursor`. |
| `list_collections()` | Lists all existing Qdrant collections. |
| `doctor` (CLI) | Health checks: Ollama model, Qdrant version/compat, collection, legacy-points report. |
| `export` / `import` (CLI) | JSONL backup/restore (`--with-vectors`, `--reembed`, `--on-conflict skip|overwrite`). |
| `reembed` (CLI) | Copy a collection into a new one with the current model (same IDs; source never mutated; resumable). |
| `migrate` (CLI) | Non-destructive system-field backfill on legacy points (resumable, `--dry-run`). |

Every data tool takes an optional `collection` string; an empty value uses
the server-configured collection (`--collection` / `COLLECTION_NAME`).

### Payload filtering

`search_memory` accepts an optional `filter` — a JSON object or a JSON string
built from payload fields. List values become a `MatchAny` condition (matches if the payload
field contains **any** of the values), scalar values become exact matches.
Multiple conditions are AND-ed together:

```json
{"tags": ["x"]}                      // payload.tags contains "x"
{"source": "doc1"}                   // exact match
{"tags": ["a", "b"], "source": "s"}  // AND of MatchAny + match
```

On startup the server connects to Ollama and Qdrant and creates the collection
automatically if it does not exist (cosine distance, dimension probed from the
embedding model). If a collection **already exists** with a different vector
dimension than the current `EMBED_MODEL` produces — whether the default
collection at startup or an ad-hoc one named in a tool call — the server fails
fast with a clear error instead of silently storing corrupt vectors — fix it
by deleting and recreating the collection, or by switching back to the
original embedding model.

Invalid `metadata`/`filter` JSON is rejected by default: the operation fails
with a single stderr line (`ArgumentError: metadata is not valid JSON (position N)`)
and nothing is written. Metadata keys must be lowercase identifiers
(`^[a-z][a-z0-9_]{0,63}$`, no leading `_`), and values must be JSON scalars or
lists of scalars. The legacy warn-and-continue behavior is available via
`--lenient` on save commands or `VM_LENIENT=1` in the environment.

## Requirements

- Python 3.11+ and [uv](https://docs.astral.sh/uv/)
- A reachable Ollama instance (default `http://192.168.X.X:11434`)
- A reachable Qdrant instance (default `http://192.168.X.X:6333`)

## Installation (CLI)

From the repo directory, install both executables as editable `uv` tools (on
PATH in `~/.local/bin`, edits to the checkout take effect immediately):

```bash
uv tool install -e .
```

This installs `vector-memory` (and the optional `vector-memory-mcp` server
executable). Verify with `vector-memory list-collections`.

## Configuration

Settings resolve in order: **CLI flags > environment variables > defaults**.

| Setting | CLI flag | Env var | Default |
|---|---|---|---|
| Ollama base URL | `--ollama-url` | `OLLAMA_URL` | `http://192.168.X.X:11434` |
| Qdrant base URL | `--qdrant-url` | `QDRANT_URL` | `http://192.168.X.X:6333` |
| Embedding model | `--embed-model` | `EMBED_MODEL` | `qwen3-embedding:8b` |
| Collection name | `--collection` | `COLLECTION_NAME` | `agent_scenarios` |

## Upgrading from mcp-ollama-qdrant

Upgrading from the old `mcp-ollama-qdrant` repo/server: collections and
memories carry over unchanged — the package rename does not touch Qdrant.
Register the new entry points (`vector-memory` CLI, `vector-memory-mcp`
server) in your client config instead of `mcp-ollama-qdrant`; the default
collection `agent_scenarios` is reused as-is.

## Running

With uv (recommended — handles the venv and sync automatically):

```bash
uv sync
uv run vector-memory-mcp        # run the stdio MCP server (or: python mcp_server.py)
uv run vector-memory search "db outage"   # one-shot CLI (no daemon)
uv run vector-memory --help               # save | save-many | search | update | delete | list-collections
```

Every CLI invocation is one-shot — there is no daemon and no CLI-to-server
RPC; the CLI calls the same core functions as the MCP server directly.

Interactive testing / inspection:

```bash
uv run mcp dev mcp_server.py
```

## CLI commands

```bash
vector-memory save "text" --project p --type decision --tags x [--metadata '{...}'] [--collection C]
vector-memory save "text" --supersedes <old-id>          # fact changed: old becomes superseded
vector-memory save-many "text A" "text B" --project p [--metadata '{...}'] [--collection C]
vector-memory search "query" [--limit N] [--filter '{"tags":["x"]}'] [--project p] [--since 7d]
           [--tag urgent --tag arch] [--type decision] [--min-score 0.2] [--recency-weight 0.5]
           [--mmr 0.7] [--brief] [--format compact] [--include-inactive] [--json] [--collection C]
vector-memory update <point-id> --text "new text" [--metadata '{...}'] [--merge-metadata]
vector-memory patch <point-id> --set '{"tags":["x"]}' [--unset key1 --unset key2] [--collection C]
vector-memory archive <point-id>[,<point-id>...]         # hide without deleting
vector-memory get <point-id>[,<point-id>...] [--system]  # full text + metadata
vector-memory list [--project p] [--limit 25] [--order-by created]
vector-memory count / stats / values project             # inventory, no embedding call
vector-memory migrate [--collection C] [--assume-model M] [--dry-run]
vector-memory reembed --collection SRC --to DST [--resume]
vector-memory export [--collection C] [--with-vectors] > backup.jsonl
vector-memory import backup.jsonl [--collection C] [--reembed] [--on-conflict skip|overwrite]
vector-memory delete <point-id>[,<point-id>...]
vector-memory delete-by-filter --filter '{"project":"p"}'    # DRY-RUN by default; --no-dry-run --yes to really delete
vector-memory delete-collection NAME --confirm NAME          # whole collection (guarded)
vector-memory list-collections
vector-memory doctor [--json]
```

Semantics worth knowing: **identical normalized text is idempotent** — saving
it again refreshes the same point (metadata merged) instead of creating a
second one; pass `--allow-duplicate` to force a new point. **Near-duplicates
are reported, never merged** (`similar <id> <score> <text>` lines + a
reminder that similarity does not imply equivalence — check numbers,
versions, negations; `--on-similar warn|skip|error`, threshold configurable
via `VM_DEDUPE_THRESHOLD`, calibrated default 0.985 — see
`docs/similarity-calibration.md`). **Superseded/archived memories are hidden
from default search** (legacy points without `_status` stay visible;
`--include-inactive` shows them annotated). `update --metadata` replaces the
whole payload metadata (`--merge-metadata` merges instead); omit `--text`
for a no-re-embed metadata update. Search hits include ID + score + metadata
+ text; `--brief`/`--format compact` truncate (preferring `summary`).
Invalid metadata/filter JSON fails the command (one stderr
`ArgumentError: ...` line, nothing written); `--lenient` / `VM_LENIENT=1`
restores the old warn-and-continue behavior. Text matching a high-confidence
secret rule is rejected (`SensitiveContentError`; the value is never echoed;
`--allow-sensitive` overrides — use only for false positives).

### System payload fields

Every save stamps `_`-prefixed system fields alongside user metadata:
`_created_ts`/`_updated_ts` (epoch floats), `_created_at`/`_updated_at`
(ISO-8601), `_content_hash` (sha256 of the normalized text), `_embed_model`,
and `_agent` (when `VM_AGENT_ID` is set). Lifecycle fields `_status`,
`_supersedes`, `_superseded_by` are reserved for the supersede/archive
lifecycle. Legacy points without these fields stay valid; use
`vector-memory migrate` to backfill them non-destructively (resumable;
`--dry-run` reports how many points would change; `--assume-model` records
the embedding model only if you are certain of what produced the vectors).
Payload indexes (keyword on `project`/`type`/`tags`/`source`/`_status`, float
on timestamps) are created idempotently when a collection is opened.

### MCP client config

Add to your client's MCP config (Claude Desktop, Hermes, etc.). The stdio
server entry point is **`vector-memory-mcp`** (the `vector-memory` command is
the CLI, not the server):

```json
{
  "mcpServers": {
    "vector-memory": {
      "command": "uv",
      "args": [
        "--directory", "/path/to/vector-memory",
        "run", "vector-memory-mcp"
      ],
      "env": {
        "OLLAMA_URL": "http://192.168.X.X:11434",
        "QDRANT_URL": "http://192.168.X.X:6333",
        "EMBED_MODEL": "qwen3-embedding:8b",
        "COLLECTION_NAME": "agent_scenarios"
      }
    }
  }
}
```

(Env entries are optional if the defaults already point at your instances.)

For Hermes `~/.hermes/config.yaml`:

```yaml
mcp:
  servers:
    vector-memory:
      command: uv
      args: ["--directory", "/path/to/vector-memory", "run", "vector-memory-mcp"]
```

## Testing

Offline unit tests (Ollama and Qdrant are mocked — no live services needed):

```bash
uv run pytest tests/
```

End-to-end smoke test against live Ollama + Qdrant (saves a few memories,
searches for them, prints similarity scores):

```bash
uv sync
uv run python scripts/live_smoke.py
```

## Notes

- All diagnostics are logged to **stderr**; stdout is reserved for the stdio
  MCP transport.
- Dependency pins: `qdrant-client>=1.15,<2`, `mcp[cli]<2`, numpy 1.x on
  Python < 3.13 — chosen for compatibility with older x86-64 hardware
  (pre-x86-64-v2) and the mcp v2 FastMCP rename. Adjust only with reason.
