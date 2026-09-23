# vector-memory

Persistent **vector memory** for AI agents backed by **Ollama** (embeddings,
tested with `qwen3-embedding:8b`) and **Qdrant** (vector store). Two entry
points over the same core (`vector_memory.core`):

- `vector-memory-mcp` — MCP stdio server (six tools, 1:1 with core ops)
- `vector-memory` — one-shot CLI (typer, 1:1 with the same core ops)

It exposes six operations:

| Tool | Description |
|---|---|
| `save_memory(text, metadata, collection, project="", type="", tags=None)` | Embeds `text` via Ollama and upserts it into Qdrant. `metadata` is optional metadata — a JSON object or a JSON string, both accepted — stored alongside the vector. `project`/`type`/`tags` are convenience kwargs merged into the same metadata keys (explicit values win). Optional `collection` targets a specific collection (created on the fly if missing; empty = server default). |
| `save_memories(texts, metadata, collection, project="", type="", tags=None)` | Batch version: embeds a list of texts in one Ollama call and upserts them as a single batch. `metadata` (object or JSON string) applies to all documents, as do `project`/`type`/`tags`. |
| `search_memory(query, limit, filter, collection)` | Embeds `query` and returns the `limit` most similar stored memories — each hit includes its point **ID**, similarity **score**, **metadata**, and text, so you can `delete_memory`/`update_memory` straight from search output. Optional `filter` is a payload filter — JSON object or JSON string (see below). |
| `update_memory(point_id, text, metadata, collection)` | Re-embeds `text` and overwrites the point in place (same ID). Empty `metadata` keeps the existing payload metadata; a JSON object or JSON string replaces it. Nonexistent IDs return an error. |
| `delete_memory(point_id, collection)` | Deletes the stored memory (point) with the given ID. |
| `list_collections()` | Lists all existing Qdrant collections. |

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

Invalid `metadata`/`filter` JSON is not silently ignored: the tool's return
string includes a warning line (e.g.
`Warning: failed to parse metadata JSON; saved with empty metadata`).

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
vector-memory save-many "text A" "text B" --project p [--metadata '{...}'] [--collection C]
vector-memory search "query" [--limit N] [--filter '{"tags":["x"]}'] [--project p] [--collection C]
vector-memory update <point-id> --text "new text" [--metadata '{...}']
vector-memory delete <point-id>
vector-memory list-collections
```

Semantics worth knowing: point IDs are uuid4 — saving identical text twice
creates two points (dedupe is the caller's job; use `update` to modify in
place). `update --metadata` replaces the whole payload metadata; pass only
`--text` to keep it. Search hits include ID + score + metadata + full text.
`--project`-scoped searches only see memories saved with a `project` metadata
field. Invalid metadata/filter JSON continues with a `Warning:` line — check
for it.

### MCP client config

Add to your client's MCP config (Claude Desktop, Hermes, etc.):

```json
{
  "mcpServers": {
    "vector-memory": {
      "command": "uv",
      "args": [
        "--directory", "/path/to/vector-memory",
        "run", "vector-memory"
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
      args: ["--directory", "/path/to/vector-memory", "run", "vector-memory"]
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
- Dependency pins: `numpy<2`, `qdrant-client<1.15`, `mcp[cli]<2` — chosen for
  compatibility with older x86-64 hardware (pre-x86-64-v2) and the mcp v2
  FastMCP rename. Adjust only with reason.
