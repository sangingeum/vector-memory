---
name: vector-memory
description: "Use for shell memory ops: vector-memory CLI, one-shot."
version: 1.0.0
---

# vector-memory (CLI)

One-shot typer CLI over `vector_memory.core`. Repo:
`/home/keum/dev/athena/vector-memory/`. Binary: `vector-memory`; package
`vector_memory`. CLI-only — no MCP variant exists. No daemon, no server RPC.

## Commands

```bash
cd /home/keum/dev/athena/vector-memory && uv run vector-memory <cmd>   # or installed console script
vector-memory save "text" --metadata '{"project":"p","type":"decision","tags":["x"]}' [--collection C]
vector-memory save-many "text A" "text B" --metadata '{"tags":["ops"]}'
vector-memory search "db outage" [--limit 3] [--filter '{"tags":["x"]}'] [--project p] [--collection C]
vector-memory update <point-id> --text "new text"        # re-embeds; --metadata replaces payload wholesale
vector-memory delete <point-id>
vector-memory list-collections
```

## Key rules

- Search hits include ID + score + metadata + full text — you can
  update/delete straight from search output.
- Always attach `project` (and `type`) metadata on save; a `--project`-scoped
  search will not see memories saved without a `project` field.
- `update --metadata` replaces the whole payload metadata (re-supply full set
  with refreshed `updated_at`); pass only `--text` to keep metadata.
- Env: `OLLAMA_URL`, `QDRANT_URL`, `EMBED_MODEL`, `COLLECTION_NAME`
  (default `agent_scenarios`).
- Each invocation pays one embedding round trip (~0.8 s GPU / ~6 s CPU,
  8B model); `save-many` amortizes it.
- Invalid metadata/filter JSON: operation continues with empty metadata /
  unfiltered search and a `Warning:` line in stdout — check for it.
- Not for: codebase indexing (use `code-indexer`), secrets, exact-match lookup.
