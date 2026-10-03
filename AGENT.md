# AGENT.md — vector-memory

Short brief for dev agents; avoid re-exploring the repo each session.

## What this is
Persistent vector memory for AI agents: Ollama embeddings
(`qwen3-embedding:8b`, 4096-dim) + Qdrant (cosine). Two thin entry points over
`vector_memory.core`: MCP stdio server `vector-memory-mcp`
(`src/vector_memory/server.py`) and one-shot CLI `vector-memory`
(`src/vector_memory/cli.py`). Every adapter function maps 1:1 to one core op.

## Commands
```bash
uv sync                       # deps (uv only; global pip is PEP 668-blocked)
uv run pytest -m "not live"   # offline suite (Qdrant mocked / local-mode)
uvx ruff check src tests      # lint (config: .ruff.toml)
uv run python scripts/live_smoke.py   # real-backend smoke (writes!)
```

## Layout
- `src/vector_memory/core.py` — the operations (single source of truth)
- `src/vector_memory/store.py` — Qdrant client, collection lifecycle, filter/metadata parsing
- `src/vector_memory/embedding.py` — Ollama client (embed / embed_many, retry/batching)
- `src/vector_memory/payload.py` — system-field conventions + payload indexes
- `src/vector_memory/validation.py` — strict input validation (VM-lenient escape)
- `src/vector_memory/errors.py` — error taxonomy (one `ErrorType: ...` line)
- `src/vector_memory/filters.py` — filter DSL v2 (ranges, $not/$or, convenience flags)
- `src/vector_memory/dedupe.py` — exact-idempotent ids + near-duplicate detection (0.985)
- `src/vector_memory/sensitive.py` — high-confidence secret guard
- `src/vector_memory/browse.py` — get/list/count/stats/values read ops
- `src/vector_memory/bulk.py` — delete-by-filter (dry-run default), delete-collection
- `src/vector_memory/migrate.py` — non-destructive system-field backfill
- `src/vector_memory/reembed.py` — model fingerprint check + reembed
- `src/vector_memory/backup.py` — export/import JSONL
- `src/vector_memory/doctor.py` — read-only health checks
- `tests/conftest.py` — fake clients (unit); `tests/local_backend.py` — local-mode Qdrant fixture (integration-style)

## Conventions & invariants
- CLI and MCP stay 1:1 over core; new params must be optional.
- Metadata: flat JSON object, keys `^[a-z][a-z0-9_]{0,63}$`, `_` prefix
  reserved for system fields. Strict validation by default; `--lenient` escape.
- Exact-duplicate saves are IDEMPOTENT (uuid5 of collection|content_hash);
  near-duplicates are reported at the calibrated 0.985 threshold, never merged.
- Active-memory filters use `must_not _status in [superseded, archived]` so
  legacy points (no `_status`) still match; superseded/archived are hidden
  from default search.
- stdout = results; stderr = diagnostics; failures = one `ErrorType:` line, exit 1.
- Tests: warn-clean (`filterwarnings=error`), live tests marked `@pytest.mark.live`.
- Production Qdrant holds real memories: never run destructive commands
  against `agent_scenarios` from tests or ad-hoc probes; use local mode.
- Docs live in `docs/` (audit, similarity calibration, migration fixture).

## Branch
`master` (tracks `origin/master`).
