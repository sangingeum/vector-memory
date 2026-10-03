# Changelog

All notable changes to vector-memory are documented here.
Format follows Keep a Changelog; versions are informal until 1.0.

## Unreleased — improvement round (2026-10-03)

### Added
- **Error taxonomy** (`vector_memory.errors`): `ArgumentError`, `ConfigError`,
  `NotFoundError`, `ConflictError`, `SensitiveContentError`, `BackendError`,
  `InternalError`. CLI failures print exactly one `ErrorType: description`
  stderr line and exit 1; MCP tools surface the same text via `isError`.
- **Strict input validation (default)**: invalid `metadata`/`filter` JSON now
  fails the operation (`ArgumentError: metadata is not valid JSON (position N)`)
  and **nothing is written** — previously the point was saved with empty
  metadata after a warning. Metadata keys must match
  `^[a-z][a-z0-9_]{0,63}$` (no leading `_`); values must be JSON scalars or
  lists of scalars. `text` must be non-empty and ≤ 20000 chars.
  Legacy behavior stays reachable via `--lenient` / `VM_LENIENT=1`.
- **System payload fields** (`_`-prefixed, never settable by callers):
  `_created_ts/_updated_ts` (epoch float, range-filterable),
  `_created_at/_updated_at` (ISO-8601), `_content_hash` (sha256 of normalized
  text), `_embed_model`, `_status` (`active|superseded|archived`; missing =
  active), `_supersedes`, `_superseded_by`, `_agent` (from `VM_AGENT_ID`),
  `_created_estimated`. Legacy points without them stay valid.
- **Payload indexes** created idempotently when a collection is opened:
  keyword on `project`/`type`/`tags`/`source`/`_status`, float on
  `_created_ts`/`_updated_ts`.
- **`vector-memory migrate [--collection C] [--assume-model M] [--dry-run]`**:
  non-destructive, resumable backfill of system fields on legacy points.
  Never touches vectors or text; re-running is a no-op. `_embed_model` is only
  recorded when `--assume-model` is passed explicitly.
- **`vector-memory patch <id> --set '{...}' [--unset key...]`** (MCP:
  `patch_metadata`): metadata-only patch using set/delete payload — no
  embedding call, vector untouched, bumps `_updated_*`.
- **`update --merge-metadata`**: merge given keys into the existing payload
  instead of replacing it. Default replace semantics unchanged (and replace
  now genuinely drops removed keys via a vector-preserving upsert).
- **Test scaffolding**: local-mode Qdrant (`QdrantClient(":memory:")`) +
  deterministic hash embedder fixtures; pytest `live` marker with
  warn-clean bar (`filterwarnings = error`); Step 0 audit
  (`docs/audit.md`), similarity calibration (`docs/similarity-calibration.md`),
  and a production-collection fixture (`docs/fixtures/agent_scenarios.jsonl`).

### Changed (intentional semantic changes)
- **Strict validation replaces warn-and-continue.** Agents that ignored
  warnings lost their metadata silently; the failure is now loud and early.
- **`save_memories([])`** now raises `ArgumentError: texts list is empty`
  instead of returning a "No texts to save." note.
- **Errors are typed.** Update/delete on missing points/collections raise
  `NotFoundError` (rendered `NotFoundError: ...`, exit 1) instead of
  free-form "Update failed:" strings.
- **All writes use `wait=True`**, so a search immediately after a save sees
  the write.

### Not yet landed (planned, see docs/audit.md §8)
- Exact-duplicate idempotent saves (`--allow-duplicate` escape hatch).
- Supersede/archive lifecycle and hiding superseded memories from search.
- Embedding-model fingerprint enforcement and `reembed`.

## Earlier
- CLI convenience options `--project/--type/--tags` on save/save-many.
- Library HTTP log noise demoted; results-only stdout.
