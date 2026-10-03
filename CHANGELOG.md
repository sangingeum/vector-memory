# Changelog

All notable changes to vector-memory are documented here.
Format follows Keep a Changelog; versions are informal until 1.0.

## Unreleased — fix round (2026-10-04)

Fixes from the two-axis review (`docs/review-improvement-round-2026-10-04.md`,
PASS-WITH-NOTES). B-1 (production fixture on GitHub) was subsequently
REMEDIATED per owner ruling: the fixture was purged from all history
(`git filter-repo`), `docs/fixtures/` is gitignored and local-only, and a
fresh fixture was regenerated locally from the production collection for
migration-verification use.

### Fixed
- **B-2 MCP isError**: `call()` now propagates typed errors — FastMCP marks
  a tool result as an error only when the handler raises; previously every
  failure reached MCP clients as a successful result. Test added.
- **B-3 legacy-duplicate catch**: before the uuid5 idempotent path, points
  matching `_content_hash` are looked up via scroll filter and refreshed —
  migrated legacy points no longer duplicate on re-save.
- **B-5 patch validation**: `_`-prefixed system keys and `text` are rejected
  in BOTH `patch --set` and `--unset` (no more system-field deletion or
  text/vector desync).
- **B-6 partial-failure contract**: implemented per VM-11 spec — a failed
  embed batch marks its indices and the loop CONTINUES; good points land;
  `BackendError: saved N/M; failed: [4, 11]` is raised after the upsert.
  Mocked-failure tests cover the spec message format.
- **B-10** non-string metadata raises `ArgumentError` (was silent `{}`).
- **B-11** model-mismatch check fetches ONE point via a scroll filter on
  `_embed_model` existing — legacy points can no longer mask a recorded
  model behind them.
- **B-13** `list_collections` raises `BackendError` (exit 1) instead of
  returning failure text on stdout.
- **B-14** `KeyError` paths in `migrate`/`delete-collection` raise
  `NotFoundError` per taxonomy.
- **B-17** unarchive of a formerly superseded point clears the stale
  `_superseded_by` link (documented in README).

### Added
- **VM-14 long-text chunking**: `save --chunk [--chunk-chars --overlap]`
  splits oversized text on sentence boundaries (Hangul-aware `다.`/`요.`/
  `?`/`!`/newlines) into points sharing `_group_id` with `_chunk_index`;
  `search --collapse-groups` returns only the best chunk per group. MCP:
  `save_chunked`.
- **VM-15 consolidate**: read-only greedy cosine clustering of active points
  producing candidate groups with a `--supersedes` suggestion; never writes.
- **B-4** `--lenient` and `--summary` are real `save` flags (docs promised
  them; env-only/absent before).
- **B-8** GitHub Actions workflow: `uv sync --frozen`, ruff, `pytest -m "not live"`.
- **B-12** `--json` on every remaining command; usage errors exit 2.
- **B-16** convenience flags (`--tag/--type/--source/--since/--before`) wired
  into `list`/`count`/`delete-by-filter`; MCP `search_memory` now exposes all
  landed search enhancements (full CLI parity).

### Documented
- **B-7** README documents the MCP surface as a deliberate CLI subset (which
  tools exist and why; administrative/destructive commands are CLI-only).
- **B-15** README Qdrant snapshot section (full-fidelity backups vs. JSONL).
- **B-9** this report claims only what is in the tree (verified by tests).

## Unreleased — improvement round (2026-10-03/04)

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
  Never touches vectors or text; re-running is a no-op.
- **Supersede/archive lifecycle**: `save --supersedes ID[,ID...]` (MCP:
  `save_superseding`), `archive`/`unarchive` (MCP tools of the same name).
  **Default search hides superseded/archived memories** (must_not form —
  legacy points without `_status` stay visible); `--include-inactive` returns
  them annotated with `status=superseded -> <new id>`. Unknown supersede
  targets fail before anything is written.
- **Embedding-model fingerprint**: every write stores `_embed_model`;
  opening/writing/searching a collection built with a different model fails
  with `ConfigError` (`VM_FORCE_MODEL_MISMATCH=1` overrides, documented
  unsafe). `vector-memory reembed --collection SRC --to DST [--resume]`
  re-embeds into a NEW collection with the same IDs/payloads; SRC is never
  mutated.
- **Duplicate handling**: identical normalized text is **idempotent** — the
  point ID for new writes is `uuid5(collection|content_hash)`, so re-saving
  the same text refreshes the existing point (metadata merged) and the
  result notes `duplicate of existing; updated`. `--allow-duplicate` forces
  a new point. Near-duplicates above the **calibrated threshold (0.985)** —
  `VM_DEDUPE_THRESHOLD` — are REPORTED, never merged:
  `similar <id> <score> <text>` lines plus a reminder that similarity does
  not imply equivalence (contradictions score in the same band as
  duplicates — see `docs/similarity-calibration.md`). `--on-similar
  warn|skip|error` (default warn; skip returns the existing id without
  writing; error raises `ConflictError`).
- **Sensitive-content guard**: high-confidence secret rules (AWS keys,
  private-key blocks, GitHub/Slack tokens, JWTs, password/api-key
  assignments) block save/update with `SensitiveContentError` — the rule is
  named, the matched VALUE is never echoed, nothing is written.
  `--allow-sensitive` / `VM_ALLOW_SENSITIVE=1` overrides.
- **Output contract**: CLI stdout = results only; stderr empty on success;
  failures = exactly one `ErrorType:` line, exit 1. `--json` on the main
  commands emits a `{"schema": 1, "ok": ...}` envelope (with
  `error`/`error_type` on failure). Search results are deterministically
  ordered (score desc, then id).
- **`vector-memory doctor [--json]`**: read-only health checks — Ollama
  reachable + model present, Qdrant reachable + version/client compat,
  default collection exists, legacy-points report (suggests `migrate`).
- **Filter DSL v2** (documented in `docs/filter-dsl.md`, validated): range
  operators (`gt/gte/lt/lte`), `$not` negation, `$or` disjunction, and
  convenience flags `--since 7d|24h|date`/`--before`/`--tag`/`--type`/
  `--source`/`--agent` that compile into the same filter object. Property
  tests compare the compiled-Qdrant result against a pure-Python reference
  evaluator.
- **Read/browse operations** (CLI + MCP): `get` (full text + metadata,
  `--system`), `list` (deterministic newest-first order with `next-cursor`
  pagination), `count`, `stats` (statuses/projects/types/created-range/
  model), `values project|type|tags|source` (schema discovery). None of them
  calls the embedder.
- **Safe bulk deletion**: `delete-by-filter` is **dry-run by default**
  (prints `would delete N points` + samples); real deletion requires
  `--no-dry-run --yes`; an empty filter is refused;
  `delete-collection NAME --confirm NAME` guards the default collection;
  `delete` accepts multiple IDs.
- **Search enhancements** (all optional, defaults unchanged): `--min-score`,
  `--recency-weight W` (application-code re-rank, `RECENCY_HALF_LIFE_DAYS`
  90-day half-life), `--mmr L` (maximal marginal relevance over vectors,
  duplicate pairs diversified), `--brief`/`--max-chars N` (per-hit
  truncation preferring the `summary` metadata, `[truncated; use: get <id>]`),
  `--format compact|json`.
- **Retry, batching, write consistency**: Ollama calls get a timeout
  (`OLLAMA_TIMEOUT`, default 120 s) and transient failures retry with
  exponential backoff (max 3); a missing model fails fast with
  `model 'X' not found on Ollama; run: ollama pull X`; `save_memories` splits
  into `EMBED_BATCH` (32) batches and reports partial failures
  (`saved 18/20; failed: [4, 11]`); all writes use `wait=True`.
- **Caller-provided summaries**: `summary` (≤ 200 chars) is a normal optional
  metadata key passed through verbatim and preferred by `--brief`. The tool
  never calls an LLM.
- **Multi-agent namespacing**: `VM_AGENT_ID` stamps `_agent` on writes;
  `--agent A` filters search by author.
- **Backup, export and import**: `export > memories.jsonl` (id/text/
  metadata/system fields, `--with-vectors`), `import memories.jsonl
  [--reembed] [--on-conflict skip|overwrite]`.
- **Test scaffolding**: local-mode Qdrant (`QdrantClient(":memory:")`) +
  deterministic hash embedder fixtures; pytest `live` marker with
  warn-clean bar; Step 0 audit, similarity calibration, production
  fixture (`docs/fixtures/agent_scenarios.jsonl`, since made local-only/gitignored); MCP stdio smoke test.

### Changed (intentional semantic changes)
- **Strict validation replaces warn-and-continue.** Agents that ignored
  warnings lost their metadata silently; the failure is now loud and early.
- **Exact-duplicate idempotency.** Identical text no longer creates a second
  point — the same point is refreshed (was: uuid4 per save).
- **Superseded/archived memories are hidden from default search** (was: all
  points always returned).
- **`save_memories([])`** raises `ArgumentError: texts list is empty`
  instead of returning a "No texts to save." note.
- **Errors are typed.** Update/delete on missing points/collections raise
  `NotFoundError` (rendered `NotFoundError: ...`, exit 1) instead of
  free-form "Update failed:" strings.
- **All writes use `wait=True`**, so a search immediately after a save sees
  the write.
- **MCP entry point fixed in docs**: the stdio server is
  `vector-memory-mcp` (README previously showed `vector-memory`, the CLI).

## Earlier
- CLI convenience options `--project/--type/--tags` on save/save-many.
- Library HTTP log noise demoted; results-only stdout.
