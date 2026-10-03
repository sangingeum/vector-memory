# vector-memory — Step 0 Audit (2026-10-03)

Read-only audit of the live backends before implementing the improvement plan.
No writes were made against any production collection; probe writes (read-your-
write, calibration) used a scratch collection that was deleted afterwards.

## 1. Payload layout (`agent_scenarios`)

- Metadata keys are stored **at the top level of the Qdrant payload**, next to
  `text`: sample keys are `['project', 'tags', 'text', 'type']`. The README
  filter examples are correct.
- **No system fields exist today** — zero points carry `_`-prefixed keys.
  There are no timestamps, no hashes, no `_status`.
- 345 points total; **15 points have no metadata at all** (only `text`);
  2 of them belong to the `solomon` collection (3 points, all bare `text`).
- Collection config: dim 4096, distance COSINE, single unnamed vector.

## 2. Entry points

- `pyproject.toml` declares two console scripts: `vector-memory` →
  `vector_memory.cli:main` and `vector-memory-mcp` → `vector_memory.server:main`.
  The MCP stdio server is `vector-memory-mcp` (root-level `mcp_server.py` is a
  legacy shim — checked: it re-exports the server entry). README's MCP config
  launching `vector-memory` is **wrong** and will be fixed under the
  entry-point work item.

## 3. Existing data

- `agent_scenarios` (production, 345 points): distinct `project` values:
  calc-engine, code-indexer, t, tooling, vector-memory, vivarium,
  vivarium-adr, vivarium-sim. `type` is free-form and highly varied (78+
  distinct values: decision, gate-verdict, run-outcome, ...). `tags` are
  free-form strings, sometimes comma-joined, sometimes raw JSON-looking
  strings (e.g. `'["affiliation","wiring",...]'`) — agents have been putting
  lists into a single tag string. This informs the recommended-schema docs
  (type as soft enum, tags validation) but legacy values must stay readable.
- Other collections on the same Qdrant: `idx_*` (code-indexer's own, 1000s of
  points), `vivarium` (3), `solomon` (2). Migration must be
  per-collection opt-in; default collection `agent_scenarios`.
- Fixture: `docs/fixtures/agent_scenarios.jsonl` (345 records: id + payload,
  vectors excluded — no sensitive vectors, payload text only).

## 4. Backend capabilities

- qdrant-client 1.19.1 (pinned `>=1.15,<2` — fine), Qdrant server 1.19.1.
  Supports `create_payload_index`, `scroll(order_by=...)`, facet, local mode.
- Ollama host: `192.168.1.103:11434` (per `embedding.py` default; NOT
  192.168.1.105 — that is Qdrant). Model `qwen3-embedding:8b` present,
  dimension 4096.
- Read-your-write verified: search immediately after upsert (without
  `wait=True`) sees the point — but `wait=True` is still adopted to make the
  guarantee explicit.

## 5. Behavior probes

- Invalid metadata JSON currently produces only a warning and saves the point
  with empty metadata (confirmed by tests / code path `parse_metadata`).
- `update` without text works (metadata-only path, no re-embed).
- Very long text: no length cap today — the 20000-char limit will be new.
- `save_memories`: single embed call (batch); no partial-failure reporting.

## 6. Similarity calibration (production model `qwen3-embedding:8b`)

Full table in `docs/similarity-calibration.md`. Headlines:

| pair | cosine |
|---|---|
| identical | 0.9997 |
| paraphrase (EN) | 0.9402 |
| **one-token contradiction (port)** | **0.9842** |
| **date contradiction** | **0.9959** |
| version contradiction | 0.9810 |
| unrelated | 0.3616 |
| Korean paraphrase | 0.9067 |

**Consequence for the dedupe threshold (VM-03):** one-token contradictions
score *higher* than paraphrases (0.98–0.996 vs 0.94) and Korean paraphrases
fall at 0.907. A threshold high enough to avoid flooding the agent with
warnings for genuinely-different memories (≥0.98) would still flag
contradictions (they score 0.98+). The plan's 0.92 default is too low —
Korean paraphrases and ordinary topic-adjacent memories would trigger it.
**Adopted default: 0.985**, configurable via `VM_DEDUPE_THRESHOLD`, with
documentation that crossing the threshold does NOT imply equivalence —
contradictions score in the same band as true duplicates.

## 7. Test suite baseline

- `uv run pytest -m "not live"`: 41 passed in 0.09 s (2026-10-03), warn-clean.
- Gaps: no integration tests over a real (local-mode) Qdrant; no filter
  semantics tests; no MCP stdio smoke test; no property tests. Addressed by
  the test-scaffolding work item.

## 8. Plan critiques (flagged before implementation)

1. **VM-03 threshold 0.92 is wrong for this model** — see §6; adopting 0.985
   from calibration data instead (plan says the threshold comes from
   calibration, so this follows the plan's own rule).
2. **VM-02 `_created_estimated: true` naming** — the plan §4 does not list
   this key; I will write `_created_estimated` as a boolean system field and
   add it to the documented system-field table (harmless, keeps provenance).
3. **VM-06 `reembed` never mutates SRC** is good, but `reembed --to DST`
   with the same configured model is a no-op worth rejecting loudly
   (ConfigError) to avoid pointless re-embedding — added as a guard.
4. **VM-10 `--recency-weight` fetch `limit*3`** — with `limit` small this
   under-samples; using `max(limit*3, 25)` floor.
5. **VM-01 `type` soft enum** — production has 78+ distinct types, so the
   known list in §4 must be *advisory only* (warning never an error), as the
   plan states. No enforcement beyond syntax, confirmed safe.
6. **Tags observed corruption**: legacy points contain stringified-JSON tags.
   Validation on *new* writes rejects non-scalar tag elements; legacy points
   are left untouched.
