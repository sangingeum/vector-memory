# Similarity calibration — `qwen3-embedding:8b` (4096-dim, cosine)

Measured 2026-10-03 against the production embedding model with representative
memory-shaped texts (including Korean). Raw probe: scratch collection on the
production Qdrant, deleted afterwards.

## Pairs

| label | text A | text B | cosine |
|---|---|---|---|
| identical | Postgres runs on port 5432 for the staging DB | (same) | 0.9997 |
| paraphrase (EN) | Postgres runs on port 5432 for the staging DB | The staging database runs PostgreSQL, listening on port 5432 | 0.9402 |
| one-token contradiction (port) | Postgres runs on port 5432 ... | Postgres runs on port 5433 ... | 0.9842 |
| date contradiction | The migration was completed on 2026-09-01 | ... 2026-10-01 | 0.9959 |
| version contradiction | We standardized on pytest 8.2 ... | ... pytest 8.4 ... | 0.9810 |
| unrelated | Postgres ... | Owner prefers walking meetings on Fridays | 0.3616 |
| Korean paraphrase | 배포는 매주 화요일에 진행한다 | 매주 화요일에 배포 작업을 수행한다 | 0.9067 |

## Findings

1. **Cosine cannot distinguish near-duplicates from one-token
   contradictions.** Contradicting facts (ports, versions, dates) score
   0.98–0.996 — *higher* than paraphrases. Any "similar memory" candidate list
   must be presented to the calling agent as candidates only, never
   auto-merged. This is why the tool only reports and never merges.
2. **Korean paraphrases sit lower (~0.91)** than English ones. A threshold of
   0.92 would under-report Korean near-duplicates but, worse, a threshold of
   0.92 catches plenty of *different* memories.

## Adopted thresholds

- **Near-duplicate warning threshold (default): 0.985**
  (`VM_DEDUPE_THRESHOLD` env / flag). At 0.985, exact/near-exact re-saves are
  caught; contradictions (0.98–0.996) are partially caught — which is
  desirable: those are exactly the pairs where the agent should double-check
  which fact is current. Below 0.985 the band is dominated by ordinary
  topic-adjacent memories (paraphrase at 0.94, Korean at 0.907), which would
  flood `warn` output.
- **`consolidate` clustering similarity floor: 0.99** — tighter, since the
  agent may act on groups and merge them.
- Always printed alongside candidates: the reminder that similarity does not
  imply equivalence (numbers, versions, negations).
