# Filter DSL

Filters are JSON objects accepted by `search` (`search_memory`), and by any
future filter-taking command. Unknown operators are rejected with
`ArgumentError` (strict mode).

## Base semantics (existing)

- A **scalar** value is an exact match: `{"source": "doc1"}`.
- A **list** value is any-of (`MatchAny`): `{"tags": ["x", "y"]}` matches a
  point whose `tags` contains `"x"` **or** `"y"`.
- Multiple keys are AND-ed: `{"tags": ["a"], "source": "s"}` requires both.

## Range operators (planned, not yet landed)

```json
{"_created_ts": {"gte": 1730000000, "lt": 1740000000}}
{"importance": {"gte": 4}}
```
Operators: `gt`, `gte`, `lt`, `lte`. The field must be numeric.

## Logical composition (planned, not yet landed)

```json
{"$not": {"type": "scratch"}}
{"$or": [{"project": "a"}, {"tags": ["urgent"]}]}
```

## Convenience flags (planned, not yet landed)

`--project`, `--type`, `--tag` (repeatable any-of), `--since 7d|24h|2026-09-01`,
`--before`, `--source` compile into the same filter object; a key conflict
with `--filter` raises `ArgumentError`.

## Status filtering (landed)

Internally, "active memories" is expressed as a `must_not` condition —
`_status in [superseded, archived]` — so legacy points without a `_status`
field remain visible. This is the documented filter form for lifecycle work;
see `docs/audit.md` and `src/vector_memory/payload.py::active_filter`.

## Validation

Strict mode (default): invalid JSON, unknown operators, range operators on
non-numeric values, and reserved `_`-prefixed user keys all raise
`ArgumentError` naming the offending key. Lenient mode (`VM_LENIENT=1`)
restores the old "ignore the filter with a Warning" behavior.
