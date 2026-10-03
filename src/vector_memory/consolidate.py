"""Consolidate: read-only candidate finder (no LLM, never writes)."""

from __future__ import annotations

import time
from typing import Any

from . import payload as _payload
from . import store as _store
from .browse import _resolve
from .errors import ArgumentError
from .filters import merge_convenience


def consolidate(
    collection: str = "",
    project: str = "",
    older_than: str = "",
    min_similarity: float = 0.99,
    limit_groups: int = 20,
    max_scan: int = 5000,
) -> str:
    """Greedy cosine clustering of ACTIVE points; outputs candidate groups.

    Purely read-only — the calling agent decides and writes any merged memory
    (e.g. with save --supersedes). High similarity does NOT imply equivalence
    (numbers, versions, negations) — the output says so.
    """
    name = _resolve(collection)
    if not 0.0 < min_similarity <= 1.0:
        raise ArgumentError("min-similarity must be in (0, 1]")
    cutoff: float | None = None
    if older_than:
        from .filters import parse_relative_time

        cutoff = parse_relative_time(older_than)
    flt = merge_convenience(None, project=project)
    flt = _payload.active_filter(flt)
    points: list[Any] = []
    offset = None
    while True:
        batch, offset = _store.qdrant.scroll(
            collection_name=name, scroll_filter=flt, limit=256, offset=offset,
            with_payload=True, with_vectors=True,
        )
        if not batch:
            break
        points.extend(batch)
        if offset is None or len(points) >= max_scan:
            break
    points = points[:max_scan]

    # Only points with vectors AND usable timestamps (if older_than asked).
    candidates = []
    for p in points:
        vector = getattr(p, "vector", None)
        if isinstance(vector, dict):
            vector = next(iter(vector.values()), None)
        if not vector:
            continue
        if cutoff is not None:
            created = (p.payload or {}).get(_payload.CREATED_TS)
            if not isinstance(created, (int, float)) or created > cutoff:
                continue
        candidates.append(p)
    if not candidates:
        return f"No candidate memories to consolidate in {name!r}."

    def _cos(a: list, b: list) -> float:
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        na = sum(x * x for x in a) ** 0.5 or 1.0
        nb = sum(y * y for y in b) ** 0.5 or 1.0
        return dot / (na * nb)

    # Greedy clustering: seed with the lowest-index unassigned point, pull in
    # all points above min_similarity, repeat.
    assigned: set[str] = set()
    groups: list[tuple[list[Any], list[float]]] = []
    for i, seed in enumerate(candidates):
        if str(seed.id) in assigned:
            continue
        seed_vec = _vec_of(seed)
        members = [seed]
        scores = []
        for other in candidates[i + 1:]:
            if str(other.id) in assigned:
                continue
            score = _cos(seed_vec, _vec_of(other))
            if score >= min_similarity:
                members.append(other)
                scores.append(score)
                assigned.add(str(other.id))
        assigned.add(str(seed.id))
        if len(members) >= 2:
            groups.append((members, scores))
        if len(groups) >= limit_groups:
            break

    if not groups:
        return (
            f"No consolidation candidates at similarity >= {min_similarity} "
            f"among {len(candidates)} points in {name!r}."
        )
    header = (f"Consolidation candidates in {name!r} "
              f"(similarity >= {min_similarity}, read-only):")
    out = [header]
    for number, (members, scores) in enumerate(groups, start=1):
        low = min(scores) if scores else 1.0
        high = max(scores) if scores else 1.0
        out.append(f"group {number} ({len(members)} memories, similarity {low:.2f}–{high:.2f})")
        ids = []
        for member in members:
            payload = member.payload or {}
            created = payload.get(_payload.CREATED_AT, "?")
            mtype = payload.get("type", "-")
            text = str(payload.get("text", ""))
            brief = (text[:80] + "…") if len(text) > 80 else text
            out.append(f"  {member.id} {created} [{mtype}] {brief!r}")
            ids.append(str(member.id))
        out.append(f"suggest: save merged memory with --supersedes {','.join(ids)}")
    out.append(
        "note: high similarity does NOT imply equivalence — verify "
        "numbers, versions, and negations before merging."
    )
    if len(points) >= max_scan:
        out.append(f"(scan capped at {max_scan} — pass --max-scan to raise)")
    return "\n".join(out)


def _vec_of(point: Any) -> list:
    vector = getattr(point, "vector", None)
    if isinstance(vector, dict):
        vector = next(iter(vector.values()), None)
    return vector


def _now_note() -> str:
    return f"(checked {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})"
