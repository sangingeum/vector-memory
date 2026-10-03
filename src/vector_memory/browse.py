"""Read and browse operations (get/list/count/stats/values)."""

from __future__ import annotations

import json
import time
from typing import Any

from . import payload as _payload
from . import store as _store
from .errors import ArgumentError, NotFoundError
from .filters import build_filter, merge_convenience


def _resolve(name: str) -> str:
    collection = name if name and name.strip() else _store.COLLECTION_NAME
    if not _store.qdrant.collection_exists(collection):
        raise NotFoundError(f"collection {collection!r} does not exist")
    return collection


def _scroll_all(collection: str, flt=None, max_points: int = 10000):
    points, offset = [], None
    while True:
        batch, offset = _store.qdrant.scroll(
            collection_name=collection, limit=256, offset=offset,
            scroll_filter=flt, with_payload=True, with_vectors=False,
        )
        if not batch:
            break
        points.extend(batch)
        if offset is None or len(points) >= max_points:
            break
    return points[:max_points]


def get_memory(point_ids: list[str], collection: str = "", with_system: bool = False) -> str:
    """Return text + metadata (+ system fields with with_system); no embedding call."""
    name = _resolve(collection)
    records = _store.qdrant.retrieve(collection_name=name, ids=point_ids, with_payload=True)
    if not records:
        raise NotFoundError(f"point(s) not found: {', '.join(point_ids)}")
    found = {str(r.id) for r in records}
    missing = [pid for pid in point_ids if str(pid) not in found]
    out = [f"Retrieved {len(records)} point(s) from {name!r}:"]
    for r in records:
        payload = dict(r.payload or {})
        text = payload.pop("text", "")
        shown = payload if with_system else {
            k: v for k, v in payload.items() if not k.startswith("_")
        }
        meta_str = json.dumps(shown, ensure_ascii=False) if shown else "{}"
        out.append(f"- [ID: {r.id}] metadata: {meta_str}")
        out.append(f"  text: {text}")
    if missing:
        out.append(f"missing: {', '.join(missing)}")
    return "\n".join(out)


def list_memories(collection: str = "", filter: str | dict[str, Any] = "",
                  limit: int = 25, order_by: str = "",
                  project: str = "", include_inactive: bool = False,
                  cursor: str = "", tag: list[str] | None = None,
                  type: str = "", source: str = "",
                  since: str = "", before: str = "") -> str:
    """Browse memories via scroll; deterministic order (created then id).

    ``cursor`` is the numeric offset from the previous page's next-cursor.
    Convenience flags (tag/type/source/since/before) compile into the same
    filter object.
    """
    name = _resolve(collection)
    flt, warning = build_filter(filter)
    flt = merge_convenience(flt, project=project, tag=tag, type=type,
                            source=source, since=since, before=before)
    if not include_inactive:
        flt = _payload.active_filter(flt)
    points = _scroll_all(name, flt, max_points=100000)

    field = {"created": _payload.CREATED_TS, "updated": _payload.UPDATED_TS}.get(order_by, _payload.CREATED_TS)
    points.sort(key=lambda p: ((p.payload or {}).get(field) or 0, str(p.id)), reverse=True)
    offset = int(cursor) if cursor.strip().isdigit() else 0
    page = points[offset:offset + max(1, limit)]
    consumed = offset + len(page)
    next_cursor = str(consumed) if len(points) > consumed else ""
    if not page:
        return _warn_append(f"No memories in {name!r} at cursor {offset}.", warning)
    out = [f"Listed {len(page)}/{len(points)} memories from {name!r}:"]
    for p in page:
        payload = p.payload or {}
        status = payload.get(_payload.STATUS)
        note = f" [status={status}]" if status else ""
        created = payload.get(_payload.CREATED_AT, "?")
        text = str(payload.get("text", ""))
        brief = (text[:100] + "…") if len(text) > 100 else text
        out.append(f"- [ID: {p.id}] {created} [{payload.get('project', '-')}/{payload.get('type', '-')}]"
                   f"{note} {brief}")
    if next_cursor:
        out.append(f"next-cursor: {next_cursor} ({len(points) - consumed} more)")
    return _warn_append("\n".join(out), warning)


def _warn_append(result: str, warning: str | None) -> str:
    if warning:
        return f"{result}\n{warning}"
    return result


def count_memories(collection: str = "", filter: str | dict[str, Any] = "",
                   project: str = "", include_inactive: bool = False,
                   tag: list[str] | None = None, type: str = "",
                   source: str = "", since: str = "", before: str = "") -> str:
    name = _resolve(collection)
    flt, _ = build_filter(filter)
    flt = merge_convenience(flt, project=project, tag=tag, type=type,
                            source=source, since=since, before=before)
    if not include_inactive:
        flt = _payload.active_filter(flt)
    n = _store.qdrant.count(collection_name=name, count_filter=flt, exact=True).count
    return f"Count: {n} in {name!r}"


def collection_stats(collection: str = "", max_scan: int = 5000) -> str:
    name = _resolve(collection)
    points = _scroll_all(name, max_points=max_scan)
    by_project: dict[str, int] = {}
    by_type: dict[str, int] = {}
    statuses: dict[str, int] = {}
    created_values: list[float] = []
    for p in points:
        payload = p.payload or {}
        proj = payload.get("project", "(none)")
        typ = payload.get("type", "(none)")
        status = payload.get(_payload.STATUS, _payload.STATUS_ACTIVE)
        by_project[proj] = by_project.get(proj, 0) + 1
        by_type[typ] = by_type.get(typ, 0) + 1
        statuses[status] = statuses.get(status, 0) + 1
        created = payload.get(_payload.CREATED_TS)
        if isinstance(created, (int, float)):
            created_values.append(created)
    info = _store.qdrant.get_collection(name)
    vectors = info.config.params.vectors
    if isinstance(vectors, dict):
        vectors = vectors.get("")
    dim = int(vectors.size) if vectors is not None else -1
    embed_models = sorted({
        p.payload.get(_payload.EMBED_MODEL)
        for p in points
        if p.payload and p.payload.get(_payload.EMBED_MODEL)
    })
    oldest = _payload.iso_now(min(created_values)) if created_values else "(no timestamps)"
    newest = _payload.iso_now(max(created_values)) if created_values else "(no timestamps)"
    lines = [
        f"Stats for {name!r} (scanned {len(points)}, dim {dim}):",
        "  status: " + ", ".join(f"{k}={v}" for k, v in sorted(statuses.items())),
        "  projects: " + ", ".join(f"{k}={v}" for k, v in sorted(by_project.items())),
        "  types: " + ", ".join(f"{k}={v}" for k, v in sorted(by_type.items())),
        f"  created range: {oldest} .. {newest}",
        f"  embed models: {', '.join(embed_models) or '(unknown)'}",
    ]
    if len(points) >= max_scan:
        lines.append(f"  (scan capped at {max_scan} — pass --max-scan to raise)")
    return "\n".join(lines)


def field_values(field: str, collection: str = "", filter: str | dict[str, Any] = "",
                 limit: int = 50, project: str = "") -> str:
    """Distinct values for project/type/tags/source (schema discovery)."""
    name = _resolve(collection)
    if field not in ("project", "type", "tags", "source"):
        raise ArgumentError(
            f"field must be one of project, type, tags, source (got {field!r})")
    flt, _ = build_filter(filter)
    flt = merge_convenience(flt, project=project)
    flt = _payload.active_filter(flt)
    points = _scroll_all(name, flt)
    values: dict[str, int] = {}
    for p in points:
        value = (p.payload or {}).get(field)
        if value is None:
            values["(missing)"] = values.get("(missing)", 0) + 1
        elif isinstance(value, list):
            for v in value:
                values[str(v)] = values.get(str(v), 0) + 1
        else:
            values[str(value)] = values.get(str(value), 0) + 1
    ordered = sorted(values.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
    if not ordered:
        return f"No values for {field!r} in {name!r}."
    out = [f"Values for {field!r} in {name!r} ({len(ordered)} distinct):"]
    for value, count in ordered:
        out.append(f"  {value} ({count})")
    return "\n".join(out)


def elapsed_note(start: float) -> str:
    return f"(took {time.time() - start:.2f}s)"
