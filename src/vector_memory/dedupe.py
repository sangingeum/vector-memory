"""Duplicate handling (exact-idempotent saves + near-duplicate detection).

Exact duplicates: point ID for new writes is
``uuid5(NAMESPACE, f"{collection}|{_content_hash}")`` — saving the same
normalized text again upserts the same point (idempotent refresh).
``--allow-duplicate`` forces a fresh uuid4 point.

Near duplicates: after embedding, the top active neighbors with cosine >=
``VM_DEDUPE_THRESHOLD`` (default 0.985 — calibrated on qwen3-embedding:8b,
see docs/similarity-calibration.md; contradictions score in the same band as
duplicates, so candidates are REPORTED, never merged) are returned to the
caller. ``--on-similar warn|skip|error`` controls the behavior.
"""

from __future__ import annotations

import os
import uuid
from typing import Any

from qdrant_client.models import FieldCondition, Filter, MatchValue

from . import payload as _payload
from . import store as _store
from .errors import ConflictError
from .validation import content_hash

# vm5-namespace for deterministic duplicate point ids
NAMESPACE_VM = uuid.uuid5(uuid.NAMESPACE_URL, "sangingeum/vector-memory")

DEDUPE_THRESHOLD = 0.985


def dedupe_threshold() -> float:
    """Configured near-duplicate threshold (env override)."""
    raw = os.environ.get("VM_DEDUPE_THRESHOLD", "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return DEDUPE_THRESHOLD


def duplicate_point_id(collection_name: str, text: str) -> str:
    """Deterministic point ID for normalized text in a collection."""
    return str(uuid5_for(collection_name, content_hash(text)))


def uuid5_for(collection_name: str, digest: str) -> uuid.UUID:
    return uuid.uuid5(NAMESPACE_VM, f"{collection_name}|{digest}")


def find_similar(
    collection_name: str, vector: list[float], *,
    project: str = "", limit: int = 3, threshold: float | None = None,
) -> list[dict[str, Any]]:
    """Active neighbors with cosine >= threshold (calibrated default 0.985)."""
    if threshold is None:
        threshold = dedupe_threshold()
    flt = _payload.active_filter()
    if project:
        flt_must = list(flt.must or [])
        flt_must.append(FieldCondition(key="project", match=MatchValue(value=project)))
        flt = Filter(must=flt_must, must_not=list(flt.must_not or []))
    try:
        response = _store.qdrant.query_points(
            collection_name=collection_name,
            query=vector,
            query_filter=flt,
            limit=limit + 1,  # +1 because the point itself may be among them
            with_payload=True,
        )
    except Exception:
        return []
    out = []
    for hit in response.points:
        if hit.score is None or hit.score < threshold:
            continue
        payload = hit.payload or {}
        text = str(payload.get("text", ""))
        out.append({
            "id": str(hit.id),
            "score": float(hit.score),
            "text": (text[:80] + "…") if len(text) > 80 else text,
            "status": payload.get(_payload.STATUS),
        })
    return out[:limit]


def find_legacy_duplicate(collection_name: str, digest: str) -> str | None:
    """ID of an existing point whose ``_content_hash`` matches *digest*.

    Catches duplicates of legacy (uuid4) points that a migration already
    stamped with ``_content_hash`` — they can never collide with the uuid5
    deterministic id. Returns the first match or None.
    """
    from qdrant_client.models import FieldCondition, MatchValue

    flt = Filter(must=[FieldCondition(
        key=_payload.CONTENT_HASH, match=MatchValue(value=digest))])
    try:
        batch, _ = _store.qdrant.scroll(
            collection_name=collection_name, scroll_filter=flt,
            limit=1, with_payload=False, with_vectors=False,
        )
    except Exception:
        return None
    return str(batch[0].id) if batch else None


def on_similar_mode() -> str:
    """Configured on-similar behavior: warn | skip | error (default warn)."""
    raw = os.environ.get("VM_ON_SIMILAR", "").strip().lower()
    if raw in ("warn", "skip", "error"):
        return raw
    return "warn"


def apply_similar_policy(similar: list[dict[str, Any]], mode: str) -> str | None:
    """Enforce the on-similar policy. Returns the existing id in 'skip' mode
    (caller must not write); raises ConflictError in 'error' mode."""
    if not similar:
        return None
    top = similar[0]
    if mode == "error":
        raise ConflictError(
            f"near-duplicate of {top['id']} (score {top['score']:.3f}): "
            f"{top['text']!r}; use update or --supersedes, or pass --on-similar warn"
        )
    if mode == "skip":
        return top["id"]
    return None
