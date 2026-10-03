"""Core operations for vector-memory (design ruling §3).

The six memory ops (save_memory, save_memories, search_memory, update_memory,
delete_memory, list_collections) live here as plain functions; both adapters —
``server.py`` (MCP, 1:1 tools) and ``cli.py`` (typer, 1:1 subcommands) — are
thin surfaces that call exactly one core op each. Embedding and Qdrant access
stay in the ``embedding`` / ``store`` modules.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from qdrant_client.models import PointStruct

from . import dedupe as _dedupe
from . import payload as _payload
from . import reembed as _fingerprint
from . import sensitive as _sensitive
from . import store as _store
from .embedding import embed, embed_many
from .errors import ArgumentError, NotFoundError, format_error
from .store import (
    COLLECTION_NAME,
    build_filter,
    ensure_collection_for,
    parse_metadata,
)
from .validation import validate_text


def call(op, *args, **kwargs) -> str:
    """Run a core op for an MCP tool; failures become ``ErrorType: ...`` text.

    The FastMCP wrapper converts a raised exception into a tool error
    (``isError``) whose content is this single-line text.
    """
    try:
        return op(*args, **kwargs)
    except Exception as exc:  # boundary error contract
        return format_error(exc)


def _append_warning(result: str, warning: str | None) -> str:
    """Append a warning line to a result string, if any."""
    if warning:
        return f"{result}\n{warning}"
    return result


def current_embed_model() -> str:
    """Read the configured embedding model at call time (tests monkeypatch it)."""
    from .embedding import EMBED_MODEL

    return EMBED_MODEL


def _agent_id() -> str:
    """Optional author id stamped as ``_agent`` (VM_AGENT_ID env)."""
    import os

    return os.environ.get("VM_AGENT_ID", "").strip()


def _apply_collection(collection: str | None) -> str:
    """Resolve the effective collection name and create it if missing."""
    name = collection if collection and collection.strip() else COLLECTION_NAME
    ensure_collection_for(name, embed, embed_model=current_embed_model())
    _payload.ensure_payload_indexes(_store.qdrant, name)
    _fingerprint.check_collection_model(name, current_embed_model())
    return name


def _apply_explicit_metadata(
    meta_dict: dict[str, Any],
    project: str = "",
    type: str = "",
    tags: list[str] | None = None,
) -> None:
    """Merge CLI convenience options into the parsed metadata dict.

    Explicit options win over same-key entries from ``--metadata`` JSON;
    keys are only written when a non-empty value was supplied.
    """
    if project:
        meta_dict["project"] = project
    if type:
        meta_dict["type"] = type
    if tags:
        meta_dict["tags"] = list(tags)


def delete_many(point_ids: list[str], collection: str = "") -> str:
    """Delete multiple points by ID (missing IDs reported)."""
    from .browse import _resolve
    from .errors import NotFoundError

    name = _resolve(collection)
    records = _store.qdrant.retrieve(collection_name=name, ids=point_ids, with_payload=False)
    found = {str(r.id) for r in records}
    missing = [pid for pid in point_ids if str(pid) not in found]
    if missing == point_ids:
        raise NotFoundError(f"point(s) not found: {', '.join(missing)}")
    if found:
        _store.qdrant.delete(collection_name=name, points_selector=sorted(found), wait=True)
    out = f"Deleted {len(found)} point(s) from {name!r}"
    if missing:
        out += f"; not found: {', '.join(missing)}"
    return out


def save_memory(text: str, metadata: str | dict[str, Any] = "{}",
                collection: str = "", project: str = "", type: str = "",
                tags: list[str] | None = None,
                supersedes: list[str] | None = None,
                allow_duplicate: bool = False,
                on_similar: str = "") -> str:
    """Save a document/scenario outcome into the vector DB.

    Raises :class:`ArgumentError` on invalid input (nothing written).
    ``supersedes`` lists point IDs replaced by this memory: those points are
    marked ``_status=superseded`` with ``_superseded_by`` pointing here, and
    this point records them under ``_supersedes``. Unknown IDs fail before
    anything is written. Identical normalized text is idempotent (same point
    refreshed) unless ``allow_duplicate``; near-duplicates (calibrated
    threshold, default 0.985) are reported by ``on_similar`` mode
    (warn|skip|error) — never merged. Text matching a high-confidence secret
    rule raises :class:`SensitiveContentError` before any write.
    """
    stripped = validate_text(text)
    _sensitive.check_text(stripped)
    meta_dict = parse_metadata(metadata)
    _apply_explicit_metadata(meta_dict, project=project, type=type, tags=tags)
    try:
        name = _apply_collection(collection)
        if supersedes:
            _validate_supersede_targets(name, supersedes)
        vector = embed(stripped)
        similar = _dedupe.find_similar(name, vector, project=project)
        skip_id = _dedupe.apply_similar_policy(similar, on_similar or _dedupe.on_similar_mode())
        if skip_id is not None:
            return _format_similar(
                _with_lenient_warning(
                    f"Skipped (near-duplicate of existing memory {skip_id}, collection: {name})"
                ),
                similar,
            )
        if allow_duplicate:
            point_id = str(uuid.uuid4())
        else:
            point_id = _dedupe.duplicate_point_id(name, stripped)
            existing = _store.qdrant.retrieve(collection_name=name, ids=[point_id], with_payload=True)
            if existing:
                # Idempotent refresh: same normalized text, same point.
                old_payload = dict(existing[0].payload or {})
                merged = dict(old_payload)
                merged.update(meta_dict)
                merged["text"] = stripped
                merged.update(_payload.system_fields_for_new_text(stripped, current_embed_model(), _agent_id()))
                _store.qdrant.upsert(
                    collection_name=name,
                    points=[PointStruct(id=point_id, vector=vector, payload=merged)],
                    wait=True,
                )
                out = _with_lenient_warning(
                    f"Memory saved (ID: {point_id}, collection: {name}; "
                    "duplicate of existing; updated)"
                )
                return _format_similar(out, similar)
        meta_dict["text"] = stripped
        meta_dict.update(_payload.system_fields_for_new_text(stripped, current_embed_model(), _agent_id()))
        if supersedes:
            meta_dict[_payload.SUPERSEDES] = list(supersedes)
        _store.qdrant.upsert(
            collection_name=name,
            points=[PointStruct(id=point_id, vector=vector, payload=meta_dict)],
            wait=True,
        )
        if supersedes:
            _mark_superseded(name, supersedes, point_id)
        return _format_similar(
            _with_lenient_warning(f"Memory saved (ID: {point_id}, collection: {name})"),
            similar,
        )
    except Exception as exc:
        raise _backend_or_internal(exc) from exc


def _format_similar(result: str, similar: list[dict[str, Any]]) -> str:
    """Append `similar` candidate lines — reported, never merged."""
    if not similar:
        return result
    lines = [result]
    for cand in similar:
        lines.append(f"similar {cand['id']} {cand['score']:.3f} \"{cand['text']}\"")
    lines.append(
        "note: similarity does not imply equivalence — check numbers, versions, "
        "negations before merging (use update or --supersedes)"
    )
    return "\n".join(lines)


def _validate_supersede_targets(collection_name: str, supersedes: list[str]) -> None:
    """All supersede targets must exist before anything is written."""
    existing = _store.qdrant.retrieve(
        collection_name=collection_name, ids=list(supersedes), with_payload=False
    )
    found = {str(r.id) for r in existing}
    missing = [pid for pid in supersedes if str(pid) not in found]
    if missing:
        raise NotFoundError(f"supersede target(s) not found: {', '.join(missing)}")


def _mark_superseded(collection_name: str, supersedes: list[str], new_id: str) -> None:
    """Flip the superseded points' lifecycle fields after the new point lands."""
    _store.qdrant.set_payload(
        collection_name=collection_name,
        payload={
            _payload.STATUS: _payload.STATUS_SUPERSEDED,
            _payload.SUPERSEDED_BY: new_id,
        },
        points=list(supersedes),
        wait=True,
    )


def set_status(point_ids: list[str], status: str, collection: str = "") -> str:
    """Archive or unarchive points (``_status`` lifecycle field)."""
    if status not in (_payload.STATUS_ARCHIVED, _payload.STATUS_ACTIVE):
        raise ArgumentError(f"status must be 'archived' or 'active' (got {status!r})")
    name = collection if collection and collection.strip() else COLLECTION_NAME
    if not _store.qdrant.collection_exists(name):
        raise NotFoundError(f"collection {name!r} does not exist")
    existing = _store.qdrant.retrieve(collection_name=name, ids=point_ids, with_payload=False)
    found = {str(r.id) for r in existing}
    missing = [pid for pid in point_ids if str(pid) not in found]
    if missing:
        raise NotFoundError(f"point(s) not found: {', '.join(missing)}")
    fields: dict[str, Any] = {_payload.STATUS: status}
    fields.update(_payload.system_fields_for_update({}, None))
    if status == _payload.STATUS_ACTIVE:
        # Unarchive clears the lifecycle marker entirely (missing = active).
        _store.qdrant.delete_payload(
            collection_name=name, keys=[_payload.STATUS], points=point_ids, wait=True
        )
    else:
        _store.qdrant.set_payload(collection_name=name, payload=fields, points=point_ids, wait=True)
    return f"Status set to {status} for {len(point_ids)} point(s) in {name!r}"


def _with_lenient_warning(result: str) -> str:
    """Legacy contract: append a pending lenient-parse warning, if any."""
    from .store import pop_last_warning

    return _append_warning(result, pop_last_warning())


def save_memories(texts: list[str], metadata: str | dict[str, Any] = "{}",
                  collection: str = "", project: str = "", type: str = "",
                  tags: list[str] | None = None) -> str:
    """Save multiple documents in one batch (single embed + upsert round trip)."""
    meta_dict = parse_metadata(metadata)
    _apply_explicit_metadata(meta_dict, project=project, type=type, tags=tags)
    if not texts:
        raise ArgumentError("texts list is empty")
    if not all(isinstance(t, str) for t in texts):
        raise ArgumentError("texts must all be strings")
    stripped = [validate_text(t, where=f"texts[{i}]") for i, t in enumerate(texts)]
    name = _apply_collection(collection)
    vectors = embed_many(stripped)
    points = []
    for text, vector in zip(stripped, vectors, strict=True):
        payload = dict(meta_dict)
        payload["text"] = text
        payload.update(_payload.system_fields_for_new_text(text, current_embed_model(), _agent_id()))
        points.append(PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload))
    _store.qdrant.upsert(collection_name=name, points=points, wait=True)
    ids = [str(p.id) for p in points]
    return _with_lenient_warning(
        f"Saved {len(points)} memories (collection: {name}, IDs: {', '.join(ids)})"
    )


def search_memory(query: str, limit: int = 3, filter: str | dict[str, Any] = "",
                  collection: str = "", project: str = "",
                  include_inactive: bool = False,
                  since: str = "", before: str = "", tag: list[str] | None = None,
                  type: str = "", source: str = "",
                  min_score: float | None = None,
                  recency_weight: float = 0.0,
                  mmr: float = 0.0,
                  brief: bool = False,
                  max_chars: int = 0,
                  output_format: str = "text") -> str:
    """Search for stored documents/scenarios similar to a query.

    Superseded/archived memories are hidden by default (legacy points without
    a ``_status`` field stay visible); ``include_inactive=True`` returns all
    with a ``status=...`` annotation. Enhancements (all optional):
    ``min_score`` (Qdrant score_threshold), ``recency_weight`` (0-1
    application-code re-rank, 90-day half-life), ``mmr`` (maximal marginal
    relevance for diversity), ``brief``/``max_chars`` (per-hit truncation,
    prefers the ``summary`` metadata), ``output_format`` compact|text.
    """
    qdrant_filter, warning = build_filter(filter)
    from .filters import merge_convenience

    qdrant_filter = merge_convenience(qdrant_filter, project=project, type=type,
                                      tag=tag, source=source, since=since, before=before)
    if not include_inactive:
        qdrant_filter = _payload.active_filter(qdrant_filter)
    # Candidate fetch: over-sample for re-ranking/MMR (floor 25 for small limits).
    fetch = max(1, limit)
    if mmr > 0:
        fetch = max(fetch * 4, 25)
    elif recency_weight > 0:
        fetch = max(fetch * 3, 25)
    try:
        name = collection if collection and collection.strip() else COLLECTION_NAME
        _fingerprint.check_collection_model(name, current_embed_model())
        query_vector = embed(query)
        response = _store.qdrant.query_points(
            collection_name=name,
            query=query_vector,
            query_filter=qdrant_filter,
            limit=fetch,
            with_payload=True,
            with_vectors=mmr > 0,
            score_threshold=min_score if min_score is not None else None,
        )
        hits = list(response.points)
    except Exception as exc:
        raise _backend_or_internal(exc) from exc

    if recency_weight > 0 and hits:
        import math
        import os

        half_life = float(os.environ.get("RECENCY_HALF_LIFE_DAYS", "90"))
        now = time.time()
        def _rerank(hit):
            created = (hit.payload or {}).get(_payload.CREATED_TS)
            age_days = max(0.0, (now - created) / 86400) if isinstance(created, (int, float)) else 0.0
            recency = math.exp(-math.log(2) * age_days / half_life)
            return hit.score * (1 - recency_weight) + recency_weight * recency
        hits.sort(key=lambda h: (-_rerank(h), str(h.id)))

    if mmr > 0 and hits:
        hits = _mmr_select(hits, limit, mmr)
    else:
        hits = hits[:max(1, limit)]

    if not hits:
        return _append_warning("No matching memories found.", warning)

    brief_chars = max_chars if max_chars > 0 else (300 if brief else 0)
    out = [f"Search results ({len(hits)} hits, collection: {name}):"]
    for hit in hits:
        payload = hit.payload or {}
        text = str(payload.get("text", ""))
        status = payload.get(_payload.STATUS)
        status_note = ""
        if status:
            superseded_by = payload.get(_payload.SUPERSEDED_BY)
            status_note = f" [status={status}" + (f" -> {superseded_by}" if superseded_by else "") + "]"
        meta = {k: v for k, v in payload.items() if k != "text" and not k.startswith("_")}
        summary_text = str(meta.get("summary", "")) if meta else ""
        if output_format == "compact":
            shown = summary_text or text
            if brief_chars and len(shown) > brief_chars:
                shown = shown[:brief_chars] + f" [truncated; use: get {hit.id}]"
            out.append(
                f"{hit.id}  {hit.score:.4f}  [{meta.get('project', '-')}/{meta.get('type', '-')}]{status_note}  {shown}"
            )
            continue
        if brief_chars:
            shown = summary_text or text
            if len(shown) > brief_chars:
                shown = shown[:brief_chars] + f" [truncated; use: get {hit.id}]"
        else:
            shown = text
        meta_str = json.dumps(meta, ensure_ascii=False) if meta else "{}"
        out.append(
            f"- [ID: {hit.id}] [score: {hit.score:.4f}]{status_note} "
            f"metadata: {meta_str} | content: {shown}"
        )
    return _append_warning("\n".join(out), warning)


def _mmr_select(hits: list, limit: int, lam: float) -> list:
    """Maximal marginal relevance: greedy diverse selection over vectors."""
    import math

    def _vec(hit):
        v = getattr(hit, "vector", None)
        if isinstance(v, dict):
            v = next(iter(v.values()), None)
        return v

    def _cos(a, b):
        if a is None or b is None:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        na = math.sqrt(sum(x * x for x in a)) or 1.0
        nb = math.sqrt(sum(y * y for y in b)) or 1.0
        return dot / (na * nb)

    selected: list = []
    candidates = list(hits)
    while candidates and len(selected) < limit:
        if not selected:
            best = candidates[0]
        else:
            best = max(
                candidates,
                key=lambda h: (
                    lam * h.score - (1 - lam) * max(
                        (_cos(_vec(h), _vec(s)) for s in selected), default=0.0
                    ),
                    str(h.id),
                ),
            )
        selected.append(best)
        candidates.remove(best)
    # Drop vectors from the output (set to None so payloads stay).
    for h in selected:
        try:
            h.vector = None
        except (AttributeError, TypeError, ValueError):
            pass
    return selected


def patch_metadata(point_id: str, set: str | dict[str, Any] = "{}",
                   unset: list[str] | None = None, collection: str = "") -> str:
    """Patch metadata without re-embedding (metadata-only ops work item).

    ``set`` merges the given keys into the payload; ``unset`` removes keys.
    No embedding call; the vector is untouched; ``_updated_*`` is bumped.
    """
    from .store import pop_last_warning

    name = collection if collection and collection.strip() else COLLECTION_NAME
    set_dict = parse_metadata(set) if _has_metadata(set) else {}
    if unset is None:
        unset = []
    if not set_dict and not unset:
        raise ArgumentError("nothing to patch — provide --set and/or --unset")
    if not _store.qdrant.collection_exists(name):
        raise NotFoundError(f"collection {name!r} does not exist")
    existing = _store.qdrant.retrieve(collection_name=name, ids=[point_id], with_payload=True)
    if not existing:
        raise NotFoundError(f"point_id {point_id} not found")
    if unset:
        _store.qdrant.delete_payload(
            collection_name=name, keys=unset, points=[point_id], wait=True
        )
    if set_dict:
        set_dict.update(_payload.system_fields_for_update(existing[0].payload or {}, None))
        _store.qdrant.set_payload(collection_name=name, payload=set_dict, points=[point_id], wait=True)
    return _append_warning(
        f"Memory patched (ID: {point_id}, collection: {name})",
        pop_last_warning(),
    )


def _backend_or_internal(exc: Exception) -> Exception:
    """Classify an exception from the Qdrant/embedding layer."""
    from .errors import BackendError

    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return BackendError(str(exc))
    return exc


def _has_metadata(metadata: str | dict[str, Any]) -> bool:
    """True if metadata carries content (non-empty dict or non-blank string)."""
    if isinstance(metadata, dict):
        return bool(metadata)
    return bool(metadata and metadata.strip())


def update_memory(point_id: str, text: str | None = None,
                  metadata: str | dict[str, Any] = "",
                  collection: str = "", merge_metadata: bool = False) -> str:
    """Update an existing memory (point) in place under the same ID.

    Default metadata semantics are REPLACE (unchanged); ``merge_metadata=True``
    merges the given keys into the existing payload instead.
    """
    if (text is None or not text.strip()) and not _has_metadata(metadata):
        raise ArgumentError("nothing to update — provide new text, new metadata, or both")
    name = collection if collection and collection.strip() else COLLECTION_NAME
    if not _store.qdrant.collection_exists(name):
        raise NotFoundError(f"collection {name!r} does not exist")
    existing = _store.qdrant.retrieve(collection_name=name, ids=[point_id], with_payload=True)
    if not existing:
        raise NotFoundError(f"point_id {point_id} not found")
    point = existing[0]
    old_payload = dict(point.payload or {})
    if _has_metadata(metadata):
        meta_dict = parse_metadata(metadata)
        if merge_metadata:
            payload: dict[str, Any] = dict(old_payload)
            payload.update(meta_dict)
        else:
            payload = dict(meta_dict)
    else:
        payload = dict(old_payload)
    if text is None or not text.strip():
        # Metadata-only update: keep the existing text and vector, rewrite the
        # payload without re-embedding. Replace mode must drop keys absent from
        # the new metadata, so the full payload is upserted (set_payload would
        # only merge and silently keep removed keys).
        if _has_metadata(metadata):
            payload["text"] = old_payload.get("text", "")
        payload.update(_payload.system_fields_for_update(old_payload, None))
        if not merge_metadata:
            vector = point.vector
            if vector is None:  # with_vectors omitted — refetch
                vector = _store.qdrant.retrieve(
                    collection_name=name, ids=[point_id], with_vectors=True
                )[0].vector
            _store.qdrant.upsert(
                collection_name=name,
                points=[PointStruct(id=point_id, vector=vector, payload=payload)],
                wait=True,
            )
        else:
            _store.qdrant.set_payload(collection_name=name, payload=payload, points=[point_id])
        return _with_lenient_warning(f"Memory updated (ID: {point_id}, collection: {name})")
    stripped = validate_text(text)
    _sensitive.check_text(stripped)
    old_payload = dict(point.payload or {})
    payload["text"] = stripped
    payload.update(_payload.system_fields_for_update(old_payload, stripped))
    vector = embed(stripped)
    _store.qdrant.upsert(
        collection_name=name,
        points=[PointStruct(id=point_id, vector=vector, payload=payload)],
        wait=True,
    )
    return _with_lenient_warning(f"Memory updated (ID: {point_id}, collection: {name})")


def delete_memory(point_id: str, collection: str = "") -> str:
    """Delete a stored memory (point) by ID."""
    name = collection if collection and collection.strip() else COLLECTION_NAME
    if not _store.qdrant.collection_exists(name):
        raise NotFoundError(f"collection {name!r} does not exist")
    existing = _store.qdrant.retrieve(collection_name=name, ids=[point_id], with_payload=False)
    if not existing:
        raise NotFoundError(f"point_id {point_id} not found")
    _store.qdrant.delete(collection_name=name, points_selector=[point_id], wait=True)
    return f"Memory deleted (ID: {point_id}, collection: {name})"


def list_collections() -> str:
    """List all collections currently present in Qdrant."""
    try:
        collections = _store.qdrant.get_collections().collections
        if not collections:
            return "No collections found."
        return "Collections: " + ", ".join(c.name for c in collections)
    except Exception as exc:
        return f"Failed to list collections: {exc}"
