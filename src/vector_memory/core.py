"""Core operations for vector-memory (design ruling §3).

The six memory ops (save_memory, save_memories, search_memory, update_memory,
delete_memory, list_collections) live here as plain functions; both adapters —
``server.py`` (MCP, 1:1 tools) and ``cli.py`` (typer, 1:1 subcommands) — are
thin surfaces that call exactly one core op each. Embedding and Qdrant access
stay in the ``embedding`` / ``store`` modules.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from qdrant_client.models import FieldCondition, Filter, MatchValue, PointStruct

from . import payload as _payload
from . import store as _store
from .embedding import EMBED_MODEL, embed, embed_many
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


def _agent_id() -> str:
    """Optional author id stamped as ``_agent`` (VM_AGENT_ID env)."""
    import os

    return os.environ.get("VM_AGENT_ID", "").strip()


def _apply_collection(collection: str | None) -> str:
    """Resolve the effective collection name and create it if missing."""
    name = collection if collection and collection.strip() else COLLECTION_NAME
    ensure_collection_for(name, embed, embed_model=EMBED_MODEL)
    _payload.ensure_payload_indexes(_store.qdrant, name)
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


def save_memory(text: str, metadata: str | dict[str, Any] = "{}",
                collection: str = "", project: str = "", type: str = "",
                tags: list[str] | None = None) -> str:
    """Save a document/scenario outcome into the vector DB.

    Raises :class:`ArgumentError` on invalid input (nothing written).
    """
    stripped = validate_text(text)
    meta_dict = parse_metadata(metadata)
    _apply_explicit_metadata(meta_dict, project=project, type=type, tags=tags)
    try:
        name = _apply_collection(collection)
        vector = embed(stripped)
        point_id = str(uuid.uuid4())
        meta_dict["text"] = stripped
        meta_dict.update(_payload.system_fields_for_new_text(stripped, EMBED_MODEL, _agent_id()))
        _store.qdrant.upsert(
            collection_name=name,
            points=[PointStruct(id=point_id, vector=vector, payload=meta_dict)],
            wait=True,
        )
        return _with_lenient_warning(f"Memory saved (ID: {point_id}, collection: {name})")
    except Exception as exc:
        raise _backend_or_internal(exc) from exc


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
        payload.update(_payload.system_fields_for_new_text(text, EMBED_MODEL, _agent_id()))
        points.append(PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload))
    _store.qdrant.upsert(collection_name=name, points=points, wait=True)
    ids = [str(p.id) for p in points]
    return _with_lenient_warning(
        f"Saved {len(points)} memories (collection: {name}, IDs: {', '.join(ids)})"
    )


def search_memory(query: str, limit: int = 3, filter: str | dict[str, Any] = "",
                  collection: str = "", project: str = "") -> str:
    """Search for stored documents/scenarios similar to a query."""
    qdrant_filter, warning = build_filter(filter)
    if project:
        proj_cond = FieldCondition(key="project", match=MatchValue(value=project))
        must: list[Any] = list(qdrant_filter.must or []) if qdrant_filter else []
        must.append(proj_cond)
        qdrant_filter = Filter(must=must)
    try:
        name = collection if collection and collection.strip() else COLLECTION_NAME
        query_vector = embed(query)
        response = _store.qdrant.query_points(
            collection_name=name,
            query=query_vector,
            query_filter=qdrant_filter,
            limit=max(1, limit),
            with_payload=True,
        )
        hits = response.points
    except Exception as exc:
        raise _backend_or_internal(exc) from exc

    if not hits:
        return _append_warning("No matching memories found.", warning)

    out = [f"Search results ({len(hits)} hits, collection: {name}):"]
    for hit in hits:
        payload = hit.payload or {}
        text = payload.get("text", "")
        meta = {k: v for k, v in payload.items() if k != "text"}
        meta_str = json.dumps(meta, ensure_ascii=False) if meta else "{}"
        out.append(
            f"- [ID: {hit.id}] [score: {hit.score:.4f}] "
            f"metadata: {meta_str} | content: {text}"
        )
    return _append_warning("\n".join(out), warning)


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
    old_payload = point.payload or {}
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
