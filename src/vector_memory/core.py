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

from .embedding import EMBED_MODEL, OLLAMA_URL, embed, embed_many
from . import store as _store
from .store import (
    COLLECTION_NAME,
    QDRANT_URL,
    build_filter,
    ensure_collection_for,
    parse_metadata,
)


def _append_warning(result: str, warning: str | None) -> str:
    """Append a warning line to a result string, if any."""
    if warning:
        return f"{result}\n{warning}"
    return result


def _apply_collection(collection: str | None) -> str:
    """Resolve the effective collection name and create it if missing."""
    name = collection if collection and collection.strip() else COLLECTION_NAME
    ensure_collection_for(name, embed, embed_model=EMBED_MODEL)
    return name


def save_memory(text: str, metadata: str | dict[str, Any] = "{}",
                collection: str = "") -> str:
    """Save a document/scenario outcome into the vector DB."""
    meta_dict, warning = parse_metadata(metadata)
    try:
        name = _apply_collection(collection)
        vector = embed(text)
        point_id = str(uuid.uuid4())
        meta_dict["text"] = text
        _store.qdrant.upsert(
            collection_name=name,
            points=[PointStruct(id=point_id, vector=vector, payload=meta_dict)],
        )
        return _append_warning(f"Memory saved (ID: {point_id}, collection: {name})", warning)
    except Exception as exc:
        return _append_warning(f"Save failed: {exc}", warning)


def save_memories(texts: list[str], metadata: str | dict[str, Any] = "{}",
                  collection: str = "") -> str:
    """Save multiple documents in one batch (single embed + upsert round trip)."""
    meta_dict, warning = parse_metadata(metadata)
    if not texts:
        return _append_warning("No texts to save.", warning)
    try:
        name = _apply_collection(collection)
        vectors = embed_many(texts)
        points = []
        for text, vector in zip(texts, vectors, strict=True):
            payload = dict(meta_dict)
            payload["text"] = text
            points.append(PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload))
        _store.qdrant.upsert(collection_name=name, points=points)
        ids = [str(p.id) for p in points]
        return _append_warning(
            f"Saved {len(points)} memories (collection: {name}, IDs: {', '.join(ids)})",
            warning,
        )
    except Exception as exc:
        return _append_warning(f"Batch save failed: {exc}", warning)


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
        return _append_warning(f"Search failed: {exc}", warning)

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


def _has_metadata(metadata: str | dict[str, Any]) -> bool:
    """True if metadata carries content (non-empty dict or non-blank string)."""
    if isinstance(metadata, dict):
        return bool(metadata)
    return bool(metadata and metadata.strip())


def update_memory(point_id: str, text: str | None = None,
                  metadata: str | dict[str, Any] = "",
                  collection: str = "") -> str:
    """Update an existing memory (point) in place under the same ID."""
    try:
        if (text is None or not text.strip()) and not _has_metadata(metadata):
            return "Error: nothing to update — provide new text, new metadata, or both."
        name = collection if collection and collection.strip() else COLLECTION_NAME
        if not _store.qdrant.collection_exists(name):
            return f"Update failed: collection {name!r} does not exist."
        existing = _store.qdrant.retrieve(collection_name=name, ids=[point_id], with_payload=True)
        if not existing:
            return f"Update failed: point_id {point_id} not found."
        point = existing[0]
        if _has_metadata(metadata):
            meta_dict, warning = parse_metadata(metadata)
            payload: dict[str, Any] = dict(meta_dict)
        else:
            warning = None
            payload = dict(point.payload or {})
        if text is None or not text.strip():
            # Metadata-only update: keep the existing text and vector, replace
            # the payload without re-embedding.
            payload["text"] = (point.payload or {}).get("text", "") if _has_metadata(metadata) else payload["text"]
            _store.qdrant.set_payload(collection_name=name, payload=payload, points=[point_id])
            return _append_warning(f"Memory updated (ID: {point_id}, collection: {name})", warning)
        payload["text"] = text
        vector = embed(text)
        _store.qdrant.upsert(
            collection_name=name,
            points=[PointStruct(id=point_id, vector=vector, payload=payload)],
        )
        return _append_warning(f"Memory updated (ID: {point_id}, collection: {name})", warning)
    except Exception as exc:
        return f"Update failed: {exc}"


def delete_memory(point_id: str, collection: str = "") -> str:
    """Delete a stored memory (point) by ID."""
    try:
        name = collection if collection and collection.strip() else COLLECTION_NAME
        if not _store.qdrant.collection_exists(name):
            return f"Delete failed: collection {name!r} does not exist."
        existing = _store.qdrant.retrieve(collection_name=name, ids=[point_id], with_payload=False)
        if not existing:
            return f"Error: point_id {point_id} not found."
        _store.qdrant.delete(collection_name=name, points_selector=[point_id])
        return f"Memory deleted (ID: {point_id}, collection: {name})"
    except Exception as exc:
        return f"Delete failed: {exc}"


def list_collections() -> str:
    """List all collections currently present in Qdrant."""
    try:
        collections = _store.qdrant.get_collections().collections
        if not collections:
            return "No collections found."
        return "Collections: " + ", ".join(c.name for c in collections)
    except Exception as exc:
        return f"Failed to list collections: {exc}"
