"""Backup: export memories to JSONL and import them back (VM-17)."""

from __future__ import annotations

import json
from typing import Any

from . import payload as _payload
from . import store as _store
from .browse import _resolve, _scroll_all
from .embedding import embed
from .errors import ArgumentError


def export_memories(collection: str = "", with_vectors: bool = False) -> str:
    """One JSON object per line: id, text, metadata (incl. system fields)."""
    name = _resolve(collection)
    points = _scroll_all(name, max_points=1000000)
    out_lines = []
    for p in points:
        payload = dict(p.payload or {})
        text = payload.pop("text", "")
        record: dict[str, Any] = {"id": str(p.id), "text": text, "metadata": payload}
        if with_vectors:

            rec = _store.qdrant.retrieve(collection_name=name, ids=[p.id], with_vectors=True)[0]
            record["vector"] = rec.vector
        out_lines.append(json.dumps(record, ensure_ascii=False))
    return "\n".join(out_lines) if out_lines else "(no memories to export)"


def import_memories(jsonl: str, collection: str = "",
                    reembed: bool = False,
                    on_conflict: str = "skip") -> str:
    """Import a JSONL export. Without vectors (or with --reembed) re-embeds."""
    name = collection if collection and collection.strip() else _store.COLLECTION_NAME
    if on_conflict not in ("skip", "overwrite"):
        raise ArgumentError("on_conflict must be 'skip' or 'overwrite'")
    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(jsonl.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ArgumentError(f"line {line_no} is not valid JSON (position {exc.pos})") from exc
        if not isinstance(record, dict) or "id" not in record or "text" not in record:
            raise ArgumentError(f"line {line_no}: each record needs 'id' and 'text'")
        records.append(record)
    if not records:
        return "Nothing to import."
    if not _store.qdrant.collection_exists(name):
        from qdrant_client.models import Distance, VectorParams

        dim = len(embed("dimension probe"))
        _store.qdrant.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
    existing_ids = {str(r.id) for r in _store.qdrant.retrieve(
        collection_name=name, ids=[r["id"] for r in records], with_payload=False)}
    imported, skipped = 0, 0
    points = []
    for record in records:
        pid = record["id"]
        if pid in existing_ids and on_conflict == "skip":
            skipped += 1
            continue
        payload = dict(record.get("metadata") or {})
        payload["text"] = record["text"]
        payload[_payload.EMBED_MODEL] = payload.get(_payload.EMBED_MODEL, "")
        vector = record.get("vector") if not reembed else None
        if not vector:
            vector = embed(record["text"])
        points.append({"id": pid, "vector": vector, "payload": payload})
        imported += 1
    if points:
        from qdrant_client.models import PointStruct

        structs = [PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"])
                   for p in points]
        _store.qdrant.upsert(collection_name=name, points=structs, wait=True)
    return f"Imported {imported} memories into {name!r} (skipped {skipped} existing)"
