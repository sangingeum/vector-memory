"""Non-destructive, resumable migration of legacy points to system fields.

For each point lacking system fields: stamp ``_created_ts``/``_updated_ts``
(migration time, ``_created_estimated: true``), compute ``_content_hash`` and
optionally set ``_embed_model`` (only with ``--assume-model``). Never touches
vectors or text; re-running is a no-op for already-migrated points.
"""

from __future__ import annotations

import time
from typing import Any

from . import payload as _payload
from . import store as _store
from .validation import content_hash


def migrate_collection(
    collection_name: str,
    *,
    assume_model: str = "",
    batch_size: int = 64,
) -> dict[str, Any]:
    """Backfill system fields on legacy points. Returns a summary dict.

    Never modifies vectors or text. Resumable: points that already carry
    ``_created_ts`` are skipped on re-run.
    """
    if not _store.qdrant.collection_exists(collection_name):
        raise KeyError(f"collection {collection_name!r} does not exist")
    _payload.ensure_payload_indexes(_store.qdrant, collection_name)

    scanned = 0
    migrated = 0
    offset = None
    while True:
        batch, offset = _store.qdrant.scroll(
            collection_name=collection_name,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not batch:
            break
        for point in batch:
            scanned += 1
            old_payload: dict[str, Any] = dict(point.payload or {})
            if _payload.CREATED_TS in old_payload:
                continue  # already migrated — resumable no-op
            now = time.time()
            new_fields: dict[str, Any] = {
                _payload.CREATED_TS: now,
                _payload.UPDATED_TS: now,
                _payload.CREATED_AT: _payload.iso_now(now),
                _payload.UPDATED_AT: _payload.iso_now(now),
                _payload.CREATED_ESTIMATED: True,
            }
            text = old_payload.get("text")
            if isinstance(text, str) and text:
                new_fields[_payload.CONTENT_HASH] = content_hash(text)
            if assume_model:
                new_fields[_payload.EMBED_MODEL] = assume_model
            _store.qdrant.set_payload(
                collection_name=collection_name,
                payload=new_fields,
                points=[point.id],
                wait=True,
            )
            migrated += 1
        if offset is None:
            break

    return {
        "collection": collection_name,
        "scanned": scanned,
        "migrated": migrated,
        "embed_model": assume_model or "(unset — pass --assume-model to record it)",
    }
