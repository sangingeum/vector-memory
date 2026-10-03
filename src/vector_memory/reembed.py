"""Embedding-model fingerprint enforcement and re-embedding (VM-06).

A dimension match does not imply model compatibility: two models with the
same vector size produce incompatible spaces. Collections record
``_embed_model`` on every write; opening/writing/searching a collection whose
recorded model differs from the configured one fails with ``ConfigError``
unless ``--force-model-mismatch`` is set (documented unsafe). Legacy points
without ``_embed_model`` count as "unknown" and never fail.
"""

from __future__ import annotations

import os
from typing import Any

from . import payload as _payload
from . import store as _store
from .errors import ConfigError


def force_model_mismatch() -> bool:
    """True when the unsafe override is requested."""
    return os.environ.get("VM_FORCE_MODEL_MISMATCH", "").strip() in ("1", "true", "yes")


def check_collection_model(collection_name: str, configured_model: str) -> None:
    """Compare the collection's recorded ``_embed_model`` with the configured one.

    Fetches ONE point filtered on ``_embed_model`` existing (per plan VM-06:
    legacy points without the field never mask a recorded model). Raises
    :class:`ConfigError` on a mismatch.
    """
    if force_model_mismatch() or not configured_model:
        return
    try:
        from qdrant_client.models import Filter, IsEmptyCondition

        flt = Filter(must_not=[IsEmptyCondition(is_empty={"key": _payload.EMBED_MODEL})])
        batch, _ = _store.qdrant.scroll(
            collection_name=collection_name, scroll_filter=flt, limit=1,
            with_payload=True, with_vectors=False,
        )
    except Exception:
        return  # collection absent / unreachable — other layers report that
    for point in batch or []:
        recorded = (point.payload or {}).get(_payload.EMBED_MODEL)
        if recorded and recorded != configured_model:
            raise ConfigError(
                f"collection {collection_name!r} was built with {recorded!r}; "
                f"configured model is {configured_model!r}. Use --collection NEW, "
                f"set EMBED_MODEL={recorded}, or run: vector-memory reembed"
            )


def reembed_collection(
    source: str, target: str, configured_model: str,
    *, batch_size: int = 32, resume: bool = False,
) -> dict[str, Any]:
    """Re-embed every point of *source* into *target* with the current model.

    Same IDs and payloads; ``_embed_model`` updated; SRC is never mutated.
    Resumable: with ``resume=True``, IDs already present in the target are
    skipped. Counts are verified at the end.
    """
    from .embedding import embed_many
    from .validation import content_hash

    if source == target:
        raise ConfigError("reembed target must differ from the source collection")
    if not _store.qdrant.collection_exists(source):
        raise ConfigError(f"source collection {source!r} does not exist")

    # Verify the target (if it exists) was not built with a different model.
    if _store.qdrant.collection_exists(target) and not force_model_mismatch():
        check_collection_model(target, configured_model)

    if not _store.qdrant.collection_exists(target):
        from qdrant_client.models import Distance, VectorParams

        dim = len(embed_many(["dimension probe"])[0])
        _store.qdrant.create_collection(
            collection_name=target,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
    _payload.ensure_payload_indexes(_store.qdrant, target)

    copied = 0
    skipped = 0
    offset = None
    while True:
        batch, offset = _store.qdrant.scroll(
            collection_name=source, limit=batch_size, offset=offset,
            with_payload=True, with_vectors=False,
        )
        if not batch:
            break
        todo = []
        if resume:
            existing_ids = {str(r.id) for r in _store.qdrant.retrieve(
                collection_name=target,
                ids=[p.id for p in batch],
                with_payload=False,
            )}
        else:
            existing_ids = set()
        for point in batch:
            if str(point.id) in existing_ids:
                skipped += 1
                continue
            payload = dict(point.payload or {})
            text = payload.get("text")
            if not isinstance(text, str) or not text:
                skipped += 1
                continue
            todo.append((point, payload, text))
        if todo:
            vectors = embed_many([t for _, _, t in todo])
            points = []
            from qdrant_client.models import PointStruct

            for (point, payload, text), vector in zip(todo, vectors, strict=True):
                payload[_payload.EMBED_MODEL] = configured_model
                payload[_payload.CONTENT_HASH] = content_hash(text)
                points.append(PointStruct(id=point.id, vector=vector, payload=payload))
            _store.qdrant.upsert(collection_name=target, points=points, wait=True)
            copied += len(points)
        if offset is None:
            break

    source_count = _store.qdrant.count(collection_name=source, exact=True).count
    target_count = _store.qdrant.count(collection_name=target, exact=True).count
    return {
        "source": source,
        "target": target,
        "copied": copied,
        "skipped": skipped,
        "source_count": source_count,
        "target_count": target_count,
        "model": configured_model,
    }
