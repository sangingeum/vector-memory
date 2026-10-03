"""B-11: mismatch check uses an _embed_model-exists filter, not arbitrary sampling."""

from __future__ import annotations

import pytest

from tests import local_backend as _lb
from tests.local_backend import HashEmbedder

local_qdrant = _lb.local_qdrant

COLL = "vm_fp_filter"


@pytest.fixture()
def env(local_qdrant, monkeypatch):
    from qdrant_client.models import PointStruct

    from vector_memory import embedding as emb
    from vector_memory import store as st
    from vector_memory.validation import content_hash

    monkeypatch.setattr(emb, "ollama_client", HashEmbedder(dim=64))
    monkeypatch.setattr(st, "qdrant", local_qdrant)
    # Legacy points FIRST (no _embed_model), a model-A point BEHIND them —
    # the arbitrary-sampling check missed exactly this shape.
    text = "recorded model behind legacy points"
    local_qdrant.create_collection(COLL, vectors_config={"size": 64, "distance": "Cosine"})
    points = [
        PointStruct(id=f"aaaaaaaa-0000-4000-8000-{i:012d}",
                    vector=[0.5] * 64,
                    payload={"text": f"legacy {i}", "_content_hash": content_hash(f"legacy {i}")})
        for i in range(20)
    ]
    points.append(PointStruct(
        id="bbbbbbbb-9999-4999-8999-999999999999", vector=[0.5] * 64,
        payload={"text": text, "_embed_model": "other-model:1b",
                 "_content_hash": content_hash(text)}))
    local_qdrant.upsert(COLL, points=points, wait=True)
    return text


def test_legacy_front_cannot_mask_recorded_model(env, monkeypatch):
    """20 legacy points before the recorded model-A point: the exists-filter
    check must still catch the mismatch."""
    from vector_memory import reembed as fp
    from vector_memory.errors import ConfigError

    with pytest.raises(ConfigError, match="other-model:1b"):
        fp.check_collection_model(COLL, "qwen3-embedding:8b")


def test_matching_model_passes(env, monkeypatch):
    from vector_memory import reembed as fp

    fp.check_collection_model(COLL, "other-model:1b")  # no raise


def test_all_legacy_passes(env, monkeypatch):
    """A collection of ONLY legacy points (no recorded model) never fails."""
    from vector_memory import reembed as fp

    fp.check_collection_model("vm_fp_legacy_only", "qwen3-embedding:8b")
