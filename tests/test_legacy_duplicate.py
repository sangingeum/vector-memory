"""B-3: legacy-duplicate catch — migrated uuid4 points must not be duplicated."""

from __future__ import annotations

import pytest

from tests import local_backend as _lb
from tests.local_backend import HashEmbedder

local_qdrant = _lb.local_qdrant

COLL = "vm_legacy_dup"


@pytest.fixture()
def env(local_qdrant, monkeypatch):
    from qdrant_client.models import PointStruct

    from vector_memory import embedding as emb
    from vector_memory import store as st
    from vector_memory.validation import content_hash

    monkeypatch.setattr(emb, "ollama_client", HashEmbedder(dim=64))
    monkeypatch.setattr(st, "qdrant", local_qdrant)
    # Simulate a migrated legacy point: uuid4 id + _content_hash stamped.
    text = "Postgres runs on port 5432 for staging"
    local_qdrant.create_collection(COLL, vectors_config={"size": 64, "distance": "Cosine"})
    local_qdrant.upsert(COLL, points=[PointStruct(
        id="aaaaaaaa-1111-4111-8111-111111111111",  # uuid4-shaped legacy id
        vector=[0.5] * 64,
        payload={"text": text, "project": "p", "_content_hash": content_hash(text)},
    )], wait=True)
    return text


def test_re_save_of_migrated_legacy_text_refreshes(env):
    from vector_memory import core
    from vector_memory import store as st

    text = env
    out = core.save_memory(text, {"project": "p", "v": 2}, collection=COLL)
    assert "duplicate of existing; updated" in out
    assert "aaaaaaaa-1111-4111-8111-111111111111" in out  # the LEGACY id
    pts, _ = st.qdrant.scroll(COLL, limit=10, with_payload=True)
    assert len(pts) == 1  # no second point created
    assert pts[0].payload["v"] == 2  # metadata refreshed in place


def test_find_legacy_duplicate_helper(env):
    from vector_memory import dedupe
    from vector_memory.validation import content_hash

    assert dedupe.find_legacy_duplicate(
        COLL, content_hash(env)) == "aaaaaaaa-1111-4111-8111-111111111111"
    assert dedupe.find_legacy_duplicate(COLL, content_hash("no such text")) is None
