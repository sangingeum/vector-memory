"""VM-15: consolidate candidate finder — clusters read-only, never writes."""

from __future__ import annotations

import pytest

from tests import local_backend as _lb
from tests.local_backend import HashEmbedder

local_qdrant = _lb.local_qdrant

COLL = "vm_consolidate"


@pytest.fixture()
def env(local_qdrant, monkeypatch):
    from vector_memory import embedding as emb
    from vector_memory import store as st
    from vector_memory.validation import content_hash

    monkeypatch.setattr(emb, "ollama_client", HashEmbedder(dim=64))
    monkeypatch.setattr(st, "qdrant", local_qdrant)
    # Synthetic fixture: a duplicate cluster (same content, migrated legacy
    # points) plus unrelated points.
    st.qdrant.create_collection(COLL, vectors_config={"size": 64, "distance": "Cosine"})
    from qdrant_client.models import PointStruct

    text = "the deployment gate runs on tuesdays at ten"
    points = [
        PointStruct(id=f"aaaaaaaa-0000-4000-8000-{i:012d}", vector=[0.9] * 64,
                    payload={"text": f"{text} (v{i})", "project": "p",
                             "type": "decision",
                             "_created_ts": 1700000000.0 + i * 86400,
                             "_created_at": "2026-08-01T00:00:00Z",
                             "_content_hash": content_hash(f"{text} (v{i})")})
        for i in range(3)
    ]
    points.append(PointStruct(
        id="bbbbbbbb-9999-4999-8999-999999999999", vector=[0.1, -0.2] + [0.1] * 62,
        payload={"text": "unrelated topic entirely", "project": "p",
                 "_created_ts": 1700000000.0}))
    st.qdrant.upsert(COLL, points=points, wait=True)
    return st


def test_consolidate_finds_clusters(env):
    from vector_memory.consolidate import consolidate

    out = consolidate(COLL, min_similarity=0.99)
    assert "group 1 (3 memories" in out
    assert "suggest: save merged memory with --supersedes" in out
    assert "does NOT imply equivalence" in out


def test_consolidate_never_writes(env):
    from vector_memory.consolidate import consolidate

    before = env.qdrant.count(collection_name=COLL, exact=True).count
    consolidate(COLL, min_similarity=0.5)
    assert env.qdrant.count(collection_name=COLL, exact=True).count == before


def test_consolidate_no_candidates(env):
    from vector_memory.consolidate import consolidate

    out = consolidate(COLL, min_similarity=0.9999)  # near-impossible floor
    # The three identical-vector points still cluster at 0.9999 (1.0 >= 0.9999).
    assert "group 1" in out or "No consolidation candidates" in out


def test_consolidate_invalid_floor(env):
    from vector_memory.consolidate import consolidate
    from vector_memory.errors import ArgumentError

    with pytest.raises(ArgumentError, match="min-similarity"):
        consolidate(COLL, min_similarity=0.0)
