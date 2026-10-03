"""Search enhancement tests (min-score, recency, MMR, brief/compact).

The hash embedder is content-addressed: querying with a doc's own text
matches that doc's vector exactly; the "weak" doc uses different text and
scores far lower. Local-mode scores are meaningful here.
"""

from __future__ import annotations

import time

import pytest

from tests import local_backend as _lb
from tests.local_backend import HashEmbedder

local_qdrant = _lb.local_qdrant

COLL = "vm_enh"
TWIN_TEXT = "fresh relevant doc"
WEAK_TEXT = "weak hit far away"
TWIN_A = "11111111-1111-1111-1111-111111111111"
TWIN_B = "22222222-2222-2222-2222-222222222222"
WEAK = "33333333-3333-3333-3333-333333333333"


@pytest.fixture()
def seeded(local_qdrant, monkeypatch):
    from qdrant_client.models import PointStruct

    from vector_memory import embedding as emb
    from vector_memory import store as st

    embedder = HashEmbedder()
    monkeypatch.setattr(emb, "ollama_client", embedder)
    monkeypatch.setattr(st, "qdrant", local_qdrant)
    local_qdrant.create_collection(COLL, vectors_config={"size": 64, "distance": "Cosine"})
    now = time.time()
    old = now - 365 * 86400
    twin_vec = embedder.embed(model=None, input=TWIN_TEXT)["embeddings"][0]
    weak_vec = embedder.embed(model=None, input=WEAK_TEXT)["embeddings"][0]
    local_qdrant.upsert(COLL, points=[
        PointStruct(id=TWIN_A, vector=twin_vec,
                    payload={"text": TWIN_TEXT, "project": "p", "type": "fact",
                             "_created_ts": now, "_created_at": "2026-10-01T00:00:00Z"}),
        PointStruct(id=TWIN_B, vector=twin_vec,
                    payload={"text": "stale but nearly identical twin doc", "project": "p",
                             "type": "fact",
                             "_created_ts": old, "_created_at": "2025-10-01T00:00:00Z"}),
        PointStruct(id=WEAK, vector=weak_vec,
                    payload={"text": WEAK_TEXT, "project": "p", "type": "fact",
                             "_created_ts": now, "_created_at": "2026-10-01T00:00:00Z"}),
    ], wait=True)
    return local_qdrant


def test_min_score_drops_weak_hits(seeded):
    from vector_memory import core

    out_all = core.search_memory(TWIN_TEXT, limit=5, collection=COLL)
    assert WEAK_TEXT in out_all
    out = core.search_memory(TWIN_TEXT, limit=5, collection=COLL, min_score=0.0)
    assert WEAK_TEXT not in out
    assert TWIN_TEXT in out


def test_recency_weight_promotes_fresh(seeded):
    from vector_memory import core

    weighted = core.search_memory(TWIN_TEXT, limit=2, collection=COLL, recency_weight=0.9)
    ids = _ids(weighted)
    assert ids[0] == TWIN_A  # fresh twin ranks above the stale one
    # Unweighted (relevance only): both twins score identically; recency breaks the tie.
    flat = core.search_memory(TWIN_TEXT, limit=2, collection=COLL)
    assert len(_ids(flat)) == 2


def test_mmr_diversifies_duplicates(seeded):
    from vector_memory import core

    plain = core.search_memory(TWIN_TEXT, limit=2, collection=COLL)
    diverse = core.search_memory(TWIN_TEXT, limit=2, collection=COLL, mmr=0.3)
    plain_ids = set(_ids(plain))
    diverse_ids = set(_ids(diverse))
    assert plain_ids == {TWIN_A, TWIN_B}  # relevance-only takes both twins
    # MMR with the identical-vector twins: only one twin + the weak hit.
    assert len(diverse_ids & {TWIN_A, TWIN_B}) == 1
    assert WEAK in diverse_ids


def test_brief_prefers_summary(seeded):
    from qdrant_client.models import PointStruct

    from vector_memory import core

    long_text = "x" * 2000
    twin_vector = seeded.retrieve(COLL, ids=[TWIN_A], with_vectors=True)[0].vector
    seeded.upsert(COLL, points=[
        PointStruct(id="44444444-4444-4444-4444-444444444444",
                    vector=twin_vector,
                    payload={"text": long_text, "summary": "short summary here",
                             "_created_ts": time.time()}),
    ], wait=True)
    out = core.search_memory(TWIN_TEXT, limit=5, collection=COLL, brief=True)
    brief_line = next(line for line in out.splitlines() if "44444444" in line)
    assert "short summary here" in brief_line
    assert long_text not in out  # full text not printed


def test_brief_output_smaller(seeded):
    from qdrant_client.models import PointStruct

    from vector_memory import core

    long_text = "long content " * 100  # > 300 chars
    twin_vector = seeded.retrieve(COLL, ids=[TWIN_A], with_vectors=True)[0].vector
    seeded.upsert(COLL, points=[
        PointStruct(id="55555555-5555-5555-5555-555555555555", vector=twin_vector,
                    payload={"text": long_text, "_created_ts": time.time()}),
    ], wait=True)
    full = core.search_memory(TWIN_TEXT, limit=3, collection=COLL)
    brief = core.search_memory(TWIN_TEXT, limit=3, collection=COLL, brief=True)
    assert len(brief) < len(full)


def test_compact_format(seeded):
    from vector_memory import core

    out = core.search_memory(TWIN_TEXT, limit=2, collection=COLL, output_format="compact")
    assert "metadata:" not in out
    line = out.splitlines()[1]
    assert line[:36]  # uuid-prefixed line


def test_defaults_unchanged(seeded):
    """No flags -> same text-mode line shape as before."""
    from vector_memory import core

    out = core.search_memory(TWIN_TEXT, limit=1, collection=COLL)
    line = next(line for line in out.splitlines() if line.startswith("- [ID:"))
    assert "[score: " in line and "metadata: " in line and "| content: " in line


def _ids(out: str) -> list[str]:
    ids = []
    for line in out.splitlines():
        if line.startswith("- [ID:"):
            ids.append(line.split("ID: ")[1].split("]")[0])
        elif len(line) > 38 and line[36:38] == "  ":
            ids.append(line[:36])
    return ids
