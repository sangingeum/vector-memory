"""VM-14: long-text chunking (save --chunk, search --collapse-groups)."""

from __future__ import annotations

import pytest

from tests import local_backend as _lb
from tests.local_backend import HashEmbedder

local_qdrant = _lb.local_qdrant

COLL = "vm_chunk"
OVER_LIMIT = "x" * 20100


@pytest.fixture()
def env(local_qdrant, monkeypatch):
    from vector_memory import embedding as emb
    from vector_memory import store as st

    monkeypatch.setattr(emb, "ollama_client", HashEmbedder(dim=64))
    monkeypatch.setattr(st, "qdrant", local_qdrant)
    return st


def test_split_text_sentence_boundaries():
    from vector_memory.chunking import split_text

    text = "첫 번째 문장입니다. 두 번째 문장입니다. Third sentence! Fourth?\nNew paragraph."
    chunks = split_text(text, chunk_chars=40, overlap=0)
    assert all(len(c) <= 40 for c in chunks)
    assert len(chunks) >= 2
    # Hangul sentence kept with its terminator, in order.
    assert chunks[0].startswith("첫 번째 문장입니다.")


def test_split_text_oversized_single_sentence_hard_split():
    from vector_memory.chunking import split_text

    chunks = split_text("y" * 5000, chunk_chars=1500, overlap=150)
    assert len(chunks) >= 3
    assert all(len(c) <= 1500 for c in chunks)


def test_oversized_text_rejected_without_chunk(env):
    from vector_memory import core
    from vector_memory.errors import ArgumentError

    with pytest.raises(ArgumentError, match="too long"):
        core.save_memory(OVER_LIMIT, {}, collection=COLL)


def test_save_chunk_creates_group(env):
    from vector_memory import core
    from vector_memory import store as st
    from vector_memory.payload import CHUNK_INDEX, GROUP_ID

    out = core.save_memory(("문단입니다. " * 4000), {"project": "p"},
                           collection=COLL, chunk=True, chunk_chars=1500)
    assert "chunks (group:" in out
    group_id = out.split("group: ")[1].split("…")[0]
    pts, _ = st.qdrant.scroll(COLL, limit=100, with_payload=True)
    assert len(pts) >= 2
    group_points = [p for p in pts if p.payload.get(GROUP_ID, "").startswith(group_id[:8])]
    assert len(group_points) == len(pts)
    indexes = sorted(p.payload[CHUNK_INDEX] for p in group_points)
    assert indexes == list(range(len(group_points)))


def test_collapse_groups_returns_best_per_group(env):
    from qdrant_client.models import PointStruct

    from vector_memory import core
    from vector_memory import store as st

    st.qdrant.create_collection("vm_collapse", vectors_config={"size": 64, "distance": "Cosine"})
    st.qdrant.upsert("vm_collapse", points=[
        # g1: two identical vectors — same score; collapse keeps the FIRST
        # encountered (score order) and drops the second.
        PointStruct(id="11111111-1111-4111-8111-111111111111", vector=[0.9] * 64,
                    payload={"text": "chunk one", "_group_id": "g1", "_chunk_index": 0}),
        PointStruct(id="22222222-2222-4222-8222-222222222222", vector=[0.9] * 64,
                    payload={"text": "chunk two", "_group_id": "g1", "_chunk_index": 1}),
        PointStruct(id="33333333-3333-4333-8333-333333333333", vector=[0.3] * 64,
                    payload={"text": "ungrouped other", "_group_id": "g2", "_chunk_index": 0}),
    ], wait=True)

    all_out = core.search_memory("anything", limit=5, collection="vm_collapse")
    assert len(_ids(all_out)) == 3
    collapsed = core.search_memory("anything", limit=5, collection="vm_collapse",
                                   collapse_groups=True)
    ids = _ids(collapsed)
    assert len(ids) == 2  # one per group
    assert len(set(ids)) == 2
    # Exactly one of the two twins survives.
    assert bool({"11111111-1111-4111-8111-111111111111"} & set(ids)) != \
        bool({"22222222-2222-4222-8222-222222222222"} & set(ids))


def _ids(out: str) -> list[str]:
    return [line.split("ID: ")[1].split("]")[0]
            for line in out.splitlines() if "ID: " in line]
