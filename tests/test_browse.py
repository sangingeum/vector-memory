"""Read/browse operations tests (get/list/count/stats/values)."""

from __future__ import annotations

import pytest

from tests import local_backend as _lb
from tests.local_backend import HashEmbedder

local_qdrant = _lb.local_qdrant

COLL = "vm_browse"


@pytest.fixture()
def seeded(local_qdrant, monkeypatch):
    from vector_memory import core

    embedding_module = HashEmbedder()
    monkeypatch.setattr("vector_memory.embedding.ollama_client", embedding_module)
    monkeypatch.setattr("vector_memory.store.qdrant", local_qdrant)
    local_qdrant.create_collection(COLL, vectors_config={"size": 64, "distance": "Cosine"})
    for i in range(30):
        core.save_memory(f"browse doc number {i}", {"project": f"p{i % 3}",
                                                    "type": "decision" if i % 2 else "fact",
                                                    "tags": [f"t{i % 4}"]},
                         collection=COLL)
    return local_qdrant


def _ids(out: str) -> list[str]:
    return [line.split("ID: ")[1].split("]")[0] for line in out.splitlines() if "ID: " in line]


def test_get_full_text_no_embedding(seeded, monkeypatch):
    from vector_memory import browse
    from vector_memory.embedding import ollama_client

    calls_before = len(ollama_client.calls)
    pts, _ = seeded.scroll(COLL, limit=1, with_payload=True)
    pid = pts[0].id
    out = browse.get_memory([pid], COLL)
    assert "text: browse doc number" in out
    assert len(ollama_client.calls) == calls_before  # no embedding call


def test_get_missing_reports_missing(seeded):
    from vector_memory import browse

    pts, _ = seeded.scroll(COLL, limit=1, with_payload=True)
    pid = pts[0].id
    out = browse.get_memory([pid, "22222222-2222-2222-2222-222222222222"], COLL)
    assert "missing:" in out


def test_list_pagination_120(seeded):
    from vector_memory import browse

    page1 = browse.list_memories(COLL, limit=25)
    assert "Listed 25/" in page1
    assert "next-cursor" in page1
    ids1 = _ids(page1)
    assert len(ids1) == 25
    cursor = next(line.split("next-cursor: ")[1].split(" ")[0]
                  for line in page1.splitlines() if "next-cursor" in line)
    page2 = browse.list_memories(COLL, limit=25, cursor=cursor)
    ids2 = _ids(page2)
    assert not set(ids1) & set(ids2)


def test_list_order_by_deterministic(seeded):
    from vector_memory import browse

    out1 = browse.list_memories(COLL, limit=5, order_by="created")
    out2 = browse.list_memories(COLL, limit=5, order_by="created")
    assert _ids(out1) == _ids(out2)


def test_count_matches_ground_truth(seeded):
    from vector_memory import browse

    out = browse.count_memories(COLL)
    assert "Count: 30" in out
    out2 = browse.count_memories(COLL, project="p0")
    assert "Count: 10" in out2


def test_stats_equals_ground_truth(seeded):
    from vector_memory import browse

    out = browse.collection_stats(COLL)
    assert "decision=15" in out and "fact=15" in out
    assert "p0=10" in out and "p1=10" in out and "p2=10" in out
    assert "active=30" in out


def test_values_discover_schema(seeded):
    from vector_memory import browse

    out = browse.field_values("project", COLL)
    assert "p0 (10)" in out and "p1 (10)" in out
    out2 = browse.field_values("tags", COLL)
    assert "t0" in out2


def test_values_invalid_field(seeded):
    from vector_memory import browse
    from vector_memory.errors import ArgumentError

    with pytest.raises(ArgumentError, match="project, type, tags, source"):
        browse.field_values("bogus", COLL)
