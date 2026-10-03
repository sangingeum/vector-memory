"""Multi-agent namespacing (VM_AGENT_ID author stamp + --agent filter)."""

from __future__ import annotations

import pytest

from tests import local_backend as _lb
from tests.local_backend import HashEmbedder

local_qdrant = _lb.local_qdrant

COLL = "vm_agent"


@pytest.fixture()
def env(local_qdrant, monkeypatch):
    from vector_memory import core
    from vector_memory import embedding as emb
    from vector_memory import store as st

    monkeypatch.setattr(emb, "ollama_client", HashEmbedder(dim=64))
    monkeypatch.setattr(st, "qdrant", local_qdrant)
    local_qdrant.create_collection(COLL, vectors_config={"size": 64, "distance": "Cosine"})
    return core, local_qdrant


def test_agent_id_stamped_on_save(env, monkeypatch):
    core, qdrant_client = env
    monkeypatch.setenv("VM_AGENT_ID", "athena")
    out = core.save_memory("authored note", {}, collection=COLL)
    assert "Memory saved" in out
    pts, _ = qdrant_client.scroll(COLL, limit=1, with_payload=True)
    assert pts[0].payload["_agent"] == "athena"


def test_no_agent_id_no_stamp(env, monkeypatch):
    core, qdrant_client = env
    monkeypatch.delenv("VM_AGENT_ID", raising=False)
    core.save_memory("anonymous note", {}, collection=COLL)
    pts, _ = qdrant_client.scroll(COLL, limit=10, with_payload=True)
    assert all("_agent" not in p.payload for p in pts)


def test_agent_filter(env, monkeypatch):
    core, _ = env
    monkeypatch.setenv("VM_AGENT_ID", "athena")
    core.save_memory("athena wrote this", {}, collection=COLL)
    monkeypatch.setenv("VM_AGENT_ID", "vera")
    core.save_memory("vera wrote this", {}, collection=COLL)
    out = core.search_memory("wrote this", collection=COLL, agent="vera")
    assert "vera wrote this" in out
    assert "athena wrote this" not in out
