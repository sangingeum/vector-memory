"""Caller-provided summaries (no LLM inside the tool)."""

from __future__ import annotations

from tests import local_backend as _lb
from tests.local_backend import HashEmbedder

local_qdrant = _lb.local_qdrant


def test_summary_stored_and_preferred_in_brief(local_qdrant, monkeypatch):
    from qdrant_client.models import PointStruct

    from vector_memory import core
    from vector_memory import embedding as emb
    from vector_memory import store as st

    monkeypatch.setattr(emb, "ollama_client", HashEmbedder(dim=64))
    monkeypatch.setattr(st, "qdrant", local_qdrant)
    import time as _t

    local_qdrant.create_collection("vm_sum", vectors_config={"size": 64, "distance": "Cosine"})
    local_qdrant.upsert("vm_sum", points=[
        PointStruct(id="11111111-1111-1111-1111-111111111111",
                    vector=[0.9] * 64,
                    payload={"text": "long text " * 200, "summary": "concise caller summary",
                             "_created_ts": _t.time()}),
    ], wait=True)
    out = core.search_memory("anything", limit=1, collection="vm_sum", brief=True)
    assert "concise caller summary" in out


def test_tool_never_calls_llm(local_qdrant, monkeypatch):
    """The embedder is the only model call; summary is passed through verbatim."""
    from vector_memory import core
    from vector_memory import embedding as emb
    from vector_memory import store as st

    embedder = HashEmbedder(dim=64)
    monkeypatch.setattr(emb, "ollama_client", embedder)
    monkeypatch.setattr(st, "qdrant", local_qdrant)
    out = core.save_memory("with summary", {"summary": "my own summary"}, collection="vm_sum2")
    assert "Memory saved" in out
    pts, _ = local_qdrant.scroll("vm_sum2", limit=1, with_payload=True)
    assert pts[0].payload["summary"] == "my own summary"
