"""Retry/batching/write-consistency tests (VM-11)."""

from __future__ import annotations

import pytest

from tests import local_backend as _lb
from tests.local_backend import HashEmbedder

local_qdrant = _lb.local_qdrant


def test_transient_failure_retries(monkeypatch):
    from vector_memory import embedding as emb

    calls = {"n": 0}

    class Flaky:
        def embed(self, model=None, input=None, **kw):
            calls["n"] += 1
            if calls["n"] < 3:
                raise ConnectionError("connection refused")
            return {"embeddings": [[0.1] * 8]}

    monkeypatch.setattr(emb, "ollama_client", Flaky())
    monkeypatch.setattr(emb, "EMBED_MODEL", "qwen3-embedding:8b")
    vec = emb.embed("retry me")
    assert vec == [0.1] * 8
    assert calls["n"] == 3  # failed twice, succeeded third


def test_model_not_found_fails_fast(monkeypatch):
    from vector_memory import embedding as emb

    class NoModel:
        def embed(self, model=None, input=None, **kw):
            raise RuntimeError("model 'nope' not found, try pulling it first")

    monkeypatch.setattr(emb, "ollama_client", NoModel())
    monkeypatch.setattr(emb, "EMBED_MODEL", "nope")
    with pytest.raises(emb.ModelNotFoundError, match="ollama pull"):
        emb.embed("x")


def test_non_transient_error_not_retried(monkeypatch):
    from vector_memory import embedding as emb

    calls = {"n": 0}

    class Broken:
        def embed(self, model=None, input=None, **kw):
            calls["n"] += 1
            raise ValueError("bad input shape")

    monkeypatch.setattr(emb, "ollama_client", Broken())
    with pytest.raises(ValueError):
        emb.embed("x")
    assert calls["n"] == 1  # no retry on a non-transient error


def test_save_memories_batched(local_qdrant, monkeypatch):
    from vector_memory import core
    from vector_memory import embedding as emb

    embedder = HashEmbedder(dim=64)
    monkeypatch.setattr(emb, "ollama_client", embedder)
    monkeypatch.setattr("vector_memory.store.qdrant", local_qdrant)
    texts = [f"batch item {i}" for i in range(70)]
    out = core.save_memories(texts, {}, collection="vm_batch")
    assert out.startswith("Saved 70 memories")
    # EMBED_BATCH=32: 3 embed calls (32+32+6).
    batch_calls = [c for c in embedder.calls if len(c) > 1]
    assert len(batch_calls) == 3
    assert len(batch_calls[0]) == 32 and len(batch_calls[-1]) == 6


def test_read_your_write_local(local_qdrant, monkeypatch):
    from vector_memory import core
    from vector_memory import embedding as emb

    monkeypatch.setattr(emb, "ollama_client", HashEmbedder(dim=64))
    monkeypatch.setattr("vector_memory.store.qdrant", local_qdrant)
    core.save_memory("read your write doc", {}, collection="vm_ryw")
    out = core.search_memory("read your write doc", collection="vm_ryw")
    assert "read your write doc" in out
