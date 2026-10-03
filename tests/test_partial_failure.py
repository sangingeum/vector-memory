"""B-6: save_memories partial-failure contract — per-item accumulation."""

from __future__ import annotations

import pytest

from tests import local_backend as _lb
from tests.local_backend import HashEmbedder

local_qdrant = _lb.local_qdrant

COLL = "vm_partial"


@pytest.fixture()
def env(local_qdrant, monkeypatch):
    from vector_memory import embedding as emb
    from vector_memory import store as st

    monkeypatch.setattr(emb, "ollama_client", HashEmbedder(dim=64))
    monkeypatch.setattr(st, "qdrant", local_qdrant)
    return emb


def test_mocked_failure_at_batch_k_reports_partial(env, monkeypatch):
    """EMBED_BATCH=32; batch 0 fails, batches 1-2 (indices 32..69) succeed:
    the result must report saved 38/70; failed: [0..31] and the good points
    must actually be stored."""
    from vector_memory import core
    from vector_memory.errors import BackendError

    real_embed_many = env.embed_many
    calls = {"n": 0}

    def flaky_many(texts):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("connection refused")
        return real_embed_many(texts)

    monkeypatch.setattr(env, "embed_many", flaky_many)
    texts = [f"partial item {i}" for i in range(70)]
    with pytest.raises(BackendError) as excinfo:
        core.save_memories(texts, {}, collection=COLL)
    message = str(excinfo.value)
    assert message.startswith("saved 38/70; failed: [")
    assert message.endswith("31]")  # indices 0..31 failed
    # The good points landed despite the failed first batch.
    from vector_memory import store as st

    count = st.qdrant.count(collection_name=COLL, exact=True).count
    assert count == 38


def test_single_batch_failure_saves_nothing_reports_all(env, monkeypatch):
    from vector_memory import core
    from vector_memory import store as st
    from vector_memory.errors import BackendError

    def always_fail(texts):
        raise ConnectionError("connection refused")

    monkeypatch.setattr(env, "embed_many", always_fail)
    with pytest.raises(BackendError) as excinfo:
        core.save_memories(["a", "b", "c"], {}, collection=COLL)
    assert "saved 0/3; failed: [0, 1, 2]" in str(excinfo.value)
    assert st.qdrant.count(collection_name=COLL, exact=True).count == 0


def test_all_success_no_error(env):
    from vector_memory import core

    out = core.save_memories([f"ok {i}" for i in range(10)], {}, collection=COLL)
    assert out.startswith("Saved 10 memories")
