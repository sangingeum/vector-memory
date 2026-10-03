"""B-16: convenience flags on list/count/delete-by-filter + MCP search parity."""

from __future__ import annotations

import pytest

from tests import local_backend as _lb
from tests.local_backend import HashEmbedder

local_qdrant = _lb.local_qdrant

COLL = "vm_flags"


@pytest.fixture()
def env(local_qdrant, monkeypatch):
    from vector_memory import core
    from vector_memory import embedding as emb
    from vector_memory import store as st

    monkeypatch.setattr(emb, "ollama_client", HashEmbedder(dim=64))
    monkeypatch.setattr(st, "qdrant", local_qdrant)
    for i in range(4):
        core.save_memory(f"flag doc {i}",
                         {"project": "p", "type": "decision" if i % 2 else "fact",
                          "tags": [f"t{i % 2}"]},
                         collection=COLL)
    return core


def test_count_convenience_flags(env):
    from vector_memory import browse

    assert "Count: 4" in browse.count_memories(COLL)
    assert "Count: 2" in browse.count_memories(COLL, type="decision")
    assert "Count: 2" in browse.count_memories(COLL, tag=["t0"])
    assert "Count: 0" in browse.count_memories(COLL, type="note")


def test_list_convenience_flags(env):
    from vector_memory import browse

    out = browse.list_memories(COLL, type="decision")
    assert out.count("- [ID:") == 2
    out2 = browse.list_memories(COLL, tag=["t0"])
    assert out2.count("- [ID:") == 2


def test_delete_by_filter_convenience_flags(env):
    from vector_memory import bulk
    from vector_memory import store as st

    out = bulk.delete_by_filter(None, COLL, type="fact")  # dry-run default
    assert "would delete 2" in out
    out2 = bulk.delete_by_filter(None, COLL, tag=["t0"], no_dry_run=True, yes=True)
    assert "Deleted 2" in out2
    assert st.qdrant.count(collection_name=COLL, exact=True).count == 2


def test_mcp_search_parity_signatures():
    """MCP search_memory exposes the landed enhancements."""
    import inspect

    import vector_memory.server as srv

    params = inspect.signature(srv.search_memory).parameters
    for name in ("min_score", "recency_weight", "mmr", "brief", "max_chars",
                 "output_format", "since", "before", "tag", "type", "source",
                 "agent", "include_inactive"):
        assert name in params, f"MCP search missing {name}"
