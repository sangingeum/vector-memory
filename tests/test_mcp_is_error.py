"""B-2: MCP isError contract — an erroring tool must surface as isError."""

from __future__ import annotations

import pytest

import vector_memory.server as srv
from tests import local_backend as _lb
from tests.local_backend import HashEmbedder

local_qdrant = _lb.local_qdrant


@pytest.fixture()
def env(local_qdrant, monkeypatch):
    from vector_memory import embedding as emb
    from vector_memory import store as st

    monkeypatch.setattr(emb, "ollama_client", HashEmbedder(dim=64))
    monkeypatch.setattr(st, "qdrant", local_qdrant)


def test_erroring_tool_raises_for_fastmcp(env):
    """FastMCP marks isError only when the handler RAISES (not when it returns
    error text) — call() must propagate, not swallow."""
    with pytest.raises(Exception, match="does not exist"):
        srv.delete_memory("00000000-0000-0000-0000-000000000000", "missing-coll")


def test_call_propagates_typed_errors(env):
    from vector_memory import core
    from vector_memory.errors import NotFoundError

    with pytest.raises(NotFoundError):
        core.call(core.delete_memory, "00000000-0000-0000-0000-000000000000",
                  "missing-coll")


def test_call_classifies_untyped_backend_errors(env, monkeypatch):
    from vector_memory import core
    from vector_memory import store as st
    from vector_memory.errors import BackendError

    class Failing:
        def info(self):
            raise RuntimeError("connect refused")

        def __getattr__(self, name):
            raise RuntimeError("connect refused")

    monkeypatch.setattr(st, "qdrant", Failing())
    with pytest.raises(BackendError, match="connect refused"):
        core.call(core.list_collections)


def test_success_still_returns_result(env):
    from vector_memory import core

    out = core.call(core.list_collections)
    assert isinstance(out, str)
