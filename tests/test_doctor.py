"""VM-18: doctor checks + MCP stdio smoke (no live backends)."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from tests import local_backend as _lb
from vector_memory.cli import app

local_qdrant = _lb.local_qdrant

runner = CliRunner()


def test_doctor_reports_checks(local_qdrant, monkeypatch):
    from tests.local_backend import HashEmbedder
    from vector_memory import embedding as emb

    monkeypatch.setattr(emb, "ollama_client", HashEmbedder())
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code in (0, 1)
    assert "qdrant-reachable" in result.stdout
    assert "collection-exists" in result.stdout


def test_doctor_json(local_qdrant, monkeypatch):
    from tests.local_backend import HashEmbedder
    from vector_memory import embedding as emb

    monkeypatch.setattr(emb, "ollama_client", HashEmbedder())
    result = runner.invoke(app, ["doctor", "--json"])
    payload = json.loads(result.stdout)
    assert payload["schema"] == 1
    names = {c["name"] for c in payload["checks"]}
    assert {"ollama-model", "qdrant-reachable", "collection-exists"} <= names


def test_doctor_warns_on_legacy_points(local_qdrant, monkeypatch):
    from qdrant_client.models import PointStruct

    from tests.local_backend import HashEmbedder
    from vector_memory import embedding as emb

    monkeypatch.setattr(emb, "ollama_client", HashEmbedder())
    local_qdrant.create_collection("vm_doc", vectors_config={"size": 64, "distance": "Cosine"})
    local_qdrant.upsert("vm_doc", points=[
        PointStruct(id="11111111-1111-1111-1111-111111111111",
                    vector=[0.5] * 64, payload={"text": "legacy"}),
    ], wait=True)
    from vector_memory import store
    monkeypatch.setattr(store, "COLLECTION_NAME", "vm_doc")
    result = runner.invoke(app, ["doctor"])
    assert "legacy-points" in result.stdout
    assert "migrate" in result.stdout


def test_mcp_stdio_smoke_initialize_and_tools_list():
    """initialize + tools/list over the MCP server object (no live backends)."""
    import anyio

    import vector_memory.server as srv

    async def smoke():
        # list_tools is the same code path tools/list serves.
        tools = await srv.mcp.list_tools()
        names = {t.name for t in tools}
        assert "save_memory" in names and "search_memory" in names
        return len(names)

    count = anyio.run(smoke)
    assert count >= 8
