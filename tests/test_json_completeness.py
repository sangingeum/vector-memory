"""B-12: --json on the remaining commands + exit codes; B-15 doc-only (README).

Behavioral checks for the output-contract completion.
"""

from __future__ import annotations

import json

from typer.testing import CliRunner

from tests import local_backend as _lb
from tests.local_backend import HashEmbedder
from vector_memory.cli import app

local_qdrant = _lb.local_qdrant

runner = CliRunner()


def _setup(local_qdrant, monkeypatch):
    monkeypatch.setattr("vector_memory.embedding.ollama_client", HashEmbedder(dim=64))
    from vector_memory import core

    for i in range(3):
        core.save_memory(f"json doc {i}", {"project": "p"}, collection="vm_json_all")
    return core


def test_json_on_browse_commands(local_qdrant, monkeypatch):
    _setup(local_qdrant, monkeypatch)
    for args in (
        ["get", _one_id(), "--collection", "vm_json_all"],
        ["list", "--collection", "vm_json_all"],
        ["count", "--collection", "vm_json_all"],
        ["stats", "--collection", "vm_json_all"],
        ["values", "project", "--collection", "vm_json_all"],
    ):
        result = runner.invoke(app, args + ["--json"])
        assert result.exit_code == 0, (args, result.stdout)
        payload = json.loads(result.stdout)
        assert payload["schema"] == 1 and payload["ok"] is True


def _one_id() -> str:
    from vector_memory import store as st

    pts, _ = st.qdrant.scroll("vm_json_all", limit=1, with_payload=True)
    return str(pts[0].id)


def test_json_on_patch_delete_archive(local_qdrant, monkeypatch):
    _setup(local_qdrant, monkeypatch)
    pid = _one_id()
    for args in (
        ["patch", pid, "--set", '{"v": 9}', "--collection", "vm_json_all"],
        ["archive", pid, "--collection", "vm_json_all"],
        ["unarchive", pid, "--collection", "vm_json_all"],
        ["migrate", "--collection", "vm_json_all"],
        ["delete", pid, "--collection", "vm_json_all"],
    ):
        result = runner.invoke(app, args + ["--json"])
        assert result.exit_code == 0, (args, result.stdout)
        payload = json.loads(result.stdout)
        assert payload["schema"] == 1 and payload["ok"] is True


def test_exit_2_on_usage_error(local_qdrant, monkeypatch):
    """Typer emits exit 2 for CLI usage errors (bad flag values); confirm."""
    result = runner.invoke(app, ["search", "q", "--limit", "not-a-number"])
    assert result.exit_code == 2


def test_json_failure_envelope_on_delete_by_filter(local_qdrant, monkeypatch):
    _setup(local_qdrant, monkeypatch)
    # delete-by-filter with an empty filter is refused -> error envelope.
    result = runner.invoke(app, ["delete-by-filter", "--collection", "vm_json_all", "--json"])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["schema"] == 1 and payload["ok"] is False
    assert payload["error_type"] == "ArgumentError"
