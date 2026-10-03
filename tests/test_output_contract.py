"""Output-contract tests (CLI stdout/stderr taxonomy, --json envelope)."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from tests import local_backend as _lb
from vector_memory.cli import app

local_qdrant = _lb.local_qdrant

runner = CliRunner()


def test_success_stdout_and_clean_stderr(local_qdrant):
    result = runner.invoke(app, ["save", "contract doc", "--collection", "vm_out"])
    assert result.exit_code == 0
    assert "Memory saved" in result.stdout
    assert "Error" not in result.stdout


def test_failure_one_stderr_line_exit_1(local_qdrant):
    result = runner.invoke(app, ["save", "doc", "--metadata", "{bad json",
                                 "--collection", "vm_out_err"])
    assert result.exit_code == 1
    assert result.stdout.strip() == ""
    err_lines = [line for line in result.stderr.splitlines() if line.strip()]
    assert len(err_lines) == 1
    assert err_lines[0].startswith("ArgumentError:")


def test_not_found_exit_1(local_qdrant):
    result = runner.invoke(
        app, ["delete", "00000000-0000-0000-0000-000000000000", "--collection", "vm_out_nf"])
    assert result.exit_code == 1
    err_lines = [line for line in result.stderr.splitlines() if line.strip()]
    assert err_lines and err_lines[0].startswith("NotFoundError:")


def test_json_envelope_success(local_qdrant):
    result = runner.invoke(app, ["save", "json doc", "--collection", "vm_out_json",
                                 "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["schema"] == 1
    assert payload["ok"] is True
    assert "Memory saved" in payload["result"]


def test_json_envelope_failure(local_qdrant):
    result = runner.invoke(app, ["delete", "00000000-0000-0000-0000-000000000000",
                                 "--collection", "vm_out_json", "--json"])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["schema"] == 1
    assert payload["ok"] is False
    assert payload["error_type"] == "NotFoundError"
