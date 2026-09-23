"""Tests for the CLI save convenience options (--project/--type/--tags).

These map to the same metadata keys the skill docs have always described;
explicit options win over same-key values given via --metadata JSON.
"""

from __future__ import annotations

import mcp_server as s
from tests.test_server import env  # reuse the fake-clients fixture


def _stored_payloads(fq) -> list[dict]:
    return list(fq.collections[s.COLLECTION_NAME]["points"].values())


def test_save_explicit_options_become_metadata(env):
    fq, _ = env
    s.save_memory("cli doc", project="athena", type="decision", tags=["a", "b"])
    (stored,) = _stored_payloads(fq)
    assert stored["project"] == "athena"
    assert stored["type"] == "decision"
    assert stored["tags"] == ["a", "b"]


def test_save_many_explicit_options_apply_to_all(env):
    fq, _ = env
    out = s.save_memories(["x", "y"], project="p", type="t", tags=["z"])
    assert out.startswith("Saved 2 memories")
    assert all(
        p["project"] == "p" and p["type"] == "t" and p["tags"] == ["z"]
        for p in _stored_payloads(fq)
    )


def test_explicit_options_win_over_metadata_json(env):
    fq, _ = env
    s.save_memory(
        "cli doc",
        metadata='{"project": "from-json", "keep": 1}',
        project="explicit",
        tags=["t1"],
    )
    (stored,) = _stored_payloads(fq)
    assert stored["project"] == "explicit"
    assert stored["keep"] == 1
    assert stored["tags"] == ["t1"]


def test_empty_options_leave_metadata_untouched(env):
    fq, _ = env
    s.save_memory("cli doc", metadata='{"project": "only-json"}')
    (stored,) = _stored_payloads(fq)
    assert stored == {"project": "only-json", "text": "cli doc"}


def test_cli_save_options_end_to_end(env):
    """Typer wiring: --project/--type/--tags reach core as merged metadata."""
    fq, _ = env

    from typer.testing import CliRunner

    from vector_memory.cli import app

    result = CliRunner().invoke(app, [
        "save", "wired doc",
        "--project", "wire-p", "--type", "wire-t",
        "--tags", "w1", "--tags", "w2",
    ])
    assert result.exit_code == 0, result.output
    assert "Memory saved" in result.output
    (stored,) = _stored_payloads(fq)
    assert stored["project"] == "wire-p"
    assert stored["type"] == "wire-t"
    assert stored["tags"] == ["w1", "w2"]
    assert stored["text"] == "wired doc"