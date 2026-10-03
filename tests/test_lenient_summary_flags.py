"""B-4: --lenient and --summary CLI flags actually exist and work."""

from __future__ import annotations

from typer.testing import CliRunner

from tests import local_backend as _lb
from vector_memory.cli import app

local_qdrant = _lb.local_qdrant

runner = CliRunner()


def test_lenient_flag_warns_and_saves(local_qdrant):
    result = runner.invoke(app, ["save", "lenient doc", "--metadata", "{bad json",
                                 "--collection", "vm_lenient", "--lenient"])
    assert result.exit_code == 0
    assert "Memory saved" in result.stdout
    assert "Warning: failed to parse metadata JSON" in result.stdout


def test_summary_flag_merges_into_metadata(local_qdrant):
    from vector_memory import store as st

    result = runner.invoke(app, ["save", "summary doc", "--summary", "concise summary",
                                 "--collection", "vm_summary"])
    assert result.exit_code == 0
    pts, _ = st.qdrant.scroll("vm_summary", limit=1, with_payload=True)
    assert pts[0].payload["summary"] == "concise summary"


def test_summary_flag_wins_over_metadata(local_qdrant):
    from vector_memory import store as st

    result = runner.invoke(app, ["save", "summary wins", "--summary", "flag value",
                                 "--metadata", '{"summary": "json value"}',
                                 "--collection", "vm_summary2"])
    assert result.exit_code == 0
    pts, _ = st.qdrant.scroll("vm_summary2", limit=1, with_payload=True)
    assert pts[0].payload["summary"] == "flag value"
