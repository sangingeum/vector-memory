"""Sensitive-content guard tests (secret-detection work item)."""

from __future__ import annotations

import pytest

from tests import local_backend as _lb

local_qdrant = _lb.local_qdrant

POSITIVES = [
    ("aws_access_key", "credentials AKIAIOSFODNN7EXAMPLE for the deploy user"),
    ("private_key_block", "key starts -----BEGIN RSA PRIVATE KEY----- here"),
    ("github_token", "token ghp_abcdefghijklmnopqrstuvwxyz012345 in env"),
    ("slack_token", "slack bot xoxb-123456789-abcdef"),
    ("jwt", "auth header eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkifQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJVadQssw5c"),
    ("password_assignment", "the password is hunter2hunter2"),
    ("api_key_assignment", "api_key = sk-1234567890abcdef1234"),
]

NEGATIVES = [
    "the password policy requires 12 characters",  # no assignment with a value
    "we rotated the API keys last Tuesday",
    "owner prefers PostgreSQL on port 5432",
    "deploy uses AWS IAM roles, no static keys",
]


@pytest.mark.parametrize("rule,text", POSITIVES)
def test_secret_rule_blocks_write(rule, text, local_qdrant):
    from vector_memory import core
    from vector_memory.errors import SensitiveContentError

    with pytest.raises(SensitiveContentError, match=f"rule: {rule}"):
        core.save_memory(text, {}, collection="vm_sec")
    # The matched value is never echoed, and nothing was written: the
    # collection was never even created (save fails before any write).
    if local_qdrant.collection_exists("vm_sec"):
        pts, _ = local_qdrant.scroll("vm_sec", limit=10, with_payload=True)
        assert len(pts) == 0


@pytest.mark.parametrize("text", NEGATIVES)
def test_benign_text_passes(text, local_qdrant):
    from vector_memory import core

    out = core.save_memory(text, {}, collection="vm_sec_ok")
    assert "Memory saved" in out


def test_allow_sensitive_override(local_qdrant, monkeypatch):
    from vector_memory import core

    monkeypatch.setenv("VM_ALLOW_SENSITIVE", "1")
    out = core.save_memory("api_key = sk-1234567890abcdef1234", {},
                           collection="vm_sec_override")
    assert "Memory saved" in out


def test_update_also_guarded(local_qdrant):
    from vector_memory import core
    from vector_memory.errors import SensitiveContentError

    out = core.save_memory("safe text", {}, collection="vm_sec_upd")
    pid = out.split("ID: ")[1].split(",")[0].strip()
    with pytest.raises(SensitiveContentError):
        core.update_memory(pid, "now with AKIAIOSFODNN7EXAMPLE", collection="vm_sec_upd")
