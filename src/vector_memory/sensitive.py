"""Sensitive-content guard (high-confidence secret detection before saving).

Memories are persistent and re-injected into prompts: a secret saved once
leaks forever. Matches fail with ``SensitiveContentError`` and nothing is
written; the matched VALUE is never echoed. Override via
``--allow-sensitive`` / ``VM_ALLOW_SENSITIVE=1``.
"""

from __future__ import annotations

import os
import re

from .errors import SensitiveContentError

# (rule name, compiled pattern). High-confidence only: a generic word like
# "password" alone must not block legitimate memory text; the value part is
# what makes these rules confident.
RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("aws_access_key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("private_key_block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("github_token", re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}")),
    ("slack_token", re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")),
    ("password_assignment", re.compile(
        r"(?i)\bpassword\s*(?:is|[:=])\s*['\"]?[^\s'\"]{8,}")),
    ("api_key_assignment", re.compile(
        r"(?i)\bapi[_-]?key\s*[:=]\s*['\"]?[^\s'\"]{8,}")),
)


def allow_sensitive() -> bool:
    """True when the override is requested."""
    return os.environ.get("VM_ALLOW_SENSITIVE", "").strip() in ("1", "true", "yes")


def check_text(text: str) -> None:
    """Raise :class:`SensitiveContentError` if text matches a secret rule.

    The error message names the RULE only — never the matched value.
    """
    if allow_sensitive():
        return
    for rule_name, pattern in RULES:
        match = pattern.search(text)
        if match:
            raise SensitiveContentError(
                f"text appears to contain a secret (rule: {rule_name}); "
                "remove it or pass --allow-sensitive"
            )
