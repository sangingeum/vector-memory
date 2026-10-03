"""Strict input validation (VM-01).

Default is strict: invalid metadata/filter JSON or a malformed metadata key
raises :class:`ArgumentError` and nothing is written. The legacy warn-and-
continue behavior stays reachable via ``--lenient`` / ``VM_LENIENT=1``.
"""

from __future__ import annotations

import json
import os
import re
import warnings
from typing import Any

from .errors import ArgumentError

MAX_TEXT_CHARS = int(os.environ.get("VM_MAX_TEXT_CHARS", "20000"))
KEY_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")

KNOWN_TYPES = (
    "decision", "fact", "preference", "procedure", "incident", "todo", "note",
)

# Keys with a documented recommended schema (advisory type-checks).
_SUMMARY_MAX = 200


def lenient() -> bool:
    """True when the legacy warn-and-continue behavior is requested."""
    return os.environ.get("VM_LENIENT", "").strip() in ("1", "true", "yes")


def _strict_fail(message: str) -> None:
    if lenient():
        warnings.warn(message, UserWarning, stacklevel=3)
        return
    raise ArgumentError(message)


def normalize_text(text: str) -> str:
    """Normalized form used for exact-duplicate detection (NFC, strip, collapse)."""
    import unicodedata

    return " ".join(unicodedata.normalize("NFC", text).split())


def content_hash(text: str) -> str:
    """sha256 hex of the normalized text (system field ``_content_hash``)."""
    import hashlib

    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def validate_text(text: Any, *, where: str = "text", allow_long: bool = False) -> str:
    """Non-empty after strip, capped at MAX_TEXT_CHARS; returns the stripped text.

    ``allow_long=True`` (chunking) skips the length cap — the caller splits.
    """
    if not isinstance(text, str):
        raise ArgumentError(f"{where} must be a string")
    stripped = text.strip()
    if not stripped:
        raise ArgumentError(f"{where} is empty")
    if not allow_long and len(stripped) > MAX_TEXT_CHARS:
        raise ArgumentError(
            f"{where} too long ({len(stripped)} > {MAX_TEXT_CHARS}); "
            "split it or use --chunk"
        )
    return stripped


def _check_scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool)) or value is None


def validate_metadata(metadata: Any, *, origin: str = "metadata") -> dict[str, Any]:
    """Validate and return a caller-supplied metadata dict.

    Rules: flat JSON object; keys ``^[a-z][a-z0-9_]{0,63}$`` without a leading
    ``_``; values are JSON scalars or lists of scalars. Known keys are
    type-checked; an unknown ``type`` value emits a stderr warning (soft enum)
    but is allowed.
    """
    if not isinstance(metadata, dict):
        _strict_fail(f"{origin} is not a JSON object (dict)")
        return {}
    for key, value in metadata.items():
        if not isinstance(key, str) or not KEY_RE.match(key):
            _strict_fail(
                f"{origin} key {key!r} is invalid: keys must match "
                "^[a-z][a-z0-9_]{0,63}$ and must not start with '_'"
            )
            continue
        if key.startswith("_"):
            _strict_fail(
                f"{origin} key {key!r} uses the reserved '_' prefix (system fields)"
            )
            continue
        if not (_check_scalar(value) or (
            isinstance(value, list) and all(_check_scalar(v) for v in value)
        )):
            _strict_fail(
                f"{origin} key {key!r} must be a JSON scalar or a list of scalars"
            )
            continue
        # Known-key type checks (advisory-only failures are warnings, not errors)
        if key == "project" and not isinstance(value, str):
            _strict_fail(f"{origin} key 'project' must be a string")
        elif key == "tags":
            if not isinstance(value, list) or not all(
                isinstance(t, str) for t in value
            ):
                _strict_fail(f"{origin} key 'tags' must be a list of strings")
        elif key == "importance" and not isinstance(value, (int, float)):
            _strict_fail(f"{origin} key 'importance' must be a number")
        elif key == "summary":
            if not isinstance(value, str):
                _strict_fail(f"{origin} key 'summary' must be a string")
            elif len(value) > _SUMMARY_MAX:
                _strict_fail(
                    f"{origin} key 'summary' too long ({len(value)} > {_SUMMARY_MAX})"
                )
        elif key == "type" and isinstance(value, str) and value not in KNOWN_TYPES:
            warnings.warn(
                f"Warning: unknown type {value!r} (known: {', '.join(KNOWN_TYPES)})",
                UserWarning,
                stacklevel=3,
            )
    return metadata


def parse_metadata_strict(
    metadata: str | dict[str, Any] | None, *, origin: str = "metadata"
) -> dict[str, Any]:
    """Parse metadata from a JSON string or a dict, validating strictly.

    A dict is accepted as-is (MCP callers often pass objects). In strict mode
    any problem raises :class:`ArgumentError`; in lenient mode the legacy
    behavior applies (empty dict + ``Warning``).
    """
    if metadata is None:
        return {}
    if isinstance(metadata, dict):
        return validate_metadata(metadata, origin=origin)
    if not isinstance(metadata, str):
        raise ArgumentError(
            f"{origin} must be a JSON object (dict) or a JSON string "
            f"(got {type(metadata).__name__})")
    if not metadata.strip():
        return {}
    try:
        parsed = json.loads(metadata)
    except json.JSONDecodeError as exc:
        _strict_fail(f"{origin} is not valid JSON (position {exc.pos})")
        return {}
    return validate_metadata(parsed, origin=origin)


def validate_filter_parsed(parsed: Any, *, origin: str = "filter") -> dict[str, Any]:
    """Structural validation of a parsed filter dict (operator checks in VM-09)."""
    if not isinstance(parsed, dict):
        _strict_fail(f"{origin} is not a JSON object (dict)")
        return {}
    return parsed
