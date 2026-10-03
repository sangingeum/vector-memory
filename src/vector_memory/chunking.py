"""Long-text chunking (Hangul-aware sentence splitting, group fields)."""

from __future__ import annotations

import re
import uuid
from typing import Any

from . import payload as _payload

# Sentence enders: Latin/CJK punctuation + Hangul sentence endings (다. 요.)
# + newlines. A sentence keeps its terminator.
_SENTENCE_RE = re.compile(r"[^.!?\n]*[.!?\n]+|[^.!?\n]+$")


def split_text(text: str, chunk_chars: int = 1500, overlap: int = 150) -> list[str]:
    """Split *text* into chunks <= chunk_chars on sentence/paragraph boundaries.

    Hangul-aware: `다.`/`요.` end sentences via the `.`; explicit newlines
    always end a sentence. Each chunk (except the first) overlaps the previous
    one by ~*overlap* chars (whole trailing sentences). The caller has already
    validated the text (chunk mode allows oversized text).
    """
    if chunk_chars <= overlap:
        raise ValueError("chunk_chars must be greater than overlap")
    sentences = [s for s in _SENTENCE_RE.findall(text) if s.strip()]
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > chunk_chars:
            # One sentence longer than the budget: hard-split it.
            if current:
                chunks.append(current)
                current = ""
            for start in range(0, len(sentence), chunk_chars - overlap):
                part = sentence[start : start + chunk_chars]
                if part.strip():
                    chunks.append(part)
            continue
        if current and len(current) + len(sentence) > chunk_chars:
            chunks.append(current)
            tail = current[-overlap:] if overlap > 0 else ""
            # Trim the tail to a sentence boundary so chunks don't cut words.
            cut = tail.find(" ")
            current = (tail[cut + 1 :] if 0 <= cut < len(tail) - 1 else tail) + sentence
        else:
            current += sentence
    if current.strip():
        chunks.append(current)
    return chunks or [text]


def group_fields(text: str) -> dict[str, Any]:
    """System fields for a chunk group: shared ``_group_id`` + ``_chunk_index``."""
    return {
        _payload.GROUP_ID: str(uuid.uuid4()),
        _payload.CHUNK_INDEX: 0,
    }


def collapse_groups(hits: list) -> list:
    """Only the best chunk per ``_group_id`` (keeps score order)."""
    seen_groups: set[str] = set()
    out = []
    for hit in hits:
        group = (hit.payload or {}).get(_payload.GROUP_ID)
        if group and group in seen_groups:
            continue
        if group:
            seen_groups.add(group)
        out.append(hit)
    return out


def group_count(hits: list) -> dict[str, int]:
    """Per-group chunk counts (for the `chunks: N` annotation)."""
    counts: dict[str, int] = {}
    for hit in hits:
        group = (hit.payload or {}).get(_payload.GROUP_ID)
        if group:
            counts[group] = counts.get(group, 0) + 1
    return counts
