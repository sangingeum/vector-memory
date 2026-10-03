"""FastMCP server: thin tool adapters over vector_memory.core.

Each tool handler is a one-line adapter calling exactly one core op; the
tools map 1:1 to core functions of the same name (existing MCP tool names are
part of the public contract and are unchanged). Embedding/Qdrant access lives
in the ``embedding`` and ``store`` modules.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

from mcp.server.fastmcp import FastMCP

from . import browse as _browse
from . import core as _core
from .embedding import EMBED_MODEL, OLLAMA_URL
from .store import COLLECTION_NAME, QDRANT_URL, qdrant

# All diagnostics go to stderr — stdout carries the stdio MCP transport.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("vector-memory")

mcp = FastMCP("Agent Vector Memory")

# Exposed for tests, which rebind the client on this module and on .store.
qdrant = qdrant  # noqa: PLW0127 - re-export for test monkeypatching


@mcp.tool()
def save_memory(text: str, metadata: str | dict[str, Any] = "{}",
                collection: str = "", allow_duplicate: bool = False,
                on_similar: str = "") -> str:
    """Save a new document or scenario outcome into the vector DB.

    metadata may be a JSON string (e.g. '{"source": "doc1", "tags": ["a"]}')
    or a JSON object — both are accepted. Identical normalized text is
    idempotent (the existing point is refreshed) unless allow_duplicate.
    Near-duplicates above the calibrated threshold are reported in the result
    (never merged); on_similar may be "warn" (default), "skip", or "error".
    """
    return _core.call(_core.save_memory, text, metadata, collection,
                      allow_duplicate=allow_duplicate, on_similar=on_similar)


@mcp.tool()
def save_superseding(text: str, supersedes: list[str],
                     metadata: str | dict[str, Any] = "{}",
                     collection: str = "") -> str:
    """Save a memory that REPLACES the listed point IDs: they are marked
    superseded (hidden from default search, kept for audit) and this memory
    records them under _supersedes. Use when a fact has changed. Unknown IDs
    fail before anything is written."""
    return _core.call(_core.save_memory, text, metadata, collection, supersedes=supersedes)


@mcp.tool()
def save_memories(texts: list[str], metadata: str | dict[str, Any] = "{}",
                  collection: str = "") -> str:
    """Save multiple documents into the vector DB in one batch.

    All texts are embedded in a single Ollama call and upserted together.
    metadata (JSON string or object) is applied to every document.
    """
    return _core.call(_core.save_memories, texts, metadata, collection)


@mcp.tool()
def search_memory(query: str, limit: int = 3, filter: str | dict[str, Any] = "",
                  collection: str = "", project: str = "",
                  include_inactive: bool = False) -> str:
    """Search the vector DB for past documents/scenarios semantically similar
    to a query. Superseded/archived memories are hidden unless
    include_inactive=True (hits are annotated with their status). filter is an
    optional payload filter (JSON string or object). project (optional) ANDs
    an exact-match condition on the payload 'project' field."""
    return _core.call(_core.search_memory, query, limit, filter, collection, project,
                      include_inactive)


@mcp.tool()
def archive(point_ids: list[str], collection: str = "") -> str:
    """Archive memories by ID: hidden from default search, kept for audit."""
    return _core.call(_core.set_status, point_ids, "archived", collection)


@mcp.tool()
def unarchive(point_ids: list[str], collection: str = "") -> str:
    """Return archived memories to active status."""
    return _core.call(_core.set_status, point_ids, "active", collection)


@mcp.tool()
def update_memory(point_id: str, text: str | None = None,
                  metadata: str | dict[str, Any] = "",
                  collection: str = "") -> str:
    """Update an existing memory (point) in place under the same ID.

    Re-embeds and overwrites when text is given; metadata-only updates keep
    the existing text and vector; both empty is an error.
    """
    return _core.call(_core.update_memory, point_id, text, metadata, collection)


@mcp.tool()
def patch_metadata(point_id: str, set: str | dict[str, Any] = "{}",
                   unset: list[str] | None = None, collection: str = "") -> str:
    """Patch metadata without re-embedding: merge keys (set) and/or remove keys (unset)."""
    return _core.call(_core.patch_metadata, point_id, set, unset, collection)


@mcp.tool()
def delete_memory(point_id: str, collection: str = "") -> str:
    """Delete a stored memory (point) from the vector DB by ID."""
    return _core.call(_core.delete_memory, point_id, collection)


@mcp.tool()
def list_collections() -> str:
    """List all collections currently present in Qdrant."""
    return _core.call(_core.list_collections)


@mcp.tool()
def get(point_ids: list[str], collection: str = "", with_system: bool = False) -> str:
    """Fetch full text + metadata for the given memory IDs (no embedding call)."""
    return _core.call(_browse.get_memory, point_ids, collection, with_system)


@mcp.tool()
def list_memories(collection: str = "", limit: int = 25, project: str = "",
                  include_inactive: bool = False) -> str:
    """Browse memories (newest first) without embedding a query."""
    return _core.call(_browse.list_memories, collection, "", limit, "", project,
                      include_inactive)


@mcp.tool()
def count(collection: str = "", project: str = "") -> str:
    """Count active memories in a collection (optionally project-scoped)."""
    return _core.call(_browse.count_memories, collection, "", project, False)


@mcp.tool()
def values(field: str, collection: str = "", limit: int = 50) -> str:
    """Discover distinct values of project/type/tags/source in a collection."""
    return _core.call(_browse.field_values, field, collection, "", limit, "")


def main() -> None:
    """CLI entry point (``vector-memory-mcp``)."""
    from .logsetup import configure_logging

    configure_logging(verbose=os.environ.get("VERBOSE", "") not in ("", "0"))
    logger.info(
        "Agent Vector Memory MCP starting — ollama=%s qdrant=%s model=%s collection=%s",
        OLLAMA_URL, QDRANT_URL, EMBED_MODEL, COLLECTION_NAME,
    )
    mcp.run(transport="stdio")
