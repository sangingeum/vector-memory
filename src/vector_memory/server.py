"""FastMCP server: thin tool adapters over vector_memory.core (ruling §3).

Each tool handler is a one-line adapter calling exactly one core op; the six
tools map 1:1 to core functions of the same name (MCP tool names are part of
the public contract and are unchanged). Embedding/Qdrant access lives in the
``embedding`` and ``store`` modules.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

from mcp.server.fastmcp import FastMCP

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
qdrant = qdrant


@mcp.tool()
def save_memory(text: str, metadata: str | dict[str, Any] = "{}",
                collection: str = "") -> str:
    """Save a new document or scenario outcome into the vector DB.

    metadata may be a JSON string (e.g. '{"source": "doc1", "tags": ["a"]}')
    or a JSON object — both are accepted. On parse failure an empty dict is
    stored instead and a warning is returned alongside the result. If
    collection is given, the memory is stored there (created automatically
    if missing; default: the server-configured collection).
    """
    return _core.save_memory(text, metadata, collection)


@mcp.tool()
def save_memories(texts: list[str], metadata: str | dict[str, Any] = "{}",
                  collection: str = "") -> str:
    """Save multiple documents into the vector DB in one batch.

    All texts are embedded in a single Ollama call and upserted together.
    metadata (JSON string or object) is applied to every document.
    """
    return _core.save_memories(texts, metadata, collection)


@mcp.tool()
def search_memory(query: str, limit: int = 3, filter: str | dict[str, Any] = "",
                  collection: str = "", project: str = "") -> str:
    """Search the vector DB for past documents/scenarios semantically similar
    to a query. filter is an optional payload filter (JSON string or object).
    project (optional) ANDs an exact-match condition on the payload 'project'
    field for project-scoped memory."""
    return _core.search_memory(query, limit, filter, collection, project)


@mcp.tool()
def update_memory(point_id: str, text: str | None = None,
                  metadata: str | dict[str, Any] = "",
                  collection: str = "") -> str:
    """Update an existing memory (point) in place under the same ID.

    Re-embeds and overwrites when text is given; metadata-only updates keep
    the existing text and vector; both empty is an error.
    """
    return _core.update_memory(point_id, text, metadata, collection)


@mcp.tool()
def delete_memory(point_id: str, collection: str = "") -> str:
    """Delete a stored memory (point) from the vector DB by ID."""
    return _core.delete_memory(point_id, collection)


@mcp.tool()
def list_collections() -> str:
    """List all collections currently present in Qdrant."""
    return _core.list_collections()


def main() -> None:
    """CLI entry point (``vector-memory-mcp``)."""
    from .logsetup import configure_logging

    configure_logging(verbose=os.environ.get("VERBOSE", "") not in ("", "0"))
    logger.info(
        "Agent Vector Memory MCP starting — ollama=%s qdrant=%s model=%s collection=%s",
        OLLAMA_URL, QDRANT_URL, EMBED_MODEL, COLLECTION_NAME,
    )
    mcp.run(transport="stdio")
