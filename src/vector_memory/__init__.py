"""Agent Vector Memory: Ollama embeddings + Qdrant vector memory.

Two entry points over the same core (vector_memory.core):
- ``vector-memory-mcp`` — MCP server (server.py, stdio transport)
- ``vector-memory``     — one-shot CLI (cli.py, typer)
"""

from .core import (
    delete_memory,
    list_collections,
    save_memories,
    save_memory,
    search_memory,
    update_memory,
)
from .embedding import EMBED_MODEL, OLLAMA_URL, embed, embed_batch, embed_many
from .server import main, mcp
from .store import COLLECTION_NAME, QDRANT_URL, qdrant

__all__ = [
    "COLLECTION_NAME",
    "EMBED_MODEL",
    "OLLAMA_URL",
    "QDRANT_URL",
    "delete_memory",
    "embed",
    "embed_batch",
    "embed_many",
    "list_collections",
    "main",
    "mcp",
    "qdrant",
    "save_memories",
    "save_memory",
    "search_memory",
    "update_memory",
]
