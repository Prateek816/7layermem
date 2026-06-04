from __future__ import annotations

import hashlib
from typing import Any, Iterator, Optional, Sequence, TYPE_CHECKING

from langchain_core.stores import BaseStore

if TYPE_CHECKING:
    from src.memory.agent_memory import AgentMemory

_VALID_STORES = frozenset({"knowledge", "entity", "workflow", "summary"})


def _content_id(text: str, prefix: str = "doc") -> str:
    """Generate a deterministic ID from text content (SHA256 hash)."""
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{h}"


class SevenLayerStore(BaseStore[str, str]):
    """LangChain key-value store backed by 7layermem vector stores.

    Maps LangChain's batch key-value interface to the 7layermem knowledge,
    entity, workflow, or summary ChromaDB stores. Keys are stored as
    metadata on the documents.

    Note:
        ``yield_keys`` performs a broad similarity search and filters by
        prefix. This is O(n) and not suitable for very large stores.

    Args:
        agent_memory: An AgentMemory instance.
        store_name: Which 7layermem store to use (default: "knowledge").
            Must be one of: knowledge, entity, workflow, summary.
    """

    def __init__(
        self,
        agent_memory: AgentMemory,
        store_name: str = "knowledge",
    ) -> None:
        if store_name not in _VALID_STORES:
            raise ValueError(
                f"store_name must be one of {sorted(_VALID_STORES)}, "
                f"got '{store_name}'"
            )
        self._agent_memory = agent_memory
        self._mm = agent_memory._mm
        self._store_name = store_name

    def mget(self, keys: Sequence[str]) -> list[Optional[str]]:
        """Get values for the given keys."""
        results: list[Optional[str]] = []
        for key in keys:
            docs = self._mm.search_store(
                self._store_name, query=key, k=1, filter={"key": key}
            )
            if docs:
                results.append(docs[0].page_content)
            else:
                results.append(None)
        return results

    def mset(self, key_value_pairs: Sequence[tuple[str, str]]) -> None:
        """Set key-value pairs in the store."""
        for key, value in key_value_pairs:
            doc_id = _content_id(f"{self._store_name}:{key}:{value}")
            self._mm.write_to_store(
                self._store_name,
                texts=[value],
                metadatas=[{"key": key}],
                ids=[doc_id],
            )

    def mdelete(self, keys: Sequence[str]) -> None:
        """Delete keys from the store."""
        for key in keys:
            docs = self._mm.search_store(
                self._store_name, query=key, k=1, filter={"key": key}
            )
            if docs and hasattr(docs[0], "id"):
                self._mm.delete_from_store(self._store_name, [docs[0].id])

    def yield_keys(self, *, prefix: str | None = None) -> Iterator[str]:
        """Iterate over keys matching the given prefix.

        Note: Uses a broad similarity search — O(n), not suitable for
        very large stores.
        """
        docs = self._mm.search_store(self._store_name, query="", k=100)
        for doc in docs:
            key = doc.metadata.get("key", "") if doc.metadata else ""
            if prefix is None or key.startswith(prefix):
                yield key
