"""
memory_manager.py

MemoryManager — unified read/write interface over StoreManager.

Covers:
  - SQLite tables  : CONVERSATIONAL_MEMORY, TOOL_LOG_MEMORY
  - Chroma stores  : SEMANTIC, WORKFLOW, TOOLBOX, ENTITY, SUMMARY
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.graphDB.entity_graph_memory import EntityGraphMemory


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(sep=" ", timespec="seconds")


def _uid() -> str:
    return str(uuid.uuid4())


def _content_id(text: str, prefix: str = "doc") -> str:
    """Generate a deterministic ID from text content (SHA256 hash)."""
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{h}"


def _dumps(obj: Any) -> str:
    """Safely JSON-serialize; fall back to str."""
    try:
        return json.dumps(obj, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(obj)


def _loads(s: Optional[str]) -> Any:
    if s is None:
        return None
    try:
        return json.loads(s)
    except (TypeError, ValueError):
        return s


# ─────────────────────────────────────────────────────────────────────────────
# MemoryManager
# ─────────────────────────────────────────────────────────────────────────────

class MemoryManager:
    """
    Unified read/write interface for all memory stores managed by StoreManager.

    SQL tables
    ----------
    write_conversation(thread_id, role, content, ...)
    read_conversations(thread_id, limit, before_timestamp)
    delete_conversation(record_id)
    update_conversation_summary_id(record_id, summary_id)

    write_tool_log(thread_id, tool_name, ...)
    read_tool_logs(thread_id, tool_name, limit)
    delete_tool_log(record_id)

    Chroma vector stores
    --------------------
    write_knowledge(texts, metadatas, ids)          → SEMANTIC_MEMORY
    search_knowledge(query, k, filter)

    write_workflow(texts, metadatas, ids)            → WORKFLOW_MEMORY
    search_workflow(query, k, filter)

    write_toolbox(texts, metadatas, ids)             → TOOLBOX_MEMORY
    search_toolbox(query, k, filter)

    write_entity(texts, metadatas, ids)              → ENTITY_MEMORY
    search_entity(query, k, filter)

    write_summary(texts, metadatas, ids)             → SUMMARY_MEMORY
    search_summary(query, k, filter)

    delete_from_store(store_name, ids)               → any Chroma store
    """

    # ------------------------------------------------------------------
    # Store aliases used in delete_from_store / _resolve_store
    # ------------------------------------------------------------------
    _STORE_ALIASES = {
        "knowledge", "knowledge_base", "semantic",
        "workflow",
        "toolbox",
        "entity",
        "summary",
    }

    def __init__(
        self,
        store_manager,             # StoreManager instance
        conversation_conn: sqlite3.Connection,
        tool_log_conn: sqlite3.Connection,
        graph_memory: Optional[EntityGraphMemory] = None,
    ):
        self._sm = store_manager
        self._conv_conn = conversation_conn
        self._tool_conn = tool_log_conn
        self._graph_memory = graph_memory

    # ══════════════════════════════════════════════════════════════════════
    # PRIVATE — SQL helpers
    # ══════════════════════════════════════════════════════════════════════

    def _conv_execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        try:
            cursor = self._conv_conn.cursor()
            cursor.execute(sql, params)
            self._conv_conn.commit()
            return cursor
        except sqlite3.Error as e:
            self._conv_conn.rollback()
            logging.error(f"SQL error in conversational memory: {e}")
            raise

    def _tool_execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        try:
            cursor = self._tool_conn.cursor()
            cursor.execute(sql, params)
            self._tool_conn.commit()
            return cursor
        except sqlite3.Error as e:
            self._tool_conn.rollback()
            logging.error(f"SQL error in tool log memory: {e}")
            raise

    # ══════════════════════════════════════════════════════════════════════
    # PRIVATE — Chroma helpers
    # ══════════════════════════════════════════════════════════════════════

    def _resolve_store(self, name: str) -> Chroma:
        """Map a friendly store name → Chroma vectorstore."""
        n = name.lower().strip()
        if n in ("knowledge", "knowledge_base", "semantic"):
            return self._sm.get_knowledge_base_store()
        if n == "workflow":
            return self._sm.get_workflow_store()
        if n == "toolbox":
            return self._sm.get_toolbox_store()
        if n == "entity":
            return self._sm.get_entity_store()
        if n == "summary":
            return self._sm.get_summary_store()
        raise ValueError(
            f"Unknown store '{name}'. "
            f"Valid names: {sorted(self._STORE_ALIASES)}"
        )

    def _write_to_store(
        self,
        store: Chroma,
        texts: list[str],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list[str]] = None,
    ) -> list[str]:
        """Add documents to a Chroma store, returning their IDs.

        When ids are not provided, generates deterministic content-hash IDs
        so the same text always maps to the same ID — ChromaDB upserts
        instead of creating duplicates.
        """
        if not texts:
            return []
        ids = ids or [_content_id(t) for t in texts]
        metadatas = metadatas or [{} for _ in texts]
        docs = [
            Document(page_content=t, metadata=m)
            for t, m in zip(texts, metadatas)
        ]
        store.add_documents(documents=docs, ids=ids)
        return ids

    def _search_store(
        self,
        store: Chroma,
        query: str,
        k: int = 5,
        filter: Optional[dict] = None,   # noqa: A002
    ) -> list[Document]:
        """Similarity search on a Chroma store."""
        kwargs: dict[str, Any] = {"k": k}
        if filter:
            kwargs["filter"] = filter
        return store.similarity_search(query, **kwargs)

    # ══════════════════════════════════════════════════════════════════════
    # CONVERSATIONAL MEMORY  (SQLite)
    # ══════════════════════════════════════════════════════════════════════

    def write_conversation(
        self,
        thread_id: str,
        role: str,
        content: str,
        metadata: Optional[dict] = None,
        record_id: Optional[str] = None,
        summary_id: Optional[str] = None,
    ) -> str:
        """
        Insert one message into CONVERSATIONAL_MEMORY.

        Returns the record id.
        """
        rid = record_id or _uid()
        table = self._sm.get_conversational_table()
        self._conv_execute(
            f"""
            INSERT INTO {table}
                (id, thread_id, role, content, timestamp, metadata,
                 created_at, summary_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rid,
                thread_id,
                role,
                content,
                _now(),
                _dumps(metadata) if metadata else None,
                _now(),
                summary_id,
            ),
        )
        return rid

    def read_conversations(
        self,
        thread_id: str,
        limit: int = 50,
        before_timestamp: Optional[str] = None,
    ) -> list[dict]:
        """
        Fetch messages for a thread, newest-last (chronological).

        Args:
            thread_id: conversation thread identifier
            limit: max rows to return
            before_timestamp: ISO string — only return messages before this
        """
        table = self._sm.get_conversational_table()
        if before_timestamp:
            cursor = self._conv_execute(
                f"""
                SELECT id, thread_id, role, content, timestamp,
                       metadata, summary_id
                FROM   {table}
                WHERE  thread_id = ? AND timestamp < ?
                ORDER  BY timestamp DESC
                LIMIT  ?
                """,
                (thread_id, before_timestamp, limit),
            )
        else:
            cursor = self._conv_execute(
                f"""
                SELECT id, thread_id, role, content, timestamp,
                       metadata, summary_id
                FROM   {table}
                WHERE  thread_id = ?
                ORDER  BY timestamp DESC
                LIMIT  ?
                """,
                (thread_id, limit),
            )
        cols = [d[0] for d in cursor.description]
        rows = [dict(zip(cols, row)) for row in cursor.fetchall()]
        # Return in chronological order
        rows.reverse()
        for r in rows:
            r["metadata"] = _loads(r["metadata"])
        return rows

    def update_conversation_summary_id(
        self,
        record_id: str,
        summary_id: str,
    ) -> None:
        """Link a conversation record to a summary stored in SUMMARY_MEMORY."""
        table = self._sm.get_conversational_table()
        self._conv_execute(
            f"UPDATE {table} SET summary_id = ? WHERE id = ?",
            (summary_id, record_id),
        )

    def delete_conversation(self, record_id: str) -> None:
        """Delete a single conversation record by id."""
        table = self._sm.get_conversational_table()
        self._conv_execute(
            f"DELETE FROM {table} WHERE id = ?",
            (record_id,),
        )

    def delete_thread_conversations(self, thread_id: str) -> int:
        """Delete all conversation records for a thread. Returns rows deleted."""
        table = self._sm.get_conversational_table()
        cursor = self._conv_execute(
            f"DELETE FROM {table} WHERE thread_id = ?",
            (thread_id,),
        )
        return cursor.rowcount

    # ══════════════════════════════════════════════════════════════════════
    # TOOL LOG MEMORY  (SQLite)
    # ══════════════════════════════════════════════════════════════════════

    def write_tool_log(
        self,
        thread_id: str,
        tool_name: str,
        tool_args: Optional[dict] = None,
        result: Optional[Any] = None,
        result_preview: Optional[str] = None,
        status: str = "success",
        error_message: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        metadata: Optional[dict] = None,
        record_id: Optional[str] = None,
    ) -> str:
        """
        Insert one tool execution log into TOOL_LOG_MEMORY.

        Returns the record id.
        """
        rid = record_id or _uid()
        table = self._sm.get_tool_log_table()
        result_str = _dumps(result) if result is not None else None
        self._tool_execute(
            f"""
            INSERT INTO {table}
                (id, thread_id, tool_call_id, tool_name, tool_args,
                 result, result_preview, status, error_message,
                 metadata, timestamp, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rid,
                thread_id,
                tool_call_id,
                tool_name,
                _dumps(tool_args) if tool_args else None,
                result_str,
                result_preview,
                status,
                error_message,
                _dumps(metadata) if metadata else None,
                _now(),
                _now(),
            ),
        )
        return rid

    def read_tool_logs(
        self,
        thread_id: str,
        tool_name: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict]:
        """
        Fetch tool logs for a thread, optionally filtered by tool_name.
        Returns rows in chronological order.
        """
        table = self._sm.get_tool_log_table()
        if tool_name:
            cursor = self._tool_execute(
                f"""
                SELECT id, thread_id, tool_call_id, tool_name, tool_args,
                       result, result_preview, status, error_message,
                       metadata, timestamp
                FROM   {table}
                WHERE  thread_id = ? AND tool_name = ?
                ORDER  BY timestamp DESC
                LIMIT  ?
                """,
                (thread_id, tool_name, limit),
            )
        else:
            cursor = self._tool_execute(
                f"""
                SELECT id, thread_id, tool_call_id, tool_name, tool_args,
                       result, result_preview, status, error_message,
                       metadata, timestamp
                FROM   {table}
                WHERE  thread_id = ?
                ORDER  BY timestamp DESC
                LIMIT  ?
                """,
                (thread_id, limit),
            )
        cols = [d[0] for d in cursor.description]
        rows = [dict(zip(cols, row)) for row in cursor.fetchall()]
        rows.reverse()
        for r in rows:
            r["tool_args"] = _loads(r["tool_args"])
            r["result"] = _loads(r["result"])
            r["metadata"] = _loads(r["metadata"])
        return rows

    def delete_tool_log(self, record_id: str) -> None:
        """Delete a single tool log record by id."""
        table = self._sm.get_tool_log_table()
        self._tool_execute(
            f"DELETE FROM {table} WHERE id = ?",
            (record_id,),
        )

    # ══════════════════════════════════════════════════════════════════════
    # SEMANTIC / KNOWLEDGE BASE  (Chroma)
    # ══════════════════════════════════════════════════════════════════════

    def write_knowledge(
        self,
        texts: list[str],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list[str]] = None,
    ) -> list[str]:
        """Upsert documents into SEMANTIC_MEMORY. Returns assigned IDs."""
        return self._write_to_store(
            self._sm.get_knowledge_base_store(), texts, metadatas, ids
        )

    def search_knowledge(
        self,
        query: str,
        k: int = 5,
        filter: Optional[dict] = None,   # noqa: A002
    ) -> list[Document]:
        """Similarity search over SEMANTIC_MEMORY."""
        return self._search_store(
            self._sm.get_knowledge_base_store(), query, k, filter
        )

    # ══════════════════════════════════════════════════════════════════════
    # WORKFLOW MEMORY  (Chroma)
    # ══════════════════════════════════════════════════════════════════════

    def write_workflow(
        self,
        texts: list[str],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list[str]] = None,
    ) -> list[str]:
        return self._write_to_store(
            self._sm.get_workflow_store(), texts, metadatas, ids
        )

    def search_workflow(
        self,
        query: str,
        k: int = 5,
        filter: Optional[dict] = None,   # noqa: A002
    ) -> list[Document]:
        return self._search_store(
            self._sm.get_workflow_store(), query, k, filter
        )

    # ══════════════════════════════════════════════════════════════════════
    # TOOLBOX MEMORY  (Chroma)
    # ══════════════════════════════════════════════════════════════════════

    def write_toolbox(
        self,
        texts: list[str],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list[str]] = None,
    ) -> list[str]:
        return self._write_to_store(
            self._sm.get_toolbox_store(), texts, metadatas, ids
        )

    def search_toolbox(
        self,
        query: str,
        k: int = 5,
        filter: Optional[dict] = None,   # noqa: A002
    ) -> list[Document]:
        return self._search_store(
            self._sm.get_toolbox_store(), query, k, filter
        )

    # ══════════════════════════════════════════════════════════════════════
    # ENTITY MEMORY  (Chroma)
    # ══════════════════════════════════════════════════════════════════════

    def write_entity(
        self,
        texts: list[str],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list[str]] = None,
        thread_id: Optional[str] = None,
    ) -> list[str]:
        if self._graph_memory:
            return self._graph_memory.write_entity_documents(
                texts=texts,
                metadatas=metadatas,
                thread_id=thread_id,
            )
        return self._write_to_store(
            self._sm.get_entity_store(), texts, metadatas, ids
        )

    def search_entity(
        self,
        query: str,
        k: int = 5,
        filter: Optional[dict] = None,   # noqa: A002
    ) -> list[Document]:
        if self._graph_memory:
            return self._graph_memory.search_entity_documents(
                query=query,
                k=k,
                filter=filter,
            )
        return self._search_store(
            self._sm.get_entity_store(), query, k, filter
        )

    def write_entity_graph(
        self,
        name: str,
        entity_type: str = "Entity",
        properties: Optional[dict] = None,
        thread_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Write a single entity to the graph DB (Neo4j only)."""
        if not self._graph_memory:
            raise RuntimeError("Graph memory not configured. Pass graph_memory to MemoryManager.")
        return self._graph_memory.write_entity(
            name=name,
            entity_type=entity_type,
            properties=properties,
            thread_id=thread_id,
        )

    def write_relationship_graph(
        self,
        source: str,
        target: str,
        rel_type: str = "RELATED_TO",
        properties: Optional[dict] = None,
    ) -> dict[str, Any]:
        """Write a relationship between two entities in the graph DB (Neo4j only)."""
        if not self._graph_memory:
            raise RuntimeError("Graph memory not configured. Pass graph_memory to MemoryManager.")
        return self._graph_memory.write_relationship(
            source=source,
            target=target,
            rel_type=rel_type,
            properties=properties,
        )

    def write_entities_from_text(
        self,
        text: str,
        thread_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Extract entities from text using LLM and store in graph DB (Neo4j only)."""
        if not self._graph_memory:
            raise RuntimeError("Graph memory not configured. Pass graph_memory to MemoryManager.")
        return self._graph_memory.write_entities_from_text(text=text, thread_id=thread_id)

    def get_entity(self, name: str) -> Optional[dict[str, Any]]:
        """Retrieve a single entity by name from the graph DB (Neo4j only)."""
        if not self._graph_memory:
            raise RuntimeError("Graph memory not configured. Pass graph_memory to MemoryManager.")
        return self._graph_memory.get_entity(name)

    def get_entity_relationships(
        self,
        name: str,
        direction: str = "both",
        rel_type: Optional[str] = None,
        depth: int = 1,
    ) -> list[dict[str, Any]]:
        """Get relationships for an entity from the graph DB (Neo4j only)."""
        if not self._graph_memory:
            raise RuntimeError("Graph memory not configured. Pass graph_memory to MemoryManager.")
        return self._graph_memory.get_entity_relationships(
            name=name,
            direction=direction,
            rel_type=rel_type,
            depth=depth,
        )

    def get_related_entities(
        self,
        name: str,
        rel_type: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Get directly connected entities from the graph DB (Neo4j only)."""
        if not self._graph_memory:
            raise RuntimeError("Graph memory not configured. Pass graph_memory to MemoryManager.")
        return self._graph_memory.get_related_entities(name=name, rel_type=rel_type)

    def delete_entity_graph(self, name: str) -> bool:
        """Delete an entity and its relationships from the graph DB (Neo4j only)."""
        if not self._graph_memory:
            raise RuntimeError("Graph memory not configured. Pass graph_memory to MemoryManager.")
        return self._graph_memory.delete_entity(name)

    def get_graph_stats(self) -> dict[str, Any]:
        """Get graph DB statistics (Neo4j only)."""
        if not self._graph_memory:
            raise RuntimeError("Graph memory not configured. Pass graph_memory to MemoryManager.")
        return self._graph_memory.get_stats()

    @property
    def has_graph_memory(self) -> bool:
        """Check if graph memory (Neo4j) is configured."""
        return self._graph_memory is not None

    # ══════════════════════════════════════════════════════════════════════
    # SUMMARY MEMORY  (Chroma)
    # ══════════════════════════════════════════════════════════════════════

    def write_summary(
        self,
        texts: list[str],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list[str]] = None,
    ) -> list[str]:
        return self._write_to_store(
            self._sm.get_summary_store(), texts, metadatas, ids
        )

    def search_summary(
        self,
        query: str,
        k: int = 5,
        filter: Optional[dict] = None,   # noqa: A002
    ) -> list[Document]:
        return self._search_store(
            self._sm.get_summary_store(), query, k, filter
        )

    # ══════════════════════════════════════════════════════════════════════
    # GENERIC CHROMA DELETE
    # ══════════════════════════════════════════════════════════════════════

    def delete_from_store(self, store_name: str, ids: list[str]) -> None:
        """
        Delete documents by ID from any Chroma store.

        store_name: one of  knowledge | knowledge_base | semantic |
                            workflow | toolbox | entity | summary
        """
        if not ids:
            return
        n = store_name.lower().strip()
        # Route entity deletes to graph DB if configured
        if n == "entity" and self._graph_memory:
            for name in ids:
                self._graph_memory.delete_entity(name)
            return
        store = self._resolve_store(store_name)
        store.delete(ids=ids)

    # ══════════════════════════════════════════════════════════════════════
    # CONVENIENCE — write a generic store by name
    # ══════════════════════════════════════════════════════════════════════

    def write_to_store(
        self,
        store_name: str,
        texts: list[str],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list[str]] = None,
    ) -> list[str]:
        """Generic write that resolves store by name."""
        store = self._resolve_store(store_name)
        return self._write_to_store(store, texts, metadatas, ids)

    def search_store(
        self,
        store_name: str,
        query: str,
        k: int = 5,
        filter: Optional[dict] = None,   # noqa: A002
    ) -> list[Document]:
        """Generic search that resolves store by name."""
        store = self._resolve_store(store_name)
        return self._search_store(store, query, k, filter)

    # ══════════════════════════════════════════════════════════════════════
    # GRAPH CLEANUP
    # ══════════════════════════════════════════════════════════════════════

    def delete_entities_by_thread(self, thread_id: str) -> int:
        """
        Delete all entities for a specific thread from the graph DB (Neo4j only).

        Returns:
            Number of entities deleted.
        """
        if not self._graph_memory:
            raise RuntimeError("Graph memory not configured. Pass graph_memory to MemoryManager.")
        return self._graph_memory.delete_entities_by_thread(thread_id)

    def close(self) -> None:
        """Close database connections and graph memory."""
        if self._graph_memory:
            self._graph_memory.close()
        if self._conv_conn:
            self._conv_conn.close()
        if self._tool_conn:
            self._tool_conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False