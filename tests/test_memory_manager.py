"""
test_memory_manager.py

Complete test suite for MemoryManager.

Tests:
  - Helper functions (_now, _uid, _dumps, _loads)
  - Conversational memory (SQLite)
  - Tool log memory (SQLite)
  - Chroma vector stores (knowledge, workflow, toolbox, entity, summary)
  - Graph DB operations (mocked EntityGraphMemory)
  - Store resolution and generic operations
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.memory.memory_manager import MemoryManager, _dumps, _loads, _now, _uid


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_store_manager():
    """Create a mock StoreManager with all required getters."""
    sm = MagicMock()
    sm.get_conversational_table.return_value = "CONVERSATIONAL_MEMORY"
    sm.get_tool_log_table.return_value = "TOOL_LOG_MEMORY"
    sm.get_knowledge_base_store.return_value = MagicMock()
    sm.get_workflow_store.return_value = MagicMock()
    sm.get_toolbox_store.return_value = MagicMock()
    sm.get_entity_store.return_value = MagicMock()
    sm.get_summary_store.return_value = MagicMock()
    return sm


@pytest.fixture
def conversation_conn():
    """In-memory SQLite connection for conversational memory."""
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE CONVERSATIONAL_MEMORY (
            id TEXT PRIMARY KEY,
            thread_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            summary_id TEXT DEFAULT NULL
        )
    """)
    conn.commit()
    yield conn
    conn.close()


@pytest.fixture
def tool_log_conn():
    """In-memory SQLite connection for tool logs."""
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE TOOL_LOG_MEMORY (
            id TEXT PRIMARY KEY,
            thread_id TEXT NOT NULL,
            tool_call_id TEXT,
            tool_name TEXT NOT NULL,
            tool_args TEXT,
            result TEXT,
            result_preview TEXT,
            status TEXT DEFAULT 'success',
            error_message TEXT,
            metadata TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    yield conn
    conn.close()


@pytest.fixture
def mock_graph_memory():
    """Create a mock EntityGraphMemory."""
    gm = MagicMock()
    gm.write_entity_documents.return_value = ["entity_1", "entity_2"]
    gm.search_entity_documents.return_value = [
        Document(
            page_content="Entity: Alice\nType: PERSON",
            metadata={"name": "Alice", "type": "PERSON", "source": "graph_db"},
        )
    ]
    gm.write_entity.return_value = {"name": "Alice", "type": "PERSON"}
    gm.write_relationship.return_value = {"source": "Alice", "rel_type": "WORKS_AT", "target": "Acme"}
    gm.write_entities_from_text.return_value = {
        "entities_written": 2,
        "relationships_written": 1,
        "entities": [],
        "relationships": [],
    }
    gm.get_entity.return_value = {"name": "Alice", "type": "PERSON"}
    gm.get_entity_relationships.return_value = [
        {"source": "Alice", "relationships": ["WORKS_AT"], "target": "Acme", "target_entity": {}}
    ]
    gm.get_related_entities.return_value = [
        {"entity": {}, "entity_name": "Acme", "relationships": ["WORKS_AT"], "direction": "outgoing"}
    ]
    gm.delete_entity.return_value = True
    gm.get_stats.return_value = {
        "entity_count": 10,
        "relationship_count": 15,
        "entity_types": {"PERSON": 5, "ORGANIZATION": 5},
    }
    return gm


@pytest.fixture
def memory(mock_store_manager, conversation_conn, tool_log_conn):
    """MemoryManager without graph memory."""
    return MemoryManager(
        store_manager=mock_store_manager,
        conversation_conn=conversation_conn,
        tool_log_conn=tool_log_conn,
    )


@pytest.fixture
def memory_with_graph(mock_store_manager, conversation_conn, tool_log_conn, mock_graph_memory):
    """MemoryManager with graph memory."""
    return MemoryManager(
        store_manager=mock_store_manager,
        conversation_conn=conversation_conn,
        tool_log_conn=tool_log_conn,
        graph_memory=mock_graph_memory,
    )


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

class TestHelpers:
    """Tests for module-level helper functions."""

    def test_now_returns_iso_string(self):
        result = _now()
        # Should be parseable ISO format
        datetime.fromisoformat(result)

    def test_uid_returns_unique_strings(self):
        ids = {_uid() for _ in range(100)}
        assert len(ids) == 100

    def test_uid_is_valid_uuid(self):
        uid = _uid()
        uuid.UUID(uid)  # Should not raise

    def test_dumps_dict(self):
        result = _dumps({"key": "value"})
        assert json.loads(result) == {"key": "value"}

    def test_dumps_list(self):
        result = _dumps([1, 2, 3])
        assert json.loads(result) == [1, 2, 3]

    def test_dumps_string(self):
        result = _dumps("hello")
        assert result == '"hello"'

    def test_dumps_fallback_for_non_serializable(self):
        result = _dumps(object())
        assert isinstance(result, str)

    def test_loads_valid_json(self):
        result = _loads('{"key": "value"}')
        assert result == {"key": "value"}

    def test_loads_none(self):
        assert _loads(None) is None

    def test_loads_invalid_json_returns_string(self):
        result = _loads("not json")
        assert result == "not json"

    def test_loads_number_string(self):
        result = _loads("42")
        assert result == 42


# ══════════════════════════════════════════════════════════════════════════════
# CONVERSATIONAL MEMORY (SQLite)
# ══════════════════════════════════════════════════════════════════════════════

class TestConversationalMemory:
    """Tests for write_conversation, read_conversations, delete_conversation."""

    def test_write_conversation_returns_id(self, memory):
        rid = memory.write_conversation(
            thread_id="t1",
            role="user",
            content="Hello",
        )
        assert isinstance(rid, str)
        assert len(rid) > 0

    def test_write_conversation_with_custom_id(self, memory):
        rid = memory.write_conversation(
            thread_id="t1",
            role="user",
            content="Hello",
            record_id="custom_id",
        )
        assert rid == "custom_id"

    def test_write_conversation_with_metadata(self, memory):
        rid = memory.write_conversation(
            thread_id="t1",
            role="assistant",
            content="Hi there",
            metadata={"source": "test"},
        )
        rows = memory.read_conversations("t1")
        assert len(rows) == 1
        assert rows[0]["metadata"] == {"source": "test"}

    def test_write_conversation_with_summary_id(self, memory):
        rid = memory.write_conversation(
            thread_id="t1",
            role="user",
            content="Hello",
            summary_id="sum_123",
        )
        rows = memory.read_conversations("t1")
        assert rows[0]["summary_id"] == "sum_123"

    def test_read_conversations_chronological_order(self, memory, conversation_conn):
        # Insert with distinct timestamps to test ordering
        conversation_conn.execute(
            "INSERT INTO CONVERSATIONAL_MEMORY (id, thread_id, role, content, timestamp, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            ("m1", "t1", "user", "First", "2025-01-01 10:00:00", "2025-01-01 10:00:00"),
        )
        conversation_conn.execute(
            "INSERT INTO CONVERSATIONAL_MEMORY (id, thread_id, role, content, timestamp, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            ("m2", "t1", "assistant", "Second", "2025-01-01 10:01:00", "2025-01-01 10:01:00"),
        )
        conversation_conn.execute(
            "INSERT INTO CONVERSATIONAL_MEMORY (id, thread_id, role, content, timestamp, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            ("m3", "t1", "user", "Third", "2025-01-01 10:02:00", "2025-01-01 10:02:00"),
        )
        conversation_conn.commit()

        rows = memory.read_conversations("t1")
        assert len(rows) == 3
        assert rows[0]["content"] == "First"
        assert rows[1]["content"] == "Second"
        assert rows[2]["content"] == "Third"

    def test_read_conversations_with_limit(self, memory):
        for i in range(10):
            memory.write_conversation(thread_id="t1", role="user", content=f"Msg {i}")

        rows = memory.read_conversations("t1", limit=3)
        assert len(rows) == 3

    def test_read_conversations_filters_by_thread(self, memory):
        memory.write_conversation(thread_id="t1", role="user", content="Thread 1")
        memory.write_conversation(thread_id="t2", role="user", content="Thread 2")

        rows_t1 = memory.read_conversations("t1")
        rows_t2 = memory.read_conversations("t2")

        assert len(rows_t1) == 1
        assert len(rows_t2) == 1
        assert rows_t1[0]["content"] == "Thread 1"
        assert rows_t2[0]["content"] == "Thread 2"

    def test_read_conversations_before_timestamp(self, memory, conversation_conn):
        # Insert with distinct timestamps
        conversation_conn.execute(
            "INSERT INTO CONVERSATIONAL_MEMORY (id, thread_id, role, content, timestamp, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            ("old", "t1", "user", "Old", "2025-01-01 10:00:00", "2025-01-01 10:00:00"),
        )
        conversation_conn.execute(
            "INSERT INTO CONVERSATIONAL_MEMORY (id, thread_id, role, content, timestamp, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            ("new", "t1", "user", "New", "2025-01-01 11:00:00", "2025-01-01 11:00:00"),
        )
        conversation_conn.commit()

        # Read only messages before 11:00
        rows = memory.read_conversations("t1", before_timestamp="2025-01-01 11:00:00")
        assert len(rows) == 1
        assert rows[0]["content"] == "Old"

    def test_read_conversations_empty_thread(self, memory):
        rows = memory.read_conversations("nonexistent")
        assert rows == []

    def test_delete_conversation(self, memory):
        rid = memory.write_conversation(thread_id="t1", role="user", content="Delete me")
        memory.delete_conversation(rid)

        rows = memory.read_conversations("t1")
        assert len(rows) == 0

    def test_delete_nonexistent_conversation(self, memory):
        # Should not raise
        memory.delete_conversation("nonexistent_id")

    def test_delete_thread_conversations(self, memory):
        memory.write_conversation(thread_id="t1", role="user", content="Msg 1")
        memory.write_conversation(thread_id="t1", role="assistant", content="Msg 2")
        memory.write_conversation(thread_id="t2", role="user", content="Other thread")

        deleted = memory.delete_thread_conversations("t1")
        assert deleted == 2

        rows_t1 = memory.read_conversations("t1")
        rows_t2 = memory.read_conversations("t2")
        assert len(rows_t1) == 0
        assert len(rows_t2) == 1

    def test_update_conversation_summary_id(self, memory):
        rid = memory.write_conversation(thread_id="t1", role="user", content="Summarize me")
        memory.update_conversation_summary_id(rid, "summary_001")

        rows = memory.read_conversations("t1")
        assert rows[0]["summary_id"] == "summary_001"

    def test_conversation_row_structure(self, memory):
        memory.write_conversation(
            thread_id="t1",
            role="user",
            content="Test content",
            metadata={"key": "value"},
        )
        rows = memory.read_conversations("t1")
        row = rows[0]

        assert "id" in row
        assert "thread_id" in row
        assert "role" in row
        assert "content" in row
        assert "timestamp" in row
        assert "metadata" in row
        assert "summary_id" in row

        assert row["thread_id"] == "t1"
        assert row["role"] == "user"
        assert row["content"] == "Test content"


# ══════════════════════════════════════════════════════════════════════════════
# TOOL LOG MEMORY (SQLite)
# ══════════════════════════════════════════════════════════════════════════════

class TestToolLogMemory:
    """Tests for write_tool_log, read_tool_logs, delete_tool_log."""

    def test_write_tool_log_returns_id(self, memory):
        rid = memory.write_tool_log(
            thread_id="t1",
            tool_name="search",
        )
        assert isinstance(rid, str)
        assert len(rid) > 0

    def test_write_tool_log_with_all_fields(self, memory):
        rid = memory.write_tool_log(
            thread_id="t1",
            tool_name="web_search",
            tool_args={"query": "python testing"},
            result={"results": [{"title": "pytest"}]},
            result_preview="Found 1 result",
            status="success",
            tool_call_id="call_123",
            metadata={"duration_ms": 150},
        )
        rows = memory.read_tool_logs("t1")
        assert len(rows) == 1
        assert rows[0]["tool_name"] == "web_search"
        assert rows[0]["tool_args"] == {"query": "python testing"}
        assert rows[0]["result"] == {"results": [{"title": "pytest"}]}
        assert rows[0]["status"] == "success"

    def test_write_tool_log_with_error(self, memory):
        rid = memory.write_tool_log(
            thread_id="t1",
            tool_name="api_call",
            status="error",
            error_message="Connection timeout",
        )
        rows = memory.read_tool_logs("t1")
        assert rows[0]["status"] == "error"
        assert rows[0]["error_message"] == "Connection timeout"

    def test_read_tool_logs_chronological(self, memory, tool_log_conn):
        # Insert with distinct timestamps
        tool_log_conn.execute(
            "INSERT INTO TOOL_LOG_MEMORY (id, thread_id, tool_name, timestamp, created_at) VALUES (?, ?, ?, ?, ?)",
            ("1", "t1", "a", "2025-01-01 10:00:00", "2025-01-01 10:00:00"),
        )
        tool_log_conn.execute(
            "INSERT INTO TOOL_LOG_MEMORY (id, thread_id, tool_name, timestamp, created_at) VALUES (?, ?, ?, ?, ?)",
            ("2", "t1", "b", "2025-01-01 10:01:00", "2025-01-01 10:01:00"),
        )
        tool_log_conn.execute(
            "INSERT INTO TOOL_LOG_MEMORY (id, thread_id, tool_name, timestamp, created_at) VALUES (?, ?, ?, ?, ?)",
            ("3", "t1", "c", "2025-01-01 10:02:00", "2025-01-01 10:02:00"),
        )
        tool_log_conn.commit()

        rows = memory.read_tool_logs("t1")
        assert len(rows) == 3
        assert [r["tool_name"] for r in rows] == ["a", "b", "c"]

    def test_read_tool_logs_with_limit(self, memory):
        for i in range(10):
            memory.write_tool_log(thread_id="t1", tool_name=f"tool_{i}")

        rows = memory.read_tool_logs("t1", limit=5)
        assert len(rows) == 5

    def test_read_tool_logs_filter_by_tool_name(self, memory):
        memory.write_tool_log(thread_id="t1", tool_name="search")
        memory.write_tool_log(thread_id="t1", tool_name="calculator")
        memory.write_tool_log(thread_id="t1", tool_name="search")

        rows = memory.read_tool_logs("t1", tool_name="search")
        assert len(rows) == 2
        assert all(r["tool_name"] == "search" for r in rows)

    def test_read_tool_logs_filters_by_thread(self, memory):
        memory.write_tool_log(thread_id="t1", tool_name="search")
        memory.write_tool_log(thread_id="t2", tool_name="calculator")

        rows_t1 = memory.read_tool_logs("t1")
        rows_t2 = memory.read_tool_logs("t2")

        assert len(rows_t1) == 1
        assert len(rows_t2) == 1

    def test_read_tool_logs_empty_thread(self, memory):
        rows = memory.read_tool_logs("nonexistent")
        assert rows == []

    def test_delete_tool_log(self, memory):
        rid = memory.write_tool_log(thread_id="t1", tool_name="delete_me")
        memory.delete_tool_log(rid)

        rows = memory.read_tool_logs("t1")
        assert len(rows) == 0

    def test_tool_log_row_structure(self, memory):
        memory.write_tool_log(
            thread_id="t1",
            tool_name="test_tool",
            tool_args={"arg": "val"},
            result="output",
            result_preview="out",
            status="success",
            tool_call_id="call_1",
            metadata={"key": "value"},
        )
        rows = memory.read_tool_logs("t1")
        row = rows[0]

        expected_keys = [
            "id", "thread_id", "tool_call_id", "tool_name", "tool_args",
            "result", "result_preview", "status", "error_message",
            "metadata", "timestamp",
        ]
        for key in expected_keys:
            assert key in row, f"Missing key: {key}"


# ══════════════════════════════════════════════════════════════════════════════
# CHROMA VECTOR STORES
# ══════════════════════════════════════════════════════════════════════════════

class TestChromaStores:
    """Tests for knowledge, workflow, toolbox, entity, summary write/search."""

    def test_write_knowledge_returns_ids(self, memory, mock_store_manager):
        ids = memory.write_knowledge(["doc1", "doc2"])
        assert len(ids) == 2
        mock_store_manager.get_knowledge_base_store().add_documents.assert_called_once()

    def test_write_knowledge_with_metadatas(self, memory, mock_store_manager):
        ids = memory.write_knowledge(
            ["doc1"],
            metadatas=[{"source": "test"}],
        )
        assert len(ids) == 1

    def test_write_knowledge_with_custom_ids(self, memory, mock_store_manager):
        ids = memory.write_knowledge(["doc1"], ids=["custom_id"])
        assert ids == ["custom_id"]

    def test_write_knowledge_empty_list(self, memory):
        ids = memory.write_knowledge([])
        assert ids == []

    def test_search_knowledge(self, memory, mock_store_manager):
        mock_store_manager.get_knowledge_base_store().similarity_search.return_value = [
            Document(page_content="result", metadata={})
        ]
        results = memory.search_knowledge("query", k=3)
        assert len(results) == 1
        mock_store_manager.get_knowledge_base_store().similarity_search.assert_called_once_with("query", k=3)

    def test_search_knowledge_with_filter(self, memory, mock_store_manager):
        mock_store_manager.get_knowledge_base_store().similarity_search.return_value = []
        memory.search_knowledge("query", filter={"source": "test"})
        mock_store_manager.get_knowledge_base_store().similarity_search.assert_called_once_with(
            "query", k=5, filter={"source": "test"}
        )

    def test_write_workflow(self, memory, mock_store_manager):
        ids = memory.write_workflow(["step1", "step2"])
        assert len(ids) == 2
        mock_store_manager.get_workflow_store().add_documents.assert_called_once()

    def test_search_workflow(self, memory, mock_store_manager):
        mock_store_manager.get_workflow_store().similarity_search.return_value = []
        memory.search_workflow("deploy")
        mock_store_manager.get_workflow_store().similarity_search.assert_called_once()

    def test_write_toolbox(self, memory, mock_store_manager):
        ids = memory.write_toolbox(["tool1"])
        assert len(ids) == 1
        mock_store_manager.get_toolbox_store().add_documents.assert_called_once()

    def test_search_toolbox(self, memory, mock_store_manager):
        mock_store_manager.get_toolbox_store().similarity_search.return_value = []
        memory.search_toolbox("calculator")
        mock_store_manager.get_toolbox_store().similarity_search.assert_called_once()

    def test_write_summary(self, memory, mock_store_manager):
        ids = memory.write_summary(["summary1"])
        assert len(ids) == 1
        mock_store_manager.get_summary_store().add_documents.assert_called_once()

    def test_search_summary(self, memory, mock_store_manager):
        mock_store_manager.get_summary_store().similarity_search.return_value = []
        memory.search_summary("topic")
        mock_store_manager.get_summary_store().similarity_search.assert_called_once()


# ══════════════════════════════════════════════════════════════════════════════
# ENTITY MEMORY — Chroma fallback (no graph)
# ══════════════════════════════════════════════════════════════════════════════

class TestEntityMemoryChroma:
    """Tests for entity operations using Chroma (no graph memory)."""

    def test_write_entity_chroma(self, memory, mock_store_manager):
        ids = memory.write_entity(["Alice is a person"])
        assert len(ids) == 1
        mock_store_manager.get_entity_store().add_documents.assert_called_once()

    def test_search_entity_chroma(self, memory, mock_store_manager):
        mock_store_manager.get_entity_store().similarity_search.return_value = [
            Document(page_content="Alice", metadata={})
        ]
        results = memory.search_entity("Alice")
        assert len(results) == 1
        mock_store_manager.get_entity_store().similarity_search.assert_called_once()


# ══════════════════════════════════════════════════════════════════════════════
# ENTITY MEMORY — Graph DB (Neo4j)
# ══════════════════════════════════════════════════════════════════════════════

class TestEntityMemoryGraph:
    """Tests for entity operations using graph memory."""

    def test_has_graph_memory_true(self, memory_with_graph):
        assert memory_with_graph.has_graph_memory is True

    def test_has_graph_memory_false(self, memory):
        assert memory.has_graph_memory is False

    def test_write_entity_routes_to_graph(self, memory_with_graph, mock_graph_memory):
        ids = memory_with_graph.write_entity(
            ["Alice is a person"],
            metadatas=[{"name": "Alice", "type": "PERSON"}],
            thread_id="t1",
        )
        assert ids == ["entity_1", "entity_2"]
        mock_graph_memory.write_entity_documents.assert_called_once_with(
            texts=["Alice is a person"],
            metadatas=[{"name": "Alice", "type": "PERSON"}],
            thread_id="t1",
        )

    def test_search_entity_routes_to_graph(self, memory_with_graph, mock_graph_memory):
        results = memory_with_graph.search_entity("Alice", k=3)
        assert len(results) == 1
        assert results[0].metadata["name"] == "Alice"
        mock_graph_memory.search_entity_documents.assert_called_once_with(
            query="Alice",
            k=3,
            filter=None,
        )

    def test_search_entity_with_filter(self, memory_with_graph, mock_graph_memory):
        memory_with_graph.search_entity("Alice", filter={"type": "PERSON"})
        mock_graph_memory.search_entity_documents.assert_called_once_with(
            query="Alice",
            k=5,
            filter={"type": "PERSON"},
        )

    def test_write_entity_graph(self, memory_with_graph, mock_graph_memory):
        result = memory_with_graph.write_entity_graph(
            name="Bob",
            entity_type="PERSON",
            properties={"role": "Engineer"},
            thread_id="t1",
        )
        assert result == {"name": "Alice", "type": "PERSON"}
        mock_graph_memory.write_entity.assert_called_once_with(
            name="Bob",
            entity_type="PERSON",
            properties={"role": "Engineer"},
            thread_id="t1",
        )

    def test_write_entity_graph_no_graph_raises(self, memory):
        with pytest.raises(RuntimeError, match="Graph memory not configured"):
            memory.write_entity_graph(name="Bob")

    def test_write_relationship_graph(self, memory_with_graph, mock_graph_memory):
        result = memory_with_graph.write_relationship_graph(
            source="Alice",
            target="Acme",
            rel_type="WORKS_AT",
            properties={"since": "2023"},
        )
        assert result["source"] == "Alice"
        mock_graph_memory.write_relationship.assert_called_once_with(
            source="Alice",
            target="Acme",
            rel_type="WORKS_AT",
            properties={"since": "2023"},
        )

    def test_write_relationship_graph_no_graph_raises(self, memory):
        with pytest.raises(RuntimeError, match="Graph memory not configured"):
            memory.write_relationship_graph(source="A", target="B")

    def test_write_entities_from_text(self, memory_with_graph, mock_graph_memory):
        result = memory_with_graph.write_entities_from_text(
            "John works at Google",
            thread_id="t1",
        )
        assert result["entities_written"] == 2
        mock_graph_memory.write_entities_from_text.assert_called_once_with(
            text="John works at Google",
            thread_id="t1",
        )

    def test_write_entities_from_text_no_graph_raises(self, memory):
        with pytest.raises(RuntimeError, match="Graph memory not configured"):
            memory.write_entities_from_text("text")

    def test_get_entity(self, memory_with_graph, mock_graph_memory):
        result = memory_with_graph.get_entity("Alice")
        assert result == {"name": "Alice", "type": "PERSON"}
        mock_graph_memory.get_entity.assert_called_once_with("Alice")

    def test_get_entity_no_graph_raises(self, memory):
        with pytest.raises(RuntimeError, match="Graph memory not configured"):
            memory.get_entity("Alice")

    def test_get_entity_relationships(self, memory_with_graph, mock_graph_memory):
        result = memory_with_graph.get_entity_relationships(
            "Alice", direction="outgoing", rel_type="WORKS_AT", depth=2
        )
        assert len(result) == 1
        mock_graph_memory.get_entity_relationships.assert_called_once_with(
            name="Alice",
            direction="outgoing",
            rel_type="WORKS_AT",
            depth=2,
        )

    def test_get_entity_relationships_no_graph_raises(self, memory):
        with pytest.raises(RuntimeError, match="Graph memory not configured"):
            memory.get_entity_relationships("Alice")

    def test_get_related_entities(self, memory_with_graph, mock_graph_memory):
        result = memory_with_graph.get_related_entities("Alice", rel_type="WORKS_AT")
        assert len(result) == 1
        assert result[0]["entity_name"] == "Acme"
        mock_graph_memory.get_related_entities.assert_called_once_with(
            name="Alice",
            rel_type="WORKS_AT",
        )

    def test_get_related_entities_no_graph_raises(self, memory):
        with pytest.raises(RuntimeError, match="Graph memory not configured"):
            memory.get_related_entities("Alice")

    def test_delete_entity_graph(self, memory_with_graph, mock_graph_memory):
        result = memory_with_graph.delete_entity_graph("Alice")
        assert result is True
        mock_graph_memory.delete_entity.assert_called_once_with("Alice")

    def test_delete_entity_graph_no_graph_raises(self, memory):
        with pytest.raises(RuntimeError, match="Graph memory not configured"):
            memory.delete_entity_graph("Alice")

    def test_get_graph_stats(self, memory_with_graph, mock_graph_memory):
        result = memory_with_graph.get_graph_stats()
        assert result["entity_count"] == 10
        assert result["relationship_count"] == 15
        mock_graph_memory.get_stats.assert_called_once()

    def test_get_graph_stats_no_graph_raises(self, memory):
        with pytest.raises(RuntimeError, match="Graph memory not configured"):
            memory.get_graph_stats()


# ══════════════════════════════════════════════════════════════════════════════
# STORE RESOLUTION & GENERIC OPERATIONS
# ══════════════════════════════════════════════════════════════════════════════

class TestStoreResolution:
    """Tests for _resolve_store, write_to_store, search_store, delete_from_store."""

    def test_resolve_knowledge_store(self, memory, mock_store_manager):
        store = memory._resolve_store("knowledge")
        assert store == mock_store_manager.get_knowledge_base_store()

    def test_resolve_knowledge_base_store(self, memory, mock_store_manager):
        store = memory._resolve_store("knowledge_base")
        assert store == mock_store_manager.get_knowledge_base_store()

    def test_resolve_semantic_store(self, memory, mock_store_manager):
        store = memory._resolve_store("semantic")
        assert store == mock_store_manager.get_knowledge_base_store()

    def test_resolve_workflow_store(self, memory, mock_store_manager):
        store = memory._resolve_store("workflow")
        assert store == mock_store_manager.get_workflow_store()

    def test_resolve_toolbox_store(self, memory, mock_store_manager):
        store = memory._resolve_store("toolbox")
        assert store == mock_store_manager.get_toolbox_store()

    def test_resolve_entity_store(self, memory, mock_store_manager):
        store = memory._resolve_store("entity")
        assert store == mock_store_manager.get_entity_store()

    def test_resolve_summary_store(self, memory, mock_store_manager):
        store = memory._resolve_store("summary")
        assert store == mock_store_manager.get_summary_store()

    def test_resolve_unknown_store_raises(self, memory):
        with pytest.raises(ValueError, match="Unknown store"):
            memory._resolve_store("nonexistent")

    def test_write_to_store_by_name(self, memory, mock_store_manager):
        ids = memory.write_to_store("knowledge", ["doc1"])
        assert len(ids) == 1
        mock_store_manager.get_knowledge_base_store().add_documents.assert_called_once()

    def test_search_store_by_name(self, memory, mock_store_manager):
        mock_store_manager.get_workflow_store().similarity_search.return_value = []
        memory.search_store("workflow", "query")
        mock_store_manager.get_workflow_store().similarity_search.assert_called_once()

    def test_delete_from_store(self, memory, mock_store_manager):
        memory.delete_from_store("entity", ["id1", "id2"])
        mock_store_manager.get_entity_store().delete.assert_called_once_with(ids=["id1", "id2"])

    def test_delete_from_store_empty_ids(self, memory, mock_store_manager):
        memory.delete_from_store("entity", [])
        mock_store_manager.get_entity_store().delete.assert_not_called()

    def test_delete_from_store_unknown_name_raises(self, memory):
        with pytest.raises(ValueError, match="Unknown store"):
            memory.delete_from_store("nonexistent", ["id1"])


# ══════════════════════════════════════════════════════════════════════════════
# INITIALIZATION
# ══════════════════════════════════════════════════════════════════════════════

class TestInitialization:
    """Tests for MemoryManager initialization."""

    def test_init_without_graph(self, mock_store_manager, conversation_conn, tool_log_conn):
        mm = MemoryManager(
            store_manager=mock_store_manager,
            conversation_conn=conversation_conn,
            tool_log_conn=tool_log_conn,
        )
        assert mm.has_graph_memory is False

    def test_init_with_graph(self, mock_store_manager, conversation_conn, tool_log_conn, mock_graph_memory):
        mm = MemoryManager(
            store_manager=mock_store_manager,
            conversation_conn=conversation_conn,
            tool_log_conn=tool_log_conn,
            graph_memory=mock_graph_memory,
        )
        assert mm.has_graph_memory is True

    def test_store_aliases_completeness(self):
        expected = {"knowledge", "knowledge_base", "semantic", "workflow", "toolbox", "entity", "summary"}
        assert MemoryManager._STORE_ALIASES == expected
