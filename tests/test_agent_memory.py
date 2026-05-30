"""
test_agent_memory.py

Tests for AgentMemory — the ultra-simple remember()/recall() interface.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.memory.agent_memory import AgentMemory, MemoryResult, _detect_type


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_memory_manager():
    """Create a mock MemoryManager."""
    mm = MagicMock()
    mm.write_conversation.return_value = "conv_id_1"
    mm.write_tool_log.return_value = "tool_id_1"
    mm.write_knowledge.return_value = ["knowledge_id_1"]
    mm.write_entity.return_value = ["entity_id_1"]
    mm.write_workflow.return_value = ["workflow_id_1"]
    mm.write_summary.return_value = ["summary_id_1"]

    mm.search_knowledge.return_value = [
        Document(page_content="Python is a language", metadata={"topic": "python"}),
    ]
    mm.search_entity.return_value = [
        Document(page_content="Alice is a developer", metadata={"name": "Alice"}),
    ]
    mm.search_workflow.return_value = [
        Document(page_content="Step 1: Run tests", metadata={}),
    ]
    mm.search_summary.return_value = [
        Document(page_content="Summary of conversation", metadata={}),
    ]

    mm.read_conversations.return_value = [
        {"role": "user", "content": "Hello", "thread_id": "t1", "timestamp": "2025-01-01"},
        {"role": "assistant", "content": "Hi there", "thread_id": "t1", "timestamp": "2025-01-01"},
    ]
    mm.read_tool_logs.return_value = [
        {"tool_name": "search", "result_preview": "Found 3 results", "status": "success"},
    ]

    return mm


@pytest.fixture
def memory(mock_memory_manager):
    """AgentMemory wrapping a mock MemoryManager."""
    return AgentMemory(mock_memory_manager)


# ══════════════════════════════════════════════════════════════════════════════
# Auto-detection
# ══════════════════════════════════════════════════════════════════════════════

class TestAutoDetect:
    """Tests for _detect_type heuristic."""

    def test_entity_detection_is_verb(self):
        assert _detect_type("Alice is a developer") == "entity"

    def test_entity_detection_was(self):
        assert _detect_type("Python was created by Guido") == "entity"

    def test_entity_detection_works(self):
        assert _detect_type("Bob works at Google") == "entity"

    def test_workflow_detection_then(self):
        assert _detect_type("First run tests, then deploy") == "workflow"

    def test_workflow_detection_step(self):
        assert _detect_type("Step 1: Install dependencies") == "workflow"

    def test_workflow_detection_next(self):
        assert _detect_type("Next, configure the database") == "workflow"

    def test_knowledge_default(self):
        assert _detect_type("RAG combines retrieval with generation") == "knowledge"

    def test_knowledge_long_text(self):
        long = "This is a very long text " * 20 + "is still knowledge"
        # Long text with "is" should still be knowledge (too long for entity)
        assert _detect_type(long) == "knowledge"


# ══════════════════════════════════════════════════════════════════════════════
# REMEMBER — routing
# ══════════════════════════════════════════════════════════════════════════════

class TestRemember:
    """Tests for remember() routing logic."""

    def test_remember_user_role(self, memory, mock_memory_manager):
        result = memory.remember("Hello!", role="user", thread_id="t1")
        mock_memory_manager.write_conversation.assert_called_once_with(
            thread_id="t1", role="user", content="Hello!", metadata=None,
        )
        assert result == "conv_id_1"

    def test_remember_assistant_role(self, memory, mock_memory_manager):
        memory.remember("Hi!", role="assistant", thread_id="t1")
        mock_memory_manager.write_conversation.assert_called_once_with(
            thread_id="t1", role="assistant", content="Hi!", metadata=None,
        )

    def test_remember_tool_role(self, memory, mock_memory_manager):
        result = memory.remember(
            "found results",
            role="tool",
            tool_name="search",
            tool_args={"q": "test"},
            result={"count": 3},
            thread_id="t1",
        )
        mock_memory_manager.write_tool_log.assert_called_once()
        assert result == "tool_id_1"

    def test_remember_explicit_entity(self, memory, mock_memory_manager):
        result = memory.remember("Alice is a dev", type="entity", thread_id="t1")
        mock_memory_manager.write_entity.assert_called_once()
        assert result == "entity_id_1"

    def test_remember_explicit_workflow(self, memory, mock_memory_manager):
        result = memory.remember("First test, then deploy", type="workflow")
        mock_memory_manager.write_workflow.assert_called_once()
        assert result == "workflow_id_1"

    def test_remember_explicit_summary(self, memory, mock_memory_manager):
        result = memory.remember("Conversation summary", type="summary")
        mock_memory_manager.write_summary.assert_called_once()
        assert result == "summary_id_1"

    def test_remember_explicit_knowledge(self, memory, mock_memory_manager):
        result = memory.remember("Python is a language", type="knowledge")
        mock_memory_manager.write_knowledge.assert_called_once()
        assert result == "knowledge_id_1"

    def test_remember_auto_detect_entity(self, memory, mock_memory_manager):
        memory.remember("Alice is a developer")
        mock_memory_manager.write_entity.assert_called_once()

    def test_remember_auto_detect_workflow(self, memory, mock_memory_manager):
        memory.remember("First run tests, then deploy")
        mock_memory_manager.write_workflow.assert_called_once()

    def test_remember_auto_detect_knowledge(self, memory, mock_memory_manager):
        memory.remember("RAG combines retrieval with generation")
        mock_memory_manager.write_knowledge.assert_called_once()

    def test_remember_with_metadata(self, memory, mock_memory_manager):
        memory.remember("fact", type="knowledge", metadata={"source": "user"})
        call_args = mock_memory_manager.write_knowledge.call_args
        assert call_args.kwargs["metadatas"][0]["source"] == "user"

    def test_remember_default_thread_id(self, memory, mock_memory_manager):
        memory.remember("Hello!", role="user")
        call_args = mock_memory_manager.write_conversation.call_args
        assert call_args.kwargs["thread_id"] == "default"


# ══════════════════════════════════════════════════════════════════════════════
# RECALL
# ══════════════════════════════════════════════════════════════════════════════

class TestRecall:
    """Tests for recall() search logic."""

    def test_recall_knowledge_only(self, memory, mock_memory_manager):
        results = memory.recall("Python", type="knowledge")
        mock_memory_manager.search_knowledge.assert_called_once_with("Python", k=5)
        assert len(results) == 1
        assert results[0].source == "knowledge"
        assert results[0].content == "Python is a language"

    def test_recall_entity_only(self, memory, mock_memory_manager):
        results = memory.recall("Alice", type="entity")
        mock_memory_manager.search_entity.assert_called_once_with("Alice", k=5)
        assert len(results) == 1
        assert results[0].source == "entity"

    def test_recall_workflow_only(self, memory, mock_memory_manager):
        results = memory.recall("deploy", type="workflow")
        mock_memory_manager.search_workflow.assert_called_once_with("deploy", k=5)
        assert len(results) == 1
        assert results[0].source == "workflow"

    def test_recall_summary_only(self, memory, mock_memory_manager):
        results = memory.recall("summary", type="summary")
        mock_memory_manager.search_summary.assert_called_once_with("summary", k=5)
        assert len(results) == 1
        assert results[0].source == "summary"

    def test_recall_conversation_only(self, memory, mock_memory_manager):
        results = memory.recall("hello", type="conversation", thread_id="t1")
        mock_memory_manager.read_conversations.assert_called_once_with(
            thread_id="t1", limit=5,
        )
        assert len(results) == 2
        assert all(r.source == "conversation" for r in results)

    def test_recall_tool_log_only(self, memory, mock_memory_manager):
        results = memory.recall("search", type="tool_log", thread_id="t1")
        mock_memory_manager.read_tool_logs.assert_called_once_with(
            thread_id="t1", limit=5,
        )
        assert len(results) == 1
        assert results[0].source == "tool_log"

    def test_recall_all_layers(self, memory, mock_memory_manager):
        results = memory.recall("test query", k=10)
        # Should search knowledge, entity, workflow, summary, and conversation
        mock_memory_manager.search_knowledge.assert_called_once()
        mock_memory_manager.search_entity.assert_called_once()
        mock_memory_manager.search_workflow.assert_called_once()
        mock_memory_manager.search_summary.assert_called_once()
        mock_memory_manager.read_conversations.assert_called_once()
        # Results should be sorted by score
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

    def test_recall_with_k_limit(self, memory, mock_memory_manager):
        results = memory.recall("test", type="knowledge", k=3)
        mock_memory_manager.search_knowledge.assert_called_once_with("test", k=3)

    def test_recall_deduplicates(self, mock_memory_manager):
        # Return same content from different stores
        mock_memory_manager.search_knowledge.return_value = [
            Document(page_content="Same content", metadata={}),
        ]
        mock_memory_manager.search_entity.return_value = [
            Document(page_content="Same content", metadata={}),
        ]
        mem = AgentMemory(mock_memory_manager)
        results = mem.recall("test", k=10)
        # Should deduplicate
        contents = [r.content for r in results]
        assert contents.count("Same content") == 1

    def test_recall_returns_memory_results(self, memory):
        results = memory.recall("test")
        assert all(isinstance(r, MemoryResult) for r in results)

    def test_recall_memory_result_fields(self, memory):
        results = memory.recall("Python", type="knowledge")
        r = results[0]
        assert r.content == "Python is a language"
        assert r.source == "knowledge"
        assert 0.0 <= r.score <= 1.0
        assert isinstance(r.metadata, dict)

    def test_recall_empty_conversation(self, mock_memory_manager):
        mock_memory_manager.read_conversations.return_value = []
        mem = AgentMemory(mock_memory_manager)
        results = mem.recall("test", type="conversation", thread_id="empty")
        assert results == []


# ══════════════════════════════════════════════════════════════════════════════
# MemoryResult
# ══════════════════════════════════════════════════════════════════════════════

class TestMemoryResult:
    """Tests for MemoryResult dataclass."""

    def test_create_minimal(self):
        r = MemoryResult(content="test", source="knowledge", score=0.9)
        assert r.content == "test"
        assert r.source == "knowledge"
        assert r.score == 0.9
        assert r.metadata == {}

    def test_create_with_metadata(self):
        r = MemoryResult(
            content="test", source="entity", score=0.8,
            metadata={"name": "Alice"},
        )
        assert r.metadata == {"name": "Alice"}


# ══════════════════════════════════════════════════════════════════════════════
# Lifecycle
# ══════════════════════════════════════════════════════════════════════════════

class TestLifecycle:
    """Tests for close() and context manager."""

    def test_close(self, memory, mock_memory_manager):
        memory.close()
        mock_memory_manager.close.assert_called_once()

    def test_context_manager(self, mock_memory_manager):
        with AgentMemory(mock_memory_manager) as mem:
            mem.remember("test", type="knowledge")
        mock_memory_manager.close.assert_called_once()
