from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from src.langchain_extension.memory import SevenLayerMemory
from src.memory.agent_memory import MemoryResult


@pytest.fixture
def mock_agent_memory():
    return MagicMock()


@pytest.fixture
def memory(mock_agent_memory):
    return SevenLayerMemory(mock_agent_memory, thread_id="test_thread")


class TestMemoryVariables:
    def test_default_memory_key(self, memory):
        assert memory.memory_variables == ["history"]

    def test_custom_memory_key(self, mock_agent_memory):
        mem = SevenLayerMemory(mock_agent_memory, memory_key="context")
        assert mem.memory_variables == ["context"]


class TestSaveContext:
    def test_saves_user_input(self, memory, mock_agent_memory):
        memory.save_context({"input": "Hello"}, {"output": "Hi"})
        mock_agent_memory.remember.assert_any_call(
            "Hello", role="user", thread_id="test_thread"
        )

    def test_saves_assistant_output(self, memory, mock_agent_memory):
        memory.save_context({"input": "Hello"}, {"output": "Hi"})
        mock_agent_memory.remember.assert_any_call(
            "Hi", role="assistant", thread_id="test_thread"
        )

    def test_no_output(self, memory, mock_agent_memory):
        memory.save_context({"input": "Hello"}, {})
        assert mock_agent_memory.remember.call_count == 1

    def test_no_input(self, memory, mock_agent_memory):
        memory.save_context({}, {"output": "Hi"})
        assert mock_agent_memory.remember.call_count == 1


class TestLoadMemoryVariables:
    def test_formats_string(self, memory, mock_agent_memory):
        mock_agent_memory.recall.return_value = [
            MemoryResult(content="fact1", source="knowledge", score=0.9, metadata={}),
            MemoryResult(content="fact2", source="entity", score=0.85, metadata={}),
        ]
        result = memory.load_memory_variables({"input": "query"})
        assert result == {"history": "[knowledge] fact1\n[entity] fact2"}

    def test_conversation_no_tag(self, memory, mock_agent_memory):
        mock_agent_memory.recall.return_value = [
            MemoryResult(content="msg", source="conversation", score=0.6, metadata={}),
        ]
        result = memory.load_memory_variables({"input": "query"})
        assert result == {"history": "msg"}

    def test_returns_docs(self, mock_agent_memory):
        mem = SevenLayerMemory(mock_agent_memory, return_docs=True)
        mock_agent_memory.recall.return_value = [
            MemoryResult(content="fact", source="knowledge", score=0.9,
                         metadata={"key": "val"}),
        ]
        result = mem.load_memory_variables({"input": "query"})
        docs = result["history"]
        assert len(docs) == 1
        assert isinstance(docs[0], Document)
        assert docs[0].page_content == "fact"
        assert docs[0].metadata["source"] == "knowledge"
        assert docs[0].metadata["key"] == "val"

    def test_extracts_query_from_input(self, memory, mock_agent_memory):
        mock_agent_memory.recall.return_value = []
        memory.load_memory_variables({"input": "Python"})
        mock_agent_memory.recall.assert_called_with(
            "Python", thread_id="test_thread", k=5
        )

    def test_empty_query(self, memory, mock_agent_memory):
        mock_agent_memory.recall.return_value = []
        memory.load_memory_variables({})
        mock_agent_memory.recall.assert_called_with(
            "", thread_id="test_thread", k=5
        )

    def test_recall_types_filter(self, mock_agent_memory):
        mem = SevenLayerMemory(mock_agent_memory, recall_types=["knowledge"])
        mock_agent_memory.recall.return_value = []
        mem.load_memory_variables({"input": "q"})
        mock_agent_memory.recall.assert_called_with(
            "q", type="knowledge", thread_id="default", k=5
        )

    def test_empty_result(self, memory, mock_agent_memory):
        mock_agent_memory.recall.return_value = []
        result = memory.load_memory_variables({"input": "q"})
        assert result == {"history": ""}


class TestClear:
    def test_clear(self, memory, mock_agent_memory):
        memory.clear()
        mock_agent_memory.clear_thread.assert_called_once_with("test_thread")
