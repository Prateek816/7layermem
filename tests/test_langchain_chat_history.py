from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.langchain_extension.chat_message_history import SevenLayerChatMessageHistory
from src.memory.agent_memory import MemoryResult


@pytest.fixture
def mock_agent_memory():
    return MagicMock()


@pytest.fixture
def history(mock_agent_memory):
    return SevenLayerChatMessageHistory(mock_agent_memory, thread_id="test_thread", k=10)


class TestAddMessages:
    def test_add_user_message_string(self, history, mock_agent_memory):
        history.add_user_message("Hello!")
        mock_agent_memory.remember.assert_called_once_with(
            "Hello!", role="user", thread_id="test_thread"
        )

    def test_add_ai_message_string(self, history, mock_agent_memory):
        history.add_ai_message("Hi there!")
        mock_agent_memory.remember.assert_called_once_with(
            "Hi there!", role="assistant", thread_id="test_thread"
        )

    def test_add_message_human(self, history, mock_agent_memory):
        history.add_message(HumanMessage(content="Test"))
        mock_agent_memory.remember.assert_called_once_with(
            "Test", role="user", thread_id="test_thread"
        )

    def test_add_message_ai(self, history, mock_agent_memory):
        history.add_message(AIMessage(content="Response"))
        mock_agent_memory.remember.assert_called_once_with(
            "Response", role="assistant", thread_id="test_thread"
        )

    def test_add_message_system(self, history, mock_agent_memory):
        history.add_message(SystemMessage(content="System prompt"))
        mock_agent_memory.remember.assert_called_once_with(
            "System prompt",
            role="user",
            thread_id="test_thread",
            metadata={"message_type": "SystemMessage"},
        )

    def test_add_messages_bulk(self, history, mock_agent_memory):
        messages = [
            HumanMessage(content="Q1"),
            AIMessage(content="A1"),
            HumanMessage(content="Q2"),
        ]
        history.add_messages(messages)
        assert mock_agent_memory.remember.call_count == 3


class TestMessagesProperty:
    def test_messages_empty(self, history, mock_agent_memory):
        mock_agent_memory.recall.return_value = []
        assert history.messages == []

    def test_messages_populated(self, history, mock_agent_memory):
        mock_agent_memory.recall.return_value = [
            MemoryResult(content="Hi", source="conversation", score=1.0,
                         metadata={"role": "user", "thread_id": "test_thread"}),
            MemoryResult(content="Hello!", source="conversation", score=1.0,
                         metadata={"role": "assistant", "thread_id": "test_thread"}),
        ]
        msgs = history.messages
        assert len(msgs) == 2
        assert isinstance(msgs[0], HumanMessage)
        assert msgs[0].content == "Hi"
        assert isinstance(msgs[1], AIMessage)
        assert msgs[1].content == "Hello!"

    def test_messages_chronological_order(self, history, mock_agent_memory):
        mock_agent_memory.recall.return_value = [
            MemoryResult(content="first", source="conversation", score=1.0,
                         metadata={"role": "user"}),
            MemoryResult(content="second", source="conversation", score=1.0,
                         metadata={"role": "assistant"}),
            MemoryResult(content="third", source="conversation", score=1.0,
                         metadata={"role": "user"}),
        ]
        msgs = history.messages
        assert [m.content for m in msgs] == ["first", "second", "third"]

    def test_messages_calls_recall_with_correct_args(self, history, mock_agent_memory):
        mock_agent_memory.recall.return_value = []
        history.messages
        mock_agent_memory.recall.assert_called_once_with(
            "", type="conversation", thread_id="test_thread", k=10
        )

    def test_messages_defaults_role_to_user(self, history, mock_agent_memory):
        mock_agent_memory.recall.return_value = [
            MemoryResult(content="msg", source="conversation", score=1.0,
                         metadata={}),
        ]
        msgs = history.messages
        assert isinstance(msgs[0], HumanMessage)


class TestClear:
    def test_clear_calls_clear_thread(self, history, mock_agent_memory):
        history.clear()
        mock_agent_memory.clear_thread.assert_called_once_with("test_thread")


class TestThreadIdIsolation:
    def test_different_threads_are_isolated(self, mock_agent_memory):
        h1 = SevenLayerChatMessageHistory(mock_agent_memory, thread_id="t1")
        h2 = SevenLayerChatMessageHistory(mock_agent_memory, thread_id="t2")

        h1.add_user_message("msg1")
        h2.add_user_message("msg2")

        calls = mock_agent_memory.remember.call_args_list
        assert calls[0].kwargs["thread_id"] == "t1"
        assert calls[1].kwargs["thread_id"] == "t2"
