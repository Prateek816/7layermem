from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

if TYPE_CHECKING:
    from src.memory.agent_memory import AgentMemory


class SevenLayerChatMessageHistory(BaseChatMessageHistory):
    """Chat history backed by the 7layermem conversation layer.

    Stores messages using AgentMemory's conversation layer (SQLite) and
    retrieves them as LangChain BaseMessage objects.

    Args:
        agent_memory: An AgentMemory instance.
        thread_id: Thread/conversation identifier (default: "default").
        k: Maximum number of messages to retrieve (default: 50).
    """

    def __init__(
        self,
        agent_memory: AgentMemory,
        thread_id: str = "default",
        k: int = 50,
    ) -> None:
        self._agent_memory = agent_memory
        self._thread_id = thread_id
        self._k = k

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Add a list of messages efficiently in one pass."""
        for message in messages:
            if isinstance(message, HumanMessage):
                self._agent_memory.remember(
                    message.content,
                    role="user",
                    thread_id=self._thread_id,
                )
            elif isinstance(message, AIMessage):
                self._agent_memory.remember(
                    message.content,
                    role="assistant",
                    thread_id=self._thread_id,
                )
            else:
                self._agent_memory.remember(
                    str(message.content),
                    role="user",
                    thread_id=self._thread_id,
                    metadata={"message_type": type(message).__name__},
                )

    def add_message(self, message: BaseMessage) -> None:
        """Add a single message to the store."""
        self.add_messages([message])

    @property
    def messages(self) -> list[BaseMessage]:  # type: ignore[override]
        """Return chat history as LangChain BaseMessage objects."""
        results = self._agent_memory.recall(
            "",
            type="conversation",
            thread_id=self._thread_id,
            k=self._k,
        )
        output: list[BaseMessage] = []
        for r in results:
            role = r.metadata.get("role", "user") if r.metadata else "user"
            if role == "assistant":
                output.append(AIMessage(content=r.content))
            else:
                output.append(HumanMessage(content=r.content))
        return output

    def clear(self) -> None:
        """Remove all messages for this thread."""
        self._agent_memory.clear_thread(self._thread_id)
