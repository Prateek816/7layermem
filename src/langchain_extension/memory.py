from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING

from langchain_classic.schema import BaseMemory
from langchain_core.documents import Document
from pydantic import PrivateAttr

if TYPE_CHECKING:
    from src.memory.agent_memory import AgentMemory


class SevenLayerMemory(BaseMemory):
    """LangChain-compatible memory wrapping all 7 layers of 7layermem.

    Implements the BaseMemory protocol for use with ConversationChain and
    similar chains. On save_context, stores user input and assistant output
    in the conversation layer. On load_memory_variables, searches across
    all memory layers and returns formatted context.

    Args:
        agent_memory: An AgentMemory instance.
        thread_id: Thread/conversation identifier (default: "default").
        k: Max results per recall (default: 5).
        memory_key: Variable name injected into chain prompts (default: "history").
        recall_types: Optional list of layer types to search (default: all layers).
        return_docs: If True, return list[Document] instead of formatted string.
    """

    memory_key: str = "history"
    _agent_memory: Any = PrivateAttr()
    _thread_id: str = PrivateAttr(default="default")
    _k: int = PrivateAttr(default=5)
    _recall_types: Optional[list[str]] = PrivateAttr(default=None)
    _return_docs: bool = PrivateAttr(default=False)

    def __init__(
        self,
        agent_memory: AgentMemory,
        thread_id: str = "default",
        k: int = 5,
        memory_key: str = "history",
        recall_types: Optional[list[str]] = None,
        return_docs: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(memory_key=memory_key, **kwargs)
        self._agent_memory = agent_memory
        self._thread_id = thread_id
        self._k = k
        self._recall_types = recall_types
        self._return_docs = return_docs

    @property
    def memory_variables(self) -> list[str]:
        return [self.memory_key]

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Search memory layers and return context for the chain."""
        query = inputs.get("input", "") or inputs.get("history", "")

        if self._recall_types and len(self._recall_types) == 1:
            results = self._agent_memory.recall(
                query,
                type=self._recall_types[0],
                thread_id=self._thread_id,
                k=self._k,
            )
        else:
            results = self._agent_memory.recall(
                query,
                thread_id=self._thread_id,
                k=self._k,
            )

        if self._return_docs:
            docs = [
                Document(
                    page_content=r.content,
                    metadata={**(r.metadata or {}), "source": r.source},
                )
                for r in results
            ]
            return {self.memory_key: docs}

        lines = []
        for r in results:
            source_tag = f"[{r.source}]" if r.source != "conversation" else ""
            lines.append(f"{source_tag} {r.content}".strip())
        return {self.memory_key: "\n".join(lines)}

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, Any]) -> None:
        """Store user input and assistant output in the conversation layer."""
        if "input" in inputs:
            self._agent_memory.remember(
                inputs["input"],
                role="user",
                thread_id=self._thread_id,
            )
        if "output" in outputs:
            self._agent_memory.remember(
                outputs["output"],
                role="assistant",
                thread_id=self._thread_id,
            )

    def clear(self) -> None:
        """Remove all conversation messages for this thread."""
        self._agent_memory.clear_thread(self._thread_id)
