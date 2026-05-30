"""
AgentMemory — ultra-simple memory interface for AI agents.

Two methods:
  - remember(text, **context) — write to memory (auto-routes to correct layer)
  - recall(query, **options)  — search memory across all layers

All 7-layer complexity is hidden behind this abstraction.

Usage:
    from src.memory import AgentMemory

    memory = AgentMemory.from_config(data_dir="./data")

    memory.remember("Alice is a developer")
    memory.remember("Hello!", role="user", thread_id="chat_1")

    results = memory.recall("Who is Alice?")
    for r in results:
        print(f"[{r.source}] {r.content} (score: {r.score})")

    memory.close()
"""

from __future__ import annotations

import os
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.memory.memory_manager import MemoryManager


# ─────────────────────────────────────────────────────────────────────────────
# MemoryResult
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MemoryResult:
    """A single result from memory recall."""
    content: str
    source: str           # "conversation" | "knowledge" | "entity" | "workflow" | "summary" | "tool_log"
    score: float          # 0.0 - 1.0 (higher = more relevant)
    metadata: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Auto-detection patterns
# ─────────────────────────────────────────────────────────────────────────────

# Entity patterns: "X is a Y", "X are Y", "X was Y", "X has Y"
_ENTITY_RE = re.compile(
    r"\b(?:is|are|was|were|has|have|had|works?|lives?|likes?|knows?|created?|built?)\b",
    re.IGNORECASE,
)

# Workflow patterns: "first...then", "step 1", "do X next", numbered lists
_WORKFLOW_RE = re.compile(
    r"\b(?:first|then|next|finally|step\s*\d|after that|before|after)\b",
    re.IGNORECASE,
)

def _detect_type(text: str) -> str:
    """Auto-detect memory type from text content.

    Returns: "entity" | "workflow" | "knowledge"
    """
    text_stripped = text.strip()

    # Short text with linking verbs → entity
    if len(text_stripped) < 200 and _ENTITY_RE.search(text_stripped):
        return "entity"

    # Multi-step / procedural text → workflow
    if _WORKFLOW_RE.search(text_stripped):
        return "workflow"

    # Default → knowledge
    return "knowledge"


# ─────────────────────────────────────────────────────────────────────────────
# AgentMemory
# ─────────────────────────────────────────────────────────────────────────────

class AgentMemory:
    """
    Ultra-simple memory interface for AI agents.

    Two methods:
      - remember(text, **context) — store in the right memory layer
      - recall(query, **options)  — search across all layers, ranked results
    """

    def __init__(self, memory_manager: "MemoryManager"):
        """Wrap an existing MemoryManager instance."""
        self._mm = memory_manager

    @classmethod
    def from_config(
        cls,
        data_dir: str = "./data",
        neo4j_uri: Optional[str] = None,
        neo4j_password: Optional[str] = None,
    ) -> "AgentMemory":
        """
        Create an AgentMemory with full auto-configuration.

        Args:
            data_dir: Directory for SQLite and Chroma data
            neo4j_uri: Optional Neo4j bolt URI (e.g., "bolt://localhost:7687")
            neo4j_password: Optional Neo4j password

        Returns:
            Configured AgentMemory instance
        """
        from src.memory.memory_manager import MemoryManager
        from src.memory.store_manager import (
            StoreManager,
            CONVERSATIONAL_TABLE,
            TOOL_LOG_TABLE,
            KNOWLEDGE_BASE_TABLE,
            WORKFLOW_TABLE,
            TOOLBOX_TABLE,
            ENTITY_TABLE,
            SUMMARY_TABLE,
            create_conversational_history_table,
            create_tool_log_table,
        )

        os.makedirs(data_dir, exist_ok=True)

        # SQLite connections
        conversation_conn = sqlite3.connect(f"{data_dir}/conversational_memory.db")
        tool_log_conn = sqlite3.connect(f"{data_dir}/tool_log_memory.db")

        # Create tables
        create_conversational_history_table(conversation_conn, CONVERSATIONAL_TABLE)
        create_tool_log_table(tool_log_conn, TOOL_LOG_TABLE)

        # Embedding model
        from langchain_huggingface import HuggingFaceEmbeddings
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # StoreManager
        store_manager = StoreManager(
            embedding_function=embedding_model,
            table_names={
                "knowledge_base": KNOWLEDGE_BASE_TABLE,
                "workflow": WORKFLOW_TABLE,
                "toolbox": TOOLBOX_TABLE,
                "entity": ENTITY_TABLE,
                "summary": SUMMARY_TABLE,
            },
            DATABASE_PATH=data_dir,
            conversational_table=CONVERSATIONAL_TABLE,
            tool_log_table=TOOL_LOG_TABLE,
        )

        # Optional: Graph memory (Neo4j)
        graph_memory = None
        if neo4j_uri and neo4j_password:
            try:
                from src.graphDB.entity_graph_memory import EntityGraphMemory
                graph_memory = EntityGraphMemory(
                    uri=neo4j_uri,
                    password=neo4j_password,
                )
            except Exception:
                pass  # Fall back to Chroma silently

        mm = MemoryManager(
            store_manager=store_manager,
            conversation_conn=conversation_conn,
            tool_log_conn=tool_log_conn,
            graph_memory=graph_memory,
        )
        return cls(mm)

    # ══════════════════════════════════════════════════════════════════════
    # REMEMBER — smart write
    # ══════════════════════════════════════════════════════════════════════

    def remember(
        self,
        text: str,
        *,
        role: Optional[str] = None,
        type: Optional[str] = None,   # noqa: A002
        thread_id: str = "default",
        tool_name: Optional[str] = None,
        tool_args: Optional[dict] = None,
        result: Optional[Any] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Store something in memory. Auto-routes to the correct layer.

        Args:
            text: The content to remember
            role: "user", "assistant", or "tool" (for conversation/tool logging)
            type: Explicit type: "knowledge", "entity", "workflow", "summary"
            thread_id: Thread/conversation ID (default: "default")
            tool_name: Tool name (required when role="tool")
            tool_args: Tool arguments (when role="tool")
            result: Tool result (when role="tool")
            metadata: Extra metadata to store

        Returns:
            ID or identifier of the stored memory

        Examples:
            memory.remember("Alice is a developer")                    # → entity
            memory.remember("Python is a language")                    # → knowledge
            memory.remember("First test, then deploy")                 # → workflow
            memory.remember("Hello!", role="user")                     # → conversation
            memory.remember("found 3 docs", role="tool", tool_name="search")  # → tool log
        """
        # ── Route by role ───────────────────────────────────────────────
        if role in ("user", "assistant"):
            return self._mm.write_conversation(
                thread_id=thread_id,
                role=role,
                content=text,
                metadata=metadata,
            )

        if role == "tool":
            return self._mm.write_tool_log(
                thread_id=thread_id,
                tool_name=tool_name or "unknown",
                tool_args=tool_args,
                result=result or text,
                result_preview=text[:200],
                metadata=metadata,
            )

        # ── Route by explicit type ──────────────────────────────────────
        t = (type or "").lower().strip()

        if t == "entity":
            return self._write_entity(text, thread_id, metadata)

        if t == "workflow":
            return self._write_workflow(text, metadata)

        if t == "summary":
            return self._write_summary(text, metadata)

        if t == "knowledge":
            return self._write_knowledge(text, metadata)

        # ── Auto-detect ─────────────────────────────────────────────────
        detected = _detect_type(text)

        if detected == "entity":
            return self._write_entity(text, thread_id, metadata)

        if detected == "workflow":
            return self._write_workflow(text, metadata)

        return self._write_knowledge(text, metadata)

    # ── Internal write helpers ──────────────────────────────────────────

    def _write_knowledge(self, text: str, metadata: Optional[dict]) -> str:
        meta = metadata or {}
        meta["source"] = meta.get("source", "agent")
        ids = self._mm.write_knowledge(
            texts=[text],
            metadatas=[meta],
        )
        return ids[0] if ids else ""

    def _write_entity(self, text: str, thread_id: str, metadata: Optional[dict]) -> str:
        meta = metadata or {}
        meta["thread_id"] = thread_id
        ids = self._mm.write_entity(
            texts=[text],
            metadatas=[meta],
            thread_id=thread_id,
        )
        return ids[0] if ids else ""

    def _write_workflow(self, text: str, metadata: Optional[dict]) -> str:
        meta = metadata or {}
        ids = self._mm.write_workflow(
            texts=[text],
            metadatas=[meta],
        )
        return ids[0] if ids else ""

    def _write_summary(self, text: str, metadata: Optional[dict]) -> str:
        meta = metadata or {}
        ids = self._mm.write_summary(
            texts=[text],
            metadatas=[meta],
        )
        return ids[0] if ids else ""

    # ══════════════════════════════════════════════════════════════════════
    # RECALL — smart search
    # ══════════════════════════════════════════════════════════════════════

    def recall(
        self,
        query: str,
        *,
        type: Optional[str] = None,   # noqa: A002
        thread_id: Optional[str] = None,
        k: int = 5,
    ) -> list[MemoryResult]:
        """
        Search memory across all layers. Returns ranked results.

        Args:
            query: What to search for
            type: Search only this layer: "knowledge", "entity", "workflow",
                  "summary", "conversation"
            thread_id: Filter by thread (for conversation search)
            k: Max results per layer (default: 5)

        Returns:
            List of MemoryResult, sorted by relevance (highest first)

        Examples:
            results = memory.recall("Who is Alice?")
            results = memory.recall("Python", type="knowledge")
            results = memory.recall("what happened?", type="conversation", thread_id="chat_1")
        """
        t = (type or "").lower().strip()
        results: list[MemoryResult] = []

        # ── Search specific layer only ───────────────────────────────────
        if t == "knowledge":
            docs = self._mm.search_knowledge(query, k=k)
            return self._docs_to_results(docs, "knowledge")

        if t == "entity":
            docs = self._mm.search_entity(query, k=k)
            return self._docs_to_results(docs, "entity")

        if t == "workflow":
            docs = self._mm.search_workflow(query, k=k)
            return self._docs_to_results(docs, "workflow")

        if t == "summary":
            docs = self._mm.search_summary(query, k=k)
            return self._docs_to_results(docs, "summary")

        if t == "conversation":
            messages = self._mm.read_conversations(
                thread_id=thread_id or "default",
                limit=k,
            )
            return [
                MemoryResult(
                    content=m["content"],
                    source="conversation",
                    score=1.0,  # Most recent = highest score
                    metadata={"role": m["role"], "thread_id": m["thread_id"]},
                )
                for m in messages
            ]

        if t == "tool_log":
            logs = self._mm.read_tool_logs(
                thread_id=thread_id or "default",
                limit=k,
            )
            return [
                MemoryResult(
                    content=log.get("result_preview") or str(log.get("result", "")),
                    source="tool_log",
                    score=1.0,
                    metadata={"tool_name": log["tool_name"], "status": log.get("status")},
                )
                for log in logs
            ]

        # ── Search all layers ────────────────────────────────────────────
        # Knowledge
        try:
            docs = self._mm.search_knowledge(query, k=k)
            results.extend(self._docs_to_results(docs, "knowledge", base_score=0.9))
        except Exception:
            pass

        # Entity
        try:
            docs = self._mm.search_entity(query, k=k)
            results.extend(self._docs_to_results(docs, "entity", base_score=0.85))
        except Exception:
            pass

        # Workflow
        try:
            docs = self._mm.search_workflow(query, k=k)
            results.extend(self._docs_to_results(docs, "workflow", base_score=0.7))
        except Exception:
            pass

        # Summary
        try:
            docs = self._mm.search_summary(query, k=k)
            results.extend(self._docs_to_results(docs, "summary", base_score=0.75))
        except Exception:
            pass

        # Conversation (recent messages)
        try:
            messages = self._mm.read_conversations(
                thread_id=thread_id or "default",
                limit=k,
            )
            for i, m in enumerate(messages):
                results.append(MemoryResult(
                    content=m["content"],
                    source="conversation",
                    score=0.6 * (1.0 - i * 0.1),  # Decay by recency
                    metadata={"role": m["role"], "thread_id": m["thread_id"]},
                ))
        except Exception:
            pass

        # ── Sort by score, deduplicate, limit ────────────────────────────
        results.sort(key=lambda r: r.score, reverse=True)
        results = self._deduplicate(results)
        return results[:k]

    # ── Internal helpers ────────────────────────────────────────────────

    def _docs_to_results(
        self,
        docs: list,
        source: str,
        base_score: float = 1.0,
    ) -> list[MemoryResult]:
        """Convert LangChain Documents to MemoryResults."""
        results = []
        for i, doc in enumerate(docs):
            score = base_score * (1.0 - i * 0.05)  # Slight decay per position
            results.append(MemoryResult(
                content=doc.page_content,
                source=source,
                score=round(max(score, 0.1), 2),
                metadata=getattr(doc, "metadata", {}),
            ))
        return results

    def _deduplicate(self, results: list[MemoryResult]) -> list[MemoryResult]:
        """Remove duplicate results based on content similarity."""
        seen = set()
        deduped = []
        for r in results:
            # Use first 100 chars as dedup key
            key = r.content[:100].lower().strip()
            if key not in seen:
                seen.add(key)
                deduped.append(r)
        return deduped

    # ══════════════════════════════════════════════════════════════════════
    # LIFECYCLE
    # ══════════════════════════════════════════════════════════════════════

    def close(self) -> None:
        """Clean shutdown of all connections."""
        self._mm.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
        return False
