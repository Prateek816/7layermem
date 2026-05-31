"""
Sample agent using the ultra-simple AgentMemory interface.

Two methods:
  - memory.remember(text) — store in the right memory layer
  - memory.recall(query)  — search across all layers

Usage:
  python agent.py                # interactive mode
  python agent.py --demo         # scripted demo
  python agent.py --neo4j bolt://localhost:7687 --neo4j-pass pass
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.memory import AgentMemory, MemoryResult


# ══════════════════════════════════════════════════════════════════════════════
# Agent
# ══════════════════════════════════════════════════════════════════════════════

class Agent:
    """A simple agent powered by AgentMemory."""

    def __init__(self, memory: AgentMemory, thread_id: str = "default"):
        self.memory = memory
        self.thread_id = thread_id

    def process(self, user_input: str) -> str:
        """
        Process a user message.

        1. Store user message
        2. Recall relevant context
        3. Generate response
        4. Store assistant response
        """
        # 1. Remember the user's message
        self.memory.remember(
            user_input,
            role="user",
            thread_id=self.thread_id,
        )

        # 2. Check for commands
        tool_result = self._check_commands(user_input)

        # 3. Recall relevant context from all memory layers
        context = self.memory.recall(user_input, k=5)

        # 4. Build response
        response = self._generate_response(user_input, context, tool_result)

        # 5. Remember the assistant's response
        self.memory.remember(
            response,
            role="assistant",
            thread_id=self.thread_id,
        )

        return response

    def _generate_response(
        self,
        user_input: str,
        context: list[MemoryResult],
        tool_result: str | None,
    ) -> str:
        """Generate a response using recalled context."""
        parts = []

        if context:
            context_lines = [
                f"  [{r.source}] {r.content[:100]}" for r in context[:3]
            ]
            parts.append("[Recalled from memory:]\n" + "\n".join(context_lines))

        if tool_result:
            parts.append(tool_result)

        if parts:
            return "\n\n".join(parts)
        return f"I received: {user_input}"

    # ── Commands ────────────────────────────────────────────────────────

    def _check_commands(self, user_input: str) -> str | None:
        """Check for slash commands."""
        lower = user_input.lower().strip()

        if lower.startswith("/remember "):
            text = user_input[10:].strip()
            self.memory.remember(text, type="knowledge", thread_id=self.thread_id)
            # Also log as tool
            self.memory.remember(
                f"Stored: {text[:50]}",
                role="tool",
                tool_name="remember",
                thread_id=self.thread_id,
            )
            return f"[Remembered: '{text[:60]}']"

        if lower.startswith("/entity "):
            name = user_input[8:].strip()
            self.memory.remember(name, type="entity", thread_id=self.thread_id)
            self.memory.remember(
                f"Stored entity: {name}",
                role="tool",
                tool_name="remember_entity",
                thread_id=self.thread_id,
            )
            return f"[Remembered entity: '{name}']"

        if lower.startswith("/workflow "):
            steps = user_input[9:].strip()
            self.memory.remember(steps, type="workflow", thread_id=self.thread_id)
            self.memory.remember(
                f"Stored workflow: {steps[:50]}",
                role="tool",
                tool_name="remember_workflow",
                thread_id=self.thread_id,
            )
            return f"[Remembered workflow: '{steps[:60]}']"

        if lower.startswith("/recall "):
            query = user_input[8:].strip()
            results = self.memory.recall(query, k=5)
            if not results:
                return "[No memories found]"
            lines = [f"  [{r.source}] {r.content[:100]}" for r in results]
            return "[Recalled:]\n" + "\n".join(lines)

        if lower.startswith("/history"):
            results = self.memory.recall("", type="conversation", thread_id=self.thread_id, k=10)
            if not results:
                return "[No conversation history]"
            lines = [f"  [{r.metadata.get('role', '?')}] {r.content[:80]}" for r in results]
            return "[Conversation history:]\n" + "\n".join(lines)

        if lower.startswith("/stats"):
            # Simple stats from memory
            conv = self.memory.recall("", type="conversation", thread_id=self.thread_id, k=100)
            knowledge = self.memory.recall("", type="knowledge", k=100)
            entity = self.memory.recall("", type="entity", k=100)
            stats = {
                "conversations": len(conv),
                "knowledge": len(knowledge),
                "entities": len(entity),
            }
            self.memory.remember(
                f"Stats: {json.dumps(stats)}",
                role="tool",
                tool_name="stats",
                thread_id=self.thread_id,
            )
            return f"[Stats]\n{json.dumps(stats, indent=2)}"

        return None


# ══════════════════════════════════════════════════════════════════════════════
# Demo
# ══════════════════════════════════════════════════════════════════════════════

def run_demo(memory: AgentMemory) -> None:
    """Run a scripted demo."""
    agent = Agent(memory, thread_id="demo")

    print("\n" + "=" * 60)
    print("AGENT MEMORY DEMO (AgentMemory)")
    print("=" * 60)

    # ── 1. Store knowledge ──────────────────────────────────────────────
    print("\n[1] Storing knowledge via remember()...")
    for fact in [
        "Python is a high-level programming language created by Guido van Rossum.",
        "LangChain is a framework for building LLM-powered applications.",
        "Neo4j is a graph database that stores data as nodes and edges.",
        "ChromaDB is an open-source vector database for embeddings.",
        "SQLite is a lightweight embedded relational database.",
    ]:
        memory.remember(fact)
    print("   Stored 5 facts")

    # ── 2. Store entities ───────────────────────────────────────────────
    print("\n[2] Storing entities via remember()...")
    memory.remember("Prateek is the developer of this agent system.", type="entity")
    memory.remember("Guido van Rossum created Python in 1991.", type="entity")
    print("   Stored 2 entities")

    # ── 3. Store workflow ───────────────────────────────────────────────
    print("\n[3] Storing workflow via remember()...")
    memory.remember("First run tests, then build Docker image, then deploy to production", type="workflow")
    print("   Stored 1 workflow")

    # ── 4. Conversation ─────────────────────────────────────────────────
    print("\n[4] Simulating conversation...")
    responses = [
        agent.process("Hello, I'm Prateek. What do you know about Python?"),
        agent.process("/remember RAG combines retrieval with generation for better answers"),
        agent.process("What entities do you know about?"),
        agent.process("/history"),
    ]
    for i, resp in enumerate(responses, 1):
        print(f"\n   Turn {i}:")
        print(f"   {resp[:200]}")

    # ── 5. Recall across all layers ─────────────────────────────────────
    print("\n[5] Recall 'database':")
    results = memory.recall("database", k=5)
    for r in results:
        print(f"   [{r.source}] {r.content[:100]} (score: {r.score})")

    print("\n[6] Recall 'Prateek':")
    results = memory.recall("Prateek", k=5)
    for r in results:
        print(f"   [{r.source}] {r.content[:100]} (score: {r.score})")

    print("\n[7] Recall workflow (type-filtered):")
    results = memory.recall("deploy", type="workflow", k=3)
    for r in results:
        print(f"   [{r.source}] {r.content[:100]} (score: {r.score})")

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# Interactive mode
# ══════════════════════════════════════════════════════════════════════════════

def run_interactive(memory: AgentMemory) -> None:
    """Run the agent in interactive mode."""
    thread_id = input("\nEnter thread ID (or press Enter for 'default'): ").strip() or "default"
    agent = Agent(memory, thread_id=thread_id)

    print(f"\nAgent ready (thread: {thread_id})")
    print("Commands:")
    print("  /remember <text>   - Store a fact in knowledge base")
    print("  /entity <name>     - Remember an entity")
    print("  /workflow <steps>  - Save workflow steps")
    print("  /recall <query>    - Search all memory")
    print("  /history           - Show conversation history")
    print("  /stats             - Show memory stats")
    print("  /quit              - Exit")
    print("  (or just type naturally)")
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
        if user_input.lower() in ("/quit", "/exit", "quit", "exit"):
            break

        response = agent.process(user_input)
        print(f"\nAgent: {response}\n")

    print("\nGoodbye!")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Sample agent with 7-layer memory")
    parser.add_argument("--demo", action="store_true", help="Run scripted demo")
    parser.add_argument("--neo4j", type=str, default=None, help="Neo4j URI")
    parser.add_argument("--neo4j-pass", type=str, default=None, help="Neo4j password")
    args = parser.parse_args()

    print("Initializing AgentMemory...")
    memory = AgentMemory.from_config(
        data_dir="./data",
        neo4j_uri=args.neo4j,
        neo4j_password=args.neo4j_pass,
    )
    print("Ready!")

    try:
        if args.demo:
            run_demo(memory)
        else:
            run_interactive(memory)
    finally:
        memory.close()


if __name__ == "__main__":
    main()
