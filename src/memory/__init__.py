"""
Memory management package.

Two ways to use:

1. Simple (recommended for most users):
   from src.memory import AgentMemory
   memory = AgentMemory.from_config(data_dir="./data")
   memory.remember("Alice is a developer")
   results = memory.recall("Who is Alice?")

2. Advanced (full control):
   from src.memory import MemoryManager
   memory = MemoryManager(store_manager=sm, ...)
   memory.write_knowledge(["Python is a language"])
   memory.write_entity(["Alice"], metadatas=[{"name": "Alice"}])
"""

from .memory_manager import MemoryManager
from .store_manager import StoreManager
from .agent_memory import AgentMemory, MemoryResult

__all__ = ["AgentMemory", "MemoryResult", "MemoryManager", "StoreManager"]
