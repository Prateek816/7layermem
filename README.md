
# 7layermem

<img width="1022" height="700" alt="Screenshot 2026-05-31 at 2 05 00 AM" src="https://github.com/user-attachments/assets/077acbbd-85f3-4c3e-a5fb-2a19ade9b2fe" />


A **7-layer memory framework for AI agents** that enables persistent, structured long-term memory across conversations, knowledge, entities, workflows, summaries, tool interactions, and more.

Built for modern agentic systems, 7layermem helps agents retain context across sessions, reduce hallucinations through memory-grounded retrieval, and perform more reliably in long-running workflows — all through a simple API consisting of just two methods: `remember()` and `recall()`.

```python
from src.memory import AgentMemory

memory = AgentMemory.from_config(data_dir="./data")

memory.remember("Alice is a developer")          # auto-routes to entity memory
memory.remember("Python is a programming language")  # auto-routes to knowledge
memory.remember("Hello!", role="user")           # conversation memory

results = memory.recall("Who is Alice?")         # searches all layers
for r in results:
    print(f"[{r.source}] {r.content} (score: {r.score})")
```

## Why 7layermem?

Most AI agents lose context between sessions. 7layermem gives your agent **persistent, structured memory** across 7 cognitive layers:

| Layer | What it stores | Backend |
|-------|---------------|---------|
| **Conversational** | Chat history (episodic memory) | SQLite |
| **Knowledge** | Facts, documents (semantic memory) | ChromaDB |
| **Entity** | People, places, things (semantic memory) | ChromaDB or Neo4j |
| **Workflow** | Step-by-step procedures (procedural memory) | ChromaDB |
| **Toolbox** | Tool descriptions and schemas | ChromaDB |
| **Summary** | Conversation summaries | ChromaDB |
| **Tool Log** | Tool execution audit trail | SQLite |

## Quick Start

### Install

```bash
pip install -r requirements.txt
```

## Installation

### Prerequisites

- Python 3.10+
- pip
- (Optional) Neo4j for graph-based entity memory

### 1. Clone the Repository

```bash
git clone https://github.com/Prateek816/7layermem.git
cd 7layermem
```

### 2. Create a Virtual Environment

#### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies

Install all required dependencies:

```bash
pip install -r requirements.txt
```


### 4. Verify Installation

Run the demo:

```bash
python agent.py --demo
```

Expected output:

```text
✅ All databases and vector stores initialized successfully.
============================================================
AGENT MEMORY DEMO
============================================================
...
============================================================
DEMO COMPLETE
============================================================
```

### 5. Run Tests

Run the complete test suite:

```bash
python -m pytest tests/ -v
```

Or run individual test suites:

```bash
python -m pytest tests/test_agent_memory.py -v
python -m pytest tests/test_memory_manager.py -v
```

### Data Directory

On first run, 7layermem automatically creates:

```text
data/
├── conversations.db
├── tool_logs.db
├── chroma/
└── embeddings/
```

No manual database setup is required.

### Neo4j Configuration (Optional)

```python
from src.memory import AgentMemory

memory = AgentMemory.from_config(
    data_dir="./data",
    neo4j_uri="bolt://localhost:7687",
    neo4j_password="password"
)
```

If Neo4j is not configured, 7layermem automatically falls back to ChromaDB for entity memory.

### Upgrade

To pull the latest changes:

```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

### Use in your agent

```python
from src.memory import AgentMemory

# Initialize (auto-creates SQLite DBs, Chroma stores, embeddings)
memory = AgentMemory.from_config(data_dir="./data")

# REMEMBER — auto-detects the right memory layer
memory.remember("Alice is a developer")                          # → entity
memory.remember("Python is a programming language")              # → knowledge
memory.remember("First run tests, then deploy")                  # → workflow
memory.remember("Hello!", role="user", thread_id="chat_1")       # → conversation
memory.remember("found 3 results", role="tool", tool_name="search")  # → tool log

# RECALL — searches all layers, returns ranked results
results = memory.recall("Who is Alice?")
for r in results:
    print(f"[{r.source}] {r.content} (score: {r.score})")

# Filter by layer
knowledge_only = memory.recall("Python", type="knowledge")
entity_only = memory.recall("Alice", type="entity")
history = memory.recall("", type="conversation", thread_id="chat_1")

# Clean shutdown
memory.close()

# Or use as context manager
with AgentMemory.from_config() as memory:
    memory.remember("fact")
    results = memory.recall("query")
```

## How Routing Works

### `remember()` — Smart Write

When you call `remember(text)`, it auto-detects which layer to store in:

| Condition | Routes to |
|-----------|-----------|
| `role="user"` or `role="assistant"` | Conversational memory (SQLite) |
| `role="tool"` | Tool log memory (SQLite) |
| `type="entity"` | Entity memory (ChromaDB/Neo4j) |
| `type="workflow"` | Workflow memory (ChromaDB) |
| `type="summary"` | Summary memory (ChromaDB) |
| `type="knowledge"` | Knowledge base (ChromaDB) |
| Text contains "is/are/was" + short | **Auto → entity** |
| Text contains "then/first/step" | **Auto → workflow** |
| Otherwise | **Auto → knowledge** |

### `recall()` — Smart Search

When you call `recall(query)`, it searches all layers and ranks results:

| Layer | Base score | Decay |
|-------|-----------|-------|
| Knowledge | 0.9 | -0.05 per position |
| Entity | 0.85 | -0.05 per position |
| Summary | 0.75 | -0.05 per position |
| Workflow | 0.7 | -0.05 per position |
| Conversation | 0.6 | -0.1 per message (recency decay) |

Results are deduplicated and sorted by score.

## Advanced Usage

### With Neo4j Graph Memory

```python
memory = AgentMemory.from_config(
    data_dir="./data",
    neo4j_uri="bolt://localhost:7687",
    neo4j_password="your_password",
)

# Entity memory now uses graph DB
memory.remember("Alice works at Google", type="entity")
memory.recall("Alice")  # includes relationship context
```

### Full MemoryManager API

For advanced use cases, use `MemoryManager` directly:

```python
from src.memory import MemoryManager

memory = MemoryManager(
    store_manager=store_manager,
    conversation_conn=conv_conn,
    tool_log_conn=tool_log_conn,
    graph_memory=graph_memory,  # optional
)

# Conversational memory
memory.write_conversation(thread_id="t1", role="user", content="Hello")
messages = memory.read_conversations("t1", limit=50)

# Knowledge base
memory.write_knowledge(["Python is a language"], metadatas=[{"topic": "python"}])
docs = memory.search_knowledge("What is Python?", k=5)

# Entity memory
memory.write_entity(["Alice is a dev"], metadatas=[{"name": "Alice"}])
entities = memory.search_entity("Alice", k=5)

# Workflow memory
memory.write_workflow(["Step 1: Install", "Step 2: Configure"])
steps = memory.search_workflow("how to install?", k=3)

# Summary memory
memory.write_summary(["The user asked about Python"])
summaries = memory.search_summary("Python", k=3)

# Tool logging
memory.write_tool_log(thread_id="t1", tool_name="search", tool_args={"q": "test"})
logs = memory.read_tool_logs("t1", tool_name="search")

# Graph-specific (Neo4j only)
memory.write_entity_graph(name="Alice", entity_type="PERSON")
memory.write_relationship_graph(source="Alice", target="Google", rel_type="WORKS_AT")
memory.get_entity_relationships("Alice", depth=2)
```

### RAG (Retrieval-Augmented Generation)

```python
from src.RAG import RAGConfig, KnowledgeIngestor, HybridRetriever

# Configure
config = RAGConfig(
    knowledge_dir=Path("knowledge"),
    data_dir=Path("data"),
)

# Ingest documents
ingestor = KnowledgeIngestor(config)
ingestor.ingest_folder()

# Hybrid search (BM25 + vector)
retriever = HybridRetriever(config)
results = retriever.retrieve("What is Python?", k=5)
```

## Project Structure

```
7layermem/
├── agent.py                    # Demo agent (interactive + scripted)
├── src/
│   ├── memory/
│   │   ├── agent_memory.py     # AgentMemory — the simple public API
│   │   ├── memory_manager.py   # MemoryManager — full control API
│   │   └── store_manager.py    # ChromaDB + SQLite store management
│   ├── graphDB/
│   │   └── entity_graph_memory.py  # Neo4j graph entity memory
│   └── RAG/
│       ├── bm25_layer.py       # BM25 lexical search
│       ├── vector_layer.py     # ChromaDB semantic search
│       ├── hybrid_retriever.py # Combined BM25 + vector
│       ├── ingestor.py         # Document ingestion
│       ├── context_store.py    # Retrieved context cache
│       └── config.py           # RAG configuration
├── tests/
│   ├── test_agent_memory.py    # 36 tests for AgentMemory
│   └── test_memory_manager.py  # 86 tests for MemoryManager
├── examples/
│   ├── demo_rag.py             # RAG demo
│   └── setup_rag.py            # Dependency installer
├── data/                       # Runtime data (SQLite, ChromaDB)
└── knowledge/                  # Sample documents for RAG
```

## Running the Demo

```bash
# Interactive mode
python agent.py

# Scripted demo
python agent.py --demo

# With Neo4j
python agent.py --demo --neo4j bolt://localhost:7687 --neo4j-pass your_password
```

## Running Tests

```bash
# All tests
python -m pytest tests/ -v

# AgentMemory tests only
python -m pytest tests/test_agent_memory.py -v

# MemoryManager tests only
python -m pytest tests/test_memory_manager.py -v
```

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  User / Agent                    │
│         memory.remember() / memory.recall()      │
└──────────────────────┬──────────────────────────┘
                       │
              ┌────────▼────────┐
              │   AgentMemory   │  ← auto-detection + routing
              │  (abstraction)  │
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │  MemoryManager  │  ← unified read/write interface
              │   (32 methods)  │
              └────────┬────────┘
                       │
       ┌───────┬───────┼───────┬────────┐
       ▼       ▼       ▼       ▼        ▼
   ┌───────┐┌──────┐┌──────┐┌───────┐┌──────┐
   │ SQLite ││Chroma││Chroma││Chroma ││ Neo4j│
   │ Conv + ││Know +││Work +││Summary││Entity│
   │ ToolLog││Entity││Tool  ││       ││Graph │
   └───────┘└──────┘└──────┘└───────┘└──────┘
```

## License

MIT
