# LangChain Extension for 7layermem

LangChain-compatible wrappers that let you plug 7layermem's 7-layer memory system into LangChain chains, agents, and chat models.

## Components

| Class | Base Class | Purpose |
|-------|-----------|---------|
| `SevenLayerChatMessageHistory` | `BaseChatMessageHistory` | Chat history for agents and chat models |
| `SevenLayerMemory` | `BaseMemory` | Memory for `ConversationChain` and similar chains |
| `SevenLayerStore` | `BaseStore` | Key-value store backed by vector stores |

## Quick Start

```python
from src.memory import AgentMemory
from src.langchain_extension import (
    SevenLayerChatMessageHistory,
    SevenLayerMemory,
    SevenLayerStore,
)

memory = AgentMemory.from_config(data_dir="./data")
```

## SevenLayerChatMessageHistory

Wraps 7layermem's conversation layer as a LangChain `BaseChatMessageHistory`. Works with LangChain agents, chat model runnables, and any chain that accepts `chat_history`.

```python
from langchain_core.messages import HumanMessage, AIMessage

history = SevenLayerChatMessageHistory(memory, thread_id="chat_42", k=50)

# Add messages
history.add_user_message("What's the capital of France?")
history.add_ai_message("Paris is the capital of France.")

# Or add message objects
history.add_message(HumanMessage(content="And its population?"))

# Retrieve history
messages = history.messages  # list[BaseMessage]
for msg in messages:
    print(f"{type(msg).__name__}: {msg.content}")

# Clear history
history.clear()
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agent_memory` | `AgentMemory` | required | 7layermem AgentMemory instance |
| `thread_id` | `str` | `"default"` | Conversation thread identifier |
| `k` | `int` | `50` | Max messages to retrieve |

### With LangChain Chat Models

```python
from langchain_core.runnables.history import RunnableWithMessageHistory

def get_history(session_id: str) -> SevenLayerChatMessageHistory:
    return SevenLayerChatMessageHistory(memory, thread_id=session_id)

chain = RunnableWithMessageHistory(
    your_chain,
    get_history,
    input_messages_key="input",
    history_messages_key="history",
)

result = chain.invoke(
    {"input": "What did we discuss earlier?"},
    config={"configurable": {"session_id": "user_123"}},
)
```

## SevenLayerMemory

Implements the `BaseMemory` protocol (`save_context` / `load_memory_variables` / `clear`) for use with `ConversationChain` and similar chains. Searches across all 7 memory layers (or a subset) to provide rich context.

```python
from langchain_classic.chains import ConversationChain
from langchain_groq import ChatGroq

seven_layer_mem = SevenLayerMemory(
    memory,
    thread_id="chain_session",
    k=10,
    memory_key="history",         # variable name in chain prompt
    recall_types=None,            # None = search all layers
    return_docs=False,            # True to get Document objects
)

chain = ConversationChain(
    llm=ChatGroq(model="llama3-8b-8192"),
    memory=seven_layer_mem,
)

chain.predict(input="Tell me about Alice")
# -> Saves to conversation layer, searches all layers for relevant context

chain.predict(input="What does Alice do?")
# -> Recalls context from knowledge, entity, workflow, etc. layers
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agent_memory` | `AgentMemory` | required | 7layermem AgentMemory instance |
| `thread_id` | `str` | `"default"` | Conversation thread identifier |
| `k` | `int` | `5` | Max results per recall |
| `memory_key` | `str` | `"history"` | Variable name injected into chain prompts |
| `recall_types` | `list[str]` or `None` | `None` | Restrict which layers to search (e.g. `["knowledge", "entity"]`) |
| `return_docs` | `bool` | `False` | If `True`, return `list[Document]` instead of formatted string |

### Layer Types

The `recall_types` parameter accepts any combination of:
- `"conversation"` - Episodic chat history (SQLite)
- `"knowledge"` - General knowledge facts (ChromaDB)
- `"entity"` - Named entities and relationships (ChromaDB or Neo4j)
- `"workflow"` - Procedural steps and processes (ChromaDB)
- `"summary"` - Summarized information (ChromaDB)
- `"tool_log"` - Tool call history (SQLite)

## SevenLayerStore

Maps LangChain's batch key-value store interface (`mget`/`mset`/`mdelete`) to 7layermem's ChromaDB vector stores. Useful for chains that need a generic store abstraction.

```python
store = SevenLayerStore(memory, store_name="knowledge")

# Set values
store.mset([
    ("user_preference", "dark mode enabled"),
    ("api_endpoint", "https://api.example.com"),
])

# Get values
values = store.mget(["user_preference", "api_endpoint"])
# -> ["dark mode enabled", "https://api.example.com"]

# Delete
store.mdelete(["api_endpoint"])

# Iterate keys
for key in store.yield_keys(prefix="user_"):
    print(key)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agent_memory` | `AgentMemory` | required | 7layermem AgentMemory instance |
| `store_name` | `str` | `"knowledge"` | Backing store: `"knowledge"`, `"entity"`, `"workflow"`, or `"summary"` |

### Limitations

`yield_keys` uses a broad similarity search and filters by prefix. This is O(n) and capped at 100 results. Not suitable for large stores where you need to enumerate all keys.

## Running Tests

```bash
python -m pytest tests/test_langchain_chat_history.py tests/test_langchain_memory.py tests/test_langchain_store.py -v
```

## Dependencies

- `langchain-core>=0.2.0` (already in 7layermem)
- `langchain-classic>=0.3.0` (for `BaseMemory`)

## Architecture

```
LangChain Chain / Agent
        |
        v
+-------------------------------+
| LangChain Extension           |
|  - SevenLayerChatMessageHistory|
|  - SevenLayerMemory           |
|  - SevenLayerStore            |
+-------------------------------+
        |
        v
+-------------------------------+
| AgentMemory                   |
|  - remember(text, **context)  |
|  - recall(query, **options)   |
+-------------------------------+
        |
        v
+-------------------------------+
| 7-Layer Memory System         |
|  Conversational (SQLite)      |
|  Knowledge (ChromaDB)         |
|  Entity (ChromaDB/Neo4j)      |
|  Workflow (ChromaDB)          |
|  Toolbox (ChromaDB)           |
|  Summary (ChromaDB)           |
|  Tool Log (SQLite)            |
+-------------------------------+
```