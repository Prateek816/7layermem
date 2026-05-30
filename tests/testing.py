import os
import sqlite3

from langchain_huggingface import HuggingFaceEmbeddings

# =========================================================
# IMPORT YOUR FILES
# =========================================================

from src.memory.memory_manager import MemoryManager
from src.memory.store_manager import StoreManager, CONVERSATIONAL_HISTORY_TABLE, TOOL_LOG_HISTORY_TABLE, create_conversational_history_table, create_tool_log_table

# your table creation functions
# =========================================================
# DATABASE PATH
# =========================================================

DATABASE_PATH = "./data"

os.makedirs(DATABASE_PATH, exist_ok=True)

# =========================================================
# TABLE NAMES
# =========================================================

CONVERSATIONAL_TABLE = "CONVERSATIONAL_MEMORY"
TOOL_LOG_TABLE = "TOOL_LOG_MEMORY"

KNOWLEDGE_BASE_TABLE = "SEMANTIC_MEMORY"
WORKFLOW_TABLE = "WORKFLOW_MEMORY"
TOOLBOX_TABLE = "TOOLBOX_MEMORY"
ENTITY_TABLE = "ENTITY_MEMORY"
SUMMARY_TABLE = "SUMMARY_MEMORY"

# =========================================================
# SQLITE CONNECTIONS
# =========================================================

conversation_conn = sqlite3.connect(
    f"{DATABASE_PATH}/conversational_memory.db"
)

tool_log_conn = sqlite3.connect(
    f"{DATABASE_PATH}/tool_log_memory.db"
)

# =========================================================
# CREATE TABLES
# =========================================================

create_conversational_history_table(
    conversation_conn,
    table_name=CONVERSATIONAL_TABLE
)

create_tool_log_table(
    tool_log_conn,
    table_name=TOOL_LOG_TABLE
)

# =========================================================
# EMBEDDING MODEL
# =========================================================

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# =========================================================
# STORE MANAGER
# =========================================================

store_manager = StoreManager(
    embedding_function=embedding_model,
    table_names={
        "knowledge_base": KNOWLEDGE_BASE_TABLE,
        "workflow": WORKFLOW_TABLE,
        "toolbox": TOOLBOX_TABLE,
        "entity": ENTITY_TABLE,
        "summary": SUMMARY_TABLE,
    },
    DATABASE_PATH=DATABASE_PATH,
    conversational_table=CONVERSATIONAL_TABLE,
    tool_log_table=TOOL_LOG_TABLE,
)

# =========================================================
# MEMORY MANAGER
# =========================================================

memory = MemoryManager(
    store_manager=store_manager,
    conversation_conn=conversation_conn,
    tool_log_conn=tool_log_conn,
)

print("\n==============================")
print("TESTING CONVERSATION MEMORY")
print("==============================")

# =========================================================
# WRITE CONVERSATIONS
# =========================================================

msg1 = memory.write_conversation(
    thread_id="thread_1",
    role="user",
    content="Hello AI",
    metadata={"source": "test"}
)

msg2 = memory.write_conversation(
    thread_id="thread_1",
    role="assistant",
    content="Hello Human"
)

print("Inserted Message IDs:")
print(msg1)
print(msg2)

# =========================================================
# READ CONVERSATIONS
# =========================================================

messages = memory.read_conversations(
    thread_id="thread_1"
)

print("\nConversation Messages:\n")

for msg in messages:
    print(msg)

# =========================================================
# TOOL LOG TEST
# =========================================================

print("\n==============================")
print("TESTING TOOL LOG MEMORY")
print("==============================")

tool_id = memory.write_tool_log(
    thread_id="thread_1",
    tool_name="search_tool",
    tool_args={"query": "AI agents"},
    result={"answer": "Found results"},
    result_preview="Found results...",
)

print("Inserted Tool Log ID:")
print(tool_id)

tool_logs = memory.read_tool_logs(
    thread_id="thread_1"
)

print("\nTool Logs:\n")

for log in tool_logs:
    print(log)

# =========================================================
# KNOWLEDGE STORE TEST
# =========================================================

print("\n==============================")
print("TESTING KNOWLEDGE STORE")
print("==============================")

knowledge_ids = memory.write_knowledge(
    texts=[
        "LangChain is a framework for AI applications.",
        "ChromaDB is a vector database.",
        "SQLite is a lightweight relational database."
    ],
    metadatas=[
        {"topic": "langchain"},
        {"topic": "vectordb"},
        {"topic": "sql"}
    ]
)

print("Knowledge IDs:")
print(knowledge_ids)

# =========================================================
# SEARCH KNOWLEDGE
# =========================================================

results = memory.search_knowledge(
    query="What is Chroma vector database?",
    k=2
)

print("\nKnowledge Search Results:\n")

for doc in results:
    print("CONTENT:", doc.page_content)
    print("METADATA:", doc.metadata)
    print()

# =========================================================
# WORKFLOW STORE TEST
# =========================================================

print("\n==============================")
print("TESTING WORKFLOW MEMORY")
print("==============================")

memory.write_workflow(
    texts=[
        "Step 1: Load model",
        "Step 2: Run inference",
        "Step 3: Save outputs"
    ]
)

workflow_results = memory.search_workflow(
    query="How to run inference?",
    k=2
)

for doc in workflow_results:
    print(doc.page_content)

# =========================================================
# ENTITY STORE TEST
# =========================================================

print("\n==============================")
print("TESTING ENTITY MEMORY")
print("==============================")

memory.write_entity(
    texts=[
        "Prateek likes AI systems.",
        "User uses MacBook M4 Air."
    ]
)

entity_results = memory.search_entity(
    query="What laptop does user use?",
    k=1
)

for doc in entity_results:
    print(doc.page_content)

# =========================================================
# DELETE TEST
# =========================================================

print("\n==============================")
print("TESTING DELETE")
print("==============================")

memory.delete_conversation(msg1)

remaining = memory.read_conversations(
    thread_id="thread_1"
)

print("\nRemaining Messages:\n")

for msg in remaining:
    print(msg)

# =========================================================
# CLOSE CONNECTIONS
# =========================================================

conversation_conn.close()
tool_log_conn.close()

print("\n✅ ALL TESTS COMPLETED SUCCESSFULLY")