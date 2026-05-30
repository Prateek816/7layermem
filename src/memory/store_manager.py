CONVERSATIONAL_TABLE   = "CONVERSATIONAL_MEMORY" # Episodic memory
KNOWLEDGE_BASE_TABLE   = "SEMANTIC_MEMORY" # Semantic memory
WORKFLOW_TABLE = "WORKFLOW_MEMORY" # Procedural memory
TOOLBOX_TABLE    = "TOOLBOX_MEMORY" # Procedural memory
ENTITY_TABLE = "ENTITY_MEMORY" # Semantic memory
SUMMARY_TABLE = "SUMMARY_MEMORY" # Semantic memory
TOOL_LOG_TABLE = "TOOL_LOG_MEMORY" # Tool execution logs

ALL_TABLES = [
    CONVERSATIONAL_TABLE, 
    KNOWLEDGE_BASE_TABLE, 
    WORKFLOW_TABLE, 
    TOOLBOX_TABLE, 
    ENTITY_TABLE, 
    SUMMARY_TABLE, 
    TOOL_LOG_TABLE]

DATABASE_PATH = "./data"

from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


import sqlite3


def table_exists(conn, table_name):
    cursor = conn.cursor()

    cursor.execute("""
        SELECT name
        FROM sqlite_master
        WHERE type='table' AND name=?
    """, (table_name,))

    return cursor.fetchone() is not None


def create_conversational_history_table(
    conn,
    table_name: str = "CONVERSATIONAL_MEMORY"
):
    """
    Create a SQLite table to store conversational history.

    Args:
        conn: SQLite database connection
        table_name: Name of the table to create
    """

    if table_exists(conn, table_name):
        print(f"  ⏭️ Table {table_name} already exists")
        return table_name

    cursor = conn.cursor()

    # Create table
    cursor.execute(f"""
        CREATE TABLE {table_name} (
            id TEXT PRIMARY KEY,
            thread_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            summary_id TEXT DEFAULT NULL
        )
    """)

    # Create index on thread_id
    cursor.execute(f"""
        CREATE INDEX IF NOT EXISTS
        idx_{table_name.lower()}_thread_id
        ON {table_name}(thread_id)
    """)

    # Create index on timestamp
    cursor.execute(f"""
        CREATE INDEX IF NOT EXISTS
        idx_{table_name.lower()}_timestamp
        ON {table_name}(timestamp)
    """)

    conn.commit()

    print(f"  ✅ Table {table_name} created successfully with indexes")

    return table_name


def create_tool_log_table(
    conn,
    table_name: str = "TOOL_LOG_MEMORY"
):
    """
    Create a SQLite table to store raw tool execution logs.
    """

    if table_exists(conn, table_name):
        print(f"  ⏭️ Table {table_name} already exists")
        return table_name

    cursor = conn.cursor()

    # Create table
    cursor.execute(f"""
        CREATE TABLE {table_name} (
            id TEXT PRIMARY KEY,
            thread_id TEXT NOT NULL,
            tool_call_id TEXT,
            tool_name TEXT NOT NULL,
            tool_args TEXT,
            result TEXT,
            result_preview TEXT,
            status TEXT DEFAULT 'success',
            error_message TEXT,
            metadata TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Indexes
    cursor.execute(f"""
        CREATE INDEX IF NOT EXISTS
        idx_{table_name.lower()}_thread_id
        ON {table_name}(thread_id)
    """)

    cursor.execute(f"""
        CREATE INDEX IF NOT EXISTS
        idx_{table_name.lower()}_tool_name
        ON {table_name}(tool_name)
    """)

    cursor.execute(f"""
        CREATE INDEX IF NOT EXISTS
        idx_{table_name.lower()}_timestamp
        ON {table_name}(timestamp)
    """)

    conn.commit()

    print(f"  ✅ Table {table_name} created successfully with indexes")

    return table_name


import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma


class StoreManager:
    """
    Manages all ChromaDB vector stores and SQL tables.
    """

    def __init__(
        self,
        embedding_function,
        table_names,
        conversational_table,
        DATABASE_PATH,
        tool_log_table=None,
    ):
        """
        Initialize all Chroma vector stores.

        Args:
            embedding_function: Embedding model
            table_names: Dict of collection names
            conversational_table: SQLite conversational table name
            DATABASE_PATH: Persistent ChromaDB storage path
            tool_log_table: SQL tool log table name
        """

        self.embedding_function = embedding_function
        self._conversational_table = conversational_table
        self._tool_log_table = tool_log_table
        self.DATABASE_PATH = DATABASE_PATH

        # Persistent Chroma Client
        self.client = chromadb.PersistentClient(
            path=DATABASE_PATH,
            settings=Settings(anonymized_telemetry=False)
        )

        # =========================================
        # Create Chroma Vector Stores
        # =========================================

        self._knowledge_base_vs = Chroma(
            client=self.client,
            collection_name=table_names["knowledge_base"],
            embedding_function=embedding_function,
            persist_directory=DATABASE_PATH,
        )

        self._workflow_vs = Chroma(
            client=self.client,
            collection_name=table_names["workflow"],
            embedding_function=embedding_function,
            persist_directory=DATABASE_PATH,
        )

        self._toolbox_vs = Chroma(
            client=self.client,
            collection_name=table_names["toolbox"],
            embedding_function=embedding_function,
            persist_directory=DATABASE_PATH,
        )

        self._entity_vs = Chroma(
            client=self.client,
            collection_name=table_names["entity"],
            embedding_function=embedding_function,
            persist_directory=DATABASE_PATH,
        )

        self._summary_vs = Chroma(
            client=self.client,
            collection_name=table_names["summary"],
            embedding_function=embedding_function,
            persist_directory=DATABASE_PATH,
        )

    # =========================================
    # Getter Methods
    # =========================================

    def get_knowledge_base_store(self):
        return self._knowledge_base_vs

    def get_workflow_store(self):
        return self._workflow_vs

    def get_toolbox_store(self):
        return self._toolbox_vs

    def get_entity_store(self):
        return self._entity_vs

    def get_summary_store(self):
        return self._summary_vs

    def get_conversational_table(self):
        return self._conversational_table

    def get_tool_log_table(self):
        return self._tool_log_table

    # =========================================
    # Optional Utility Functions
    # =========================================

    def list_all_collections(self):
        """
        List all ChromaDB collections.
        """
        return self.client.list_collections()

    def delete_collection(self, collection_name):
        """
        Delete a collection.
        """
        self.client.delete_collection(collection_name)

    def reset_database(self):
        """
        Delete all collections.
        """
        collections = self.client.list_collections()

        for collection in collections:
            self.client.delete_collection(collection.name)

        print("All collections deleted.")


import os
os.makedirs(DATABASE_PATH, exist_ok=True)

CONVERSATIONAL_DB_PATH = f"{DATABASE_PATH}/conversational_memory.db"
TOOL_LOG_DB_PATH = f"{DATABASE_PATH}/tool_log_memory.db"

conversation_conn = sqlite3.connect(CONVERSATIONAL_DB_PATH)
tool_log_conn = sqlite3.connect(TOOL_LOG_DB_PATH)


CONVERSATIONAL_HISTORY_TABLE = create_conversational_history_table(
    conversation_conn,
    table_name=CONVERSATIONAL_TABLE
)
TOOL_LOG_HISTORY_TABLE = create_tool_log_table(
    tool_log_conn,
    table_name=TOOL_LOG_TABLE
)

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
    conversational_table=CONVERSATIONAL_HISTORY_TABLE,
    tool_log_table=TOOL_LOG_HISTORY_TABLE,
)

print("\n✅ All databases and vector stores initialized successfully.")