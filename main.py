from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
import sqlite3
import chromadb
import uuid
import json as json_lib
from typing import List
from pydantic import BaseModel, Field
from datetime import datetime

load_dotenv()

# ====UTILITIES====
database_connection = chromadb.Client()
conn = sqlite3.connect('my_data.db')

# Table names for each memory type
CONVERSATIONAL_TABLE = "CONVERSATIONAL_MEMORY"   # Episodic memory
KNOWLEDGE_BASE_TABLE = "SEMANTIC_MEMORY"          # Semantic memory
WORKFLOW_TABLE       = "WORKFLOW_MEMORY"           # Procedural memory
TOOLBOX_TABLE        = "TOOLBOX_MEMORY"            # Procedural memory
ENTITY_TABLE         = "ENTITY_MEMORY"             # Semantic memory
SUMMARY_TABLE        = "SUMMARY_MEMORY"            # Semantic memory
TOOL_LOG_TABLE       = "TOOL_LOG_MEMORY"           # Tool execution logs

ALL_TABLES = [
    CONVERSATIONAL_TABLE,
    KNOWLEDGE_BASE_TABLE,
    WORKFLOW_TABLE,
    TOOLBOX_TABLE,
    ENTITY_TABLE,
    SUMMARY_TABLE,
    TOOL_LOG_TABLE,
]

# Module-level LLM instance (created once, reused across calls)
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.0,
    api_key=os.getenv("GROQ_API_KEY"),
)


class Entity(BaseModel):
    name: str = Field(description="Name of the entity")
    type: str = Field(description="Type of entity: PERSON, PLACE, or SYSTEM")
    description: str = Field(description="Brief description of the entity")


class EntityList(BaseModel):
    entities: List[Entity] = Field(
        description="List of extracted entities", default_factory=list
    )


# ── SQL table helpers ────────────────────────────────────────────────────────

def create_conversational_history_table(conn, table_name: str = "CONVERSATIONAL_MEMORY") -> str:
    """Create (or recreate) the conversational history table."""
    cur = conn.cursor()
    cur.execute(f"DROP TABLE IF EXISTS {table_name}")
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id          TEXT PRIMARY KEY DEFAULT (
                            lower(hex(randomblob(4))) || '-' ||
                            lower(hex(randomblob(2))) || '-4' ||
                            substr(lower(hex(randomblob(2))), 2) || '-' ||
                            substr('89ab', abs(random()) % 4 + 1, 1) ||
                            substr(lower(hex(randomblob(2))), 2) || '-' ||
                            lower(hex(randomblob(6)))
                        ),
            thread_id   TEXT NOT NULL,
            role        TEXT NOT NULL,
            content     TEXT NOT NULL,
            timestamp   TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
            metadata    TEXT,
            created_at  TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
            summary_id  TEXT DEFAULT NULL
        )
    """)
    cur.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_{table_name.lower()}_thread_id
        ON {table_name}(thread_id)
    """)
    cur.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_{table_name.lower()}_timestamp
        ON {table_name}(timestamp)
    """)
    conn.commit()
    cur.close()
    print(f"Table {table_name} created successfully with indexes")
    return table_name


def table_exists(conn, table_name: str) -> bool:
    """Return True if the table already exists in the SQLite database."""
    cur = conn.cursor()
    cur.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    )
    exists = cur.fetchone() is not None
    cur.close()
    return exists


def create_tool_log_table(conn, table_name: str = "TOOL_LOG_MEMORY") -> str:
    """Create the tool-log table (skipped if it already exists)."""
    if table_exists(conn, table_name):
        print(f"  ⏭️ Table {table_name} already exists (using existing table)")
        return table_name

    cur = conn.cursor()
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id              TEXT PRIMARY KEY DEFAULT (
                                lower(hex(randomblob(4))) || '-' ||
                                lower(hex(randomblob(2))) || '-4' ||
                                substr(lower(hex(randomblob(2))), 2) || '-' ||
                                substr('89ab', abs(random()) % 4 + 1, 1) ||
                                substr(lower(hex(randomblob(2))), 2) || '-' ||
                                lower(hex(randomblob(6)))
                            ),
            thread_id       TEXT NOT NULL,
            tool_call_id    TEXT,
            tool_name       TEXT NOT NULL,
            tool_args       TEXT,
            result          TEXT,
            result_preview  TEXT,
            status          TEXT DEFAULT 'success',
            error_message   TEXT,
            metadata        TEXT,
            timestamp       TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
            created_at      TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now'))
        )
    """)
    cur.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_{table_name.lower()}_thread_id
        ON {table_name}(thread_id)
    """)
    cur.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_{table_name.lower()}_tool_name
        ON {table_name}(tool_name)
    """)
    cur.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_{table_name.lower()}_timestamp
        ON {table_name}(timestamp)
    """)
    conn.commit()
    cur.close()
    print(f"  ✅ Table {table_name} created successfully with indexes")
    return table_name


CONVERSATION_HISTORY_TABLE = create_conversational_history_table(conn, CONVERSATIONAL_TABLE)
TOOL_LOG_HISTORY_TABLE     = create_tool_log_table(conn, TOOL_LOG_TABLE)


# ── StoreManager ─────────────────────────────────────────────────────────────

class StoreManager:
    """
    Manages all stores (ChromaDB vector collections + SQL table names).

    Args:
        client:               ChromaDB client
        table_names:          Dict with keys: knowledge_base, workflow, toolbox, entity, summary
        conversational_table: Name of the conversational history SQL table
        tool_log_table:       Name of the SQL tool-log table
    """

    def __init__(
        self,
        client,
        table_names: dict,
        conversational_table: str,
        tool_log_table: str | None = None,
    ):
        self.client                = client
        self._conversational_table = conversational_table
        self._tool_log_table       = tool_log_table
        self.table_names           = table_names

        # Vector stores (ChromaDB collections)
        self._knowledge_base_vs = self.client.get_or_create_collection(
            name=self.table_names["knowledge_base"]
        )
        self._workflow_vs = self.client.get_or_create_collection(
            name=self.table_names["workflow"]
        )
        self._toolbox_vs = self.client.get_or_create_collection(
            name=self.table_names["toolbox"]
        )
        self._entity_vs = self.client.get_or_create_collection(
            name=self.table_names["entity"]
        )
        self._summary_vs = self.client.get_or_create_collection(
            name=self.table_names["summary"]
        )

    # Getters
    @property
    def get_knowledge_base_store(self):
        return self._knowledge_base_vs

    @property
    def get_workflow_store(self):
        return self._workflow_vs

    @property
    def get_toolbox_store(self):
        return self._toolbox_vs

    @property
    def get_entity_store(self):
        return self._entity_vs

    @property
    def get_summary_store(self):
        return self._summary_vs

    @property
    def get_conversational_table(self):
        return self._conversational_table

    @property
    def get_tool_log_table(self):
        return self._tool_log_table


# ── MemoryManager ─────────────────────────────────────────────────────────────

class MemoryManager:
    """
    Memory manager for AI agents using ChromaDB (vector) and SQLite (relational).

    Manages 7 memory types:
        Conversational  — chat history per thread          (SQL)
        Tool Log        — raw tool execution metadata      (SQL)
        Knowledge Base  — searchable reference documents   (Vector)
        Workflow        — past execution patterns          (Vector)
        Toolbox         — available tool definitions       (Vector)
        Entity          — people, places, systems          (Vector)
        Summary         — compressed context snapshots     (Vector)
    """

    def __init__(
        self,
        conn,
        conversational_table: str,
        knowledge_base_vs,
        workflow_vs,
        toolbox_vs,
        entity_vs,
        summary_vs,
        tool_log_table: str | None = None,
    ):
        self.conn                  = conn
        self._conversational_table = conversational_table
        self._knowledge_base_vs    = knowledge_base_vs
        self._workflow_vs          = workflow_vs
        self._toolbox_vs           = toolbox_vs
        self._entity_vs            = entity_vs
        self._summary_vs           = summary_vs
        self._tool_log_table       = tool_log_table

    # ── Conversational memory (SQL) ──────────────────────────────────────────

    def write_conversational_memory(self, content: str, role: str, thread_id: str) -> str:
        """Persist a message to the conversational history table."""
        thread_id = str(thread_id)
        record_id = str(uuid.uuid4())

        cur = self.conn.cursor()
        cur.execute(
            f"""
            INSERT INTO {self._conversational_table}
                (id, thread_id, role, content, metadata, timestamp)
            VALUES
                (?, ?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%f', 'now'))
            """,
            (record_id, thread_id, role, content, "{}"),
        )
        self.conn.commit()
        cur.close()
        return record_id

    def read_conversational_memory(self, thread_id: str, limit: int = 10) -> str:
        """Return unsummarized conversation history for a thread, oldest first."""
        thread_id = str(thread_id)

        cur = self.conn.cursor()
        cur.execute(
            f"""
            SELECT role, content, timestamp
            FROM {self._conversational_table}
            WHERE thread_id = ? AND summary_id IS NULL
            ORDER BY timestamp ASC
            LIMIT ?
            """,
            (thread_id, limit),
        )
        results = cur.fetchall()
        cur.close()

        messages = [f"[{ts}] [{role}] {content}" for role, content, ts in results]
        messages_formatted = "\n".join(messages) or "(No unsummarized messages found for this thread.)"

        return f"""## Conversation Memory

{messages_formatted}"""

    def mark_as_summarized(self, thread_id: str, summary_id: str) -> None:
        """Mark all unsummarized messages in a thread as belonging to a summary."""
        thread_id = str(thread_id)

        cur = self.conn.cursor()
        cur.execute(
            f"""
            UPDATE {self._conversational_table}
            SET summary_id = ?
            WHERE thread_id = ? AND summary_id IS NULL
            """,
            (summary_id, thread_id),
        )
        self.conn.commit()
        count = cur.rowcount
        cur.close()
        print(f"  📦 Marked {count} messages as summarized (summary_id: {summary_id})")

    # ── Tool log memory (SQL) ────────────────────────────────────────────────

    def write_tool_log(
        self,
        thread_id: str,
        tool_name: str,
        tool_args,
        result: str,
        status: str = "success",
        tool_call_id: str | None = None,
        error_message: str | None = None,
        metadata: dict | None = None,
    ) -> str | None:
        """Persist a raw tool execution log entry."""
        if not self._tool_log_table:
            return None

        thread_id = str(thread_id)

        tool_args_str = (
            json_lib.dumps(tool_args, ensure_ascii=False)
            if isinstance(tool_args, (dict, list))
            else ("" if tool_args is None else str(tool_args))
        )
        result_str  = "" if result is None else str(result)
        preview     = result_str.encode("utf-8")[:2000].decode("utf-8", errors="ignore")
        metadata_str = json_lib.dumps(metadata, ensure_ascii=False) if metadata else "{}"
        log_id      = str(uuid.uuid4())

        cur = self.conn.cursor()
        cur.execute(
            f"""
            INSERT INTO {self._tool_log_table}
                (id, thread_id, tool_call_id, tool_name, tool_args, result, result_preview,
                 status, error_message, metadata, timestamp)
            VALUES
                (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%f', 'now'))
            """,
            (
                log_id, thread_id, tool_call_id, tool_name,
                tool_args_str, result_str, preview,
                status, error_message, metadata_str,
            ),
        )
        self.conn.commit()
        cur.close()
        return log_id

    def read_tool_logs(self, thread_id: str, limit: int = 20) -> list[dict]:
        """Return recent tool logs for a thread, newest first."""
        if not self._tool_log_table:
            return []

        thread_id = str(thread_id)
        cur = self.conn.cursor()
        cur.execute(
            f"""
            SELECT id, tool_call_id, tool_name, tool_args, result_preview,
                   status, error_message, metadata, timestamp
            FROM {self._tool_log_table}
            WHERE thread_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (thread_id, limit),
        )
        rows = cur.fetchall()
        cur.close()

        return [
            {
                "id":             log_id,
                "tool_call_id":   tool_call_id,
                "tool_name":      tool_name,
                "tool_args":      tool_args,
                "result_preview": result_preview,
                "status":         status,
                "error_message":  error_message,
                "metadata":       metadata,
                "timestamp":      ts,
            }
            for log_id, tool_call_id, tool_name, tool_args, result_preview,
                status, error_message, metadata, ts in rows
        ]

    # ── Knowledge base (Vector) ──────────────────────────────────────────────

    def write_knowledge_base(self, text: str | list[str], metadata: dict | list[dict]) -> None:
        """Store one or more documents in the knowledge base."""
        texts     = [str(t) for t in text] if isinstance(text, list) else [str(text)]
        metadatas = metadata if isinstance(metadata, list) else [metadata] * len(texts)

        if len(texts) != len(metadatas):
            raise ValueError(
                f"Batch length mismatch: {len(texts)} texts vs {len(metadatas)} metadata rows"
            )

        sanitised = [
            {
                k: v if isinstance(v, (str, int, float, bool)) else json_lib.dumps(v)
                for k, v in (m if isinstance(m, dict) else {}).items()
            }
            for m in metadatas
        ]

        self._knowledge_base_vs.add(
            ids=[str(uuid.uuid4()) for _ in texts],
            documents=texts,
            metadatas=sanitised,
        )

    def read_knowledge_base(self, query: str, k: int = 3) -> str:
        """Semantic search over the knowledge base."""
        # FIX: guard against empty collection (ChromaDB raises on empty query)
        if self._knowledge_base_vs.count() == 0:
            return """"""

        results   = self._knowledge_base_vs.query(
            query_texts=[query],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        documents = results["documents"][0]
        distances = results["distances"][0]

        if not documents:
            content = "(No relevant knowledge base passages found.)"
        else:
            passages = [
                f"[relevance: {round(1 - dist, 4)}] {doc}"
                for doc, dist in zip(documents, distances)
            ]
            content = "\n\n".join(passages)

        return f"""## Knowledge Base Memory
### Retrieved passages
{content}"""

    # ── Workflow memory (Vector) ─────────────────────────────────────────────

    def write_workflow(self, query: str, steps: list, final_answer: str, success: bool = True) -> None:
        """Store a completed workflow for future reference."""
        steps_text = "\n".join([f"Step {i+1}: {s}" for i, s in enumerate(steps)])
        text = f"Query: {query}\nSteps:\n{steps_text}\nAnswer: {final_answer[:200]}"

        self._workflow_vs.add(
            ids=[str(uuid.uuid4())],
            documents=[text],
            metadatas=[{
                "query":     query,
                "success":   str(success),
                "num_steps": len(steps),
                "timestamp": datetime.now().isoformat(),
            }],
        )

    def read_workflow(self, query: str, k: int = 3) -> str:
        """Search for similar past workflows."""
        NO_RESULTS = """## Workflow Memory
(No relevant workflows found.)"""

        if self._workflow_vs.count() == 0:
            return NO_RESULTS

        results   = self._workflow_vs.query(
            query_texts=[query],
            n_results=k,
            where={"num_steps": {"$gt": 0}},
            include=["documents", "metadatas", "distances"],
        )
        documents = results["documents"][0]

        if not documents:
            return NO_RESULTS

        content = "\n---\n".join(documents)
        return f"""## Workflow Memory

{content}"""

    # ── Toolbox (Vector) ─────────────────────────────────────────────────────

    def write_toolbox(self, text: str, metadata: dict) -> None:
        """Store a tool definition."""
        sanitised = {
            k: v if isinstance(v, (str, int, float, bool)) else json_lib.dumps(v)
            for k, v in metadata.items()
        }
        self._toolbox_vs.add(
            ids=[str(uuid.uuid4())],
            documents=[text],
            metadatas=[sanitised],
        )

    def read_toolbox(self, query: str, k: int = 3) -> list[dict]:
        """Return relevant tool schemas (OpenAI-compatible format)."""
        if self._toolbox_vs.count() == 0:
            return []

        results   = self._toolbox_vs.query(
            query_texts=[query],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        metadatas = results["metadatas"][0]

        type_mapping = {
            "<class 'str'>":   "string",
            "<class 'int'>":   "integer",
            "<class 'float'>": "number",
            "<class 'bool'>":  "boolean",
            "str":   "string",
            "int":   "integer",
            "float": "number",
            "bool":  "boolean",
        }

        tools: list[dict] = []
        seen: set[str]    = set()

        for meta in metadatas:
            tool_name = meta.get("name", "tool")
            if tool_name in seen:
                continue
            seen.add(tool_name)

            raw_params = meta.get("parameters", "{}")
            if isinstance(raw_params, str):
                try:
                    stored_params = json_lib.loads(raw_params)
                except json_lib.JSONDecodeError:
                    stored_params = {}
            else:
                stored_params = raw_params

            properties: dict = {}
            required:   list = []
            for param_name, param_info in stored_params.items():
                json_type = type_mapping.get(param_info.get("type", "string"), "string")
                properties[param_name] = {"type": json_type}
                if "default" not in param_info:
                    required.append(param_name)

            tools.append({
                "type": "function",
                "function": {
                    "name":        tool_name,
                    "description": meta.get("description", ""),
                    "parameters": {
                        "type":       "object",
                        "properties": properties,
                        "required":   required,
                    },
                },
            })

        return tools

    # ── Entity memory (Vector) ───────────────────────────────────────────────

    def extract_entities(self, text: str) -> list[dict]:
        """Use the module-level LLM to extract named entities from text."""
        prompt = f'Extract entities from: "{text[:500]}". If none found, return empty list.'
        try:
            structured_llm = llm.with_structured_output(EntityList)
            result = structured_llm.invoke(prompt)
            return [
                {"name": e.name, "type": e.type, "description": e.description}
                for e in result.entities
            ]
        except Exception as e:
            print(f"  ⚠️ Entity extraction failed: {e}")
            return []

    def write_entity(
        self,
        name: str,
        entity_type: str,
        description: str,
        text: str = None,
    ) -> list[dict]:
        """Store a single entity directly, or extract-and-store from raw text."""
        if text:
            entities = self.extract_entities(text)
            for e in entities:
                self._entity_vs.add(
                    ids=[str(uuid.uuid4())],
                    documents=[f"{e['name']} ({e['type']}): {e['description']}"],
                    metadatas=[{"name": e["name"], "type": e["type"], "description": e["description"]}],
                )
            return entities
        else:
            self._entity_vs.add(
                ids=[str(uuid.uuid4())],
                documents=[f"{name} ({entity_type}): {description}"],
                metadatas=[{"name": name, "type": entity_type, "description": description}],
            )
            return [{"name": name, "type": entity_type, "description": description}]

    def read_entity(self, query: str, k: int = 5) -> str:
        """Search for relevant entities."""
        HEADER = """## Entity Memory
### Retrieved entities"""

        # Guard: empty collection
        if self._entity_vs.count() == 0:
            return f"{HEADER}\n(No entities stored yet.)"

        try:
            results = self._entity_vs.query(query_texts=[query], n_results=k)
        except Exception as e:
            print(f"⚠️ Entity search failed: {e}")
            return f"{HEADER}\n(Search unavailable.)"

        if not results:
            return f"{HEADER}\n(No entities found.)"

        docs   = results.get("documents", [[]])[0]
        metas  = results.get("metadatas", [[]])[0]
        lines  = []
        for meta in metas:
            name        = meta.get("name", "?")
            entity_type = meta.get("type", "")
            description = meta.get("description", "")
            type_tag    = f"[{entity_type}] " if entity_type else ""
            lines.append(f"• {type_tag}{name}: {description}")

        return f"{HEADER}\n" + "\n".join(lines)

    # ── Summary memory (Vector) ──────────────────────────────────────────────

    def write_summary(
        self,
        summary_id: str,
        full_content: str,
        summary: str,
        description: str,
        thread_id: str | None = None,
    ) -> str:
        """Store a compressed conversation summary."""
        metadata: dict = {
            "id":           summary_id,
            "full_content": full_content,
            "summary":      summary,
            "description":  description,
        }
        if thread_id is not None:
            metadata["thread_id"] = str(thread_id)

        # FIX: use native ChromaDB API, not LangChain .add_texts()
        try:
            self._summary_vs.add(
                ids=[str(uuid.uuid4())],
                documents=[f"{summary_id}: {description}"],
                metadatas=[metadata],
            )
        except Exception as e:
            print(f"⚠️ write_summary failed: {e}")

        return summary_id

    def read_summary_memory(self, summary_id: str, thread_id: str | None = None) -> str:
        """Retrieve a specific summary by ID (just-in-time retrieval)."""
        # FIX: use native ChromaDB API, not LangChain .similarity_search()
        if thread_id is not None:
            where = {"$and": [{"id": {"$eq": summary_id}}, {"thread_id": {"$eq": str(thread_id)}}]}
        else:
            where = {"id": {"$eq": summary_id}}

        try:
            results = self._summary_vs.query(
                query_texts=[summary_id],
                n_results=1,
                where=where,
                include=["metadatas"],
            )
        except Exception as e:
            print(f"⚠️ read_summary_memory failed: {e}")
            return f"Summary {summary_id} could not be retrieved."

        metas = results.get("metadatas", [[]])[0]
        if not metas:
            scope = f" for thread {thread_id}" if thread_id is not None else ""
            return f"Summary {summary_id} not found{scope}."

        return metas[0].get("summary", "No summary content.")

    def read_summary_context(
        self, query: str = "", k: int = 10, thread_id: str | None = None
    ) -> str:
        """Return available summary stubs (IDs + descriptions) for the context window."""
        HEADER = """## Summary Memory
### Available summaries"""

        if self._summary_vs.count() == 0:
            scope = f"(No summaries available for thread {thread_id}.)" if thread_id else "(No summaries available.)"
            return f"{HEADER}\n{scope}"

        # FIX: pass the where filter to the query; guard empty query string
        where = {"thread_id": {"$eq": str(thread_id)}} if thread_id is not None else None
        query_texts = [query if query else "summary"]

        try:
            kwargs: dict = {"query_texts": query_texts, "n_results": k}
            if where is not None:
                kwargs["where"] = where
            results = self._summary_vs.query(**kwargs)
        except Exception as e:
            print(f"⚠️ read_summary_context failed: {e}")
            return f"{HEADER}\n(Search unavailable.)"

        lines = [HEADER, "Use expand_summary(id) to retrieve the detailed underlying conversation."]
        if thread_id is not None:
            lines.append(f"Scope: thread_id = {thread_id}")

        docs  = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        for meta in metas:
            sid  = meta.get("id", "?")
            desc = meta.get("description", "No description")
            lines.append(f"  • [ID: {sid}] {desc}")

        return "\n".join(lines)

    def read_conversations_by_summary_id(self, summary_id: str) -> str:
        """Return all original messages that belong to a given summary."""
        # FIX: SQLite cursors are not context managers — use plain assignment
        try:
            cur = self.conn.cursor()
            cur.execute(
                f"""
                SELECT id, role, content, timestamp
                FROM {self._conversational_table}
                WHERE summary_id = ?
                ORDER BY timestamp ASC
                """,
                (summary_id,),
            )
            rows = cur.fetchall()
            cur.close()
        except Exception as e:
            print(f"⚠️ read_conversations_by_summary_id failed: {e}")
            return f"Could not retrieve conversations for summary_id: {summary_id}"

        if not rows:
            return f"No conversations found for summary_id: {summary_id}"

        lines = [
            f"## Expanded Conversations for Summary ID: {summary_id}",
            f"Total messages: {len(rows)}\n",
        ]
        for _msg_id, role, content, timestamp in rows:
            # FIX: SQLite timestamps are plain strings, not datetime objects
            ts_str = timestamp or "Unknown"
            lines.append(f"[{ts_str}] [{role.upper()}]")
            lines.append(content)
            lines.append("")

        return "\n".join(lines)


# ── Initialise stores ────────────────────────────────────────────────────────

store_manager = StoreManager(
    client=database_connection,
    table_names={
        "knowledge_base": KNOWLEDGE_BASE_TABLE,
        "workflow":        WORKFLOW_TABLE,
        "toolbox":         TOOLBOX_TABLE,
        "entity":          ENTITY_TABLE,
        "summary":         SUMMARY_TABLE,
    },
    conversational_table=CONVERSATION_HISTORY_TABLE,
    tool_log_table=TOOL_LOG_HISTORY_TABLE,
)

conversation_table = store_manager.get_conversational_table
knowledge_base_vs  = store_manager.get_knowledge_base_store
workflow_vs        = store_manager.get_workflow_store
toolbox_vs         = store_manager.get_toolbox_store
entity_vs          = store_manager.get_entity_store
summary_vs         = store_manager.get_summary_store
tool_log_table     = store_manager.get_tool_log_table

print("✅ All stores loaded via StoreManager")
print("Knowledge Base Store:", knowledge_base_vs)
print("Workflow Store:",       workflow_vs)
print("Toolbox Store:",        toolbox_vs)
print("Entity Store:",         entity_vs)
print("Summary Store:",        summary_vs)
print("Conversational Table:", conversation_table)
print("Tool Log Table:",       tool_log_table)

memory_manager = MemoryManager(
    conn=conn,
    conversational_table=conversation_table,
    knowledge_base_vs=knowledge_base_vs,
    workflow_vs=workflow_vs,
    toolbox_vs=toolbox_vs,
    entity_vs=entity_vs,
    summary_vs=summary_vs,
    tool_log_table=TOOL_LOG_HISTORY_TABLE,
)


# ── Agent entry point ────────────────────────────────────────────────────────

def call_agent(query: str, thread_id: str = "1", max_iterations: int = 10) -> str:
    """Agent loop: build memory context → call LLM → persist results."""
    thread_id = str(thread_id)
    steps:     list = []
    summaries: list = []

    print("\n" + "=" * 50)
    print("🧠 BUILDING CONTEXT...")

    memory_context  = ""
    memory_context += memory_manager.read_conversational_memory(thread_id) + "\n\n"
    memory_context += memory_manager.read_knowledge_base(query)             + "\n\n"
    memory_context += memory_manager.read_workflow(query)                   + "\n\n"
    memory_context += memory_manager.read_entity(query)                     + "\n\n"
    memory_context += memory_manager.read_summary_context(query, thread_id=thread_id) + "\n\n"

    context = f"# Question\n{query}\n\n{memory_context}"

    # Persist user message and extract entities before calling the LLM
    memory_manager.write_conversational_memory(query, "user", thread_id)
    try:
        memory_manager.write_entity("", "", "", text=query)
    except Exception:
        pass

    AGENT_SYSTEM_PROMPT = (
        "You are an intelligent agent designed to answer the user's question "
        "by leveraging available tools and memory."
    )

    messages = [
        SystemMessage(content=AGENT_SYSTEM_PROMPT),
        HumanMessage(content=context),
    ]

    print("\n🤖 AGENT LOOP")
    response     = llm.invoke(messages)   # returns AIMessage
    final_answer = response.content       # FIX: extract string from AIMessage

    # ── TODO: insert tool-calling loop here ─────────────────────────────────
    # The commented-out block in the original can be re-enabled here once
    # execute_tool() and the rest of the scaffolding are implemented.
    # ────────────────────────────────────────────────────────────────────────

    # Persist workflow only when real steps were executed
    # FIX: removed `steps = 0` placeholder — steps stays as [] so this is always skipped
    if steps:
        memory_manager.write_workflow(query, steps, final_answer)

    try:
        memory_manager.write_entity("", "", "", text=query)
    except Exception:
        pass

    memory_manager.write_conversational_memory(final_answer, "assistant", thread_id)

    print("\n" + "=" * 50 + f"\n💬 ANSWER:\n{final_answer}\n" + "=" * 50)
    return final_answer


if __name__ == "__main__":
    test_query = "What are the key differences between supervised and unsupervised learning?"
    call_agent(test_query)
    call_agent("advantages")
    call_agent("My name is Prateek Rastogi and i am a Software Engineer")
    call_agent("who am I")
