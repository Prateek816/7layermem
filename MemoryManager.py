from langchain_groq import ChatGroq
import chromadb

from pydantic import BaseModel, Field
from typing import List

from datetime import datetime
import uuid
import json as json_lib
import os
from dotenv import load_dotenv
load_dotenv()  

#OUTPUT VALIDATION SCHEMAS
class Entity(BaseModel):
    name: str = Field(description="Name of the entity")
    type: str = Field(description="Type of entity: PERSON, PLACE, or SYSTEM")
    description: str = Field(description="Brief description of the entity")

class EntityList(BaseModel):
    entities: List[Entity] = Field(description="List of extracted entities", default_factory=list)
class MemoryManager:
    """
    A simplified memory manager for AI agents using ChromaDB Vector Database and SQLite for SQL.
    
    Manages 7 types of memory:
    - Conversational: Chat history per thread (SQL table)
    - Tool Log: Raw tool execution outputs and metadata (SQL table)
    - Knowledge Base: Searchable documents (Vector store)
    - Workflow: Execution patterns (Vector store)
    - Toolbox: Available tools (Vector store)
    - Entity: People, places, systems (Vector store)
    - Summary: Storing compressed context window
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
        tool_log_table: str | None = None
    ):
        self.conn = conn
        self._conversational_table = conversational_table
        self._knowledge_base_vs = knowledge_base_vs
        self._workflow_vs = workflow_vs
        self._toolbox_vs = toolbox_vs
        self._entity_vs = entity_vs
        self._summary_vs = summary_vs
        self._tool_log_table = tool_log_table

    def write_conversational_memory(self, content: str, role: str, thread_id: str) -> str:
        """Store a message in conversational history."""
        import uuid

        thread_id = str(thread_id)
        record_id = str(uuid.uuid4())  # generate UUID in Python since SQLite has no RETURNING clause support in older versions

        cur = self.conn.cursor()
        cur.execute(f"""
            INSERT INTO {self._conversational_table} 
                (id, thread_id, role, content, metadata, timestamp)
            VALUES 
                (?, ?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%f', 'now'))
        """, (record_id, thread_id, role, content, "{}"))

        self.conn.commit()
        cur.close()

        return record_id
    
    def read_conversational_memory(self, thread_id: str, limit: int = 10) -> str:
        """Read conversation history for a thread (excludes summarized messages)."""
        thread_id = str(thread_id)

        cur = self.conn.cursor()
        cur.execute(f"""
            SELECT role, content, timestamp 
            FROM {self._conversational_table}
            WHERE thread_id = ? AND summary_id IS NULL
            ORDER BY timestamp ASC
            LIMIT ?
        """, (thread_id, limit))

        results = cur.fetchall()
        cur.close()

        messages = [
            f"[{ts}] [{role}] {content}"
            for role, content, ts in results
        ]

        messages_formatted = '\n'.join(messages)
        if not messages_formatted:
            messages_formatted = "(No unsummarized messages found for this thread.)"

        return f"""## Conversation Memory
    ### What this memory is
    Chronological, unsummarized messages from the current thread. This memory captures user intent, constraints, and commitments made in recent turns.
    ### How you should leverage it
    - Preserve continuity with prior decisions, terminology, and user preferences.
    - Resolve references like "that", "previous step", or "the paper above" using earlier turns.
    - If older context conflicts with newer user instructions, prioritize the latest user direction.
    ### Retrieved messages

    {messages_formatted}"""

    def mark_as_summarized(self, thread_id: str, summary_id: str):
        """Mark all unsummarized messages in a thread as summarized."""
        thread_id = str(thread_id)

        cur = self.conn.cursor()
        cur.execute(f"""
            UPDATE {self._conversational_table}
            SET summary_id = ?
            WHERE thread_id = ? AND summary_id IS NULL
        """, (summary_id, thread_id))

        self.conn.commit()
        cur.close()

        print(f"  📦 Marked {cur.rowcount} messages as summarized (summary_id: {summary_id})")

    # ==================== TOOL LOG MEMORY (SQL) ====================
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
        """Persist raw tool execution logs for auditing and just-in-time retrieval."""
        if not self._tool_log_table:
            return None

        thread_id = str(thread_id)

        if isinstance(tool_args, (dict, list)):
            tool_args_str = json_lib.dumps(tool_args, ensure_ascii=False)
        else:
            tool_args_str = "" if tool_args is None else str(tool_args)

        result_str = "" if result is None else str(result)

        # Truncate preview to 2000 UTF-8 bytes (mirrors original Oracle VARCHAR2 limit)
        preview = result_str.encode("utf-8")[:2000].decode("utf-8", errors="ignore")

        metadata_str = json_lib.dumps(metadata, ensure_ascii=False) if metadata else "{}"

        log_id = str(uuid.uuid4())

        cur = self.conn.cursor()
        cur.execute(f"""
            INSERT INTO {self._tool_log_table}
                (id, thread_id, tool_call_id, tool_name, tool_args, result, result_preview,
                status, error_message, metadata, timestamp)
            VALUES
                (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%f', 'now'))
        """, (
            log_id,
            thread_id,
            tool_call_id,
            tool_name,
            tool_args_str,
            result_str,
            preview,
            status,
            error_message,
            metadata_str,
        ))

        self.conn.commit()
        cur.close()

        return log_id
    
    def read_tool_logs(self, thread_id: str, limit: int = 20) -> list[dict]:
        """Read recent tool logs for a thread, newest first."""
        if not self._tool_log_table:
            return []

        thread_id = str(thread_id)

        cur = self.conn.cursor()
        cur.execute(f"""
            SELECT id, tool_call_id, tool_name, tool_args, result_preview,
                status, error_message, metadata, timestamp
            FROM {self._tool_log_table}
            WHERE thread_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (thread_id, limit))

        rows = cur.fetchall()
        cur.close()

        logs = []
        for log_id, tool_call_id, tool_name, tool_args, result_preview, status, error_message, metadata, ts in rows:
            logs.append({
                "id":            log_id,
                "tool_call_id":  tool_call_id,
                "tool_name":     tool_name,
                "tool_args":     tool_args,
                "result_preview": result_preview,
                "status":        status,
                "error_message": error_message,
                "metadata":      metadata,
                "timestamp":     ts,  # already an ISO 8601 string in SQLite — no .isoformat() needed
            })
        return logs
    
    #=================KNOWLEDGE BASE MEMORY (VECTOR DB)==================

    def write_knowledge_base(self, text: str | list[str], metadata: dict | list[dict]):
        """
        Store knowledge-base content with metadata.
        Supports:
        - Single record: text=str, metadata=dict
        - Batch insert: text=list[str], metadata=list[dict]
        """
        # ── Normalise inputs to lists ────────────────────────────────────────
        if isinstance(text, list):
            texts = [str(t) for t in text]
        else:
            texts = [str(text)]

        if isinstance(metadata, list):
            metadatas = metadata
        else:
            metadatas = [metadata] * len(texts)

        if len(texts) != len(metadatas):
            raise ValueError(
                f"Knowledge-base batch length mismatch: "
                f"{len(texts)} texts vs {len(metadatas)} metadata rows"
            )

        # ── Sanitise metadata: ChromaDB only accepts str/int/float/bool values ─
        sanitised_metadatas = []
        for m in metadatas:
            sanitised_metadatas.append({
                k: v if isinstance(v, (str, int, float, bool)) else json_lib.dumps(v)
                for k, v in (m if isinstance(m, dict) else {}).items()
            })

        # ── Generate unique IDs for each document ───────────────────────────
        ids = [str(uuid.uuid4()) for _ in texts]

        # ── Add to ChromaDB collection ───────────────────────────────────────
        # ChromaDB will use the collection's embedding_function automatically
        self._knowledge_base_vs.add(
            ids=ids,
            documents=texts,
            metadatas=sanitised_metadatas,
        )

    def read_knowledge_base(self, query: str, k: int = 3) -> str:
        """Search knowledge base for relevant content."""

        results = self._knowledge_base_vs.query(
            query_texts=[query],      # ChromaDB embeds this automatically
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )

        # results["documents"] is a list of lists — one list per query
        # since we pass a single query, take index [0]
        documents = results["documents"][0]   # list[str]
        distances = results["distances"][0]   # list[float]

        if not documents:
            content = "(No relevant knowledge base passages found.)"
        else:
            passages = []
            for doc, dist in zip(documents, distances):
                relevance = round(1 - dist, 4)  # cosine: closer to 1 = more relevant
                passages.append(f"[relevance: {relevance}] {doc}")
            content = "\n\n".join(passages)

        return f"""## Knowledge Base Memory
    ### What this memory is
    Retrieved background documents and previously ingested reference material relevant to the current query.
    ### How you should leverage it
    - Ground responses in these passages when making factual or technical claims.
    - Prefer concrete details from this memory over unsupported assumptions.
    - If evidence is missing or ambiguous, state uncertainty and request clarification or additional retrieval.
    ### Retrieved passages
    {content}"""

    def write_workflow(self, query: str, steps: list, final_answer: str, success: bool = True):
        """Store a completed workflow pattern for future reference."""

        # ── Format steps as text ─────────────────────────────────────────────
        steps_text = "\n".join([f"Step {i+1}: {s}" for i, s in enumerate(steps)])
        text = f"Query: {query}\nSteps:\n{steps_text}\nAnswer: {final_answer[:200]}"

        # ── Metadata: ChromaDB only accepts str/int/float/bool ───────────────
        metadata = {
            "query":     query,
            "success":   str(success),        # bool → str to be safe across versions
            "num_steps": len(steps),          # int ✅
            "timestamp": datetime.now().isoformat()  # str ✅
        }

        self._workflow_vs.add(
            ids=[str(uuid.uuid4())],
            documents=[text],
            metadatas=[metadata],
        )


    def read_workflow(self, query: str, k: int = 3) -> str:
        """Search for similar past workflows with at least 1 step."""

        NO_RESULTS_RESPONSE = """## Workflow Memory
    ### What this memory is
    Past task trajectories that include query context, ordered steps taken, and prior outcomes.
    ### How you should leverage it
    - Use these workflows as reusable execution patterns for planning and tool orchestration.
    - Adapt step sequences to the current task rather than copying blindly.
    - Reuse successful patterns first, then adjust when task scope or constraints differ.
    ### Retrieved workflows
    (No relevant workflows found.)"""

        # ── Check collection has documents before querying ───────────────────
        if self._workflow_vs.count() == 0:
            return NO_RESULTS_RESPONSE

        # ── Query with ChromaDB native filter (num_steps > 0) ────────────────
        results = self._workflow_vs.query(
            query_texts=[query],
            n_results=k,
            where={"num_steps": {"$gt": 0}},   # ChromaDB uses 'where', not 'filter'
            include=["documents", "metadatas", "distances"]
        )

        documents = results["documents"][0]    # list[str] for first (only) query

        if not documents:
            return NO_RESULTS_RESPONSE

        content = "\n---\n".join(documents)

        return f"""## Workflow Memory
    ### What this memory is
    Past task trajectories that include query context, ordered steps taken, and prior outcomes.
    ### How you should leverage it
    - Use these workflows as reusable execution patterns for planning and tool orchestration.
    - Adapt step sequences to the current task rather than copying blindly.
    - Reuse successful patterns first, then adjust when task scope or constraints differ.
    ### Retrieved workflows

    {content}"""

    def write_toolbox(self, text: str, metadata: dict):
        """Store a tool definition in the toolbox."""

        # ── Sanitise metadata: ChromaDB only accepts str/int/float/bool ──────
        sanitised_metadata = {}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)):
                sanitised_metadata[k] = v
            else:
                sanitised_metadata[k] = json_lib.dumps(v)  # serialise complex types

        self._toolbox_vs.add(
            ids=[str(uuid.uuid4())],
            documents=[text],
            metadatas=[sanitised_metadata],
        )


    def read_toolbox(self, query: str, k: int = 3) -> list[dict]:
        """Find relevant tools and return OpenAI-compatible schemas."""

        # ── Guard: empty collection ──────────────────────────────────────────
        if self._toolbox_vs.count() == 0:
            return []

        results = self._toolbox_vs.query(
            query_texts=[query],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )

        metadatas = results["metadatas"][0]   # list[dict] for first (only) query

        tools = []
        seen_tool_names: set[str] = set()

        for meta in metadatas:
            tool_name = meta.get("name", "tool")
            if tool_name in seen_tool_names:
                continue
            seen_tool_names.add(tool_name)

            # ── Deserialise parameters (stored as JSON string in ChromaDB) ───
            raw_params = meta.get("parameters", "{}")
            if isinstance(raw_params, str):
                try:
                    stored_params = json_lib.loads(raw_params)
                except json_lib.JSONDecodeError:
                    stored_params = {}
            else:
                stored_params = raw_params  # already a dict (shouldn't happen but safe)

            # ── Build OpenAI-compatible parameter schema ─────────────────────
            type_mapping = {
                "<class 'str'>":   "string",
                "<class 'int'>":   "integer",
                "<class 'float'>": "number",
                "<class 'bool'>":  "boolean",
                "str":             "string",
                "int":             "integer",
                "float":           "number",
                "bool":            "boolean",
            }

            properties = {}
            required = []

            for param_name, param_info in stored_params.items():
                param_type = param_info.get("type", "string")
                json_type  = type_mapping.get(param_type, "string")
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
                        "required":   required
                    }
                }
            })

        return tools

     # ==================== ENTITY (Vector Store) ====================

    def extract_entities(self,text: str) -> list[dict]:
        prompt = f'Extract entities from: "{text[:500]}". If none found, return empty list.'

        try:
            llm = ChatGroq(
                model="llama-3.1-8b-instant",
                temperature=0.0,
                api_key=os.getenv("GROQ_API_KEY")
            )

            structured_llm = llm.with_structured_output(EntityList)
            result = structured_llm.invoke(prompt)

            return [
                {
                    "name":        entity.name,
                    "type":        entity.type,
                    "description": entity.description
                }
                for entity in result.entities
            ]

        except Exception as e:
            print(f"  ⚠️ Entity extraction failed: {e}")
            return []

    #removed the need of llm_client argument since the extract_entities method now handled by CHATgroq inside it
    def write_entity(
        self,
        name: str,
        entity_type: str,
        description: str,
        #llm_client=None,
        text: str = None 
    ):
        """Store an entity OR extract and store entities from text."""

        if text :
            # ── Extract entities from text via LLM then store each ───────────
            entities = self.extract_entities(text)
            for e in entities:
                self._entity_vs.add(
                    ids=[str(uuid.uuid4())],
                    documents=[f"{e['name']} ({e['type']}): {e['description']}"],
                    metadatas=[{
                        "name":        e["name"],
                        "type":        e["type"],
                        "description": e["description"]
                    }]
                )
            return entities

        else:
            # ── Store single entity directly ──────────────────────────────────
            self._entity_vs.add(
                ids=[str(uuid.uuid4())],
                documents=[f"{name} ({entity_type}): {description}"],
                metadatas=[{
                    "name":        name,
                    "type":        entity_type,
                    "description": description
                }]
            )
            return [{
                "name":        name,
                "type":        entity_type,
                "description": description
            }]


    def read_entity(self, query: str, k: int = 5) -> str:
        """Search for relevant entities in ChromaDB."""
        _ENTITY_MEMORY_HEADER = """\
## Entity Memory
### What this memory is
Entity-level context such as people, organizations, systems, tools, and other named items previously identified in conversations or documents.
### How you should leverage it
- Use entities to disambiguate references and maintain consistent naming.
- Preserve important attributes (roles, relationships, descriptions) across turns.
- Personalize and contextualize responses using relevant known entities.
### Retrieved entities"""
        try:
            results = self._entity_vs.similarity_search_with_relevance_scores(
                query,
                k=k,
                score_threshold=0.3   # filter out low-confidence matches
            )
        except Exception as e:
            print(f"⚠️ Entity search failed: {e}")
            return f"{_ENTITY_MEMORY_HEADER}\n(Search unavailable.)"

        if not results:
            return f"{_ENTITY_MEMORY_HEADER}\n(No entities found.)"

        lines = []
        for doc, score in results:
            meta = doc.metadata or {}
            name        = meta.get("name", "?")
            entity_type = meta.get("type", "")
            description = meta.get("description", "")

            type_tag = f"[{entity_type}] " if entity_type else ""
            confidence = f"(score: {score:.2f})"
            lines.append(f"• {type_tag}{name}: {description} {confidence}")

        return f"{_ENTITY_MEMORY_HEADER}\n" + "\n".join(lines)

    # ==================== SUMMARY (Vector Store) ====================

    


    def write_summary(
        self,
        summary_id: str,
        full_content: str,
        summary: str,
        description: str,
        thread_id: str | None = None,
    ) -> str:
        """Store a summary with its original content."""
        metadata = {
            "id":           summary_id,
            "full_content": full_content,
            "summary":      summary,
            "description":  description,
        }
        if thread_id is not None:
            metadata["thread_id"] = str(thread_id)

        try:
            self._summary_vs.add_texts(
                texts=[f"{summary_id}: {description}"],
                metadatas=[metadata],
            )
        except Exception as e:
            print(f"⚠️ write_summary failed: {e}")

        return summary_id


    def read_summary_memory(self, summary_id: str, thread_id: str | None = None) -> str:
        """Retrieve a specific summary by ID (just-in-time retrieval)."""
        # ChromaDB requires $and for multiple conditions
        if thread_id is not None:
            where = {"$and": [{"id": {"$eq": summary_id}}, {"thread_id": {"$eq": str(thread_id)}}]}
        else:
            where = {"id": {"$eq": summary_id}}

        try:
            results = self._summary_vs.similarity_search(
                query=summary_id,
                k=1,                  # we only ever use results[0]
                filter=where,
            )
        except Exception as e:
            print(f"⚠️ read_summary_memory failed: {e}")
            return f"Summary {summary_id} could not be retrieved."

        if not results:
            scope = f" for thread {thread_id}" if thread_id is not None else ""
            return f"Summary {summary_id} not found{scope}."

        return results[0].metadata.get("summary", "No summary content.")


    def read_summary_context(
        self, query: str = "", k: int = 10, thread_id: str | None = None
    ) -> str:
        """Get available summaries for context window (IDs + descriptions only)."""
        _SUMMARY_MEMORY_HEADER = """\
## Summary Memory
### What this memory is
Compressed snapshots of older conversation windows preserved to retain long-range context.
### How you should leverage it
- Use summaries to maintain continuity when full historical messages are not in the active context window.
- Call expand_summary(id) before depending on exact quotes, fine-grained details, or step-by-step chronology.
### Available summaries"""
        # ChromaDB filter syntax
        where = {"thread_id": {"$eq": str(thread_id)}} if thread_id is not None else None

        try:
            results = self._summary_vs.similarity_search_with_relevance_scores(
                query=query or "summary",
                k=k,
                filter=where,
                score_threshold=0.0,   # return all; let caller decide relevance
            )
        except Exception as e:
            print(f"⚠️ read_summary_context failed: {e}")
            return f"{_SUMMARY_MEMORY_HEADER}\n(Search unavailable.)"

        if not results:
            scope = f"(No summaries available for thread {thread_id}.)" if thread_id else "(No summaries available.)"
            return f"{_SUMMARY_MEMORY_HEADER}\n{scope}"

        lines = [
            _SUMMARY_MEMORY_HEADER,
            "Use expand_summary(id) to retrieve the detailed underlying conversation.",
        ]
        if thread_id is not None:
            lines.append(f"Scope: thread_id = {thread_id}")

        for doc, score in results:
            meta = doc.metadata or {}
            sid  = meta.get("id", "?")
            desc = meta.get("description", "No description")
            lines.append(f"  • [ID: {sid}] {desc} (score: {score:.2f})")

        return "\n".join(lines)


    def read_conversations_by_summary_id(self, summary_id: str) -> str:
        """
        Retrieve all original conversations summarized under a given summary_id.
        Returns conversations ordered by timestamp.
        """
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT id, role, content, timestamp
                    FROM {self._conversational_table}
                    WHERE summary_id = :summary_id
                    ORDER BY timestamp ASC
                    """,
                    {"summary_id": summary_id},
                )
                rows = cur.fetchall()
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
            ts_str = timestamp.strftime("%Y-%m-%d %H:%M:%S") if timestamp else "Unknown"
            lines.append(f"[{ts_str}] [{role.upper()}]")
            lines.append(content)
            lines.append("")

        return "\n".join(lines)