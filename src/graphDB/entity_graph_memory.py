"""

EntityGraphMemory — Neo4j-backed entity memory using LangChain.

Stores entities as nodes and relationships as edges in a graph database,
enabling rich relationship queries that vector stores cannot provide.

Requires:
  - neo4j Python driver
  - langchain-community (for Neo4jGraph)
  - langchain-groq or equivalent LLM for entity extraction
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Optional

from langchain_community.graphs import Neo4jGraph
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(sep=" ", timespec="seconds")


# ─────────────────────────────────────────────────────────────────────────────
# Entity extraction prompt
# ─────────────────────────────────────────────────────────────────────────────

ENTITY_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Extract entities and their relationships from the text.

Return a JSON object with this structure:
{{
  "entities": [
    {{
      "name": "entity name",
      "type": "PERSON|ORGANIZATION|LOCATION|CONCEPT|EVENT|OTHER",
      "properties": {{"key": "value"}}
    }}
  ],
  "relationships": [
    {{
      "source": "source entity name",
      "target": "target entity name",
      "type": "RELATIONSHIP_TYPE",
      "properties": {{"key": "value"}}
    }}
  ]
}}

Only extract meaningful entities. Be precise with names."""),
    ("human", "Text: {text}")
])


# ─────────────────────────────────────────────────────────────────────────────
# EntityGraphMemory
# ─────────────────────────────────────────────────────────────────────────────

class EntityGraphMemory:
    """
    Neo4j-backed entity memory for storing and retrieving entities
    and their relationships using a graph database.

    Features:
    - Store entities as nodes with labels and properties
    - Store relationships as edges between entities
    - Query entities by name, type, or properties
    - Traverse relationships to find connected entities
    - Hybrid search: graph traversal + vector similarity (if embeddings configured)

    Environment variables (or pass directly):
        NEO4J_URI      - bolt://localhost:7687 (default)
        NEO4J_USERNAME - neo4j (default)
        NEO4J_PASSWORD - required
        NEO4J_DATABASE - neo4j (default)
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
        llm: Optional[BaseChatModel] = None,
    ):
        self._uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self._username = username or os.getenv("NEO4J_USERNAME", "neo4j")
        self._password = password or os.getenv("NEO4J_PASSWORD", "")
        self._database = database or os.getenv("NEO4J_DATABASE", "neo4j")

        self._graph = Neo4jGraph(
            url=self._uri,
            username=self._username,
            password=self._password,
            database=self._database,
        )

        self._llm = llm or ChatGroq(model="openai/gpt-oss-120b", temperature=0)

        # Create constraints/indexes on init
        self._init_schema()

    def _init_schema(self) -> None:
        """Create indexes and constraints for entity nodes."""
        try:
            self._graph.query(
                "CREATE CONSTRAINT entity_name IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE e.name IS UNIQUE"
            )
        except Exception:
            # Constraint may already exist or Neo4j version doesn't support IF NOT EXISTS
            pass

    # ══════════════════════════════════════════════════════════════════════
    # ENTITY EXTRACTION
    # ══════════════════════════════════════════════════════════════════════

    def extract_entities(self, text: str) -> dict[str, Any]:
        """
        Use LLM to extract entities and relationships from text.

        Returns:
            {
                "entities": [{"name": ..., "type": ..., "properties": {...}}],
                "relationships": [{"source": ..., "target": ..., "type": ..., "properties": {...}}]
            }
        """
        chain = ENTITY_EXTRACTION_PROMPT | self._llm
        response = chain.invoke({"text": text})

        try:
            return json.loads(str(response.content))
        except (json.JSONDecodeError, AttributeError):
            return {"entities": [], "relationships": []}

    # ══════════════════════════════════════════════════════════════════════
    # WRITE OPERATIONS
    # ══════════════════════════════════════════════════════════════════════

    def write_entity(
        self,
        name: str,
        entity_type: str = "Entity",
        properties: Optional[dict[str, Any]] = None,
        thread_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Create or update an entity node in the graph.

        Args:
            name: Entity name (used as unique identifier)
            entity_type: Label for the node (e.g., PERSON, ORGANIZATION)
            properties: Additional properties to store on the node
            thread_id: Optional thread ID for scoping

        Returns:
            The created/updated entity properties
        """
        props = properties or {}
        props["name"] = name
        props["updated_at"] = _now()
        if thread_id:
            props["thread_id"] = thread_id

        # Sanitize entity_type for Cypher label (no spaces/special chars)
        safe_type = "".join(c for c in entity_type if c.isalnum() or c == "_").upper()
        if not safe_type:
            safe_type = "Entity"

        # Build property string for Cypher
        prop_str = ", ".join(f"e.{k} = ${k}" for k in props)

        query = f"""
        MERGE (e:{safe_type} {{name: $name}})
        ON CREATE SET e.created_at = $created_at, {prop_str}
        ON MATCH SET {prop_str}
        RETURN e
        """
        props["created_at"] = _now()

        result = self._graph.query(query, params=props)
        return result[0]["e"] if result else props

    def write_relationship(
        self,
        source: str,
        target: str,
        rel_type: str = "RELATED_TO",
        properties: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Create a relationship between two entities.

        Args:
            source: Source entity name
            target: Target entity name
            rel_type: Relationship type (e.g., WORKS_AT, KNOWS)
            properties: Additional properties for the relationship

        Returns:
            The created relationship info
        """
        props = properties or {}
        props["updated_at"] = _now()

        safe_rel = "".join(c for c in rel_type if c.isalnum() or c == "_").upper()
        if not safe_rel:
            safe_rel = "RELATED_TO"

        prop_str = ", ".join(f"r.{k} = ${k}" for k in props)

        query = f"""
        MATCH (a:Entity {{name: $source}})
        MATCH (b:Entity {{name: $target}})
        MERGE (a)-[r:{safe_rel}]->(b)
        SET {prop_str}
        RETURN a.name AS source, type(r) AS rel_type, b.name AS target, r
        """
        props["source"] = source
        props["target"] = target

        result = self._graph.query(query, params=props)
        return result[0] if result else {"source": source, "rel_type": safe_rel, "target": target}

    def write_entities_from_text(
        self,
        text: str,
        thread_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Extract entities and relationships from text using LLM,
        then store them in the graph.

        Args:
            text: Raw text to extract entities from
            thread_id: Optional thread ID for scoping

        Returns:
            {
                "entities_written": int,
                "relationships_written": int,
                "entities": [...],
                "relationships": [...]
            }
        """
        extracted = self.extract_entities(text)

        entities_written = []
        for entity in extracted.get("entities", []):
            result = self.write_entity(
                name=entity["name"],
                entity_type=entity.get("type", "Entity"),
                properties=entity.get("properties", {}),
                thread_id=thread_id,
            )
            entities_written.append(result)

        rels_written = []
        for rel in extracted.get("relationships", []):
            result = self.write_relationship(
                source=rel["source"],
                target=rel["target"],
                rel_type=rel.get("type", "RELATED_TO"),
                properties=rel.get("properties", {}),
            )
            rels_written.append(result)

        return {
            "entities_written": len(entities_written),
            "relationships_written": len(rels_written),
            "entities": entities_written,
            "relationships": rels_written,
        }

    # ══════════════════════════════════════════════════════════════════════
    # READ / QUERY OPERATIONS
    # ══════════════════════════════════════════════════════════════════════

    def get_entity(self, name: str) -> Optional[dict[str, Any]]:
        """
        Retrieve a single entity by name.

        Returns:
            Entity properties dict or None if not found.
        """
        query = """
        MATCH (e:Entity {name: $name})
        RETURN e
        """
        result = self._graph.query(query, params={"name": name})
        return result[0]["e"] if result else None

    def search_entity(
        self,
        query: str,
        entity_type: Optional[str] = None,
        thread_id: Optional[str] = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Search entities by name substring or properties.

        Args:
            query: Search term (matched against entity name)
            entity_type: Optional filter by entity type/label
            thread_id: Optional filter by thread ID
            limit: Max results

        Returns:
            List of entity property dicts
        """
        where_clauses = ["toLower(e.name) CONTAINS toLower($query)"]
        params: dict[str, Any] = {"query": query, "limit": limit}

        if thread_id:
            where_clauses.append("e.thread_id = $thread_id")
            params["thread_id"] = thread_id

        where_str = " AND ".join(where_clauses)

        if entity_type:
            safe_type = "".join(c for c in entity_type if c.isalnum() or c == "_").upper()
            query_str = f"""
            MATCH (e:{safe_type})
            WHERE {where_str}
            RETURN e
            LIMIT $limit
            """
        else:
            query_str = f"""
            MATCH (e:Entity)
            WHERE {where_str}
            RETURN e
            LIMIT $limit
            """

        result = self._graph.query(query_str, params=params)
        return [record["e"] for record in result]

    def get_entity_relationships(
        self,
        name: str,
        direction: str = "both",
        rel_type: Optional[str] = None,
        depth: int = 1,
    ) -> list[dict[str, Any]]:
        """
        Get all relationships for an entity.

        Args:
            name: Entity name
            direction: "outgoing", "incoming", or "both"
            rel_type: Optional filter by relationship type
            depth: Traversal depth (1 = direct neighbors only)

        Returns:
            List of relationship records: {source, relationship, target}
        """
        safe_depth = max(1, min(depth, 5))  # Cap depth at 5

        if rel_type:
            safe_rel = "".join(c for c in rel_type if c.isalnum() or c == "_").upper()
            rel_pattern = f":{safe_rel}"
        else:
            rel_pattern = ""

        if direction == "outgoing":
            path_pattern = f"(a:Entity {{name: $name}})-[r{rel_pattern}*1..{safe_depth}]->(b)"
        elif direction == "incoming":
            path_pattern = f"(a:Entity {{name: $name}})<-[r{rel_pattern}*1..{safe_depth}]-(b)"
        else:  # both
            path_pattern = f"(a:Entity {{name: $name}})-[r{rel_pattern}*1..{safe_depth}]-(b)"

        query = f"""
        MATCH {path_pattern}
        RETURN a.name AS source,
               [rel IN r | type(rel)] AS relationships,
               b.name AS target,
               b AS target_entity
        LIMIT 50
        """
        result = self._graph.query(query, params={"name": name})
        return result

    def get_related_entities(
        self,
        name: str,
        rel_type: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Get directly connected entities (convenience wrapper).

        Returns:
            List of {entity, relationship, direction}
        """
        outgoing = self.get_entity_relationships(name, direction="outgoing", rel_type=rel_type, depth=1)
        incoming = self.get_entity_relationships(name, direction="incoming", rel_type=rel_type, depth=1)

        results = []
        for record in outgoing:
            results.append({
                "entity": record.get("target_entity"),
                "entity_name": record.get("target"),
                "relationships": record.get("relationships", []),
                "direction": "outgoing",
            })
        for record in incoming:
            results.append({
                "entity": record.get("target_entity"),
                "entity_name": record.get("source"),
                "relationships": record.get("relationships", []),
                "direction": "incoming",
            })

        return results

    # ══════════════════════════════════════════════════════════════════════
    # CONVENIENCE — compatible interface with MemoryManager
    # ══════════════════════════════════════════════════════════════════════

    def write_entity_documents(
        self,
        texts: list[str],
        metadatas: Optional[list[dict]] = None,
        thread_id: Optional[str] = None,
    ) -> list[str]:
        """
        Write entities from a list of text documents.
        Each text is processed with LLM entity extraction.

        Compatible interface with MemoryManager.write_entity().

        Returns:
            List of entity names written
        """
        written = []
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
            entity_name = metadata.get("name", f"entity_{i}")

            self.write_entity(
                name=entity_name,
                entity_type=metadata.get("type", "Entity"),
                properties={"description": text, **metadata},
                thread_id=thread_id,
            )
            written.append(entity_name)

        return written

    def search_entity_documents(
        self,
        query: str,
        k: int = 5,
        filter: Optional[dict] = None,
    ) -> list[Document]:
        """
        Search entities and return as LangChain Documents.

        Compatible interface with MemoryManager.search_entity().

        Returns:
            List of Document objects with entity info
        """
        entity_type = filter.get("type") if filter else None
        thread_id = filter.get("thread_id") if filter else None

        entities = self.search_entity(
            query=query,
            entity_type=entity_type,
            thread_id=thread_id,
            limit=k,
        )

        documents = []
        for entity in entities:
            # Get relationships for context
            rels = self.get_related_entities(entity.get("name", ""))

            rel_text = ""
            if rels:
                rel_lines = []
                for rel in rels:
                    rel_name = rel.get("entity_name", "unknown")
                    rel_types = rel.get("relationships", [])
                    direction = rel.get("direction", "")
                    arrow = "->" if direction == "outgoing" else "<-"
                    rel_lines.append(f"  {arrow} {', '.join(rel_types)} {rel_name}")
                rel_text = "\nRelationships:\n" + "\n".join(rel_lines)

            doc_text = f"Entity: {entity.get('name', '')}\nType: {entity.get('type', 'Entity')}\n{entity.get('description', '')}{rel_text}"

            documents.append(Document(
                page_content=doc_text,
                metadata={
                    "name": entity.get("name", ""),
                    "type": entity.get("type", "Entity"),
                    "thread_id": entity.get("thread_id", ""),
                    "source": "graph_db",
                },
            ))

        return documents

    # ══════════════════════════════════════════════════════════════════════
    # DELETE OPERATIONS
    # ══════════════════════════════════════════════════════════════════════

    def delete_entity(self, name: str) -> bool:
        """
        Delete an entity and all its relationships.

        Returns:
            True if entity was deleted, False if not found.
        """
        query = """
        MATCH (e:Entity {name: $name})
        DETACH DELETE e
        RETURN count(e) AS deleted
        """
        result = self._graph.query(query, params={"name": name})
        return result[0]["deleted"] > 0 if result else False

    def delete_entities_by_thread(self, thread_id: str) -> int:
        """
        Delete all entities for a specific thread.

        Returns:
            Number of entities deleted.
        """
        query = """
        MATCH (e:Entity {thread_id: $thread_id})
        DETACH DELETE e
        RETURN count(e) AS deleted
        """
        result = self._graph.query(query, params={"thread_id": thread_id})
        return result[0]["deleted"] if result else 0

    # ══════════════════════════════════════════════════════════════════════
    # GRAPH STATISTICS
    # ══════════════════════════════════════════════════════════════════════

    def get_stats(self) -> dict[str, Any]:
        """
        Get graph statistics.

        Returns:
            {entity_count, relationship_count, entity_types}
        """
        entity_count = self._graph.query("MATCH (e:Entity) RETURN count(e) AS count")
        rel_count = self._graph.query("MATCH ()-[r]->() RETURN count(r) AS count")
        types = self._graph.query("MATCH (e:Entity) RETURN DISTINCT labels(e) AS labels, count(*) AS count")

        return {
            "entity_count": entity_count[0]["count"] if entity_count else 0,
            "relationship_count": rel_count[0]["count"] if rel_count else 0,
            "entity_types": {str(t["labels"]): t["count"] for t in types} if types else {},
        }

    def close(self) -> None:
        """Close the Neo4j connection."""
        # Neo4jGraph uses a driver internally; access via _driver if available
        driver = getattr(self._graph, "_driver", None)
        if driver and hasattr(driver, "close"):
            driver.close()
