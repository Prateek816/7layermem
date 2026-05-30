"""
Context Store - Persistent storage for retrieved contexts.

Stores retrieved documents and search contexts for persistent retrieval
without re-running the retrieval process.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class RetrievedContext:
    """Represents a retrieved document with metadata."""
    doc_id: str
    content: str
    score: float
    source: str
    retrieved_at: str
    metadata: Dict[str, Any]


@dataclass
class SearchContext:
    """Represents a complete search context with query and results."""
    context_id: str
    query: str
    retrieved_contexts: List[RetrievedContext]
    retrieved_at: str
    metadata: Dict[str, Any]


class ContextStore:
    """
    Persistent storage for retrieved contexts.

    Features:
    - Store search results with timestamps
    - Retrieve previous contexts by query
    - Batch operations for efficiency
    """

    def __init__(self, db_path: Path):
        """
        Initialize context store.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Table for retrieved documents
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS retrieved_contexts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                context_id TEXT NOT NULL,
                doc_id TEXT NOT NULL,
                content TEXT NOT NULL,
                score REAL NOT NULL,
                source TEXT,
                retrieved_at TEXT NOT NULL,
                metadata TEXT,
                UNIQUE(context_id, doc_id)
            )
        """)

        # Table for search contexts (queries)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS search_contexts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                context_id TEXT UNIQUE NOT NULL,
                query TEXT NOT NULL,
                retrieved_at TEXT NOT NULL,
                metadata TEXT
            )
        """)

        # Indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_context_id
            ON retrieved_contexts(context_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_query
            ON search_contexts(query)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_retrieved_at
            ON retrieved_contexts(retrieved_at)
        """)

        conn.commit()
        conn.close()

    def _get_conn(self) -> sqlite3.Connection:
        """Get database connection."""
        return sqlite3.connect(self.db_path)

    def _serialize_metadata(self, metadata: Optional[Dict]) -> str:
        """Serialize metadata to JSON string."""
        return json.dumps(metadata or {}, ensure_ascii=False)

    def _deserialize_metadata(self, json_str: Optional[str]) -> Dict:
        """Deserialize JSON string to metadata dict."""
        if not json_str:
            return {}
        try:
            return json.loads(json_str)
        except (TypeError, ValueError):
            return {}

    def store_search_context(
        self,
        context_id: str,
        query: str,
        retrieved_contexts: List[RetrievedContext],
        metadata: Optional[Dict[str, Any]] = None,
        retrieved_at: Optional[str] = None,
    ) -> None:
        """
        Store a search context with all retrieved documents.

        Args:
            context_id: Unique identifier for this context
            query: The search query
            retrieved_contexts: List of retrieved documents
            metadata: Optional metadata about the search
            retrieved_at: Optional timestamp (defaults to now)
        """
        if retrieved_at is None:
            retrieved_at = datetime.now(timezone.utc).isoformat()

        conn = self._get_conn()
        cursor = conn.cursor()

        try:
            # Store search context
            cursor.execute("""
                INSERT OR REPLACE INTO search_contexts
                (context_id, query, retrieved_at, metadata)
                VALUES (?, ?, ?, ?)
            """, (context_id, query, retrieved_at, self._serialize_metadata(metadata)))

            # Store retrieved contexts
            for ctx in retrieved_contexts:
                cursor.execute("""
                    INSERT OR REPLACE INTO retrieved_contexts
                    (context_id, doc_id, content, score, source, retrieved_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    context_id,
                    ctx.doc_id,
                    ctx.content,
                    ctx.score,
                    ctx.source,
                    retrieved_at,
                    self._serialize_metadata(ctx.metadata),
                ))

            conn.commit()
        finally:
            conn.close()

    def get_search_context(self, context_id: str) -> Optional[SearchContext]:
        """
        Retrieve a complete search context by ID.

        Args:
            context_id: Unique identifier of the context

        Returns:
            SearchContext object or None if not found
        """
        conn = self._get_conn()
        cursor = conn.cursor()

        try:
            # Get search context
            cursor.execute("""
                SELECT context_id, query, retrieved_at, metadata
                FROM search_contexts
                WHERE context_id = ?
            """, (context_id,))
            row = cursor.fetchone()

            if not row:
                return None

            ctx_id, query, retrieved_at, metadata_json = row
            metadata = self._deserialize_metadata(metadata_json)

            # Get retrieved contexts
            cursor.execute("""
                SELECT doc_id, content, score, source, retrieved_at, metadata
                FROM retrieved_contexts
                WHERE context_id = ?
                ORDER BY score DESC
            """, (context_id,))

            contexts = []
            for row in cursor.fetchall():
                doc_id, content, score, source, ctx_retrieved_at, ctx_metadata_json = row
                contexts.append(RetrievedContext(
                    doc_id=doc_id,
                    content=content,
                    score=score,
                    source=source,
                    retrieved_at=ctx_retrieved_at,
                    metadata=self._deserialize_metadata(ctx_metadata_json),
                ))

            return SearchContext(
                context_id=ctx_id,
                query=query,
                retrieved_contexts=contexts,
                retrieved_at=retrieved_at,
                metadata=metadata,
            )

        finally:
            conn.close()

    def search_by_query(
        self,
        query: str,
        limit: int = 10,
    ) -> List[SearchContext]:
        """
        Find search contexts by query substring.

        Args:
            query: Query to search for (substring match)
            limit: Maximum results to return

        Returns:
            List of matching SearchContext objects
        """
        conn = self._get_conn()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT context_id, query, retrieved_at, metadata
                FROM search_contexts
                WHERE query LIKE ? COLLATE NOCASE
                ORDER BY retrieved_at DESC
                LIMIT ?
            """, (f"%{query}%", limit))

            contexts = []
            for row in cursor.fetchall():
                ctx_id, q, retrieved_at, metadata_json = row
                metadata = self._deserialize_metadata(metadata_json)

                # Get associated retrieved contexts
                cursor.execute("""
                    SELECT doc_id, content, score, source, retrieved_at, metadata
                    FROM retrieved_contexts
                    WHERE context_id = ?
                    ORDER BY score DESC
                """, (ctx_id,))

                retrieved_contexts = []
                for ctx_row in cursor.fetchall():
                    doc_id, content, score, source, ctx_retrieved_at, ctx_metadata_json = ctx_row
                    retrieved_contexts.append(RetrievedContext(
                        doc_id=doc_id,
                        content=content,
                        score=score,
                        source=source,
                        retrieved_at=ctx_retrieved_at,
                        metadata=self._deserialize_metadata(ctx_metadata_json),
                    ))

                contexts.append(SearchContext(
                    context_id=ctx_id,
                    query=q,
                    retrieved_contexts=retrieved_contexts,
                    retrieved_at=retrieved_at,
                    metadata=metadata,
                ))

            return contexts

        finally:
            conn.close()

    def delete_context(self, context_id: str) -> None:
        """Delete a search context and its retrieved documents."""
        conn = self._get_conn()
        cursor = conn.cursor()

        try:
            cursor.execute("DELETE FROM retrieved_contexts WHERE context_id = ?", (context_id,))
            cursor.execute("DELETE FROM search_contexts WHERE context_id = ?", (context_id,))
            conn.commit()
        finally:
            conn.close()

    def get_recent_contexts(self, limit: int = 20) -> List[SearchContext]:
        """Get the most recent search contexts."""
        conn = self._get_conn()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT context_id, query, retrieved_at, metadata
                FROM search_contexts
                ORDER BY retrieved_at DESC
                LIMIT ?
            """, (limit,))

            contexts = []
            for row in cursor.fetchall():
                ctx_id, q, retrieved_at, metadata_json = row
                metadata = self._deserialize_metadata(metadata_json)

                # Get associated retrieved contexts
                cursor.execute("""
                    SELECT doc_id, content, score, source, retrieved_at, metadata
                    FROM retrieved_contexts
                    WHERE context_id = ?
                    ORDER BY score DESC
                """, (ctx_id,))

                retrieved_contexts = []
                for ctx_row in cursor.fetchall():
                    doc_id, content, score, source, ctx_retrieved_at, ctx_metadata_json = ctx_row
                    retrieved_contexts.append(RetrievedContext(
                        doc_id=doc_id,
                        content=content,
                        score=score,
                        source=source,
                        retrieved_at=ctx_retrieved_at,
                        metadata=self._deserialize_metadata(ctx_metadata_json),
                    ))

                contexts.append(SearchContext(
                    context_id=ctx_id,
                    query=q,
                    retrieved_contexts=retrieved_contexts,
                    retrieved_at=retrieved_at,
                    metadata=metadata,
                ))

            return contexts

        finally:
            conn.close()

    def cleanup_old_contexts(self, days: int = 30) -> int:
        """Delete contexts older than specified days. Returns number deleted."""
        conn = self._get_conn()
        cursor = conn.cursor()

        try:
            import datetime

            cutoff = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days)).isoformat()

            # Get context_ids to delete
            cursor.execute("""
                SELECT DISTINCT context_id FROM retrieved_contexts
                WHERE retrieved_at < ?
            """, (cutoff,))
            old_ids = [row[0] for row in cursor.fetchall()]

            if old_ids:
                placeholders = ",".join("?" for _ in old_ids)
                cursor.execute(f"""
                    DELETE FROM retrieved_contexts WHERE context_id IN ({placeholders})
                """, old_ids)
                cursor.execute(f"""
                    DELETE FROM search_contexts WHERE context_id IN ({placeholders})
                """, old_ids)
                conn.commit()
                return len(old_ids)

            return 0

        finally:
            conn.close()

    def get_stats(self) -> dict:
        """Get statistics about the context store."""
        conn = self._get_conn()
        cursor = conn.cursor()

        try:
            cursor.execute("SELECT COUNT(*) FROM search_contexts")
            num_contexts = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM retrieved_contexts")
            num_documents = cursor.fetchone()[0]

            return {
                "num_search_contexts": num_contexts,
                "num_retrieved_documents": num_documents,
                "db_path": str(self.db_path),
            }

        finally:
            conn.close()
