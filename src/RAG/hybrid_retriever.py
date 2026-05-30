"""
Hybrid Retriever - Combines BM25 and Vector retrieval.

The main entry point for the Hybrid RAG system, coordinating
between BM25 layer (lexical) and Vector layer (semantic),
with persistent context storage.
"""

import hashlib
import uuid
import datetime as dt
from datetime import datetime, timezone
from typing import List, Optional, Tuple, Dict, Any

from .config import RAGConfig
from .bm25_layer import BM25Layer
from .vector_layer import VectorLayer
from .context_store import ContextStore, RetrievedContext, SearchContext


class HybridRetriever:
    """
    Main hybrid retrieval system combining BM25 and vector search.

    Features:
    - Simultaneous BM25 and vector retrieval
    - Score normalization and fusion
    - Persistent context storage
    - Query deduplication (fast path for repeated queries)
    """

    def __init__(
        self,
        config: RAGConfig,
        bm25_layer: BM25Layer,
        vector_layer: VectorLayer,
        context_store: ContextStore,
    ):
        """
        Initialize hybrid retriever.

        Args:
            config: RAG configuration
            bm25_layer: BM25 retrieval layer
            vector_layer: Vector retrieval layer
            context_store: Persistent context store
        """
        self.config = config
        self.bm25_layer = bm25_layer
        self.vector_layer = vector_layer
        self.context_store = context_store

    def _normalize_scores(
        self,
        bm25_results: List[Tuple[int, float, str]],
        vector_results: List[Tuple[Any, float]],  # (Document, score)
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Normalize BM25 and vector scores to 0-1 range.

        Args:
            bm25_results: BM25 results (doc_index, score, text)
            vector_results: Vector results (Document, score)

        Returns:
            Normalized score dictionaries keyed by doc_id
        """
        # Get max scores for normalization
        if bm25_results:
            max_bm25 = max(score for _, score, _ in bm25_results)
        else:
            max_bm25 = 1.0

        if vector_results:
            max_vector = max(score for _, score in vector_results)
        else:
            max_vector = 1.0

        # Normalize BM25 scores
        bm25_normalized = {}
        if max_bm25 > 0:
            for idx, score, _ in bm25_results:
                doc_id = f"doc_{idx}_chunk_0"  # Assume doc_id format
                bm25_normalized[doc_id] = score / max_bm25

        # Normalize vector scores
        vector_normalized = {}
        if max_vector > 0:
            for doc, score in vector_results:
                doc_id = doc.metadata.get("doc_id", str(uuid.uuid4()))
                vector_normalized[doc_id] = score / max_vector

        return bm25_normalized, vector_normalized

    def _hybrid_score(
        self,
        bm25_normalized: dict,
        vector_normalized: dict,
        doc_id: str,
    ) -> float:
        """
        Calculate fused hybrid score.

        Args:
            bm25_normalized: Dict of doc_id -> normalized BM25 score
            vector_normalized: Dict of doc_id -> normalized vector score
            doc_id: Document ID

        Returns:
            Weighted hybrid score
        """
        bm25_weight = self.config.bm25_weight
        vector_weight = self.config.vector_weight

        # Get scores for this doc_id
        bm25_score = bm25_normalized.get(doc_id, 0.0)
        vector_score = vector_normalized.get(doc_id, 0.0)

        return (bm25_weight * bm25_score) + (vector_weight * vector_score)

    def search(
        self,
        query: str,
        k: Optional[int] = None,
        use_cache: bool = True,
        store_results: bool = True,
    ) -> List[RetrievedContext]:
        """
        Perform hybrid search combining BM25 + Vector.

        Args:
            query: Search query string
            k: Number of results (defaults to config.default_k)
            use_cache: Check context store for cached results
            store_results: Store results in context store

        Returns:
            List of retrieved contexts sorted by hybrid score
        """
        k = k or self.config.default_k

        # Generate cache key
        cache_key = self._generate_cache_key(query, k)

        # Fast path: check cache
        if use_cache:
            cached = self.context_store.get_search_context(cache_key)
            if cached and cached.retrieved_contexts:
                print(f"✓ Cache hit: {len(cached.retrieved_contexts)} results")
                return cached.retrieved_contexts

        print(f"Performing hybrid search for: '{query}'")

        # Parallel retrieval
        bm25_results = []
        vector_results = []

        # Use thread pool for parallel execution if needed
        try:
            bm25_results = self.bm25_layer.search(query, k=k * 2)
        except Exception as e:
            print(f"BM25 search error: {e}")

        try:
            vector_results = self.vector_layer.search(query, k=k * 2)
        except Exception as e:
            print(f"Vector search error: {e}")

        print(f"  BM25 found {len(bm25_results)} results")
        print(f"  Vector found {len(vector_results)} results")

        # Normalize scores
        bm25_normalized, vector_normalized = self._normalize_scores(bm25_results, vector_results)

        # Get all unique document IDs
        all_doc_ids = set()

        # BM25 doc IDs (for standard format)
        for idx, _, _ in bm25_results:
            # Try to get actual doc_id from vector results
            doc_metadata = None
            for doc, _ in vector_results:
                if doc.metadata.get("chunk_index") == 0:
                    doc_metadata = doc.metadata
                    break
            doc_id = doc_metadata.get("doc_id") if doc_metadata else f"doc_{idx}"
            all_doc_ids.add(doc_id)

        # Vector doc IDs
        for doc, _ in vector_results:
            doc_id = doc.metadata.get("doc_id", str(uuid.uuid4()))
            all_doc_ids.add(doc_id)

        # Calculate hybrid scores
        doc_scores = {}
        for doc_id in all_doc_ids:
            bm25_score = bm25_normalized.get(doc_id, 0)
            vector_score = vector_normalized.get(doc_id, 0)
            doc_scores[doc_id] = self._hybrid_score(
                bm25_normalized, vector_normalized, doc_id
            )

        # Create retrieved contexts
        retrieved_contexts = []

        # Add BM25 results
        for idx, bm25_score, text in bm25_results:
            doc_id = f"doc_{idx}"  # Fallback
            # Try to find matching vector result
            has_vector_match = False
            for doc, vec_score in vector_results:
                if doc.metadata.get("chunk_index") == 0:
                    doc_id = doc.metadata.get("doc_id", doc_id)
                    has_vector_match = True
                    break

            hybrid_score = doc_scores.get(doc_id, bm25_score / 2)  # Lower weight if no vector match

            if hybrid_score > 0:
                retrieved_contexts.append(
                    RetrievedContext(
                        doc_id=doc_id,
                        content=text,
                        score=hybrid_score,
                        source="hybrid" if has_vector_match else "bm25",
                        retrieved_at=datetime.now(timezone.utc).isoformat(),
                        metadata={"bm25_score": bm25_score},
                    )
                )

        # Add vector-only results
        for doc, vec_score in vector_results:
            doc_id = doc.metadata.get("doc_id", str(uuid.uuid4()))
            if doc_id not in [ctx.doc_id for ctx in retrieved_contexts]:
                hybrid_score = doc_scores.get(doc_id, vec_score / 2)

                if hybrid_score > 0 and vec_score > 0:
                    retrieved_contexts.append(
                        RetrievedContext(
                            doc_id=doc_id,
                            content=doc.page_content,
                            score=hybrid_score,
                            source="vector" if bm25_normalized.get(doc_id, 0) == 0 else "hybrid",
                            retrieved_at=datetime.now(timezone.utc).isoformat(),
                            metadata={
                                "vector_score": vec_score,
                                "source_file": doc.metadata.get("source"),
                            },
                        )
                    )

        # Sort by hybrid score descending
        retrieved_contexts.sort(key=lambda x: x.score, reverse=True)

        # Limit to k results
        retrieved_contexts = retrieved_contexts[:k]

        print(f"  Hybrid search returned {len(retrieved_contexts)} results")

        # Store results
        if store_results and retrieved_contexts:
            self.context_store.store_search_context(
                context_id=cache_key,
                query=query,
                retrieved_contexts=retrieved_contexts,
                metadata={
                    "num_bm25_results": len(bm25_results),
                    "num_vector_results": len(vector_results),
                    "k": k,
                },
            )

        return retrieved_contexts

    def search_by_context_id(
        self,
        context_id: str,
    ) -> Optional[SearchContext]:
        """
        Retrieve a previously stored search context.

        Args:
            context_id: ID of the context to retrieve

        Returns:
            SearchContext or None if not found
        """
        return self.context_store.get_search_context(context_id)

    def search_by_query_substring(
        self,
        query_substring: str,
        limit: int = 10,
    ) -> List[SearchContext]:
        """
        Find previous search contexts matching a query substring.

        Args:
            query_substring: Substring to search for in previous queries
            limit: Maximum results to return

        Returns:
            List of matching search contexts
        """
        return self.context_store.search_by_query(query_substring, limit=limit)

    def get_recent_searches(self, limit: int = 20) -> List[SearchContext]:
        """Get the most recent search contexts."""
        return self.context_store.get_recent_contexts(limit=limit)

    def _generate_cache_key(self, query: str, k: int) -> str:
        """Generate a deterministic cache key for a query."""
        query_normalized = query.lower().strip()
        key_string = f"{query_normalized}|k={k}"
        return hashlib.md5(key_string.encode()).hexdigest()[:16]

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the hybrid retriever."""
        bm25_stats = self.bm25_layer.get_stats()
        vector_stats = self.vector_layer.get_stats()
        context_stats = self.context_store.get_stats()

        return {
            "bm25": bm25_stats,
            "vector": vector_stats,
            "context_store": context_stats,
            "config": {
                "default_k": self.config.default_k,
                "bm25_weight": self.config.bm25_weight,
                "vector_weight": self.config.vector_weight,
            },
        }
