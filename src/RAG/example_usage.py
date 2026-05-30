"""
Example usage of the Hybrid RAG system.

This demonstrates how to:
1. Ingest documents from the knowledge folder
2. Perform hybrid search
3. Retrieve cached results
4. Get statistics
"""

from pathlib import Path

from .config import RAGConfig
from .bm25_layer import BM25Layer
from .vector_layer import VectorLayer
from .context_store import ContextStore
from .ingestor import KnowledgeIngestor
from .hybrid_retriever import HybridRetriever


def main():
    """Demo the Hybrid RAG system."""
    # Initialize configuration
    config = RAGConfig(
        knowledge_dir=Path("./knowledge"),
        data_dir=Path("./data"),
    )

    print("=== Hybrid RAG Demo ===\n")
    print(f"Knowledge folder: {config.knowledge_dir}")
    print(f"Data folder: {config.data_dir}\n")

    # Initialize all components
    print("Initializing components...")
    bm25_layer = BM25Layer(
        index_path=config.bm25_index_path,
        k1=config.k1,
        b=config.b,
    )

    vector_layer = VectorLayer(
        persist_directory=config.vector_store_path,
        collection_name=config.vector_collection_name,
        embedding_model=config.embedding_model,
    )

    context_store = ContextStore(config.context_db_path)

    ingestor = KnowledgeIngestor(
        config=config,
        bm25_layer=bm25_layer,
        vector_layer=vector_layer,
    )

    retriever = HybridRetriever(
        config=config,
        bm25_layer=bm25_layer,
        vector_layer=vector_layer,
        context_store=context_store,
    )

    print("✓ Components initialized\n")

    # Step 1: Ingest documents
    print("=== Step 1: Ingesting Documents ===")
    print("Looking for documents in knowledge folder...")

    stats = ingestor.ingest_folder(force_rebuild=False)
    print(f"Ingestion stats: {stats}\n")

    # Step 2: Perform search
    print("=== Step 2: Performing Hybrid Search ===")
    query = "example query about your topic"

    results = retriever.search(
        query=query,
        k=5,
        use_cache=True,
        store_results=True,
    )

    print(f"\nSearch results for '{query}':")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result.score:.4f}")
        print(f"   Source: {result.source}")
        print(f"   Content: {result.content[:200]}...")

    # Step 3: Cache hit demonstration
    print("\n=== Step 3: Cache Hit (fast path) ===")
    print("Performing same query again...")

    cached_results = retriever.search(
        query=query,
        k=5,
        use_cache=True,
        store_results=True,
    )

    print(f"Retrieved {len(cached_results)} results from cache")

    # Step 4: Get statistics
    print("\n=== Step 4: Statistics ===")
    stats = retriever.get_stats()
    print(f"BM25: {stats['bm25']}")
    print(f"Vector: {stats['vector']}")
    print(f"Context Store: {stats['context_store']}")

    # Step 5: Retrieve previous search
    print("\n=== Step 5: Retrieve Previous Search ===")
    recent_searches = retriever.get_recent_searches(limit=3)

    for search_ctx in recent_searches:
        print(f"\nQuery: {search_ctx.query}")
        print(f"Results: {len(search_ctx.retrieved_contexts)}")
        print(f"Time: {search_ctx.retrieved_at}")

    print("\n=== Demo Complete ===")


def quick_search_demo():
    """Quick demo focusing only on search (assuming data is already indexed)."""
    from .config import default_config as config

    # Initialize only search components
    bm25_layer = BM25Layer(
        index_path=config.bm25_index_path,
        k1=config.k1,
        b=config.b,
    )

    # Try to load existing BM25 index
    if not bm25_layer._load_index():
        print("No BM25 index found. Please run ingest first.")
        return

    vector_layer = VectorLayer(
        persist_directory=config.vector_store_path,
        collection_name=config.vector_collection_name,
        embedding_model=config.embedding_model,
    )

    context_store = ContextStore(config.context_db_path)

    retriever = HybridRetriever(
        config=config,
        bm25_layer=bm25_layer,
        vector_layer=vector_layer,
        context_store=context_store,
    )

    # Quick search
    query = "machine learning"
    results = retriever.search(query=query, k=5)

    print(f"Results for '{query}':")
    for i, result in enumerate(results, 1):
        print(f"{i}. [{result.score:.4f}] {result.content[:100]}...")


if __name__ == "__main__":
    main()
