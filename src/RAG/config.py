"""
Configuration settings for the Hybrid RAG system.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class RAGConfig:
    """Configuration for the Hybrid RAG system."""

    # Base paths
    data_dir: Path = Path("./data")
    knowledge_dir: Path = Path("./knowledge")

    # BM25 configuration
    bm25_index_path: Path = Path("./data/bm25_index.pkl")
    k1: float = 1.5  # BM25 term frequency saturation parameter
    b: float = 0.75  # BM25 document length normalization

    # Vector configuration
    vector_store_path: Path = Path("./data/vector_store")
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_collection_name: str = "hybrid_rag"

    # Context store configuration
    context_db_path: Path = Path("./data/retrieved_contexts.db")

    # Retrieval configuration
    default_k: int = 5  # Default number of documents to retrieve
    bm25_weight: float = 0.5  # Weight for BM25 score (0-1)
    vector_weight: float = 0.5  # Weight for vector score (0-1)
    rerank_threshold: float = 0.7  # Threshold for reranking

    # Ingestion configuration
    chunk_size: int = 512
    chunk_overlap: int = 128

    def __post_init__(self):
        """Ensure all directories exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "RAGConfig":
        """Create config from dictionary."""
        return cls(**config_dict)

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "data_dir": str(self.data_dir),
            "knowledge_dir": str(self.knowledge_dir),
            "bm25_index_path": str(self.bm25_index_path),
            "k1": self.k1,
            "b": self.b,
            "vector_store_path": str(self.vector_store_path),
            "embedding_model": self.embedding_model,
            "vector_collection_name": self.vector_collection_name,
            "context_db_path": str(self.context_db_path),
            "default_k": self.default_k,
            "bm25_weight": self.bm25_weight,
            "vector_weight": self.vector_weight,
            "rerank_threshold": self.rerank_threshold,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }


# Default configuration instance
default_config = RAGConfig()
