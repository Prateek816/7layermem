"""
Hybrid RAG System - BM25 + Vector Retrieval

A hybrid retrieval system that combines:
1. BM25 (lexical) retrieval - fast, exact keyword matching
2. Vector (semantic) retrieval - semantic similarity search
3. Persistent storage for indices and retrieved contexts
"""

from .config import RAGConfig
from .bm25_layer import BM25Layer
from .vector_layer import VectorLayer
from .context_store import ContextStore
from .ingestor import KnowledgeIngestor
from .hybrid_retriever import HybridRetriever

__version__ = "1.0.0"
__all__ = [
    "RAGConfig",
    "BM25Layer",
    "VectorLayer",
    "ContextStore",
    "KnowledgeIngestor",
    "HybridRetriever",
]
