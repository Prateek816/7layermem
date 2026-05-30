# Sample Knowledge Document

This is a sample document to demonstrate the Hybrid RAG system's ingestion capabilities.

## Introduction

The Hybrid RAG system combines two powerful retrieval methods:

1. **BM25 (Best Match 25)**: A statistical retrieval method that ranks documents based on term frequency and inverse document frequency. It excels at exact keyword matching and is very fast.

2. **Vector Search**: Uses machine learning embeddings to understand semantic meaning, making it excellent for finding conceptually similar content even without exact keyword matches.

## System Architecture

The system is designed with clean separation of concerns across multiple layers:

- **BM25 Layer**: Handles lexical/keyword-based retrieval with persistent indexing
- **Vector Layer**: Manages semantic similarity search using embeddings
- **Context Store**: Provides persistent storage for retrieved results
- **Hybrid Coordinator**: Fuses both retrieval methods with configurable weights

## Use Cases

### 1. Documentation Search
Find relevant sections in technical documentation by matching both specific terms and semantic concepts.

### 2. Code Retrieval
Search for code snippets by function name, comments, or conceptual patterns.

### 3. Research Knowledge Base
Find research papers and articles based on topic keywords or related concepts.

## Key Benefits

- **Speed**: BM25 provides instant keyword matching
- **Accuracy**: Vector search understands context and meaning
- **Persistence**: No need to recompute indices on every query
- **Hybrid Scoring**: Best of both worlds with configurable blending

## Configuration Example

```python
from RAG import RAGConfig

config = RAGConfig(
    k1=1.5,           # BM25 term frequency parameter
    b=0.75,           # BM25 document length normalization
    bm25_weight=0.5,  # Weight for BM25 in hybrid scoring
    chunk_size=512,   # Document chunk size
)
```

## Additional Information

For more details, see the README in the RAG directory or check the example usage scripts.

---
*This document is part of the knowledge base for demonstrating the Hybrid RAG system.*
