# Hybrid RAG System

A persistent hybrid retrieval system combining BM25 (lexical) and Vector (semantic) search for enhanced document retrieval.

## Features

- **Persistent BM25 Index**: No recomputation on repeated queries
- **Vector Similarity Search**: Semantic understanding with ChromaDB
- **Hybrid Scoring**: Weighted fusion of both retrieval methods
- **Context Storage**: Persist retrieved results for fast cache hits
- **Incremental Updates**: Add documents without rebuilding entire index

## Architecture

```
RAG/
├── __init__.py          # Package initialization
├── config.py            # Configuration settings
├── bm25_layer.py        # Lexical retrieval (BM25)
├── vector_layer.py      # Semantic retrieval (ChromaDB)
├── context_store.py     # Persistent context storage
├── ingestor.py          # Knowledge folder ingestion
├── hybrid_retriever.py  # Main hybrid retrieval coordinator
└── example_usage.py     # Usage examples
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r RAG/requirements.txt
```

### 2. Prepare Knowledge Folder

Create a `knowledge` folder with your documents:

```
knowledge/
├── docs/
│   ├── guide1.md
│   └── guide2.txt
├── code/
│   └── example.py
└── data/
    └── info.json
```

### 3. Ingest Documents

```python
from pathlib import Path
from RAG import RAGConfig, KnowledgeIngestor, BM25Layer, VectorLayer

# Initialize
config = RAGConfig(knowledge_dir=Path("./knowledge"))
bm25_layer = BM25Layer(index_path=config.bm25_index_path)
vector_layer = VectorLayer(persist_directory=config.vector_store_path)
ingestor = KnowledgeIngestor(config, bm25_layer, vector_layer)

# Ingest all documents
stats = ingestor.ingest_folder()
```

### 4. Perform Hybrid Search

```python
from RAG import HybridRetriever, ContextStore

context_store = ContextStore(config.context_db_path)
retriever = HybridRetriever(config, bm25_layer, vector_layer, context_store)

# Search
results = retriever.search("your query", k=5)

# Results are automatically cached for fast repeats
```

## Configuration

Edit `RAGConfig` in `config.py`:

```python
config = RAGConfig(
    knowledge_dir=Path("./knowledge"),
    data_dir=Path("./data"),
    chunk_size=512,
    chunk_overlap=128,
    bm25_weight=0.5,    # BM25 vs Vector weight
    vector_weight=0.5,
    default_k=5,        # Default results per search
)
```

## Component Details

### BM25 Layer (`bm25_layer.py`)
- `BM25Okapi` implementation with persistent storage
- Saves index to pickle file for fast loading
- Supports incremental document addition

### Vector Layer (`vector_layer.py`)
- ChromaDB-based vector storage
- Persistent with automatic saving
- Supports semantic similarity search

### Context Store (`context_store.py`)
- SQLite database for retrieved contexts
- Enables fast cache hits for repeated queries
- Supports query history and context retrieval

### Ingestor (`ingestor.py`)
- Recursive folder scanning
- Smart text chunking with overlap
- Supports multiple file formats

### Hybrid Retriever (`hybrid_retriever.py`)
- Combines BM25 and Vector scores
- Normalized scoring with configurable weights
- Automatic cache management

## File Format Support

- **Text**: `.txt`, `.md`, `.markdown`
- **Code**: `.py`, `.js`, `.java`, `.cpp`, `.c`, `.h`
- **Structured**: `.json`
- **Config**: `.yaml`, `.yml`

## Performance Tips

1. **Chunk sizing**: Larger chunks for dense content, smaller for structured data
2. **Weight tuning**: Adjust BM25/Vector weights based on use case
3. **Cache policy**: Reuse contexts for identical or similar queries
4. **Batch ingestion**: Process multiple files together for efficiency

## Example Usage

See `example_usage.py` for complete examples:

```bash
python -m RAG.example_usage
```

## Integration with 7-Layer Memory

This Hybrid RAG system can integrate with your existing 7-layer memory architecture:

```python
from memory_manager import MemoryManager
from RAG.config import RAGConfig

# Share data directory
rag_config = RAGConfig(data_dir=Path(MemoryManager.DATA_DIR))
```

## License

MIT License - Free to use and modify.
