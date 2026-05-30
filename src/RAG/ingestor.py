"""
Knowledge Ingestor - Handles ingestion from knowledge folder.

Processes documents from the knowledge folder and indexes them
into both BM25 and vector stores for hybrid retrieval.
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass

from .config import RAGConfig
from .bm25_layer import BM25Layer
from .vector_layer import VectorLayer


@dataclass
class DocumentChunk:
    """Represents a processed document chunk."""
    content: str
    metadata: Dict[str, Any]
    doc_id: str
    source: str


class KnowledgeIngestor:
    """
    Ingestion handler for knowledge documents.

    Features:
    - Reads documents from knowledge folder
    - Split into chunks with overlap
    - Indexes to both BM25 and vector stores
    - Tracks document metadata and source
    """

    def __init__(
        self,
        config: RAGConfig,
        bm25_layer: BM25Layer,
        vector_layer: VectorLayer,
    ):
        """
        Initialize ingestor.

        Args:
            config: RAG configuration
            bm25_layer: BM25 index layer
            vector_layer: Vector store layer
        """
        self.config = config
        self.bm25_layer = bm25_layer
        self.vector_layer = vector_layer

    def _read_file(self, file_path: Path) -> str:
        """Read content from a file based on its extension."""
        try:
            if file_path.suffix.lower() == ".txt":
                with open(file_path, "r", encoding="utf-8") as f:
                    return f.read()
            elif file_path.suffix.lower() in [".md", ".markdown"]:
                with open(file_path, "r", encoding="utf-8") as f:
                    return f.read()
            elif file_path.suffix.lower() == ".json":
                import json
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return json.dumps(data, ensure_ascii=False, indent=2)
            elif file_path.suffix.lower() in [".py", ".js", ".java", ".cpp", ".c", ".h"]:
                with open(file_path, "r", encoding="utf-8") as f:
                    return f.read()
            else:
                # Try to read as text
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        return f.read()
                except:
                    return ""
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return ""

    def _chunk_text(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int,
        doc_id: str,
        source: str,
    ) -> List[DocumentChunk]:
        """
        Split text into overlapping chunks.

        Args:
            text: Text to chunk
            chunk_size: Maximum chunk size
            chunk_overlap: Overlap between chunks
            doc_id: Base document ID
            source: Source file path

        Returns:
            List of DocumentChunk objects
        """
        if not text.strip():
            return []

        # Split by paragraphs or sentences for better chunks
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = []
        current_length = 0

        chunk_index = 0
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_length = len(para)

            if current_length + para_length > chunk_size and current_chunk:
                # Current chunk is full, save it
                chunk_content = "\n\n".join(current_chunk)
                chunks.append(DocumentChunk(
                    content=chunk_content,
                    metadata={"chunk_index": chunk_index, "source": source},
                    doc_id=f"{doc_id}_chunk_{chunk_index}",
                    source=source,
                ))
                chunk_index += 1

                # Start new chunk with overlap
                if chunk_overlap > 0:
                    # Keep last X characters for overlap
                    overlap_text = chunk_content[-chunk_overlap:] if len(chunk_content) > chunk_overlap else chunk_content
                    current_chunk = [overlap_text]
                    current_length = len(overlap_text)
                else:
                    current_chunk = []
                    current_length = 0

            current_chunk.append(para)
            current_length += para_length

        # Add final chunk
        if current_chunk:
            chunk_content = "\n\n".join(current_chunk)
            chunks.append(DocumentChunk(
                content=chunk_content,
                metadata={"chunk_index": chunk_index, "source": source},
                doc_id=f"{doc_id}_chunk_{chunk_index}",
                source=source,
            ))

        return chunks

    def ingest_folder(
        self,
        folder_path: Optional[Path] = None,
        subfolder: str = "",
        force_rebuild: bool = False,
    ) -> Dict[str, Any]:
        """
        Ingest all documents from a folder.

        Args:
            folder_path: Path to folder (defaults to config.knowledge_dir)
            subfolder: Subfolder within knowledge directory
            force_rebuild: If True, rebuild indices from scratch

        Returns:
            Ingestion statistics dictionary
        """
        folder_path = folder_path or self.config.knowledge_dir / subfolder

        if not folder_path.exists():
            print(f"Warning: Folder {folder_path} does not exist")
            return {"message": f"Folder not found: {folder_path}"}

        # Get all files
        files = []
        for ext in ["*.txt", "*.md", "*.markdown", "*.json", "*.py", "*.js", "*.java"]:
            files.extend(folder_path.glob(f"**/{ext}"))

        if not files:
            print(f"No supported files found in {folder_path}")
            return {"message": f"No files found in {folder_path}"}

        print(f"Found {len(files)} files to ingest")

        all_chunks = []
        all_docs = []
        all_ids = []

        for file_path in files:
            print(f"  Processing: {file_path.name}")
            content = self._read_file(file_path)

            if content.strip():
                # Create chunks
                chunks = self._chunk_text(
                    content,
                    chunk_size=self.config.chunk_size,
                    chunk_overlap=self.config.chunk_overlap,
                    doc_id=file_path.stem,
                    source=str(file_path.relative_to(self.config.knowledge_dir.parent)),
                )

                all_chunks.extend(chunks)
                all_docs.extend([chunk.content for chunk in chunks])
                all_ids.extend([chunk.doc_id for chunk in chunks])

        if not all_chunks:
            return {"message": "No content found in files"}

        # Index in BM25
        print(f"Indexing {len(all_docs)} chunks in BM25...")
        self.bm25_layer.build_index(
            documents=all_docs,
            doc_ids=all_ids,
            rebuild=force_rebuild,
        )
        self.bm25_layer.save()

        # Index in vector store
        print(f"Indexing {len(all_docs)} chunks in vector store...")
        metadatas = [
            {
                **chunk.metadata,
                "doc_id": chunk.doc_id,
                "source": chunk.source,
            }
            for chunk in all_chunks
        ]
        self.vector_layer.add_documents(
            documents=all_docs,
            metadatas=metadatas,
            ids=all_ids,
        )

        stats = {
            "num_files": len(files),
            "num_chunks": len(all_chunks),
            "folder": str(folder_path),
            "bm25_stats": self.bm25_layer.get_stats(),
            "vector_stats": self.vector_layer.get_stats(),
        }

        print(f"✓ Ingestion complete: {stats}")
        return stats

    def ingest_file(
        self,
        file_path: Path,
        doc_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Ingest a single file.

        Args:
            file_path: Path to the file
            doc_id: Optional document ID (defaults to filename stem)

        Returns:
            Ingestion statistics
        """
        doc_id = doc_id or file_path.stem

        content = self._read_file(file_path)
        if not content.strip():
            return {"message": f"No content in {file_path}"}

        chunks = self._chunk_text(
            content,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            doc_id=doc_id,
            source=str(file_path.relative_to(self.config.knowledge_dir.parent)),
        )

        # Index in BM25
        self.bm25_layer.add_documents(
            documents=[chunk.content for chunk in chunks],
            doc_ids=[chunk.doc_id for chunk in chunks],
        )
        self.bm25_layer.save()

        # Index in vector store
        metadatas = [
            {
                **chunk.metadata,
                "doc_id": chunk.doc_id,
                "source": chunk.source,
            }
            for chunk in chunks
        ]
        self.vector_layer.add_documents(
            documents=[chunk.content for chunk in chunks],
            metadatas=metadatas,
            ids=[chunk.doc_id for chunk in chunks],
        )

        stats = {
            "file": str(file_path),
            "num_chunks": len(chunks),
            "doc_id": doc_id,
        }

        print(f"✓ Ingested {file_path.name}: {len(chunks)} chunks")
        return stats

    def reindex_all(self) -> Dict[str, Any]:
        """Rebuild all indices from scratch."""
        return self.ingest_folder(force_rebuild=True)

    def get_stats(self) -> Dict[str, Any]:
        """Get ingestion statistics from both stores."""
        return {
            "bm25": self.bm25_layer.get_stats(),
            "vector": self.vector_layer.get_stats(),
        }
