"""
Vector Layer - Persistent semantic retrieval using embeddings.

Handles vector similarity search with persistent ChromaDB storage.
"""

from pathlib import Path
from typing import List, Optional, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
from chromadb.config import Settings


class VectorLayer:
    """
    Persistent vector similarity retrieval layer.

    Features:
    - Semantic similarity search using embeddings
    - Persistent ChromaDB storage
    - Efficient vector operations
    """

    def __init__(
        self,
        persist_directory: Path,
        collection_name: str = "hybrid_rag",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """
        Initialize vector layer.

        Args:
            persist_directory: Path to persist ChromaDB data
            collection_name: Name of the ChromaDB collection
            embedding_model: HuggingFace embedding model name
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.embedding_model = embedding_model

        # Initialize embedding function
        self.embedding_function = HuggingFaceEmbeddings(
            model_name=embedding_model
        )

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(persist_directory),
            settings=Settings(anonymized_telemetry=False)
        )

        # Initialize Chroma store
        self.vector_store = Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=self.embedding_function,
            persist_directory=str(persist_directory),
        )

    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Add documents to the vector store.

        Args:
            documents: List of document texts
            metadatas: Optional list of metadata dicts
            ids: Optional list of document IDs

        Returns:
            List of assigned document IDs
        """
        # Create LangChain Document objects
        docs = []
        for i, text in enumerate(documents):
            metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
            doc_id = ids[i] if ids and i < len(ids) else f"doc_{i}"
            docs.append(Document(page_content=text, metadata=metadata, id=doc_id))

        # Add to vector store
        inserted_ids = self.vector_store.add_documents(docs)
        return inserted_ids

    def search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[dict] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Search for similar documents.

        Args:
            query: Search query text
            k: Number of results to return
            filter: Optional metadata filter

        Returns:
            List of (Document, similarity_score) tuples
        """
        if filter:
            results = self.vector_store.similarity_search_with_relevance_scores(
                query, k=k, filter=filter
            )
        else:
            results = self.vector_store.similarity_search_with_relevance_scores(
                query, k=k
            )
        return results

    def get_by_ids(self, ids: List[str]) -> List[Document]:
        """Get documents by their IDs."""
        return self.vector_store.get_by_ids(ids)

    def delete_documents(self, ids: List[str]) -> None:
        """Delete documents by IDs."""
        if ids:
            self.vector_store.delete(ids=ids)

    def clear_collection(self) -> None:
        """Clear all documents from the collection."""
        collection = self.client.get_collection(self.collection_name)
        collection.delete()

    def get_stats(self) -> dict:
        """Get statistics about the vector store."""
        collection = self.client.get_collection(self.collection_name)
        count = collection.count()
        return {
            "num_documents": count,
            "collection_name": self.collection_name,
            "embedding_model": self.embedding_model,
            "persist_directory": str(self.persist_directory),
        }

    def persist(self) -> None:
        """Persist the vector store to disk."""
        # ChromaDB persists automatically, but this method exists for compatibility
        pass
