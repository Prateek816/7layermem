"""
BM25 Layer - Persistent lexical retrieval using BM25 scoring.

Handles loading, saving, and searching a BM25 index.
No recomputation of the index on repeated queries.
"""

import pickle
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk

# Download required NLTK data
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")


class BM25Layer:
    """
    Persistent BM25 retrieval layer.

    Features:
    - Persistent index storage (no recomputation)
    - Fast lexical matching
    - Supports incremental updates
    """

    def __init__(
        self,
        index_path: Path,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        """
        Initialize BM25 layer.

        Args:
            index_path: Path to save/load the BM25 index
            k1: Term frequency saturation parameter (higher = less saturation)
            b: Document length normalization (0-1)
        """
        self.index_path = Path(index_path)
        self.k1 = k1
        self.b = b
        self.bm25: Optional[BM25Okapi] = None
        self.corpus: List[str] = []
        self.tokenized_corpus: List[List[str]] = []
        self.doc_ids: List[str] = []  # Document IDs for reference

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text using NLTK."""
        # Simple preprocessing
        text = re.sub(r"[^\w\s]", " ", text.lower())
        tokens = word_tokenize(text)
        return [t for t in tokens if t.isalnum()]

    def build_index(
        self,
        documents: List[str],
        doc_ids: Optional[List[str]] = None,
        rebuild: bool = False,
    ) -> None:
        """
        Build or rebuild the BM25 index from documents.

        Args:
            documents: List of document texts
            doc_ids: Optional list of document IDs (defaults to indices)
            rebuild: If True, rebuild even if index exists
        """
        if not rebuild and self._load_index():
            return

        self.corpus = documents
        self.tokenized_corpus = [self._tokenize(doc) for doc in documents]
        self.doc_ids = doc_ids or [str(i) for i in range(len(documents))]

        # Build BM25 index
        self.bm25 = BM25Okapi(
            self.tokenized_corpus,
            k1=self.k1,
            b=self.b,
        )
        print(f"✓ Built BM25 index with {len(documents)} documents")

    def search(
        self,
        query: str,
        k: int = 10,
    ) -> List[Tuple[int, float, str]]:
        """
        Search the BM25 index.

        Args:
            query: Search query string
            k: Number of top results to return

        Returns:
            List of (doc_index, score, doc_text) tuples sorted by score
        """
        if self.bm25 is None:
            if not self._load_index():
                raise RuntimeError("BM25 index not built or loaded")

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        # Get top k indices
        top_indices = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_indices:
            score = scores[idx]
            if score > 0:  # Only include non-zero scoring docs
                results.append((idx, float(score), self.corpus[idx]))

        return results

    def _save_index(self) -> bool:
        """Save the BM25 index to disk."""
        if self.bm25 is None:
            return False

        try:
            data = {
                "corpus": self.corpus,
                "doc_ids": self.doc_ids,
                "tokenized_corpus": self.tokenized_corpus,
                "k1": self.k1,
                "b": self.b,
                # BM25Okapi state
                "idf": self.bm25.idf,
                "doc_len": self.bm25.doc_len,
                "avgdl": self.bm25.average_length(),
            }

            with open(self.index_path, "wb") as f:
                pickle.dump(data, f)

            print(f"✓ Saved BM25 index to {self.index_path}")
            return True
        except Exception as e:
            print(f"✗ Failed to save BM25 index: {e}")
            return False

    def _load_index(self) -> bool:
        """Load the BM25 index from disk."""
        if not self.index_path.exists():
            return False

        try:
            with open(self.index_path, "rb") as f:
                data = pickle.load(f)

            self.corpus = data["corpus"]
            self.doc_ids = data.get("doc_ids", [str(i) for i in range(len(self.corpus))])
            self.tokenized_corpus = data["tokenized_corpus"]

            # Recreate BM25Okapi with saved state
            self.bm25 = BM25Okapi(
                corpus=None,  # Will set manually
                k1=data["k1"],
                b=data["b"],
            )
            # Manually set the BM25 state
            self.bm25.idf = data["idf"]
            self.bm25.doc_len = data["doc_len"]
            self.bm25.average_length = lambda: data["avgdl"]
            self.bm25.corpus = self.tokenized_corpus  # Set corpus for queries

            print(f"✓ Loaded BM25 index from {self.index_path} ({len(self.corpus)} docs)")
            return True
        except Exception as e:
            print(f"✗ Failed to load BM25 index: {e}")
            return False

    def save(self) -> bool:
        """Public method to save index."""
        return self._save_index()

    def add_documents(
        self,
        documents: List[str],
        doc_ids: Optional[List[str]] = None,
    ) -> None:
        """Add documents to existing index and rebuild."""
        if doc_ids and len(doc_ids) != len(documents):
            raise ValueError("doc_ids length must match documents length")

        # Append to existing corpus
        self.corpus.extend(documents)
        self.doc_ids.extend(doc_ids or [str(i) for i in range(len(self.corpus), len(self.corpus) + len(documents))])

        # Rebuild index with all documents
        self.build_index(self.corpus, self.doc_ids, rebuild=True)

    @property
    def is_loaded(self) -> bool:
        """Check if BM25 index is loaded."""
        return self.bm25 is not None

    def get_stats(self) -> dict:
        """Get statistics about the BM25 index."""
        return {
            "num_documents": len(self.corpus),
            "index_path": str(self.index_path),
            "is_loaded": self.is_loaded,
            "k1": self.k1,
            "b": self.b,
        }
