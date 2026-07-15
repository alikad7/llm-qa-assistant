import os
import pickle
import re
from typing import List, Tuple

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


class BM25Store:
    """Handles BM25 keyword index creation, persistence, and loading."""

    def __init__(self, index_dir: str = "data/indexes/bm25") -> None:
        self.index_dir = index_dir
        self.index_file = os.path.join(index_dir, "bm25_index.pkl")
        self.documents_file = os.path.join(index_dir, "documents.pkl")
        self.bm25 = None
        self.documents: List[Document] = []

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenizes text for BM25 retrieval.

        This is intentionally simple and stable:
        - lowercases text
        - keeps only alphanumeric tokens
        - removes very short tokens
        """
        text = text.lower()
        tokens = re.findall(r"\b[a-zA-Z0-9]+\b", text)
        return [token for token in tokens if len(token) > 1]

    def create_and_save_index(self, documents: List[Document]) -> None:
        """Creates a BM25 index from documents and saves it to disk."""
        if not documents:
            raise ValueError("Cannot create BM25 index from an empty document list.")

        print(f"Creating BM25 index for {len(documents)} chunks...")

        self.documents = documents
        tokenized_corpus = [self._tokenize(doc.page_content) for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

        os.makedirs(self.index_dir, exist_ok=True)

        with open(self.index_file, "wb") as f:
            pickle.dump(self.bm25, f)

        with open(self.documents_file, "wb") as f:
            pickle.dump(self.documents, f)

        print(f"✅ BM25 index saved to {self.index_dir}")

    def load_index(self) -> None:
        """Loads BM25 index and documents from disk."""
        if not os.path.exists(self.index_file):
            raise FileNotFoundError(f"BM25 index not found at {self.index_file}")

        if not os.path.exists(self.documents_file):
            raise FileNotFoundError(f"BM25 documents not found at {self.documents_file}")

        with open(self.index_file, "rb") as f:
            self.bm25 = pickle.load(f)

        with open(self.documents_file, "rb") as f:
            self.documents = pickle.load(f)

        print(f"✅ BM25 index loaded from {self.index_dir}")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        """Searches BM25 index and returns top-k documents with scores."""
        if self.bm25 is None or not self.documents:
            raise ValueError("BM25 index is not loaded. Call load_index() first.")

        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda idx: scores[idx],
            reverse=True,
        )[:top_k]

        return [
            (self.documents[idx], float(scores[idx]))
            for idx in ranked_indices
            if scores[idx] > 0
        ]

    