import os
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.retrieval.embeddings import EmbeddingManager


class VectorStoreManager:
    """Handles FAISS index creation, persistence, and loading."""

    def __init__(self, index_path: str = "data/indexes/faiss_index") -> None:
        self.index_path = index_path
        self.embedding_manager = EmbeddingManager()

    def create_and_save_index(self, chunks: List[Document]) -> FAISS:
        """Creates a FAISS index from documents and saves it to disk."""
        print(f"Creating FAISS index for {len(chunks)} chunks...")

        embeddings = self.embedding_manager.get_embeddings()
        vector_db = FAISS.from_documents(chunks, embeddings)

        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)

        vector_db.save_local(self.index_path)
        print(f"✅ FAISS index saved to {self.index_path}")
        return vector_db

    def load_index(self) -> FAISS:
        """Loads the FAISS index from disk."""
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"FAISS index not found at {self.index_path}")

        embeddings = self.embedding_manager.get_embeddings()
        return FAISS.load_local(
            self.index_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )
