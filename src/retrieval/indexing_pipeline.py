from typing import List

from langchain_core.documents import Document
from src.retrieval.vector_store import VectorStoreManager
from src.retrieval.bm25_store import BM25Store


class IndexingPipeline:
    """Orchestrates the build process of both FAISS and BM25 indexes."""

    def __init__(
        self,
        faiss_path: str = "data/indexes/faiss_index",
        bm25_dir: str = "data/indexes/bm25",
    ) -> None:
        self.vector_manager = VectorStoreManager(index_path=faiss_path)
        self.bm25_store = BM25Store(index_dir=bm25_dir)

    def build_and_persist(self, chunks: List[Document]) -> None:
        """
        Builds both Vector and BM25 indexes from the provided chunks
        and persists them to disk.
        """
        if not chunks:
            raise ValueError("No chunks provided to build indexes.")

        print(f"🎬 Starting hybrid index generation for {len(chunks)} chunks...\n")

        try:
            self.vector_manager.create_and_save_index(chunks)
        except Exception as e:
            print(f"❌ Failed to build FAISS index: {e}")
            raise

        try:
            self.bm25_store.create_and_save_index(chunks)
        except Exception as e:
            print(f"❌ Failed to build BM25 index: {e}")
            raise

        print("\n Indexing pipeline completed successfully. Both indexes are persisted.")
