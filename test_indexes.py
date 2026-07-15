from pathlib import Path

from src.retrieval.vector_store import VectorStoreManager
from src.retrieval.bm25_store import BM25Store


PROJECT_ROOT = Path(__file__).resolve().parent
INDEX_DIR = PROJECT_ROOT / "data" / "indexes"

FAISS_INDEX_DIR = INDEX_DIR / "faiss_index"
BM25_INDEX_DIR = INDEX_DIR / "bm25"


def print_result(title: str, results):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    if not results:
        print("No results found.")
        return

    for rank, item in enumerate(results, start=1):
        if isinstance(item, tuple):
            document, score = item
        else:
            document = item
            score = None

        print(f"\nRank: {rank}")

        if score is not None:
            print(f"Score: {score:.4f}")

        print("Metadata:")
        for key, value in document.metadata.items():
            print(f"  {key}: {value}")

        preview = document.page_content[:700].replace("\n", " ")
        print("\nContent preview:")
        print(preview)


def main():
    query = "What is software testing?"

    print(f"Project root: {PROJECT_ROOT}")
    print(f"FAISS index path: {FAISS_INDEX_DIR}")
    print(f"BM25 index path: {BM25_INDEX_DIR}")
    print(f"Test query: {query}")

    print("\nLoading FAISS index...")
    vector_store_manager = VectorStoreManager(index_path=str(FAISS_INDEX_DIR))
    faiss_index = vector_store_manager.load_index()

    print("Searching FAISS index...")
    faiss_results = faiss_index.similarity_search_with_score(query, k=5)
    print_result("FAISS RESULTS", faiss_results)

    print("\nLoading BM25 index...")
    bm25_store = BM25Store(index_dir=str(BM25_INDEX_DIR))
    bm25_store.load_index()

    print("Searching BM25 index...")
    bm25_results = bm25_store.search(query=query, top_k=5)
    print_result("BM25 RESULTS", bm25_results)


if __name__ == "__main__":
    main()
