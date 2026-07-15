#test_hybrid_retriever.py
from pathlib import Path
import traceback

from src.retrieval.hybrid_retriever import HybridRetriever


# Define project root and index paths
PROJECT_ROOT = Path(__file__).resolve().parent
INDEX_DIR = PROJECT_ROOT / "data" / "indexes"

FAISS_INDEX_PATH = INDEX_DIR / "faiss_index"
BM25_INDEX_DIR = INDEX_DIR / "bm25"


def print_hybrid_search_results(query: str, results: list, top_k: int = 5) -> None:
    """Print formatted results from the hybrid retriever."""
    print("\n" + "=" * 80)
    print(f"HYBRID RETRIEVAL RESULTS FOR QUERY: '{query}' (Top {top_k})")
    print("=" * 80)

    if not results:
        print("No results found.")
        return

    for rank, result in enumerate(results, start=1):
        doc = result.document
        score = result.score
        faiss_score = result.faiss_score
        bm25_score = result.bm25_score
        source = result.source

        print(f"\nRank: {rank}")
        print(
            f"Final Score: {score:.4f} "
            f"(FAISS: {faiss_score:.4f}, BM25: {bm25_score:.4f})"
        )
        print(f"Source: {source}")

        print("Metadata:")
        for key, value in doc.metadata.items():
            print(f"  {key}: {value}")

        # Display content preview, limiting length and handling newlines
        content_preview = doc.page_content[:700].replace("\n", " ").strip()
        if len(content_preview) > 700:
            content_preview = content_preview[:700] + "..."
        print("\nContent preview:")
        print(content_preview)
        print("-" * 40)


def main():
    try:
        # Initialize the Hybrid Retriever
        print("Initializing HybridRetriever...")
        retriever = HybridRetriever(
            faiss_index_path=str(FAISS_INDEX_PATH),
            bm25_index_dir=str(BM25_INDEX_DIR),
            # Use default weights and penalties for now
        )
        print("HybridRetriever initialized successfully.")

        # Define test queries
        queries = [
            "What is software testing?",
            "What is regression testing?",
            "What is boundary value analysis?",
            "What is a test plan?",
            "What are the benefits of automated testing?",
            "How to handle test data?",
        ]

        # Run retrieval for each query
        for query in queries:
            print(f"\nRunning search for query: '{query}'")
            # Retrieve top 5 results, using 20 candidates internally
            results = retriever.search(query=query, top_k=5, candidate_k=20)
            print_hybrid_search_results(query=query, results=results, top_k=5)

    except FileNotFoundError as e:
        print(
            f"\nError: Index files not found. Please ensure you have run "
            f"'python build_index.py' first. Details: {e}"
        )
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
