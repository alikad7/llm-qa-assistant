# FILE: test_hybrid_retriever_reranker.py

from __future__ import annotations

from langchain_core.documents import Document

from src.retrieval.hybrid_retriever import (
    HybridRetriever,
    HybridSearchResult,
)
from src.retrieval.post_processing import MMRPostProcessor


class MockQueryAnalysis:
    needs_retrieval = True
    semantic_query = "regression testing"
    bm25_query = "regression testing"


class MockQueryAnalyzer:
    def analyze(self, query: str) -> MockQueryAnalysis:
        return MockQueryAnalysis()


class MockReranker:
    def __init__(self) -> None:
        self.called = False
        self.received_query: str | None = None
        self.received_top_k: int | None = None
        self.received_results_count: int | None = None

    def rerank(
        self,
        query,
        results,
        *,
        top_k,
    ):
        self.called = True
        self.received_query = query
        self.received_top_k = top_k
        self.received_results_count = len(results)

        reranked = sorted(
            results,
            key=lambda result: (
                "regression"
                in result.document.page_content.lower()
            ),
            reverse=True,
        )

        return reranked[:top_k]


class FailingReranker:
    def __init__(self) -> None:
        self.called = False

    def rerank(
        self,
        query,
        results,
        *,
        top_k,
    ):
        self.called = True
        raise RuntimeError("Simulated reranker failure")


class MockPostProcessor:
    def __init__(self) -> None:
        self.called = False
        self.received_top_k: int | None = None
        self.received_results_count: int | None = None

    def process(
        self,
        results,
        *,
        top_k,
    ):
        self.called = True
        self.received_top_k = top_k
        self.received_results_count = len(results)
        return list(results)[:top_k]


class FailingPostProcessor:
    def __init__(self) -> None:
        self.called = False

    def process(
        self,
        results,
        *,
        top_k,
    ):
        self.called = True
        raise RuntimeError("Simulated post-processing failure")


def make_result(
    content: str,
    score: float,
    source: str = "mock-document",
) -> HybridSearchResult:
    return HybridSearchResult(
        document=Document(
            page_content=content,
            metadata={"source": source},
        ),
        score=score,
        faiss_score=score,
        bm25_score=score,
        source="faiss+bm25",
    )


def build_test_retriever(
    reranker=None,
    post_processor=None,
) -> HybridRetriever:
    """
    Construct a lightweight HybridRetriever without loading real indexes.
    """
    retriever = HybridRetriever.__new__(HybridRetriever)

    retriever.query_analyzer = MockQueryAnalyzer()
    retriever.reranker = reranker
    retriever.post_processor = post_processor

    retriever._search_faiss = lambda query, k: []
    retriever._search_bm25 = lambda query, k: []

    retriever._merge_faiss_results = (
        lambda merged_results, faiss_results: None
    )
    retriever._merge_bm25_results = (
        lambda merged_results, bm25_results: None
    )

    retriever._rank_merged_results = lambda merged_results: [
        make_result(
            content="General software testing content.",
            score=0.90,
            source="doc-1",
        ),
        make_result(
            content=(
                "Regression testing verifies that previously working "
                "functionality remains unchanged."
            ),
            score=0.40,
            source="doc-2",
        ),
        make_result(
            content=(
                "Regression testing verifies that previously working "
                "functionality remains unchanged."
            ),
            score=0.30,
            source="doc-2",
        ),
    ]

    return retriever


def test_successful_reranking() -> None:
    mock_reranker = MockReranker()
    retriever = build_test_retriever(reranker=mock_reranker)

    results = retriever.search(
        query="What is regression testing?",
        top_k=1,
        candidate_k=2,
    )

    assert mock_reranker.called is True
    assert mock_reranker.received_query == (
        "What is regression testing?"
    )
    assert mock_reranker.received_top_k == 2
    assert mock_reranker.received_results_count == 2

    assert len(results) == 1
    assert (
        "regression"
        in results[0].document.page_content.lower()
    )

    print("PASS: HybridRetriever called the reranker.")
    print(
        "Top result:",
        results[0].document.page_content,
    )


def test_reranker_fallback() -> None:
    failing_reranker = FailingReranker()
    retriever = build_test_retriever(reranker=failing_reranker)

    results = retriever.search(
        query="What is regression testing?",
        top_k=1,
        candidate_k=2,
    )

    assert failing_reranker.called is True
    assert len(results) == 1

    assert results[0].document.page_content == (
        "General software testing content."
    )

    assert results[0].score == 0.90

    print(
        "PASS: Reranker failure safely falls back "
        "to hybrid ranking."
    )


def test_post_processor_is_called() -> None:
    mock_reranker = MockReranker()
    mock_post_processor = MockPostProcessor()

    retriever = build_test_retriever(
        reranker=mock_reranker,
        post_processor=mock_post_processor,
    )

    results = retriever.search(
        query="What is regression testing?",
        top_k=1,
        candidate_k=3,
    )

    assert mock_reranker.called is True
    assert mock_post_processor.called is True
    assert mock_post_processor.received_top_k == 1
    assert mock_post_processor.received_results_count == 3

    assert len(results) == 1

    print("PASS: HybridRetriever called the post-processor.")


def test_post_processor_fallback() -> None:
    failing_post_processor = FailingPostProcessor()

    retriever = build_test_retriever(
        reranker=None,
        post_processor=failing_post_processor,
    )

    results = retriever.search(
        query="What is regression testing?",
        top_k=1,
        candidate_k=2,
    )

    assert failing_post_processor.called is True
    assert len(results) == 1
    assert results[0].document.page_content == (
        "General software testing content."
    )

    print(
        "PASS: Post-processor failure safely falls back "
        "to top_k results."
    )


def test_real_mmr_post_processor_deduplicates_results() -> None:
    post_processor = MMRPostProcessor(
        lambda_param=0.7,
        min_score_threshold=0.0,
        max_context_chars=None,
        deduplicate_by_content=True,
    )

    retriever = build_test_retriever(
        reranker=None,
        post_processor=post_processor,
    )

    results = retriever.search(
        query="What is regression testing?",
        top_k=3,
        candidate_k=3,
    )

    contents = [
        result.document.page_content
        for result in results
    ]

    assert len(contents) == len(set(contents))

    print("PASS: MMRPostProcessor deduplicated duplicate content.")


def main() -> None:
    test_successful_reranking()
    test_reranker_fallback()
    test_post_processor_is_called()
    test_post_processor_fallback()
    test_real_mmr_post_processor_deduplicates_results()

    print("\nAll HybridRetriever reranker/MMR tests passed.")


if __name__ == "__main__":
    main()
