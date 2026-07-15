# FILE: src/retrieval/hybrid_retriever.py

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from langchain_core.documents import Document

from src.query_analyzer.fallback_query_analyzer import FallbackQueryAnalyzer
from src.retrieval.bm25_store import BM25Store
from src.retrieval.post_processing import MMRPostProcessor
from src.retrieval.vector_store import VectorStoreManager


@dataclass
class HybridSearchResult:
    document: Document
    score: float
    faiss_score: float
    bm25_score: float
    source: str


class HybridRetriever:
    """
    Hybrid retriever that combines semantic FAISS retrieval with lexical BM25 retrieval,
    then optionally applies reranking and MMR-based post-processing.
    """

    LOW_VALUE_SECTION_KEYWORDS = (
        "recommended reading",
        "references",
        "related articles",
        "further reading",
        "thoughts on",
        "comments",
        "reply",
        "leave a reply",
    )

    def __init__(
        self,
        faiss_index_path: str | Path = "data/indexes/faiss_index",
        bm25_index_dir: str | Path = "data/indexes/bm25",
        faiss_weight: float = 0.45,
        bm25_weight: float = 0.55,
        low_value_section_penalty: float = 0.20,
        short_chunk_penalty: float = 0.50,
        short_chunk_threshold: int = 250,
        min_score_threshold: float = 0.0,
        query_analyzer=None,
        reranker=None,
        post_processor: MMRPostProcessor | None = None,
    ) -> None:
        self.faiss_index_path = Path(faiss_index_path)
        self.bm25_index_dir = Path(bm25_index_dir)

        self.faiss_weight = faiss_weight
        self.bm25_weight = bm25_weight
        self.low_value_section_penalty = low_value_section_penalty
        self.short_chunk_penalty = short_chunk_penalty
        self.short_chunk_threshold = short_chunk_threshold
        self.min_score_threshold = min_score_threshold
        self.query_analyzer = query_analyzer or FallbackQueryAnalyzer()
        self.reranker = reranker
        self.post_processor = post_processor

        self.vector_store_manager = VectorStoreManager(
            index_path=str(self.faiss_index_path)
        )
        self.faiss_index = self.vector_store_manager.load_index()

        self.bm25_store = BM25Store(index_dir=str(self.bm25_index_dir))
        self.bm25_store.load_index()

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 20,
    ) -> List[HybridSearchResult]:
        query = (query or "").strip()
        if not query:
            return []

        analysis = self.query_analyzer.analyze(query)
        print("ANALYSIS:", analysis)

        if not analysis.needs_retrieval:
            return []

        candidate_k = max(candidate_k, top_k)

        faiss_results = self._search_faiss(
            query=analysis.semantic_query,
            k=candidate_k,
        )
        print("FAISS_COUNT:", len(faiss_results))

        bm25_results = self._search_bm25(
            query=analysis.bm25_query,
            k=candidate_k,
        )
        print("BM25_COUNT:", len(bm25_results))

        merged_results: Dict[str, Dict[str, Any]] = {}

        self._merge_faiss_results(
            merged_results=merged_results,
            faiss_results=faiss_results,
        )
        print("MERGED_Faiss_COUNT:", len(merged_results))

        self._merge_bm25_results(
            merged_results=merged_results,
            bm25_results=bm25_results,
        )
        print("MERGED_bm25_COUNT:", len(merged_results))

        ranked_results = self._rank_merged_results(merged_results)
        print("RANKED_COUNT:", len(ranked_results))

        working_results = ranked_results[:candidate_k]

        if self.reranker is not None:
            try:
                working_results = self.reranker.rerank(
                    query=query,
                    results=working_results,
                    top_k=candidate_k,
                )
            except Exception:
                working_results = ranked_results[:candidate_k]

        if self.post_processor is not None:
            try:
                working_results = self.post_processor.process(
                    working_results,
                    top_k=top_k,
                )
            except Exception:
                working_results = working_results[:top_k]
        else:
            working_results = working_results[:top_k]
        print("FINAL_COUNT:", len(working_results))
        return working_results[:top_k]
        
    

    def _search_faiss(self, query: str, k: int) -> List[Tuple[Document, float]]:
        return self.faiss_index.similarity_search_with_score(query, k=k)

    def _search_bm25(self, query: str, k: int) -> List[Tuple[Document, float]]:
        return self.bm25_store.search(query=query, top_k=k)

    def _merge_faiss_results(
        self,
        merged_results: Dict[str, Dict[str, Any]],
        faiss_results: List[Tuple[Document, float]],
    ) -> None:
        for document, raw_score in faiss_results:
            key = self._document_key(document)
            normalized_score = self._normalize_faiss_score(raw_score)

            if key not in merged_results:
                merged_results[key] = {
                    "document": document,
                    "faiss_score": 0.0,
                    "bm25_score": 0.0,
                    "sources": set(),
                }

            merged_results[key]["faiss_score"] = max(
                merged_results[key]["faiss_score"],
                normalized_score,
            )
            merged_results[key]["sources"].add("faiss")

    def _merge_bm25_results(
        self,
        merged_results: Dict[str, Dict[str, Any]],
        bm25_results: List[Tuple[Document, float]],
    ) -> None:
        max_bm25_score = max((score for _, score in bm25_results), default=0.0)

        for document, raw_score in bm25_results:
            key = self._document_key(document)
            normalized_score = self._normalize_bm25_score(
                score=raw_score,
                max_score=max_bm25_score,
            )

            if key not in merged_results:
                merged_results[key] = {
                    "document": document,
                    "faiss_score": 0.0,
                    "bm25_score": 0.0,
                    "sources": set(),
                }

            merged_results[key]["bm25_score"] = max(
                merged_results[key]["bm25_score"],
                normalized_score,
            )
            merged_results[key]["sources"].add("bm25")

    def _rank_merged_results(
        self,
        merged_results: Dict[str, Dict[str, Any]],
    ) -> List[HybridSearchResult]:
        ranked_results: List[HybridSearchResult] = []

        for item in merged_results.values():
            document = item["document"]
            faiss_score = float(item["faiss_score"])
            bm25_score = float(item["bm25_score"])

            final_score = (
                self.faiss_weight * faiss_score
                + self.bm25_weight * bm25_score
            )

            final_score = self._apply_penalties(
                document=document,
                score=final_score,
            )

            source = "+".join(sorted(item["sources"]))

            final_score = self._apply_source_penalty(
                score=final_score,
                sources=item["sources"],
            )

            if final_score < self.min_score_threshold:
                continue

            ranked_results.append(
                HybridSearchResult(
                    document=document,
                    score=final_score,
                    faiss_score=faiss_score,
                    bm25_score=bm25_score,
                    source=source,
                )
            )

        ranked_results.sort(key=lambda result: result.score, reverse=True)
        return ranked_results

    def _normalize_faiss_score(self, score: float) -> float:
        distance = max(float(score), 0.0)
        normalized = 1.0 / (1.0 + distance)
        return self._clamp_01(normalized)

    def _normalize_bm25_score(self, score: float, max_score: float) -> float:
        score = float(score)
        max_score = float(max_score)

        if max_score <= 0:
            return 0.0

        normalized = score / max_score
        return self._clamp_01(normalized)

    def _apply_penalties(self, document: Document, score: float) -> float:
        penalized_score = score

        if self._is_low_value_section(document):
            penalized_score *= self.low_value_section_penalty

        if self._is_short_chunk(document):
            penalized_score *= self.short_chunk_penalty

        return self._clamp_01(penalized_score)

    def _is_low_value_section(self, document: Document) -> bool:
        metadata = document.metadata or {}

        section_title = str(
            metadata.get("section_title")
            or metadata.get("title")
            or metadata.get("heading")
            or ""
        ).lower()

        page_content_prefix = (document.page_content or "")[:300].lower()
        searchable_text = f"{section_title}\n{page_content_prefix}"

        return any(
            keyword in searchable_text
            for keyword in self.LOW_VALUE_SECTION_KEYWORDS
        )

    def _is_short_chunk(self, document: Document) -> bool:
        return len((document.page_content or "").strip()) < self.short_chunk_threshold

    def _apply_source_penalty(self, score: float, sources: set[str]) -> float:
        if sources == {"faiss"}:
            score *= 0.75

        return self._clamp_01(score)

    def _document_key(self, document: Document) -> str:
        metadata = document.metadata or {}

        source = metadata.get("source") or metadata.get("file_path") or ""
        chunk_index = metadata.get("chunk_index")

        if chunk_index is not None:
            return f"{source}::{chunk_index}"

        stable_text_hash = hashlib.md5(
            (document.page_content or "").encode("utf-8")
        ).hexdigest()

        return f"{source}::{stable_text_hash}"

    @staticmethod
    def _clamp_01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))
