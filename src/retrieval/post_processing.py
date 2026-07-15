from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Protocol, Sequence, TypeVar


class DocumentLike(Protocol):
    page_content: str
    metadata: dict


class SearchResultLike(Protocol):
    document: DocumentLike
    score: float


ResultT = TypeVar("ResultT", bound=SearchResultLike)


class MMRPostProcessor:
    def __init__(
        self,
        *,
        lambda_param: float = 0.7,
        min_score_threshold: float = 0.0,
        max_context_chars: int | None = None,
        deduplicate_by_content: bool = True,
    ) -> None:
        if not 0.0 <= lambda_param <= 1.0:
            raise ValueError("lambda_param must be between 0 and 1")

        self.lambda_param = lambda_param
        self.min_score_threshold = float(min_score_threshold)
        self.max_context_chars = max_context_chars
        self.deduplicate_by_content = deduplicate_by_content

    def process(
        self,
        results: Sequence[ResultT],
        *,
        top_k: int = 5,
    ) -> list[ResultT]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        if not results:
            return []

        filtered_results = self._filter_by_score(results)
        deduplicated_results = self._deduplicate(filtered_results)

        if not deduplicated_results:
            return []

        mmr_selected = self._apply_mmr(deduplicated_results, top_k=top_k)
        trimmed_results = self._apply_context_budget(mmr_selected)

        return trimmed_results[:top_k]

    def _filter_by_score(
        self,
        results: Sequence[ResultT],
    ) -> list[ResultT]:
        return [
            result
            for result in results
            if float(result.score) >= self.min_score_threshold
        ]

    def _deduplicate(
        self,
        results: Sequence[ResultT],
    ) -> list[ResultT]:
        if not self.deduplicate_by_content:
            return list(results)

        unique_results: list[ResultT] = []
        seen_keys: set[str] = set()

        for result in results:
            key = self._result_key(result)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            unique_results.append(result)

        return unique_results

    def _apply_mmr(
        self,
        results: Sequence[ResultT],
        *,
        top_k: int,
    ) -> list[ResultT]:
        if len(results) <= top_k:
            return list(results)

        selected: list[ResultT] = []
        remaining = list(results)

        first = max(remaining, key=lambda item: float(item.score))
        selected.append(first)
        remaining.remove(first)

        while remaining and len(selected) < top_k:
            best_candidate = None
            best_mmr_score = float("-inf")

            for candidate in remaining:
                relevance = float(candidate.score)
                diversity_penalty = self._max_similarity_to_selected(
                    candidate=candidate,
                    selected=selected,
                )

                mmr_score = (
                    self.lambda_param * relevance
                    - (1.0 - self.lambda_param) * diversity_penalty
                )

                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_candidate = candidate

            if best_candidate is None:
                break

            selected.append(best_candidate)
            remaining.remove(best_candidate)

        return selected

    def _max_similarity_to_selected(
        self,
        *,
        candidate: ResultT,
        selected: Sequence[ResultT],
    ) -> float:
        if not selected:
            return 0.0

        candidate_tokens = self._tokenize(candidate.document.page_content)

        if not candidate_tokens:
            return 0.0

        similarities = []

        for item in selected:
            selected_tokens = self._tokenize(item.document.page_content)
            similarity = self._jaccard_similarity(candidate_tokens, selected_tokens)
            similarities.append(similarity)

        return max(similarities, default=0.0)

    def _apply_context_budget(
        self,
        results: Sequence[ResultT],
    ) -> list[ResultT]:
        if self.max_context_chars is None:
            return list(results)

        budgeted_results: list[ResultT] = []
        total_chars = 0

        for result in results:
            content = (result.document.page_content or "").strip()
            content_length = len(content)

            if not budgeted_results and content_length > self.max_context_chars:
                truncated_content = content[: self.max_context_chars].strip()
                result.document.page_content = truncated_content
                budgeted_results.append(result)
                break

            if total_chars + content_length > self.max_context_chars:
                break

            budgeted_results.append(result)
            total_chars += content_length

        return budgeted_results

    def _result_key(self, result: ResultT) -> str:
        metadata = getattr(result.document, "metadata", {}) or {}

        source = metadata.get("source") or metadata.get("file_path") or ""
        chunk_index = metadata.get("chunk_index")

        if chunk_index is not None:
            return f"{source}::{chunk_index}"

        content = (result.document.page_content or "").strip()
        content_hash = hashlib.md5(content.encode("utf-8")).hexdigest()
        return f"{source}::{content_hash}"

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        text = (text or "").lower().strip()
        if not text:
            return set()

        return {token for token in text.split() if token}

    @staticmethod
    def _jaccard_similarity(tokens_a: set[str], tokens_b: set[str]) -> float:
        if not tokens_a or not tokens_b:
            return 0.0

        intersection = len(tokens_a & tokens_b)
        union = len(tokens_a | tokens_b)

        if union == 0:
            return 0.0

        return intersection / union
