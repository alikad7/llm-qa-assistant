from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Protocol, Sequence, TypeVar

from sentence_transformers import CrossEncoder


class DocumentLike(Protocol):
    page_content: str


class RerankableResult(Protocol):
    document: DocumentLike
    score: float


ResultT = TypeVar("ResultT", bound=RerankableResult)


class CrossEncoderReranker:
    """
    Rerank retrieval results using a CrossEncoder model (local path or Hugging Face ID).
    """

    def __init__(
        self,
        model_name: str | Path,
        *,
        batch_size: int = 16,
        device: str | None = None,
        normalize_scores: bool = True,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        self.model_name = str(model_name)
        self.batch_size = batch_size
        self.device = device
        self.normalize_scores = normalize_scores

        print(f"Loading CrossEncoder model: {self.model_name}...")
        self.model = CrossEncoder(
            self.model_name,
            device=self.device,
        )
        print("✅ CrossEncoder model loaded successfully.")

    def rerank(
        self,
        query: str,
        results: Sequence[ResultT],
        *,
        top_k: int = 5,
    ) -> list[ResultT]:
        query = (query or "").strip()

        if not query:
            raise ValueError("query must not be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        if not results:
            return []

        pairs: list[tuple[str, str]] = []

        for index, result in enumerate(results):
            page_content = (
                getattr(result.document, "page_content", "") or ""
            ).strip()

            if not page_content:
                raise ValueError(
                    "Cannot rerank a result with empty page_content. "
                    f"Result index: {index}"
                )

            pairs.append((query, page_content))

        predicted_scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        scores = [float(score) for score in predicted_scores]

        if len(scores) != len(results):
            raise RuntimeError(
                "CrossEncoder returned a different number of scores "
                f"than input results: scores={len(scores)}, "
                f"results={len(results)}"
            )

        if self.normalize_scores:
            scores = [self._sigmoid(score) for score in scores]

        reranked_results = [
            replace(result, score=score)
            for result, score in zip(results, scores)
        ]

        reranked_results.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        return reranked_results[:top_k]

    @staticmethod
    def _sigmoid(value: float) -> float:
        import math

        if value >= 0:
            z = math.exp(-value)
            return 1.0 / (1.0 + z)

        z = math.exp(value)
        return z / (1.0 + z)
