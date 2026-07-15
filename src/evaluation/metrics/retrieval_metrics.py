from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence
from urllib.parse import urlsplit, urlunsplit


@dataclass(frozen=True)
class RetrievalMetricsResult:
    hit_rate_at_k: float
    precision_at_k: float
    recall_at_k: float
    mrr_at_k: float
    ndcg_at_k: float
    total_questions: int
    questions_with_hits: int


def normalize_source(source: str) -> str:
    normalized_source = (source or "").strip()

    if not normalized_source:
        return ""

    try:
        parts = urlsplit(normalized_source)

        if parts.scheme and parts.netloc:
            normalized_path = parts.path.rstrip("/") or "/"

            return urlunsplit(
                (
                    parts.scheme.lower(),
                    parts.netloc.lower(),
                    normalized_path,
                    parts.query,
                    "",
                )
            ).rstrip("/")

    except ValueError:
        pass

    return normalized_source.rstrip("/").lower()


def _normalized_unique_sources(
    sources: Sequence[str],
) -> list[str]:
    normalized_sources: list[str] = []
    seen_sources: set[str] = set()

    for source in sources:
        normalized_source = normalize_source(source)

        if not normalized_source:
            continue

        if normalized_source in seen_sources:
            continue

        seen_sources.add(normalized_source)
        normalized_sources.append(normalized_source)

    return normalized_sources


def hit_rate_at_k(
    retrieved_sources: Sequence[str],
    expected_sources: Sequence[str],
) -> float:
    retrieved_set = set(
        _normalized_unique_sources(retrieved_sources)
    )
    expected_set = set(
        _normalized_unique_sources(expected_sources)
    )

    if not expected_set:
        return 0.0

    return (
        1.0
        if retrieved_set.intersection(expected_set)
        else 0.0
    )


def precision_at_k(
    retrieved_sources: Sequence[str],
    expected_sources: Sequence[str],
) -> float:
    retrieved = _normalized_unique_sources(
        retrieved_sources
    )
    expected_set = set(
        _normalized_unique_sources(expected_sources)
    )

    if not retrieved or not expected_set:
        return 0.0

    relevant_count = sum(
        1
        for source in retrieved
        if source in expected_set
    )

    return relevant_count / len(retrieved)


def recall_at_k(
    retrieved_sources: Sequence[str],
    expected_sources: Sequence[str],
) -> float:
    retrieved_set = set(
        _normalized_unique_sources(retrieved_sources)
    )
    expected_set = set(
        _normalized_unique_sources(expected_sources)
    )

    if not expected_set:
        return 0.0

    matched_sources = retrieved_set.intersection(
        expected_set
    )

    return len(matched_sources) / len(expected_set)


def mrr_at_k(
    retrieved_sources: Sequence[str],
    expected_sources: Sequence[str],
) -> float:
    expected_set = set(
        _normalized_unique_sources(expected_sources)
    )

    if not expected_set:
        return 0.0

    seen_sources: set[str] = set()
    unique_rank = 0

    for source in retrieved_sources:
        normalized_source = normalize_source(source)

        if (
            not normalized_source
            or normalized_source in seen_sources
        ):
            continue

        seen_sources.add(normalized_source)
        unique_rank += 1

        if normalized_source in expected_set:
            return 1.0 / unique_rank

    return 0.0


def ndcg_at_k(
    retrieved_sources: Sequence[str],
    expected_sources: Sequence[str],
) -> float:
    retrieved = _normalized_unique_sources(
        retrieved_sources
    )
    expected_set = set(
        _normalized_unique_sources(expected_sources)
    )

    if not retrieved or not expected_set:
        return 0.0

    dcg = 0.0

    for rank, source in enumerate(
        retrieved,
        start=1,
    ):
        relevance = (
            1.0 if source in expected_set else 0.0
        )

        dcg += relevance / math.log2(rank + 1)

    ideal_relevant_count = min(
        len(expected_set),
        len(retrieved),
    )

    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(
            1,
            ideal_relevant_count + 1,
        )
    )

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def aggregate_retrieval_metrics(
    evaluation_rows: Iterable[dict],
) -> RetrievalMetricsResult:
    rows = list(evaluation_rows)

    if not rows:
        return RetrievalMetricsResult(
            hit_rate_at_k=0.0,
            precision_at_k=0.0,
            recall_at_k=0.0,
            mrr_at_k=0.0,
            ndcg_at_k=0.0,
            total_questions=0,
            questions_with_hits=0,
        )

    hit_scores: list[float] = []
    precision_scores: list[float] = []
    recall_scores: list[float] = []
    mrr_scores: list[float] = []
    ndcg_scores: list[float] = []

    for row in rows:
        retrieved_sources = row.get(
            "retrieved_sources",
            [],
        )
        expected_sources = row.get(
            "expected_sources",
            [],
        )

        hit_scores.append(
            hit_rate_at_k(
                retrieved_sources,
                expected_sources,
            )
        )
        precision_scores.append(
            precision_at_k(
                retrieved_sources,
                expected_sources,
            )
        )
        recall_scores.append(
            recall_at_k(
                retrieved_sources,
                expected_sources,
            )
        )
        mrr_scores.append(
            mrr_at_k(
                retrieved_sources,
                expected_sources,
            )
        )
        ndcg_scores.append(
            ndcg_at_k(
                retrieved_sources,
                expected_sources,
            )
        )

    total_questions = len(rows)
    questions_with_hits = sum(
        1
        for score in hit_scores
        if score > 0.0
    )

    return RetrievalMetricsResult(
        hit_rate_at_k=(
            sum(hit_scores) / total_questions
        ),
        precision_at_k=(
            sum(precision_scores) / total_questions
        ),
        recall_at_k=(
            sum(recall_scores) / total_questions
        ),
        mrr_at_k=(
            sum(mrr_scores) / total_questions
        ),
        ndcg_at_k=(
            sum(ndcg_scores) / total_questions
        ),
        total_questions=total_questions,
        questions_with_hits=questions_with_hits,
    )
