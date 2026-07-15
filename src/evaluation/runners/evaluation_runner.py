from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.config import settings
from src.evaluation.metrics.retrieval_metrics import (
    aggregate_retrieval_metrics,
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.post_processing import MMRPostProcessor
from src.retrieval.reranker import CrossEncoderReranker


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "datasets"
    / "evaluation_dataset.json"
)
REPORTS_DIR = (
    PROJECT_ROOT
    / "evaluation"
    / "reports"
)


def build_retriever() -> HybridRetriever:
    reranker = CrossEncoderReranker(
        model_name=settings.local_reranker_model,
        batch_size=settings.reranker_batch_size,
        device=settings.reranker_device,
        normalize_scores=(
            settings.reranker_normalize_scores
        ),
    )

    post_processor = MMRPostProcessor(
        lambda_param=settings.mmr_lambda_param,
        min_score_threshold=(
            settings.mmr_min_score_threshold
        ),
        max_context_chars=(
            settings.mmr_max_context_chars
        ),
        deduplicate_by_content=(
            settings.mmr_deduplicate_by_content
        ),
    )

    return HybridRetriever(
        faiss_index_path=settings.faiss_index_path,
        bm25_index_dir=settings.bm25_index_dir,
        reranker=reranker,
        post_processor=post_processor,
    )


def load_dataset(
    dataset_path: Path,
) -> list[dict]:
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {dataset_path}"
        )

    with dataset_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if isinstance(data, dict):
        data = data.get("questions", data)

    if not isinstance(data, list):
        raise ValueError(
            "Dataset must be a JSON array"
        )

    return data


def extract_source(result) -> str:
    metadata = result.document.metadata or {}

    return str(
        metadata.get("source")
        or metadata.get("url")
        or metadata.get("file_path")
        or metadata.get("filename")
        or "unknown"
    )


def run_retrieval_evaluation() -> Path:
    dataset = load_dataset(DATASET_PATH)
    retriever = build_retriever()

    evaluation_rows: list[dict] = []

    for item in dataset:
        question_id = item["id"]
        question = item["question"]
        expected_sources = item.get(
            "expected_sources",
            [],
        )

        try:
            results = retriever.search(
                query=question,
                top_k=settings.retrieval_top_k,
                candidate_k=(
                    settings.retrieval_candidate_k
                ),
            )

            retrieved_sources = [
                extract_source(result)
                for result in results
            ]

            row_hit_rate = hit_rate_at_k(
                retrieved_sources,
                expected_sources,
            )
            row_precision = precision_at_k(
                retrieved_sources,
                expected_sources,
            )
            row_recall = recall_at_k(
                retrieved_sources,
                expected_sources,
            )
            row_mrr = mrr_at_k(
                retrieved_sources,
                expected_sources,
            )
            row_ndcg = ndcg_at_k(
                retrieved_sources,
                expected_sources,
            )

            evaluation_rows.append(
                {
                    "id": question_id,
                    "question": question,
                    "language": item.get(
                        "language",
                        "en",
                    ),
                    "expected_sources": (
                        expected_sources
                    ),
                    "retrieved_sources": (
                        retrieved_sources
                    ),
                    "retrieved_count": len(
                        retrieved_sources
                    ),
                    "metrics": {
                        "hit_rate_at_k": round(
                            row_hit_rate,
                            4,
                        ),
                        "precision_at_k": round(
                            row_precision,
                            4,
                        ),
                        "recall_at_k": round(
                            row_recall,
                            4,
                        ),
                        "mrr_at_k": round(
                            row_mrr,
                            4,
                        ),
                        "ndcg_at_k": round(
                            row_ndcg,
                            4,
                        ),
                        "source_level_context_precision": (
                            round(
                                row_precision,
                                4,
                            )
                        ),
                        "source_level_context_recall": (
                            round(
                                row_recall,
                                4,
                            )
                        ),
                    },
                    "status": "ok",
                }
            )

        except Exception as exc:
            evaluation_rows.append(
                {
                    "id": question_id,
                    "question": question,
                    "language": item.get(
                        "language",
                        "en",
                    ),
                    "expected_sources": (
                        expected_sources
                    ),
                    "retrieved_sources": [],
                    "retrieved_count": 0,
                    "metrics": {
                        "hit_rate_at_k": 0.0,
                        "precision_at_k": 0.0,
                        "recall_at_k": 0.0,
                        "mrr_at_k": 0.0,
                        "ndcg_at_k": 0.0,
                        "source_level_context_precision": (
                            0.0
                        ),
                        "source_level_context_recall": (
                            0.0
                        ),
                    },
                    "status": "error",
                    "error": str(exc),
                }
            )

    successful_rows = [
        row
        for row in evaluation_rows
        if row.get("status") == "ok"
    ]

    metrics = aggregate_retrieval_metrics(
        successful_rows
    )

    report = {
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "dataset_path": str(DATASET_PATH),
        "retrieval_top_k": (
            settings.retrieval_top_k
        ),
        "retrieval_candidate_k": (
            settings.retrieval_candidate_k
        ),
        "summary": {
            "total_dataset_questions": len(
                dataset
            ),
            "successful_questions": len(
                successful_rows
            ),
            "failed_questions": (
                len(evaluation_rows)
                - len(successful_rows)
            ),
            "hit_rate_at_k": round(
                metrics.hit_rate_at_k,
                4,
            ),
            "precision_at_k": round(
                metrics.precision_at_k,
                4,
            ),
            "recall_at_k": round(
                metrics.recall_at_k,
                4,
            ),
            "mrr_at_k": round(
                metrics.mrr_at_k,
                4,
            ),
            "ndcg_at_k": round(
                metrics.ndcg_at_k,
                4,
            ),
            "source_level_context_precision": (
                round(
                    metrics.precision_at_k,
                    4,
                )
            ),
            "source_level_context_recall": (
                round(
                    metrics.recall_at_k,
                    4,
                )
            ),
            "questions_with_hits": (
                metrics.questions_with_hits
            ),
        },
        "results": evaluation_rows,
    }

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    report_path = (
        REPORTS_DIR
        / f"retrieval_report_{timestamp}.json"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return report_path


if __name__ == "__main__":
    report_file = run_retrieval_evaluation()

    print(
        "Evaluation report saved to: "
        f"{report_file}"
    )
