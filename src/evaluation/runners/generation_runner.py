from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.evaluation.metrics.generation_metrics import (
    GenerationEvaluator,
)
from src.evaluation.metrics.retrieval_metrics import (
    precision_at_k,
    recall_at_k,
)
from src.generation.context_builder import ContextBuilder
from src.generation.generator import AnswerGenerator


class GenerationEvaluationRunner:
    def __init__(
        self,
        *,
        retriever: Any,
        generator: AnswerGenerator,
        dataset_path: str | Path,
        report_output_dir: str | Path = (
            "src/evaluation/reports"
        ),
        max_context_chars: int = 6000,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.dataset_path = Path(dataset_path)
        self.report_output_dir = Path(
            report_output_dir
        )
        self.context_builder = ContextBuilder(
            max_context_chars=max_context_chars
        )
        self.evaluator = GenerationEvaluator()

        self.report_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    @staticmethod
    def _extract_document_source(
        source: Any,
    ) -> str:
        metadata = source.metadata or {}

        return str(
            metadata.get("source")
            or metadata.get("url")
            or metadata.get("file_path")
            or metadata.get("filename")
            or "unknown"
        )

    def run_evaluation(
        self,
        answer_language: str = "en",
    ) -> dict[str, Any]:
        if not self.dataset_path.exists():
            raise FileNotFoundError(
                "Dataset not found at: "
                f"{self.dataset_path}"
            )

        with self.dataset_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            dataset = json.load(file)

        questions = (
            dataset.get("questions", dataset)
            if isinstance(dataset, dict)
            else dataset
        )

        results: list[dict[str, Any]] = []

        summary = {
            "total_evaluated": 0,
            "successful_runs": 0,
            "failed_runs": 0,
            "avg_citation_precision": 0.0,
            "avg_citation_coverage": 0.0,
            "avg_citation_recall": 0.0,
            "avg_faithfulness": 0.0,
            "avg_source_level_context_precision": (
                0.0
            ),
            "avg_source_level_context_recall": 0.0,
            "avg_confidence_score": 0.0,
            "total_invalid_citations": 0,
            "total_hallucinations": 0,
            "total_insufficient_responses": 0,
        }

        totals = {
            "citation_precision": 0.0,
            "citation_coverage": 0.0,
            "faithfulness": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "confidence": 0.0,
        }

        invalid_citation_count = 0
        insufficient_count = 0

        for index, item in enumerate(
            questions,
            start=1,
        ):
            query = (
                item.get("question")
                or item.get("query")
                or ""
            ).strip()

            if not query:
                continue

            summary["total_evaluated"] += 1

            expected_sources = item.get(
                "expected_sources",
                [],
            )
            item_language = item.get(
                "language",
                answer_language,
            )

            record: dict[str, Any] = {
                "id": item.get("id", index),
                "question": query,
                "language": item_language,
                "expected_sources": expected_sources,
                "reference_answer": item.get(
                    "reference_answer",
                    item.get(
                        "expected_answer",
                        "",
                    ),
                ),
            }

            try:
                retrieved_results = (
                    self.retriever.search(query)
                )

                built_context = (
                    self.context_builder.build(
                        retrieved_results
                    )
                )

                citation_source_ids = [
                    source.source_id
                    for source in built_context.sources
                ]

                retrieved_document_sources = [
                    self._extract_document_source(
                        source
                    )
                    for source in built_context.sources
                ]

                record["retrieved_sources"] = (
                    retrieved_document_sources
                )
                record["citation_source_ids"] = (
                    citation_source_ids
                )
                record[
                    "retrieved_source_details"
                ] = [
                    {
                        "source_id": (
                            source.source_id
                        ),
                        "document_source": (
                            self._extract_document_source(
                                source
                            )
                        ),
                        "score": source.score,
                        "metadata": source.metadata,
                    }
                    for source in built_context.sources
                ]
                record["context_length"] = (
                    built_context.total_chars
                )
                record["context_text"] = (
                    built_context.context_text
                )

                generated_answer = (
                    self.generator.generate_answer(
                        question=query,
                        built_context=built_context,
                        answer_language=item_language,
                    )
                )

                record["generated_answer"] = (
                    generated_answer
                )

                generation_metrics = (
                    self.evaluator.evaluate(
                        answer=generated_answer,
                        retrieved_sources=(
                            citation_source_ids
                        ),
                        context_text=(
                            built_context.context_text
                        ),
                    )
                )

                context_precision = precision_at_k(
                    retrieved_document_sources,
                    expected_sources,
                )
                context_recall = recall_at_k(
                    retrieved_document_sources,
                    expected_sources,
                )

                record["metrics"] = {
                    "citation_precision": (
                        generation_metrics
                        .citation_precision
                    ),
                    "citation_coverage": (
                        generation_metrics
                        .citation_coverage
                    ),
                    "citation_recall": (
                        generation_metrics
                        .citation_coverage
                    ),
                    "faithfulness_score": (
                        generation_metrics
                        .faithfulness_score
                    ),
                    "source_level_context_precision": (
                        context_precision
                    ),
                    "source_level_context_recall": (
                        context_recall
                    ),
                    "confidence_score": (
                        generation_metrics
                        .confidence_score
                    ),
                    "invalid_citation_flag": (
                        generation_metrics
                        .invalid_citation_flag
                    ),
                    "hallucination_flag": (
                        generation_metrics
                        .invalid_citation_flag
                    ),
                    "is_insufficient": (
                        generation_metrics
                        .is_insufficient
                    ),
                    "response_length": (
                        generation_metrics
                        .response_length
                    ),
                    "extracted_citations": (
                        generation_metrics
                        .extracted_citations
                    ),
                }

                record["status"] = "success"

                totals["citation_precision"] += (
                    generation_metrics
                    .citation_precision
                )
                totals["citation_coverage"] += (
                    generation_metrics
                    .citation_coverage
                )
                totals["faithfulness"] += (
                    generation_metrics
                    .faithfulness_score
                )
                totals["context_precision"] += (
                    context_precision
                )
                totals["context_recall"] += (
                    context_recall
                )
                totals["confidence"] += (
                    generation_metrics
                    .confidence_score
                )

                if (
                    generation_metrics
                    .invalid_citation_flag
                ):
                    invalid_citation_count += 1

                if generation_metrics.is_insufficient:
                    insufficient_count += 1

                summary["successful_runs"] += 1

            except Exception as exc:
                record["status"] = "failed"
                record["error"] = str(exc)
                summary["failed_runs"] += 1

            results.append(record)

        success_count = summary["successful_runs"]

        if success_count > 0:
            summary[
                "avg_citation_precision"
            ] = round(
                totals["citation_precision"]
                / success_count,
                4,
            )
            summary[
                "avg_citation_coverage"
            ] = round(
                totals["citation_coverage"]
                / success_count,
                4,
            )
            summary[
                "avg_citation_recall"
            ] = summary[
                "avg_citation_coverage"
            ]
            summary["avg_faithfulness"] = round(
                totals["faithfulness"]
                / success_count,
                4,
            )
            summary[
                "avg_source_level_context_precision"
            ] = round(
                totals["context_precision"]
                / success_count,
                4,
            )
            summary[
                "avg_source_level_context_recall"
            ] = round(
                totals["context_recall"]
                / success_count,
                4,
            )
            summary[
                "avg_confidence_score"
            ] = round(
                totals["confidence"]
                / success_count,
                4,
            )

        summary[
            "total_invalid_citations"
        ] = invalid_citation_count

        summary[
            "total_hallucinations"
        ] = invalid_citation_count

        summary[
            "total_insufficient_responses"
        ] = insufficient_count

        report = {
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
            "summary": summary,
            "details": results,
        }

        report_filename = (
            "generation_eval_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            ".json"
        )

        report_path = (
            self.report_output_dir
            / report_filename
        )

        with report_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                report,
                file,
                indent=2,
                ensure_ascii=False,
            )

        return report
