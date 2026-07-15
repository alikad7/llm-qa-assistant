# FILE: run_generation_evaluation.py

from __future__ import annotations

from src.config import settings
from src.evaluation.runners.generation_runner import GenerationEvaluationRunner
from src.generation.generator import AnswerGenerator
from src.retrieval.hybrid_retriever import HybridRetriever


def main() -> None:
    retriever = HybridRetriever(
        faiss_index_path=settings.faiss_index_path,
        bm25_index_dir=settings.bm25_index_dir,
    )

    generator = AnswerGenerator(
        api_key=settings.gapgpt_api_key,
        base_url=settings.gapgpt_base_url,
        model=settings.chat_model,
        temperature=settings.generation_temperature,
        max_tokens=settings.generation_max_tokens,
        timeout_seconds=settings.generation_timeout_seconds,
    )

    runner = GenerationEvaluationRunner(
        retriever=retriever,
        generator=generator,
        dataset_path="src/evaluation/datasets/evaluation_dataset.json",
        report_output_dir="src/evaluation/reports",
        max_context_chars=settings.context_max_chars,
    )

    report = runner.run_evaluation(
        answer_language=settings.default_answer_language
    )

    print("Generation evaluation completed successfully.")
    print("Summary:")
    print(report["summary"])


if __name__ == "__main__":
    main()
