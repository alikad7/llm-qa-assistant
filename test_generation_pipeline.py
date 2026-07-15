# FILE: test_generation_pipeline.py

from __future__ import annotations

from langchain_core.documents import Document

from src.generation.context_builder import ContextBuilder
from src.generation.generator import AnswerGenerator
from src.generation.prompts import build_rag_messages
from src.retrieval.hybrid_retriever import HybridSearchResult


def make_result(
    content: str,
    score: float,
    source: str,
) -> HybridSearchResult:
    return HybridSearchResult(
        document=Document(
            page_content=content,
            metadata={
                "source": source,
                "section_title": "Regression Testing",
            },
        ),
        score=score,
        faiss_score=score,
        bm25_score=score,
        source="faiss+bm25",
    )


def test_context_builder_builds_sources() -> None:
    builder = ContextBuilder(
        max_context_chars=1000,
        include_scores=True,
    )

    results = [
        make_result(
            content=(
                "Regression testing verifies that existing functionality "
                "still works after changes."
            ),
            score=0.91,
            source="software-testing-notes.md",
        )
    ]

    built_context = builder.build(results)

    assert "[S1]" in built_context.context_text
    assert "software-testing-notes.md" in built_context.context_text
    assert "Regression testing verifies" in built_context.context_text
    assert len(built_context.sources) == 1
    assert built_context.sources[0].source_id == "S1"

    print("PASS: ContextBuilder created context with sources.")


def test_prompt_supports_persian_language() -> None:
    messages = build_rag_messages(
        question="Regression testing چیست؟",
        context="[S1] Regression testing verifies existing behavior.",
        answer_language="fa",
    )

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "Persian/Farsi" in messages[0]["content"]

    print("PASS: Prompt supports Persian answers.")


def test_prompt_supports_english_language() -> None:
    messages = build_rag_messages(
        question="What is regression testing?",
        context="[S1] Regression testing verifies existing behavior.",
        answer_language="en",
    )

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "English" in messages[0]["content"]

    print("PASS: Prompt supports English answers.")


def test_generator_empty_context_fallback_fa() -> None:
    generator = AnswerGenerator(
        api_key="test-key",
        base_url="https://example.com/v1",
        model="test-model",
    )

    builder = ContextBuilder()
    built_context = builder.build([])

    answer = generator.generate_answer(
        question="Regression testing چیست؟",
        built_context=built_context,
        answer_language="fa",
    )

    assert "زمینه کافی" in answer

    print("PASS: Generator returns Persian fallback for empty context.")


def test_generator_empty_context_fallback_en() -> None:
    generator = AnswerGenerator(
        api_key="test-key",
        base_url="https://example.com/v1",
        model="test-model",
    )

    builder = ContextBuilder()
    built_context = builder.build([])

    answer = generator.generate_answer(
        question="What is regression testing?",
        built_context=built_context,
        answer_language="en",
    )

    assert "not sufficient" in answer

    print("PASS: Generator returns English fallback for empty context.")


def main() -> None:
    test_context_builder_builds_sources()
    test_prompt_supports_persian_language()
    test_prompt_supports_english_language()
    test_generator_empty_context_fallback_fa()
    test_generator_empty_context_fallback_en()

    print("\nAll Step 7 generation tests passed.")


if __name__ == "__main__":
    main()
