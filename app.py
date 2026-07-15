from __future__ import annotations

from typing import Any

import streamlit as st

from src.config import settings
from src.generation.context_builder import ContextBuilder
from src.generation.generator import AnswerGenerator
from src.retrieval.hybrid_retriever import (
    HybridRetriever,
    HybridSearchResult,
)
from src.retrieval.post_processing import MMRPostProcessor
from src.retrieval.reranker import CrossEncoderReranker
from src.evaluation.metrics.generation_metrics import (
    GenerationEvaluator,
)


st.set_page_config(
    page_title="Software Testing Knowledge Assistant",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)


INSUFFICIENT_CONTEXT_MESSAGE = (
    "The retrieved context is not sufficient to answer this "
    "question accurately."
)


@st.cache_resource(show_spinner=False)
def get_retriever() -> HybridRetriever:
    reranker = CrossEncoderReranker(
        model_name=settings.local_reranker_model,
        batch_size=settings.reranker_batch_size,
        device=settings.reranker_device,
        normalize_scores=settings.reranker_normalize_scores,
    )

    post_processor = MMRPostProcessor(
        lambda_param=settings.mmr_lambda_param,
        min_score_threshold=settings.mmr_min_score_threshold,
        max_context_chars=settings.mmr_max_context_chars,
        deduplicate_by_content=settings.mmr_deduplicate_by_content,
    )

    return HybridRetriever(
        faiss_index_path=settings.faiss_index_path,
        bm25_index_dir=settings.bm25_index_dir,
        reranker=reranker,
        post_processor=post_processor,
    )


@st.cache_resource(show_spinner=False)
def get_context_builder() -> ContextBuilder:
    return ContextBuilder(
        max_context_chars=settings.context_max_chars,
        include_scores=settings.context_include_scores,
    )


@st.cache_resource(show_spinner=False)
def get_answer_generator() -> AnswerGenerator:
    return AnswerGenerator(
        api_key=settings.gapgpt_api_key,
        base_url=settings.gapgpt_base_url,
        model=settings.chat_model,
        temperature=settings.generation_temperature,
        max_tokens=settings.generation_max_tokens,
        timeout_seconds=settings.generation_timeout_seconds,
    )


@st.cache_resource(show_spinner=False)
def get_generation_evaluator() -> GenerationEvaluator:
    return GenerationEvaluator()


def get_document_source(result: HybridSearchResult) -> str:
    metadata = result.document.metadata or {}

    return str(
        metadata.get("source")
        or metadata.get("file_path")
        or metadata.get("filename")
        or "unknown"
    )


def get_section_title(result: HybridSearchResult) -> str | None:
    metadata = result.document.metadata or {}

    section_title = (
        metadata.get("section_title")
        or metadata.get("title")
        or metadata.get("heading")
    )

    if section_title is None:
        return None

    return str(section_title)


def format_score(score: Any) -> str:
    if score is None:
        return "N/A"

    try:
        return f"{float(score):.4f}"
    except (TypeError, ValueError):
        return str(score)


def clean_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    cleaned_metadata: dict[str, Any] = {}

    for key, value in metadata.items():
        if value is None:
            continue

        if isinstance(value, (str, int, float, bool)):
            cleaned_metadata[str(key)] = value
        else:
            cleaned_metadata[str(key)] = str(value)

    return cleaned_metadata


def is_insufficient_answer(answer: str) -> bool:
    normalized_answer = " ".join(answer.lower().split())

    insufficient_patterns = (
        "retrieved context is not sufficient",
        "context is insufficient",
        "insufficient context",
        "not enough information in the provided context",
        "provided context does not contain enough information",
        "cannot answer accurately based on the provided context",
        "cannot be answered from the provided context",
    )

    return any(
        pattern in normalized_answer
        for pattern in insufficient_patterns
    )


def display_source(
    result: HybridSearchResult,
    position: int,
) -> None:
    source_id = f"S{position}"
    document_source = get_document_source(result)
    section_title = get_section_title(result)
    metadata = clean_metadata(result.document.metadata or {})
    content = result.document.page_content.strip()

    expander_title = (
        f"[{source_id}] {document_source} — "
        f"Score: {format_score(result.score)}"
    )

    if section_title:
        expander_title += f" — {section_title}"

    with st.expander(expander_title, expanded=False):
        score_column_1, score_column_2, score_column_3 = st.columns(3)

        score_column_1.metric(
            "Final score",
            format_score(result.score),
        )
        score_column_2.metric(
            "FAISS score",
            format_score(result.faiss_score),
        )
        score_column_3.metric(
            "BM25 score",
            format_score(result.bm25_score),
        )

        st.markdown("#### Source details")

        source_details = {
            "Citation ID": source_id,
            "Document source": document_source,
            "Retrieval source": str(result.source),
        }

        if section_title:
            source_details["Section"] = section_title

        st.json(source_details)

        if metadata:
            st.markdown("#### Metadata")
            st.json(metadata)

        st.markdown("#### Content preview")

        if content:
            st.text(content)
        else:
            st.caption("No content is available for this source.")


def display_sources(
    results: list[HybridSearchResult],
) -> None:
    st.subheader("Retrieved Sources")

    if not results:
        st.info("No relevant sources were retrieved.")
        return

    st.caption(
        f"{len(results)} source(s) were retrieved. "
        "Open each item to inspect its scores, metadata, and content."
    )

    for position, result in enumerate(results, start=1):
        display_source(
            result=result,
            position=position,
        )


def display_sidebar() -> None:
    with st.sidebar:
        st.header("Configuration")

        st.markdown("**Answer language:** English")
        st.markdown(f"**Chat model:** `{settings.chat_model}`")
        st.markdown(
            f"**Retrieved results:** "
            f"`{settings.retrieval_top_k}`"
        )
        st.markdown(
            f"**Candidate results:** "
            f"`{settings.retrieval_candidate_k}`"
        )
        st.markdown(
            f"**Maximum context size:** "
            f"`{settings.context_max_chars:,}` characters"
        )

        st.divider()

        st.header("How to use")

        st.markdown(
            """
1. Enter a software testing question in English.
2. Select **Ask Question**.
3. Review the generated answer.
4. Inspect the retrieved sources and scores.
            """
        )

        st.divider()

        if st.button(
            "Clear current result",
            use_container_width=True,
        ):
            st.session_state.pop("query", None)
            st.session_state.pop("answer", None)
            st.session_state.pop("results", None)
            st.session_state.pop("context_length", None)
            st.session_state.pop("answer_metrics", None)
            st.rerun()


def initialize_session_state() -> None:
    if "query" not in st.session_state:
        st.session_state.query = ""

    if "answer" not in st.session_state:
        st.session_state.answer = None

    if "results" not in st.session_state:
        st.session_state.results = []

    if "context_length" not in st.session_state:
        st.session_state.context_length = 0

    if "answer_metrics" not in st.session_state:
        st.session_state.answer_metrics = None


def initialize_pipeline() -> tuple[
    HybridRetriever,
    ContextBuilder,
    AnswerGenerator,
]:
    retriever = get_retriever()
    context_builder = get_context_builder()
    answer_generator = get_answer_generator()

    return retriever, context_builder, answer_generator


def run_query(
    *,
    query: str,
    retriever: HybridRetriever,
    context_builder: ContextBuilder,
    answer_generator: AnswerGenerator,
) -> tuple[
    str,
    list[HybridSearchResult],
    int,
    dict[str, Any],
]:
    results = retriever.search(
        query=query,
        top_k=settings.retrieval_top_k,
        candidate_k=settings.retrieval_candidate_k,
    )

    if not results:
        return (
            INSUFFICIENT_CONTEXT_MESSAGE,
            [],
            0,
            {
                "confidence_score": 0.0,
                "faithfulness_score": 0.0,
                "citation_precision": 0.0,
                "citation_coverage": 0.0,
                "invalid_citation_flag": False,
                "is_insufficient": True,
                "extracted_citations": [],
            },
        )

    built_context = context_builder.build(results)

    answer = answer_generator.generate_answer(
        question=query,
        built_context=built_context,
    )

    evaluator = get_generation_evaluator()

    citation_source_ids = [
        source.source_id
        for source in built_context.sources
    ]

    metrics = evaluator.evaluate(
        answer=answer,
        retrieved_sources=citation_source_ids,
        context_text=built_context.context_text,
    )

    answer_metrics = {
        "confidence_score": (
            metrics.confidence_score
        ),
        "faithfulness_score": (
            metrics.faithfulness_score
        ),
        "citation_precision": (
            metrics.citation_precision
        ),
        "citation_coverage": (
            metrics.citation_coverage
        ),
        "invalid_citation_flag": (
            metrics.invalid_citation_flag
        ),
        "is_insufficient": (
            metrics.is_insufficient
        ),
        "extracted_citations": (
            metrics.extracted_citations
        ),
    }

    return (
        answer,
        results,
        len(built_context.context_text),
        answer_metrics,
    )


def display_answer(
    *,
    answer: str,
    results: list[HybridSearchResult],
    context_length: int,
    answer_metrics: dict[str, Any] | None,
) -> None:
    st.subheader("Answer")

    metrics = answer_metrics or {}
    confidence_score = float(
        metrics.get("confidence_score", 0.0)
    )
    confidence_percentage = (
        confidence_score * 100.0
    )

    is_insufficient = (
        is_insufficient_answer(answer)
        or bool(metrics.get("is_insufficient", False))
    )

    answer_column, confidence_column = (
        st.columns([4, 1])
    )

    with answer_column:
        if is_insufficient:
            st.warning(answer)
        else:
            st.markdown(answer)

    with confidence_column:
        st.metric(
            "Confidence",
            f"{confidence_percentage:.1f}%",
        )

        if is_insufficient:
            st.error("Low")
        elif confidence_score >= 0.80:
            st.success("High")
        elif confidence_score >= 0.55:
            st.warning("Medium")
        else:
            st.error("Low")

    st.divider()

    (
        metric_column_1,
        metric_column_2,
        metric_column_3,
        metric_column_4,
    ) = st.columns(4)

    metric_column_1.metric(
        "Retrieved sources",
        len(results),
    )
    metric_column_2.metric(
        "Context characters",
        f"{context_length:,}",
    )
    metric_column_3.metric(
        "Context overlap",
        (
            f"{float(metrics.get('faithfulness_score', 0.0)):.3f}"
        ),
    )
    metric_column_4.metric(
        "Answer status",
        (
            "Insufficient context"
            if is_insufficient
            else "Completed"
        ),
    )

    with st.expander(
        "Answer + Metrics",
        expanded=False,
    ):
        st.json(
            {
                "confidence_score": (
                    confidence_score
                ),
                "context_overlap_score": (
                    metrics.get(
                        "faithfulness_score",
                        0.0,
                    )
                ),
                "citation_precision": (
                    metrics.get(
                        "citation_precision",
                        0.0,
                    )
                ),
                "citation_coverage": (
                    metrics.get(
                        "citation_coverage",
                        0.0,
                    )
                ),
                "invalid_citation_flag": (
                    metrics.get(
                        "invalid_citation_flag",
                        False,
                    )
                ),
                "is_insufficient": (
                    is_insufficient
                ),
                "extracted_citations": (
                    metrics.get(
                        "extracted_citations",
                        [],
                    )
                ),
            }
        )

    st.divider()

    display_sources(results)


def main() -> None:
    initialize_session_state()
    display_sidebar()

    st.title("🧪 LLM Software Testing Knowledge Assistant")

    st.markdown(
        "Ask software testing questions in English and receive "
        "context-grounded answers with traceable sources."
    )

    try:
        with st.spinner("Initializing the QA pipeline..."):
            retriever, context_builder, answer_generator = (
                initialize_pipeline()
            )
    except Exception as exc:
        st.error("Failed to initialize the QA pipeline.")
        st.exception(exc)
        st.stop()

    with st.form(
        key="question_form",
        clear_on_submit=False,
    ):
        query = st.text_area(
            "Question",
            value=st.session_state.query,
            placeholder=(
                "Example: What is the difference between "
                "verification and validation?"
            ),
            height=120,
            max_chars=1000,
        )

        submitted = st.form_submit_button(
            "Ask Question",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        normalized_query = query.strip()
        st.session_state.answer_metrics = None

        if not normalized_query:
            st.warning("Please enter a question.")
        else:
            st.session_state.query = normalized_query
            st.session_state.answer = None
            st.session_state.results = []
            st.session_state.context_length = 0

            try:
                with st.spinner(
                    "Retrieving sources and generating the answer..."
                ):
                    (
                        answer,
                        results,
                        context_length,
                        answer_metrics,
                    ) = run_query(
                        query=normalized_query,
                        retriever=retriever,
                        context_builder=context_builder,
                        answer_generator=answer_generator,
                    )

                st.session_state.answer = answer
                st.session_state.results = results
                st.session_state.context_length = context_length
                st.session_state.answer_metrics = answer_metrics

            except Exception as exc:
                st.error(
                    "The question could not be processed. "
                    "Please review the error details below."
                )
                st.exception(exc)

    if st.session_state.answer is not None:
        display_answer(
            answer=st.session_state.answer,
            results=st.session_state.results,
            context_length=(
                st.session_state.context_length
            ),
            answer_metrics=(
                st.session_state.answer_metrics
            ),
        )


if __name__ == "__main__":
    main()
