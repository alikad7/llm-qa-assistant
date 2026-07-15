import json
import logging
from typing import Any, Optional

from src.query_analyzer.fallback_query_analyzer import FallbackQueryAnalyzer
from src.query_analyzer.prompts import (
    QUERY_ANALYZER_SYSTEM_PROMPT,
    build_query_analyzer_user_prompt,
)
from src.query_analyzer.schemas import QueryAnalysis, QueryDomain, QueryIntent

logger = logging.getLogger(__name__)


class LLMQueryAnalyzer:
    """
    LLM-based query analyzer for domain classification, intent detection,
    retrieval query generation, and domain guardrail enforcement.
    """

    CODE_FENCE = chr(96) * 3

    def __init__(
        self,
        llm_client: Any,
        model: str = "gpt-4o-mini",
        fallback_analyzer: Optional[FallbackQueryAnalyzer] = None,
        temperature: float = 0.0,
    ) -> None:
        self.llm_client = llm_client
        self.model = model
        self.fallback_analyzer = fallback_analyzer or FallbackQueryAnalyzer()
        self.temperature = temperature

    def analyze(self, query: str) -> QueryAnalysis:
        try:
            response_text = self._call_llm(query)
            payload = self._extract_json(response_text)
            analysis = self._validate_payload(payload)
            return self._finalize_guardrail(analysis)
        except Exception:
            logger.exception(
                "LLM query analysis failed; falling back to rule-based analyzer for query=%r",
                query,
            )
            fallback_analysis = self.fallback_analyzer.analyze(query)
            return self._finalize_guardrail(fallback_analysis)

    def _call_llm(self, query: str) -> str:
        response = self.llm_client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {
                    "role": "system",
                    "content": QUERY_ANALYZER_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": build_query_analyzer_user_prompt(query),
                },
            ],
        )

        message = response.choices[0].message
        content = message.content

        if content is None:
            raise ValueError("LLM returned empty content")

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            text_parts = []

            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))

            joined = "\n".join(part for part in text_parts if part).strip()
            if joined:
                return joined

        raise ValueError(f"Unsupported LLM response content type: {type(content)!r}")

    def _extract_json(self, response_text: str) -> dict:
        cleaned_text = response_text.strip()

        if cleaned_text.startswith(self.CODE_FENCE):
            lines = cleaned_text.splitlines()

            if lines and lines[0].startswith(self.CODE_FENCE):
                lines = lines[1:]

            if lines and lines[-1].strip() == self.CODE_FENCE:
                lines = lines[:-1]

            cleaned_text = "\n".join(lines).strip()

        try:
            return json.loads(cleaned_text)
        except json.JSONDecodeError:
            start_index = cleaned_text.find("{")
            end_index = cleaned_text.rfind("}")

            if start_index == -1 or end_index == -1 or end_index <= start_index:
                raise

            return json.loads(cleaned_text[start_index : end_index + 1])

    def _validate_payload(self, payload: dict) -> QueryAnalysis:
        if hasattr(QueryAnalysis, "model_validate"):
            return QueryAnalysis.model_validate(payload)

        return QueryAnalysis.parse_obj(payload)

    def _finalize_guardrail(self, analysis: QueryAnalysis) -> QueryAnalysis:
        original_query = analysis.original_query or ""
        normalized_query = analysis.normalized_query or ""
        contains_testing_term = self.fallback_analyzer.contains_testing_term(
            original_query or normalized_query
        )

        if contains_testing_term and analysis.domain != QueryDomain.SOFTWARE_TESTING:
            intent = self.fallback_analyzer.detect_intent_from_query(original_query)
            normalized = normalized_query or self.fallback_analyzer.normalize_query(original_query)
            bm25_query = self.fallback_analyzer.build_bm25_query_from_query(original_query, intent)
            semantic_query = self.fallback_analyzer.build_semantic_query_from_query(
                original_query, intent
            )

            return self._copy_analysis(
                analysis,
                normalized_query=normalized,
                domain=QueryDomain.SOFTWARE_TESTING,
                intent=intent,
                is_in_domain=True,
                needs_retrieval=True,
                bm25_query=bm25_query,
                semantic_query=semantic_query,
                reason=(
                    "Guardrail override: the query contains clear software testing or QA terms."
                ),
                confidence=max(float(analysis.confidence), 0.85),
                language=analysis.language or self.fallback_analyzer.detect_language(original_query),
            )

        if analysis.domain == QueryDomain.OUT_OF_DOMAIN:
            return self._copy_analysis(
                analysis,
                intent=QueryIntent.OUT_OF_DOMAIN,
                is_in_domain=False,
                needs_retrieval=False,
                bm25_query="",
                semantic_query="",
            )

        if analysis.domain == QueryDomain.UNCERTAIN:
            return self._copy_analysis(
                analysis,
                is_in_domain=False,
                needs_retrieval=False,
                bm25_query="",
                semantic_query="",
            )

        if analysis.domain == QueryDomain.SOFTWARE_TESTING:
            bm25_query = analysis.bm25_query.strip()
            semantic_query = analysis.semantic_query.strip()

            if not bm25_query:
                bm25_query = analysis.normalized_query or analysis.original_query

            if not semantic_query:
                semantic_query = analysis.normalized_query or analysis.original_query

            return self._copy_analysis(
                analysis,
                is_in_domain=True,
                needs_retrieval=True,
                bm25_query=bm25_query,
                semantic_query=semantic_query,
            )

        return analysis

    def _copy_analysis(self, analysis: QueryAnalysis, **updates: Any) -> QueryAnalysis:
        if hasattr(analysis, "model_copy"):
            return analysis.model_copy(update=updates)

        return analysis.copy(update=updates)
