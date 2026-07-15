from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class GenerationMetricsResult:
    citation_precision: float
    citation_coverage: float
    invalid_citation_flag: bool
    faithfulness_score: float
    confidence_score: float
    is_insufficient: bool
    response_length: int
    extracted_citations: list[str]

    @property
    def citation_recall(self) -> float:
        return self.citation_coverage

    @property
    def hallucination_flag(self) -> bool:
        return self.invalid_citation_flag


class GenerationEvaluator:
    STOPWORDS = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "by",
        "for",
        "from",
        "has",
        "have",
        "if",
        "in",
        "into",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "these",
        "this",
        "those",
        "to",
        "was",
        "were",
        "while",
        "with",
    }

    def __init__(
        self,
        *,
        insufficient_phrases: Sequence[str] = (
            "not sufficient",
            "insufficient context",
            "not enough information",
            "no context provided",
            "cannot answer accurately",
            "cannot be answered",
        ),
    ) -> None:
        self.insufficient_phrases = tuple(insufficient_phrases)
        self.citation_pattern = re.compile(
            r"\[S(\d+)\]",
            re.IGNORECASE,
        )

    def evaluate(
        self,
        *,
        answer: str,
        retrieved_sources: list[str],
        context_text: str,
    ) -> GenerationMetricsResult:
        cleaned_answer = answer.strip()
        response_length = len(cleaned_answer)

        is_insufficient = self._check_insufficient(cleaned_answer)

        extracted_citations = self._extract_citations(cleaned_answer)

        (
            citation_precision,
            citation_coverage,
        ) = self._calculate_citation_metrics(
            answer=cleaned_answer,
            extracted_citations=extracted_citations,
            available_sources=retrieved_sources,
            is_insufficient=is_insufficient,
        )

        faithfulness_score = self._calculate_faithfulness(
            answer=cleaned_answer,
            context_text=context_text,
            is_insufficient=is_insufficient,
        )

        invalid_citation_flag = self._check_invalid_citations(
            extracted_citations=extracted_citations,
            available_sources=retrieved_sources,
        )

        confidence_score = self.calculate_confidence_score(
            faithfulness=faithfulness_score,
            citation_precision=citation_precision,
            citation_coverage=citation_coverage,
            is_insufficient=is_insufficient,
            invalid_citation=invalid_citation_flag,
        )

        return GenerationMetricsResult(
            citation_precision=citation_precision,
            citation_coverage=citation_coverage,
            invalid_citation_flag=invalid_citation_flag,
            faithfulness_score=faithfulness_score,
            confidence_score=confidence_score,
            is_insufficient=is_insufficient,
            response_length=response_length,
            extracted_citations=extracted_citations,
        )

    @staticmethod
    def calculate_confidence_score(
        *,
        faithfulness: float,
        citation_precision: float,
        citation_coverage: float,
        is_insufficient: bool,
        invalid_citation: bool,
    ) -> float:
        if is_insufficient:
            return 0.0

        score = (
            0.45 * faithfulness
            + 0.25 * citation_precision
            + 0.30 * citation_coverage
        )

        if invalid_citation:
            score *= 0.5

        return round(
            max(0.0, min(score, 1.0)),
            4,
        )

    def _check_insufficient(
        self,
        answer: str,
    ) -> bool:
        lowered_answer = answer.lower()
        return any(
            phrase.lower() in lowered_answer
            for phrase in self.insufficient_phrases
        )

    def _extract_citations(
        self,
        answer: str,
    ) -> list[str]:
        matches = self.citation_pattern.findall(answer)
        unique_citations = {
            f"S{int(number)}"
            for number in matches
        }
        return sorted(
            unique_citations,
            key=lambda citation: int(citation[1:]),
        )

    def _extract_claim_sentences(
        self,
        answer: str,
    ) -> list[str]:
        sentences = re.split(
            r"(?<=[.!?])\s+|\n+",
            answer.strip(),
        )
        return [
            sentence.strip()
            for sentence in sentences
            if self._is_factual_sentence(sentence)
        ]

    def _is_factual_sentence(
        self,
        sentence: str,
    ) -> bool:
        text_without_citations = self.citation_pattern.sub(
            "",
            sentence,
        ).strip()
        words = re.findall(
            r"[A-Za-z0-9]+",
            text_without_citations,
        )
        return len(words) >= 4

    def _sentence_has_valid_citation(
        self,
        *,
        sentence: str,
        available_sources: set[str],
    ) -> bool:
        sentence_citations = {
            f"S{int(number)}"
            for number in self.citation_pattern.findall(sentence)
        }
        return bool(sentence_citations.intersection(available_sources))

    def _calculate_citation_metrics(
        self,
        *,
        answer: str,
        extracted_citations: list[str],
        available_sources: list[str],
        is_insufficient: bool,
    ) -> tuple[float, float]:
        if is_insufficient:
            return 0.0, 0.0

        available_source_set = set(available_sources)
        if not available_source_set:
            return 0.0, 0.0

        valid_citations = {
            citation
            for citation in extracted_citations
            if citation in available_source_set
        }

        citation_precision = (
            len(valid_citations) / len(extracted_citations)
            if extracted_citations
            else 0.0
        )

        claim_sentences = self._extract_claim_sentences(answer)
        if not claim_sentences:
            citation_coverage = 0.0
        else:
            cited_claims = sum(
                self._sentence_has_valid_citation(
                    sentence=sentence,
                    available_sources=available_source_set,
                )
                for sentence in claim_sentences
            )
            citation_coverage = cited_claims / len(claim_sentences)

        return citation_precision, citation_coverage

    def _calculate_faithfulness(
        self,
        *,
        answer: str,
        context_text: str,
        is_insufficient: bool,
    ) -> float:
        if is_insufficient:
            return 0.0

        if (
            not context_text.strip()
            or not answer.strip()
        ):
            return 0.0

        answer_without_citations = self.citation_pattern.sub(
            "",
            answer,
        )

        answer_tokens = self._tokenize(answer_without_citations)
        context_tokens = self._tokenize(context_text)

        if not answer_tokens:
            return 0.0

        overlap = answer_tokens.intersection(context_tokens)
        return len(overlap) / len(answer_tokens)

    @classmethod
    def _tokenize(cls, text: str) -> set[str]:
        words = re.findall(
            r"[a-z0-9]+",
            text.lower(),
        )
        return {
            word
            for word in words
            if len(word) > 2 and word not in cls.STOPWORDS
        }

    @staticmethod
    def _check_invalid_citations(
        *,
        extracted_citations: list[str],
        available_sources: list[str],
    ) -> bool:
        available_source_set = set(available_sources)
        return any(
            citation not in available_source_set
            for citation in extracted_citations
        )
