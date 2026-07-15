from __future__ import annotations
import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List
from langchain_core.documents import Document


@dataclass
class FilterStats:
    total: int = 0
    kept: int = 0
    low_value_section: int = 0
    too_short: int = 0
    navigation: int = 0
    duplicate: int = 0


class DocumentQualityFilter:
    """Removes low-value sections and duplicate content before indexing."""

    LOW_VALUE_SECTION_EXACT = {
        "recommended reading",
        "related posts",
        "related articles",
        "you may also like",
        "further reading",
        "popular articles",
        "advertisement",
        "references",
    }

    LOW_VALUE_SECTION_PATTERNS = (
        re.compile(r"\bthoughts?\s+on\b", re.IGNORECASE),
        re.compile(r"\bcomments?\b", re.IGNORECASE),
        re.compile(r"\bleave\s+a\s+reply\b", re.IGNORECASE),
        re.compile(r"\breader\s+interactions?\b", re.IGNORECASE),
    )

    NAVIGATION_PHRASES = (
        "recommended reading",
        "related articles",
        "you may also like",
        "previous article",
        "next article",
    )

    def __init__(self, min_document_length: int = 120) -> None:
        self.min_document_length = min_document_length

    def filter_documents(
        self,
        documents: Iterable[Document],
    ) -> tuple[List[Document], FilterStats]:
        documents = list(documents)
        stats = FilterStats(total=len(documents))
        filtered: List[Document] = []
        seen_content: set[str] = set()

        for document in documents:
            text = self._normalize_text(document.page_content)
            section = self._normalize_text(
                str(document.metadata.get("section_title") or "")
            ).lower()

            if self._is_low_value_section(section):
                stats.low_value_section += 1
                continue

            if len(text) < self.min_document_length:
                stats.too_short += 1
                continue

            if self._looks_like_navigation(text):
                stats.navigation += 1
                continue

            content_hash = self._content_hash(text)
            if content_hash in seen_content:
                stats.duplicate += 1
                continue

            seen_content.add(content_hash)
            document.page_content = text
            filtered.append(document)

        stats.kept = len(filtered)
        return filtered, stats

    def _is_low_value_section(self, section: str) -> bool:
        if not section:
            return False

        if section in self.LOW_VALUE_SECTION_EXACT:
            return True

        return any(
            pattern.search(section)
            for pattern in self.LOW_VALUE_SECTION_PATTERNS
        )

    def _looks_like_navigation(self, text: str) -> bool:
        normalized = text.lower()
        words = re.findall(r"[a-z0-9]+", normalized)

        if len(words) < 20:
            return True

        phrase_hits = sum(
            normalized.count(phrase)
            for phrase in self.NAVIGATION_PHRASES
        )

        # Navigation chunks usually contain repeated article names and phrases.
        if phrase_hits >= 3:
            return True

        if len(words) >= 30:
            word_counts = Counter(words)
            repeated_words = sum(
                count - 1
                for count in word_counts.values()
                if count >= 4
            )

            if repeated_words / len(words) > 0.35:
                return True

        return False

    def _normalize_text(self, text: str) -> str:
        text = text or ""
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _content_hash(self, text: str) -> str:
        canonical = text.lower()
        canonical = re.sub(r"[^a-z0-9]+", " ", canonical)
        canonical = re.sub(r"\s+", " ", canonical).strip()
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
