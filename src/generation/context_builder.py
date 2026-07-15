from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence


class DocumentLike(Protocol):
    page_content: str
    metadata: dict[str, Any]


class SearchResultLike(Protocol):
    document: DocumentLike
    score: float


@dataclass(frozen=True)
class ContextSource:
    source_id: str
    content: str
    score: float
    metadata: dict[str, Any]


@dataclass(frozen=True)
class BuiltContext:
    context_text: str
    sources: list[ContextSource]
    total_chars: int


class ContextBuilder:
    def __init__(
        self,
        *,
        max_context_chars: int = 6000,
        include_scores: bool = True,
    ) -> None:
        if max_context_chars <= 0:
            raise ValueError("max_context_chars must be greater than zero")

        self.max_context_chars = max_context_chars
        self.include_scores = include_scores

    def build(
        self,
        results: Sequence[SearchResultLike],
    ) -> BuiltContext:
        if not results:
            return BuiltContext(
                context_text="",
                sources=[],
                total_chars=0,
            )

        context_blocks: list[str] = []
        sources: list[ContextSource] = []
        used_chars = 0

        for index, result in enumerate(results, start=1):
            content = self._clean_text(result.document.page_content)

            if not content:
                continue

            metadata = dict(result.document.metadata or {})
            source_id = f"S{index}"

            block = self._format_source_block(
                source_id=source_id,
                content=content,
                score=float(result.score),
                metadata=metadata,
            )

            remaining_chars = self.max_context_chars - used_chars

            if remaining_chars <= 0:
                break

            if len(block) > remaining_chars:
                if not context_blocks:
                    block = block[:remaining_chars].rstrip()
                    content = content[:remaining_chars].rstrip()
                else:
                    break

            context_blocks.append(block)
            used_chars += len(block)

            sources.append(
                ContextSource(
                    source_id=source_id,
                    content=content,
                    score=float(result.score),
                    metadata=metadata,
                )
            )

        return BuiltContext(
            context_text="\n\n".join(context_blocks),
            sources=sources,
            total_chars=used_chars,
        )

    def _format_source_block(
        self,
        *,
        source_id: str,
        content: str,
        score: float,
        metadata: dict[str, Any],
    ) -> str:
        source_name = self._get_source_name(metadata)
        section_title = self._get_section_title(metadata)

        header_parts = [f"[{source_id}]", f"source={source_name}"]

        if section_title:
            header_parts.append(f"section={section_title}")

        if self.include_scores:
            header_parts.append(f"score={score:.4f}")

        header = " | ".join(header_parts)

        return f"{header}\n{content}"

    @staticmethod
    def _get_source_name(metadata: dict[str, Any]) -> str:
        source = (
            metadata.get("source")
            or metadata.get("file_path")
            or metadata.get("filename")
            or metadata.get("document_id")
            or "unknown"
        )

        return str(source)

    @staticmethod
    def _get_section_title(metadata: dict[str, Any]) -> str | None:
        section = (
            metadata.get("section_title")
            or metadata.get("title")
            or metadata.get("heading")
        )

        if section is None:
            return None

        return str(section)

    @staticmethod
    def _clean_text(text: str) -> str:
        lines = [
            line.strip()
            for line in (text or "").splitlines()
            if line.strip()
        ]

        return "\n".join(lines)
