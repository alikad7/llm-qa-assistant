from __future__ import annotations
from typing import Any, Dict, List
from langchain_core.documents import Document
from src.processing.quality_filter import DocumentQualityFilter


class DocumentProcessor:
    def __init__(self) -> None:
        self.quality_filter = DocumentQualityFilter(
            min_document_length=120,
        )

    def _normalize_text(self, text: str) -> str:
        if not text:
            return ""

        return " ".join(text.split()).strip()

    def _normalize_metadata(
        self,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        metadata = metadata.copy() if metadata else {}

        return {
            "source": metadata.get("source", ""),
            "type": metadata.get("type", "unknown"),
            "page": metadata.get("page"),
            "domain": metadata.get("domain"),
            "file_path": metadata.get("file_path"),
            "section_title": metadata.get("section_title"),
            "section_level": metadata.get("section_level"),
            "page_title": metadata.get("page_title"),
        }

    def to_langchain_documents(
        self,
        raw_documents: List[Dict[str, Any]],
    ) -> List[Document]:
        documents: List[Document] = []

        for item in raw_documents:
            content = self._normalize_text(item.get("content", ""))
            metadata = self._normalize_metadata(
                item.get("metadata", {})
            )

            if not content:
                continue

            documents.append(
                Document(
                    page_content=content,
                    metadata=metadata,
                )
            )

        filtered_documents, stats = (
            self.quality_filter.filter_documents(documents)
        )

        print("\nDocument quality report")
        print(f"  Input documents: {stats.total}")
        print(f"  Kept documents: {stats.kept}")
        print(
            "  Removed low-value sections: "
            f"{stats.low_value_section}"
        )
        print(f"  Removed short documents: {stats.too_short}")
        print(f"  Removed navigation documents: {stats.navigation}")
        print(f"  Removed duplicate documents: {stats.duplicate}")

        return filtered_documents
