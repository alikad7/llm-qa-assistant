from __future__ import annotations

import hashlib
import re
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentChunker:
    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        min_chunk_length: int = 120,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_length = min_chunk_length

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                "? ",
                "! ",
                "; ",
                ": ",
                " ",
                "",
            ],
            keep_separator=True,
        )

    def split_documents(
        self,
        documents: List[Document],
    ) -> List[Document]:
        if not documents:
            print("Warning: No documents provided to chunker.")
            return []

        print(
            f"Splitting {len(documents)} documents "
            f"(size={self.chunk_size}, overlap={self.chunk_overlap})..."
        )

        raw_chunks = self.splitter.split_documents(documents)

        filtered_chunks: List[Document] = []
        seen_hashes: set[str] = set()

        removed_invalid = 0
        removed_toc = 0
        removed_duplicate = 0

        for chunk in raw_chunks:
            cleaned_text = self._normalize_chunk_text(
                chunk.page_content or ""
            )

            if not self._is_valid_chunk(cleaned_text):
                removed_invalid += 1
                continue

            if self._looks_like_table_of_contents(cleaned_text):
                removed_toc += 1
                continue

            content_hash = self._content_hash(cleaned_text)
            if content_hash in seen_hashes:
                removed_duplicate += 1
                continue

            seen_hashes.add(content_hash)

            chunk.page_content = cleaned_text
            filtered_chunks.append(chunk)

        for index, chunk in enumerate(filtered_chunks):
            chunk.metadata["chunk_index"] = index
            chunk.metadata["chunk_length"] = len(
                chunk.page_content
            )

        print(f"Generated {len(filtered_chunks)} chunks.")
        print(f"Removed invalid chunks: {removed_invalid}")
        print(f"Removed TOC-like chunks: {removed_toc}")
        print(f"Removed duplicate chunks: {removed_duplicate}")

        return filtered_chunks

    def _normalize_chunk_text(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text

    def _content_hash(self, text: str) -> str:
        canonical = text.lower()
        canonical = re.sub(r"[^a-z0-9]+", " ", canonical)
        canonical = re.sub(r"\s+", " ", canonical).strip()
        return hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest()

    def _is_valid_chunk(self, text: str) -> bool:
        if not text:
            return False

        if len(text) < self.min_chunk_length:
            return False

        alphanumeric_chars = re.findall(r"[A-Za-z0-9]", text)
        alphabetic_chars = re.findall(r"[A-Za-z]", text)

        if len(alphanumeric_chars) < 30:
            return False

        if len(alphabetic_chars) < 20:
            return False

        return True

    def _looks_like_table_of_contents(self, text: str) -> bool:
        lines = text.splitlines()
        nonempty_lines = [
            line.strip()
            for line in lines
            if line.strip()
        ]

        if not nonempty_lines:
            return False

        toc_line_count = 0

        for line in nonempty_lines:
            if re.search(r"\.{2,}\s*\d+$", line):
                toc_line_count += 1
                continue

            if re.match(
                r"^chapter\s+\d+",
                line,
                re.IGNORECASE,
            ):
                toc_line_count += 1

        return (
            toc_line_count / len(nonempty_lines)
        ) > 0.3
