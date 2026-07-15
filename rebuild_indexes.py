#File rebuild_indexes.py
from __future__ import annotations

import shutil
from pathlib import Path

from src.ingestion.pdf_loader import PDFLoader
from src.ingestion.web_loader import WebLoader
from src.processing.chunker import DocumentChunker
from src.processing.document_processor import DocumentProcessor
from src.retrieval.indexing_pipeline import IndexingPipeline


PROJECT_ROOT = Path(__file__).resolve().parent
PDF_DIR = PROJECT_ROOT / "data" / "pdfs"
URL_LIST = PROJECT_ROOT / "data" / "urls.txt"
INDEX_DIR = PROJECT_ROOT / "data" / "indexes"
FAISS_PATH = INDEX_DIR / "faiss_index"
BM25_DIR = INDEX_DIR / "bm25"


def load_raw_documents() -> list[dict]:
    raw_documents: list[dict] = []

    pdf_loader = PDFLoader(data_dir=str(PDF_DIR))
    web_loader = WebLoader(url_list_file=str(URL_LIST))

    print("Loading PDF documents...")
    pdf_documents = pdf_loader.load()
    print(f"Loaded {len(pdf_documents)} raw PDF documents.")
    raw_documents.extend(pdf_documents)

    print("Loading web documents...")
    web_documents = web_loader.load()
    print(f"Loaded {len(web_documents)} raw web documents.")
    raw_documents.extend(web_documents)

    return raw_documents


def validate_chunks(chunks) -> None:
    forbidden_sections = {
        "recommended reading",
    }

    invalid_chunks = []

    for chunk in chunks:
        section = str(
            chunk.metadata.get("section_title") or ""
        ).strip().lower()

        if section in forbidden_sections:
            invalid_chunks.append(chunk)
            continue

        if "thoughts on" in section:
            invalid_chunks.append(chunk)

    if invalid_chunks:
        examples = [
            {
                "source": item.metadata.get("source"),
                "section": item.metadata.get("section_title"),
                "text": item.page_content[:100],
            }
            for item in invalid_chunks[:5]
        ]

        raise RuntimeError(
            "Low-value chunks passed the quality filter. "
            f"Count={len(invalid_chunks)}, examples={examples}"
        )


def main() -> None:
    raw_documents = load_raw_documents()

    if not raw_documents:
        raise RuntimeError(
            "No input documents were loaded. "
            "Check data/pdfs and data/urls.txt."
        )

    processor = DocumentProcessor()
    documents = processor.to_langchain_documents(
        raw_documents
    )

    chunker = DocumentChunker(
        chunk_size=800,
        chunk_overlap=150,
        min_chunk_length=120,
    )
    chunks = chunker.split_documents(documents)

    if not chunks:
        raise RuntimeError(
            "No chunks remain after quality filtering."
        )

    validate_chunks(chunks)

    print(f"\nFinal chunk count: {len(chunks)}")
    print("Chunk validation passed.")

    if INDEX_DIR.exists():
        print(f"Removing old indexes: {INDEX_DIR}")
        shutil.rmtree(INDEX_DIR)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    pipeline = IndexingPipeline(
        faiss_path=str(FAISS_PATH),
        bm25_dir=str(BM25_DIR),
    )

    pipeline.build_and_persist(chunks)

    print("\nIndex rebuild completed successfully.")
    print(f"FAISS: {FAISS_PATH}")
    print(f"BM25: {BM25_DIR}")


if __name__ == "__main__":
    main()
