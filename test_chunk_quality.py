# scripts/inspect_chunks.py
from src.ingestion.pdf_loader import PDFLoader
from src.ingestion.web_loader import WebLoader
from src.processing.document_processor import DocumentProcessor
from src.processing.chunker import DocumentChunker


def main():
    raw_data = []

    pdf_loader = PDFLoader(data_dir="data/pdfs")
    raw_data.extend(pdf_loader.load())

    web_loader = WebLoader(url_list_file="data/urls.txt")
    raw_data.extend(web_loader.load())

    processor = DocumentProcessor()
    documents = processor.to_langchain_documents(raw_data)

    chunker = DocumentChunker(chunk_size=800, chunk_overlap=150)
    chunks = chunker.split_documents(documents)

    print(f"Total documents: {len(documents)}")
    print(f"Total chunks: {len(chunks)}")
    print("=" * 80)

    for i, chunk in enumerate(chunks[:10]):
        print(f"\nCHUNK #{i}")
        print(f"Length: {len(chunk.page_content)}")
        print(f"Metadata: {chunk.metadata}")
        print("-" * 80)
        print(chunk.page_content[:1000])
        print("=" * 80)


if __name__ == "__main__":
    main()
