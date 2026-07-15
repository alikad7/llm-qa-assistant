#build_index.py
from src.ingestion.pdf_loader import PDFLoader
from src.ingestion.web_loader import WebLoader
from src.processing.document_processor import DocumentProcessor
from src.retrieval.indexing_pipeline import IndexingPipeline
from src.processing.chunker import DocumentChunker


def run_build_pipeline() -> None:
    print("🚀 Initializing Ingestion, Processing, and Indexing Pipeline...")

    raw_data = []

    # 1. Ingestion
    print("📂 Loading PDFs from data/pdfs...")
    pdf_loader = PDFLoader(data_dir="data/pdfs")
    pdf_raw_data = pdf_loader.load()
    raw_data.extend(pdf_raw_data)

    print("🌐 Loading URLs from data/urls.txt...")
    web_loader = WebLoader(url_list_file="data/urls.txt")
    web_raw_data = web_loader.load()
    raw_data.extend(web_raw_data)

    print(f"📊 Total raw documents loaded: {len(raw_data)}")

    if not raw_data:
        print("❌ No documents loaded. Exiting pipeline.")
        return

    # 2. Normalization
    print("🧹 Normalizing raw documents...")
    processor = DocumentProcessor()
    documents = processor.to_langchain_documents(raw_data)

    print(f"📄 Total normalized documents: {len(documents)}")

    if not documents:
        print("❌ No valid documents after normalization. Exiting pipeline.")
        return

    # 3. Chunking (استراتژی خرد کردن مستندات)
    print("✂️ Starting Chunking Process...")
    # تنظیمات پیشنهادی برای مستندات تست نرم‌افزار
    chunker = DocumentChunker(chunk_size=800, chunk_overlap=150)
    chunks = chunker.split_documents(documents)

    if not chunks:
        print("❌ No chunks generated. Exiting pipeline.")
        return

    # 4. Indexing (Vector + Keyword)
    print(f"📦 Building hybrid indexes for {len(chunks)} chunks...")
    pipeline = IndexingPipeline(
        faiss_path="data/indexes/faiss_index",
        bm25_dir="data/indexes/bm25",
    )
    pipeline.build_and_persist(chunks)

    print("✅ Build index pipeline completed successfully.")


if __name__ == "__main__":
    run_build_pipeline()
