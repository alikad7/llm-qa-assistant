import os
from dotenv import load_dotenv

# لود کردن تنظیمات محیطی
load_dotenv()

from src.ingestion import PDFLoader, WebLoader
from src.processing import DocumentProcessor, DocumentChunker

def run_test():
    print("🚀 Starting Ingestion and Processing Test (Updated Paths)...\n")

    # تنظیم مسیرهای جدید بر اساس ساختار شما
    pdf_dir = os.path.join("data", "pdfs")
    url_file = os.path.join("data", "urls.txt")

    # بررسی وجود فایل‌ها و پوشه‌ها
    if not os.path.exists(pdf_dir):
        print(f"❌ Error: PDF directory '{pdf_dir}' not found. Please create it and add your PDFs.")
        return

    if not os.path.exists(url_file):
        print(f"❌ Error: URL list file '{url_file}' not found. Please create it.")
        return

    # 1. Ingestion Phase
    print("--- Phase 1: Ingestion ---")
    pdf_loader = PDFLoader(data_dir=pdf_dir)
    web_loader = WebLoader(url_list_file=url_file)

    raw_pdf_docs = pdf_loader.load()
    raw_web_docs = web_loader.load()
    
    all_raw_docs = raw_pdf_docs + raw_web_docs
    print(f"✅ Loaded {len(raw_pdf_docs)} PDF sections from '{pdf_dir}'.")
    print(f"✅ Loaded {len(raw_web_docs)} Web sections from '{url_file}'.")
    print(f"Total raw document units: {len(all_raw_docs)}\n")

    if not all_raw_docs:
        print("⚠ No documents loaded. Please add some PDF files or URLs to proceed.")
        return

    # 2. Processing Phase
    print("--- Phase 2: Processing (Normalization) ---")
    processor = DocumentProcessor()
    langchain_docs = processor.to_langchain_documents(all_raw_docs)
    print(f"✅ Converted to {len(langchain_docs)} LangChain Documents.\n")

    # 3. Chunking Phase
    print("--- Phase 3: Chunking (Title-Aware) ---")
    chunker = DocumentChunker(chunk_size=800, chunk_overlap=100)
    final_chunks = chunker.split_documents(langchain_docs)
    print(f"✅ Generated {len(final_chunks)} total chunks.\n")

    # 4. بررسی نمونه‌های خروجی
    print("--- Phase 4: Sample Inspection ---")
    
    # نمونه وب
    web_samples = [c for c in final_chunks if c.metadata.get("type") == "web"]
    if web_samples:
        sample = web_samples[0]
        print("🌐 [Web Chunk Sample]")
        print(f"Metadata: {sample.metadata}")
        print(f"Content Preview (first 200 chars):\n{sample.page_content[:200]}...")
        print("-" * 40)
    else:
        print("ℹ No Web chunks available to display.")

    # نمونه پی‌دی‌اف
    pdf_samples = [c for c in final_chunks if c.metadata.get("type") == "pdf"]
    if pdf_samples:
        sample = pdf_samples[0]
        print("📄 [PDF Chunk Sample]")
        print(f"Metadata: {sample.metadata}")
        print(f"Content Preview (first 200 chars):\n{sample.page_content[:200]}...")
        print("-" * 40)
    else:
        print("ℹ No PDF chunks available to display.")

if __name__ == "__main__":
    run_test()
