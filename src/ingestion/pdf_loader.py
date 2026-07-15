import os
import re
from typing import List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader
from src.ingestion.utils import clean_text

class PDFLoader:
    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir

    def load(self) -> List[Dict[str, Any]]:
        documents = []
        if not os.path.exists(self.data_dir):
            print(f"Directory {self.data_dir} does not exist.")
            return documents

        for file_name in os.listdir(self.data_dir):
            if file_name.lower().endswith(".pdf"):
                file_path = os.path.join(self.data_dir, file_name)
                try:
                    # استفاده از PyPDFLoader با تنظیمات پیش‌فرض
                    loader = PyPDFLoader(file_path)
                    pages = loader.load()
                    
                    for page in pages:
                        raw_content = page.page_content
                        
                        # اعمال پاک‌سازی اصلاح شده
                        cleaned_content = clean_text(raw_content)
                        
                        if cleaned_content.strip():
                            documents.append({
                                "content": cleaned_content,
                                "metadata": {
                                    "source": file_name,
                                    "type": "pdf",
                                    "page": page.metadata.get("page", 0) + 1,
                                    "file_path": file_path
                                }
                            })
                except Exception as e:
                    print(f"Error loading PDF {file_name}: {e}")
        return documents
