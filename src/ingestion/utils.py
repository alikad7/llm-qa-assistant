import re

def clean_text(text: str) -> str:
    """Professional noise filtering for RAG documents."""
    if not text:
        return ""

    # حذف کاراکترهای غیرقابل چاپ
    text = "".join(ch for ch in text if ch.isprintable() or ch in ["\n", "\r", "\t"])
    
    # استانداردسازی فاصله‌ها (تبدیل همه فواصل به یک اسپیس)
    text = re.sub(r"[ \t]+", " ", text)
    
    # استانداردسازی newlineها
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    # حذف جداکننده‌های طولانی
    text = re.sub(r"([_*=-])\1{3,}", "", text)

    return text.strip()
