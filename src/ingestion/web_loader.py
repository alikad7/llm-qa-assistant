import os
from typing import List, Dict, Any
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup, Tag

from .utils import clean_text


class WebLoader:
    def __init__(self, url_list_file: str):
        self.url_list_file = url_list_file

    def _read_urls(self) -> List[str]:
        if not os.path.exists(self.url_list_file):
            return []

        with open(self.url_list_file, "r", encoding="utf-8") as f:
            return [
                line.strip()
                for line in f
                if line.strip() and not line.strip().startswith("#")
            ]

    def _extract_section_documents(self, soup: BeautifulSoup, url: str) -> List[Dict[str, Any]]:
        documents: List[Dict[str, Any]] = []
        domain = urlparse(url).netloc

        for element in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
            element.decompose()

        body = soup.body or soup
        page_title = clean_text(soup.title.get_text()) if soup.title else ""

        current_title = page_title if page_title else "Untitled Section"
        current_level = "title"
        current_text_parts: List[str] = []

        def flush_section():
            nonlocal current_text_parts, current_title, current_level

            section_text = clean_text(" ".join(current_text_parts))
            if section_text:
                documents.append(
                    {
                        "content": section_text,
                        "metadata": {
                            "source": url,
                            "domain": domain,
                            "type": "web",
                            "section_title": current_title,
                            "section_level": current_level,
                            "page_title": page_title,
                        },
                    }
                )
            current_text_parts = []

        for node in body.descendants:
            if not isinstance(node, Tag):
                continue

            if node.name in {"h1", "h2", "h3"}:
                flush_section()
                heading_text = clean_text(node.get_text(" ", strip=True))
                if heading_text:
                    current_title = heading_text
                    current_level = node.name

            elif node.name in {"p", "li"}:
                text = clean_text(node.get_text(" ", strip=True))
                if text:
                    current_text_parts.append(text)

        flush_section()

        if not documents:
            fallback_text = clean_text(body.get_text(separator=" "))
            if fallback_text:
                documents.append(
                    {
                        "content": fallback_text,
                        "metadata": {
                            "source": url,
                            "domain": domain,
                            "type": "web",
                            "section_title": page_title or "Full Page Content",
                            "section_level": "page",
                            "page_title": page_title,
                        },
                    }
                )

        return documents

    def load(self) -> List[Dict[str, Any]]:
        documents: List[Dict[str, Any]] = []
        urls = self._read_urls()

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Assistant/1.0"
        }

        for url in urls:
            try:
                response = requests.get(url, headers=headers, timeout=15)
                response.raise_for_status()

                soup = BeautifulSoup(response.text, "html.parser")
                page_docs = self._extract_section_documents(soup, url)
                documents.extend(page_docs)

            except Exception as e:
                print(f"Error fetching URL {url}: {e}")

        return documents
