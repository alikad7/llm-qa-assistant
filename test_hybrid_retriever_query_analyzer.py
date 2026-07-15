#test_hybrid_retriever_query_analyzer.py
import os
from typing import Any
from dataclasses import dataclass
from src.retrieval.hybrid_retriever import HybridRetriever
from src.query_analyzer.llm_query_analyzer import LLMQueryAnalyzer

# --- Mocking Infrastructure ---

@dataclass
class MockMessage:
    content: str

@dataclass
class MockChoice:
    message: MockMessage

@dataclass
class MockResponse:
    choices: list[MockChoice]

class MockLLMClient:
    """شبیه‌ساز کلاینت OpenAI برای تست اینتگریشن بدون هزینه"""
    def __init__(self, responses: dict[str, str]):
        self.responses = responses
        self.chat = self # برای سازگاری با ساختار client.chat.completions

    def completions(self):
        return self

    def create(self, messages: list, **kwargs) -> MockResponse:
        # استخراج کوئری کاربر از پیام‌ها
        user_content = messages[-1]["content"]
        
        # پیدا کردن پاسخ شبیه‌سازی شده بر اساس کلیدواژه
        selected_response = self.responses.get("default")
        for key in self.responses:
            if key in user_content.lower():
                selected_response = self.responses[key]
                break
        
        return MockResponse(choices=[MockChoice(message=MockMessage(content=selected_response))])

# --- تعریف پاسخ‌های شبیه‌سازی شده LLM ---

MOCK_RESPONSES = {
    "testing": """{
        "original_query": "What is software testing?",
        "normalized_query": "definition of software testing",
        "domain": "software_testing",
        "intent": "definition",
        "is_in_domain": true,
        "needs_retrieval": true,
        "bm25_query": "software testing definition",
        "semantic_query": "what is software testing",
        "reason": "Technical query about testing.",
        "confidence": 0.95
    }""",
    "pizza": """{
        "original_query": "طرز تهیه پیتزا",
        "normalized_query": "pizza recipe",
        "domain": "out_of_domain",
        "intent": "out_of_domain",
        "is_in_domain": false,
        "needs_retrieval": false,
        "bm25_query": "",
        "semantic_query": "",
        "reason": "Cooking is out of scope.",
        "confidence": 0.99
    }""",
    "default": """{
        "original_query": "unknown",
        "normalized_query": "unknown",
        "domain": "uncertain",
        "intent": "general",
        "is_in_domain": false,
        "needs_retrieval": false,
        "bm25_query": "",
        "semantic_query": "",
        "reason": "Not clear.",
        "confidence": 0.50
    }"""
}

# --- اجرای تست ---

def run_integration_test():
    print("🚀 Starting Integration Test: LLMQueryAnalyzer + HybridRetriever\n")

    # 1. تنظیم Mock Client و Analyzer
    mock_client = MockLLMClient(MOCK_RESPONSES)
    llm_analyzer = LLMQueryAnalyzer(llm_client=mock_client)

    # 2. تنظیم Retriever با Analyzer تزریق شده
    # نکته: مسیر ایندکس‌ها باید با ساختار پروژه شما (data/indexes) همخوانی داشته باشد
    try:
        retriever = HybridRetriever(query_analyzer=llm_analyzer)
        print(
        "QUERY ANALYZER:",
        type(retriever.query_analyzer).__name__,
            )

    except Exception as e:
        print(f"❌ Error loading indexes: {e}")
        print("Make sure you have built the indexes first (Step 4).")
        return

    test_cases = [
        ("What is software testing?", "Should trigger retrieval"),
        ("طرز تهیه پیتزا یه؟", "Should NOT trigger retrieval (Out of Domain)"),
        ("Hello!", "Should NOT trigger retrieval (Uncertain/General)")
    ]

    for query, expectation in test_cases:
        print("=" * 80)
        print(f"INPUT QUERY: {query}")
        print(f"EXPECTATION: {expectation}")
        print("-" * 40)

        # اجرای عملیات جستجو (که در دل خود analyze را فراخوانی می‌کند)
        results = retriever.search(query, top_k=3)

        # استخراج آنالیز برای نمایش در لاگ تست
        analysis = retriever.query_analyzer.analyze(query)
        
        print(f"ANALYSIS -> Domain: {analysis.domain}, Needs Retrieval: {analysis.needs_retrieval}")
        print(f"RETRIEVED DOCUMENTS: {len(results)}")

        for i, res in enumerate(results, 1):
            print(f"  [{i}] Score: {res.score:.4f} | Source: {res.source} | Content: {res.document.page_content[:100]}...")
        
        print("\n")

if __name__ == "__main__":
    run_integration_test()
