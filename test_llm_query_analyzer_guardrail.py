import json

from src.query_analyzer.llm_query_analyzer import LLMQueryAnalyzer
from src.query_analyzer.schemas import QueryDomain, QueryIntent


class MockMessage:
    def __init__(self, content):
        self.content = content


class MockChoice:
    def __init__(self, content):
        self.message = MockMessage(content)


class MockResponse:
    def __init__(self, content):
        self.choices = [MockChoice(content)]


class MockCompletions:
    def __init__(self, content):
        self.content = content

    def create(self, **kwargs):
        return MockResponse(self.content)


class MockChat:
    def __init__(self, content):
        self.completions = MockCompletions(content)


class MockLLMClient:
    def __init__(self, content):
        self.chat = MockChat(content)


def run_case(name, query, llm_payload):
    client = MockLLMClient(json.dumps(llm_payload))
    analyzer = LLMQueryAnalyzer(llm_client=client)

    result = analyzer.analyze(query)

    print(f"\n=== {name} ===")
    print("domain:", result.domain)
    print("intent:", result.intent)
    print("is_in_domain:", result.is_in_domain)
    print("needs_retrieval:", result.needs_retrieval)
    print("bm25_query:", repr(result.bm25_query))
    print("semantic_query:", repr(result.semantic_query))


def main():
    run_case(
        "Out of domain with wrong retrieval flags",
        "How to make pizza?",
        {
            "original_query": "How to make pizza?",
            "normalized_query": "how to make pizza",
            "domain": QueryDomain.OUT_OF_DOMAIN.value,
            "intent": QueryIntent.GENERAL.value,
            "is_in_domain": True,
            "needs_retrieval": True,
            "bm25_query": "pizza recipe",
            "semantic_query": "how to make pizza at home",
            "reason": "This is about cooking.",
            "confidence": 0.90,
        },
    )

    run_case(
        "Uncertain with wrong retrieval flags",
        "testing?",
        {
            "original_query": "testing?",
            "normalized_query": "testing",
            "domain": QueryDomain.UNCERTAIN.value,
            "intent": QueryIntent.GENERAL.value,
            "is_in_domain": True,
            "needs_retrieval": True,
            "bm25_query": "testing",
            "semantic_query": "testing meaning",
            "reason": "The query is ambiguous.",
            "confidence": 0.90,
        },
    )

    run_case(
        "Software testing with empty retrieval queries",
        "What is regression testing?",
        {
            "original_query": "What is regression testing?",
            "normalized_query": "what is regression testing",
            "domain": QueryDomain.SOFTWARE_TESTING.value,
            "intent": QueryIntent.DEFINITION.value,
            "is_in_domain": False,
            "needs_retrieval": False,
            "bm25_query": "",
            "semantic_query": "",
            "reason": "This is about software testing.",
            "confidence": 0.90,
        },
    )


if __name__ == "__main__":
    main()
