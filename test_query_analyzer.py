from src.query_analyzer.fallback_query_analyzer import FallbackQueryAnalyzer


analyser = FallbackQueryAnalyzer()

queries = [
    "What is software testing?",
    "How to write test cases?",
    "فرق verification و validation چیست؟",
    "طرز تهیه پیتزا چیه؟",
]

for query in queries:
    result = analyser.analyze(query)
    print("=" * 80)
    print("Query:", query)
    print(result.model_dump())
