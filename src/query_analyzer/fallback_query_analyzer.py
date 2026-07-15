import re
from src.query_analyzer.schemas import QueryAnalysis, QueryDomain, QueryIntent

class FallbackQueryAnalyzer:
    """
    Rule-based fallback analyzer used when LLM-based query analysis
    is unavailable or fails.
    """

    DOMAIN_KEYWORDS = {
        "software testing",
        "testing",
        "qa",
        "qc",
        "quality assurance",
        "quality control",
        "test",
        "test case",
        "test cases",
        "test scenario",
        "test scenarios",
        "test suite",
        "test suites",
        "test data",
        "test plan",
        "test strategy",
        "test coverage",
        "test design",
        "test execution",
        "test management",
        "test automation",
        "automation testing",
        "manual testing",
        "exploratory testing",
        "unit testing",
        "integration testing",
        "system testing",
        "acceptance testing",
        "uat",
        "smoke testing",
        "sanity testing",
        "regression testing",
        "performance testing",
        "load testing",
        "stress testing",
        "security testing",
        "usability testing",
        "api testing",
        "ui testing",
        "functional testing",
        "non functional testing",
        "black box testing",
        "white box testing",
        "grey box testing",
        "bug",
        "bugs",
        "defect",
        "defects",
        "error guessing",
        "verification",
        "validation",
        "v model",
        "stlc",
        "sdlc",
        "boundary value analysis",
        "boundary value",
        "bva",
        "equivalence partitioning",
        "equivalence class",
        "decision table",
        "state transition",
        "requirements traceability matrix",
        "rtm",
        "traceability matrix",
        "bug report",
        "defect report",
        "test report",
        "test summary report",
        "jira testing",
        "selenium",
        "cypress",
        "playwright",
        "postman testing",
    }

    QUESTION_PHRASES = (
        "what is",
        "what are",
        "what does",
        "define",
        "explain",
        "how to",
        "how do",
        "how can",
        "how should",
        "when should",
        "why is",
        "فرق",
        "تفاوت",
        "چیست",
        "چیه",
        "چطور",
        "چگونه",
    )

    def analyze(self, query: str) -> QueryAnalysis:
        normalized_query = self._normalize(query)
        is_in_domain = self._is_in_domain(normalized_query)

        if not is_in_domain:
            return QueryAnalysis(
                original_query=query,
                normalized_query=normalized_query,
                domain=QueryDomain.OUT_OF_DOMAIN,
                intent=QueryIntent.OUT_OF_DOMAIN,
                is_in_domain=False,
                needs_retrieval=False,
                bm25_query="",
                semantic_query="",
                reason="The query does not appear to be related to software testing or QA.",
                confidence=0.65,
                language=self._detect_language(query),
            )

        intent = self._detect_intent(normalized_query)

        return QueryAnalysis(
            original_query=query,
            normalized_query=normalized_query,
            domain=QueryDomain.SOFTWARE_TESTING,
            intent=intent,
            is_in_domain=True,
            needs_retrieval=True,
            bm25_query=self._build_bm25_query(normalized_query, intent),
            semantic_query=self._build_semantic_query(normalized_query, intent),
            reason="The query contains software testing or QA related concepts.",
            confidence=0.70,
            language=self._detect_language(query),
        )

    def contains_testing_term(self, query: str) -> bool:
        return self._is_in_domain(self._normalize(query))

    def detect_intent_from_query(self, query: str) -> QueryIntent:
        return self._detect_intent(self._normalize(query))

    def build_bm25_query_from_query(self, query: str, intent: QueryIntent) -> str:
        return self._build_bm25_query(self._normalize(query), intent)

    def build_semantic_query_from_query(self, query: str, intent: QueryIntent) -> str:
        return self._build_semantic_query(self._normalize(query), intent)

    def detect_language(self, query: str) -> str:
        return self._detect_language(query)

    def normalize_query(self, query: str) -> str:
        return self._normalize(query)

    def _normalize(self, query: str) -> str:
        normalized = query.strip().lower()
        normalized = re.sub(r"[^\w\s]", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def _strip_question_phrases(self, normalized_query: str) -> str:
        cleaned_query = normalized_query

        for phrase in self.QUESTION_PHRASES:
            if cleaned_query.startswith(phrase + " "):
                cleaned_query = cleaned_query[len(phrase):].strip()
                break

        return cleaned_query

    def _is_in_domain(self, normalized_query: str) -> bool:
        return any(keyword in normalized_query for keyword in self.DOMAIN_KEYWORDS)

    def _detect_intent(self, normalized_query: str) -> QueryIntent:
        if normalized_query.startswith(("what is", "what are", "define", "explain", "چیست", "چیه")):
            return QueryIntent.DEFINITION

        if normalized_query.startswith(
            ("how to", "how do", "how can", "how should", "چطور", "چگونه")
        ):
            return QueryIntent.HOW_TO

        if (
            "difference" in normalized_query
            or "compare" in normalized_query
            or "vs" in normalized_query
            or "versus" in normalized_query
            or "فرق" in normalized_query
            or "تفاوت" in normalized_query
        ):
            return QueryIntent.COMPARISON

        if "example" in normalized_query or "sample" in normalized_query or "template" in normalized_query:
            return QueryIntent.EXAMPLE

        if (
            "best practice" in normalized_query
            or "best practices" in normalized_query
            or "should include" in normalized_query
            or "checklist" in normalized_query
        ):
            return QueryIntent.BEST_PRACTICES

        if (
            "tool" in normalized_query
            or "tools" in normalized_query
            or "framework" in normalized_query
            or "selenium" in normalized_query
            or "cypress" in normalized_query
            or "playwright" in normalized_query
            or "postman" in normalized_query
            or "jira" in normalized_query
        ):
            return QueryIntent.TOOLING

        if "error" in normalized_query or "fail" in normalized_query or "issue" in normalized_query:
            return QueryIntent.TROUBLESHOOTING

        return QueryIntent.GENERAL

    def _build_bm25_query(self, normalized_query: str, intent: QueryIntent) -> str:
        core_query = self._strip_question_phrases(normalized_query)

        if core_query in {"testing", "software testing", "test"}:
            return (
                "software testing definition purpose verification validation "
                "quality assurance defects test design test execution"
            )

        if "test plan" in core_query:
            return "test plan scope objectives strategy resources schedule risks entry exit criteria deliverables"

        if "test strategy" in core_query:
            return "test strategy objectives scope levels types approach environment risks tools"

        if "test coverage" in core_query:
            return "test coverage requirements code coverage risk coverage traceability metrics"

        if "requirements traceability matrix" in core_query or "traceability matrix" in core_query or "rtm" in core_query:
            return "requirements traceability matrix rtm requirements test cases coverage mapping traceability"

        if "boundary value analysis" in core_query or "boundary value" in core_query or "bva" in core_query:
            return "boundary value analysis bva test design boundaries min max valid invalid inputs"

        if "equivalence partitioning" in core_query or "equivalence class" in core_query:
            return "equivalence partitioning equivalence class test design valid invalid partitions inputs"

        if intent == QueryIntent.DEFINITION:
            return f"{core_query} definition purpose basics software testing qa"

        if intent == QueryIntent.HOW_TO:
            return f"{core_query} steps process guide software testing qa"

        if intent == QueryIntent.EXAMPLE:
            return f"{core_query} example sample template software testing"

        if intent == QueryIntent.BEST_PRACTICES:
            return f"{core_query} best practices checklist guidelines software testing"

        if intent == QueryIntent.COMPARISON:
            return f"{core_query} difference comparison software testing"

        return core_query

    def _build_semantic_query(self, normalized_query: str, intent: QueryIntent) -> str:
        core_query = self._strip_question_phrases(normalized_query)

        if core_query in {"testing", "software testing", "test"}:
            return "definition and purpose of software testing in quality assurance"

        if "test plan" in core_query:
            return "what a software testing test plan should include"

        if "test strategy" in core_query:
            return "what a software testing test strategy includes and how it is structured"

        if "test coverage" in core_query:
            return "meaning and types of test coverage in software testing"

        if "requirements traceability matrix" in core_query or "traceability matrix" in core_query or "rtm" in core_query:
            return "requirements traceability matrix in software testing and how it maps requirements to test cases"

        if "boundary value analysis" in core_query or "boundary value" in core_query or "bva" in core_query:
            return "boundary value analysis in software testing with examples"

        if "equivalence partitioning" in core_query or "equivalence class" in core_query:
            return "equivalence partitioning in software testing with examples"

        if intent == QueryIntent.DEFINITION:
            return f"definition and explanation of {core_query} in software testing"

        if intent == QueryIntent.HOW_TO:
            return f"how to perform {core_query} in software testing"

        if intent == QueryIntent.BEST_PRACTICES:
            return f"best practices for {core_query} in software testing"

        if intent == QueryIntent.COMPARISON:
            return f"comparison and differences related to {core_query} in software testing"

        if intent == QueryIntent.EXAMPLE:
            return f"examples of {core_query} in software testing"

        return core_query

    def _detect_language(self, query: str) -> str:
        has_fa = bool(re.search(r"[\u0600-\u06FF]", query))
        has_en = bool(re.search(r"[a-zA-Z]", query))

        if has_fa and has_en:
            return "mixed"

        if has_fa:
            return "fa"

        if has_en:
            return "en"

        return "unknown"
