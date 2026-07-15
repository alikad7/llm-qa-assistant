import re

class QueryProcessor:
    """
    Handles query preprocessing before retrieval.
    """

    GENERIC_TESTING_QUERIES = {
        "software testing": "software testing basics definition verification validation quality assurance defects",
        "testing": "software testing definition types process",
        "test": "software testing test case test process",
    }

    def normalize(self, query: str) -> str:
        if not query:
            return query

        normalized = query.strip().lower()

        normalized = re.sub(r"[^\w\s]", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()

        question_prefix_patterns = [
            r"^what is\s+",
            r"^what are\s+",
            r"^what does\s+",
            r"^how to\s+",
            r"^how do i\s+",
            r"^how can i\s+",
            r"^how does\s+",
            r"^define\s+",
            r"^explain\s+",
            r"^tell me about\s+",
        ]

        simplified = normalized

        for pattern in question_prefix_patterns:
            simplified = re.sub(pattern, "", simplified).strip()

        if len(simplified) < 3:
            return normalized

        return simplified

    def expand_if_needed(self, query: str) -> str:
        normalized_query = self.normalize(query)

        if normalized_query in self.GENERIC_TESTING_QUERIES:
            return self.GENERIC_TESTING_QUERIES[normalized_query]

        tokens = normalized_query.split()

        if len(tokens) <= 2 and "testing" in tokens:
            return f"{normalized_query} basics definition quality assurance defects"

        return normalized_query

    def process(self, query: str) -> str:
        return self.expand_if_needed(query)
