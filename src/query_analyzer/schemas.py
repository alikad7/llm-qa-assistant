from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class QueryDomain(str, Enum):
    SOFTWARE_TESTING = "software_testing"
    OUT_OF_DOMAIN = "out_of_domain"
    UNCERTAIN = "uncertain"


class QueryIntent(str, Enum):
    DEFINITION = "definition"
    HOW_TO = "how_to"
    COMPARISON = "comparison"
    EXAMPLE = "example"
    TROUBLESHOOTING = "troubleshooting"
    BEST_PRACTICES = "best_practices"
    TOOLING = "tooling"
    GENERAL = "general"
    OUT_OF_DOMAIN = "out_of_domain"


class QueryAnalysis(BaseModel):
    original_query: str = Field(description="The original user query.")
    normalized_query: str = Field(description="Cleaned and normalized version of the query.")

    domain: QueryDomain = Field(description="Detected domain of the query.")
    intent: QueryIntent = Field(description="Detected intent of the query.")

    is_in_domain: bool = Field(description="Whether the query belongs to software testing or QA.")
    needs_retrieval: bool = Field(description="Whether retrieval should be executed.")

    bm25_query: str = Field(description="Keyword-focused query for BM25 retrieval.")
    semantic_query: str = Field(description="Meaning-focused query for vector retrieval.")

    reason: str = Field(description="Short reason for the classification.")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0 and 1.")

    language: Optional[str] = Field(default=None, description="Detected language, such as fa or en.")
