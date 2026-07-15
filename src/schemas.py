from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field


RetrieverType = Literal["dense", "sparse", "hybrid"]
AnswerLanguage = Literal["fa", "en"]


class RetrievalDebugInfo(BaseModel):
    retriever: RetrieverType
    fusion_method: Optional[str] = None
    rank: Optional[int] = None
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    dense_score: Optional[float] = None
    dense_distance: Optional[float] = None
    bm25_score: Optional[float] = None
    rrf_score: Optional[float] = None


class RetrievalResult(BaseModel):
    id: str
    content: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    retrieval: RetrievalDebugInfo


class AnswerSource(BaseModel):
    source_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QuestionAnswerRequest(BaseModel):
    question: str
    answer_language: AnswerLanguage = "fa"
    top_k: int = 5
    candidate_k: int = 20


class QuestionAnswerResponse(BaseModel):
    question: str
    answer: str
    answer_language: AnswerLanguage
    sources: list[AnswerSource] = Field(default_factory=list)
