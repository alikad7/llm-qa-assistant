from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_ROOT = PROJECT_ROOT / "models"

EMBEDDING_HF_ID = "sentence-transformers/ms-marco-MiniLM-L-6-v2"
RERANKER_HF_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"

EMBEDDING_LOCAL_PATH = MODELS_ROOT / "ms-marco-MiniLM-L-6-v2"
RERANKER_LOCAL_PATH = (
    MODELS_ROOT
    / "cross-encoder"
    / "ms-marco-MiniLM-L-6-v2"
)


def _is_sentence_transformer_model(path: Path) -> bool:
    """Check whether the path contains a Sentence Transformer model."""
    return (
        path.is_dir()
        and (path / "modules.json").is_file()
        and (
            (path / "config_sentence_transformers.json").is_file()
            or (path / "sentence_bert_config.json").is_file()
        )
    )


def _is_cross_encoder_model(path: Path) -> bool:
    """Check whether the path contains a Cross-Encoder model."""
    return (
        path.is_dir()
        and (path / "config.json").is_file()
        and (
            (path / "model.safetensors").is_file()
            or (path / "pytorch_model.bin").is_file()
        )
    )


def resolve_embedding_model() -> str:
    """Use the local embedding model when it is available."""
    if _is_sentence_transformer_model(EMBEDDING_LOCAL_PATH):
        return str(EMBEDDING_LOCAL_PATH)

    return EMBEDDING_HF_ID


def resolve_reranker_model() -> str:
    """Use the local reranker model when it is available."""
    if _is_cross_encoder_model(RERANKER_LOCAL_PATH):
        return str(RERANKER_LOCAL_PATH)

    return RERANKER_HF_ID


class Settings(BaseSettings):
    # GapGPT API
    gapgpt_api_key: str
    gapgpt_base_url: str = "https://api.gapgpt.app/v1"

    # Generation and API embedding models
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # Local models
    local_embedding_model: str = resolve_embedding_model()
    local_reranker_model: str = resolve_reranker_model()

    # Cross-encoder reranker settings
    reranker_batch_size: int = 16
    reranker_device: str | None = None
    reranker_normalize_scores: bool = True

    # MMR post-processing parameters
    mmr_lambda_param: float = 0.7
    mmr_min_score_threshold: float = 0.10
    mmr_max_context_chars: int | None = 6000
    mmr_deduplicate_by_content: bool = True

    # Persisted index paths
    faiss_index_path: str = str(
        PROJECT_ROOT / "data" / "indexes" / "faiss_index"
    )
    bm25_index_dir: str = str(
        PROJECT_ROOT / "data" / "indexes" / "bm25"
    )

    # Retrieval parameters
    retrieval_top_k: int = 5
    retrieval_candidate_k: int = 20

    # Context construction settings
    context_max_chars: int = 6000
    context_include_scores: bool = True

    # Answer generation parameters
    generation_temperature: float = 0.2
    generation_max_tokens: int = 700
    generation_timeout_seconds: int = 120

    default_answer_language: str = "en"

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
