from langchain_huggingface import HuggingFaceEmbeddings
from src.config import settings


class EmbeddingManager:
    """Manages the lifecycle and loading of the embedding model."""

    def __init__(self) -> None:
        self.model_name = settings.local_embedding_model
        self.model_kwargs = {"device": "cpu"}
        self.encode_kwargs = {"normalize_embeddings": True}
        self._embeddings = None

    def get_embeddings(self) -> HuggingFaceEmbeddings:
        """Loads and returns the HuggingFaceEmbeddings instance."""
        if self._embeddings is None:
            print(f"Loading embedding model: {self.model_name}...")
            try:
                self._embeddings = HuggingFaceEmbeddings(
                    model_name=self.model_name,
                    model_kwargs=self.model_kwargs,
                    encode_kwargs=self.encode_kwargs,
                )
                print("✅ Embedding model loaded successfully.")
            except Exception as e:
                print(f"❌ Error loading embedding model '{self.model_name}': {e}")
                raise
        return self._embeddings
