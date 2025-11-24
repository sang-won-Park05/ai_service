# ai_service/llm/utils/config.py
import os
from pathlib import Path
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parents[2]  # ai_service/ 기준 루트

class Settings(BaseModel):
    # OpenAI
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str | None = os.getenv("OPENAI_BASE_URL") or None
    openai_model_chat: str = os.getenv("OPENAI_MODEL_CHAT", "gpt-4o-mini")
    openai_model_embedding: str = os.getenv(
        "OPENAI_MODEL_EMBEDDING", "text-embedding-3-large"
    )

    # OpenSearch
    opensearch_index: str = os.getenv("OPENSEARCH_INDEX", "medinote_v3")
    opensearch_vector_field: str = os.getenv("OPENSEARCH_VECTOR_FIELD", "embedding")
    opensearch_content_field: str = os.getenv("OPENSEARCH_CONTENT_FIELD", "content")

    # RAG / 챗봇
    retriever_top_k: int = int(os.getenv("RETRIEVER_TOP_K", "5"))
    max_history_turns: int = int(os.getenv("MAX_HISTORY_TURNS", "8"))
    max_history_chars: int = int(os.getenv("MAX_HISTORY_CHARS", "4000"))
    max_context_chars: int = int(os.getenv("MAX_CONTEXT_CHARS", "8000"))

    # 기타
    env: str = os.getenv("APP_ENV", "local")


settings = Settings()
