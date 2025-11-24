# ai_service/llm/embeddings.py
from functools import lru_cache
from typing import List

from openai import OpenAI
from .config import settings


@lru_cache()
def get_openai_client() -> OpenAI:
    """
    OpenAI 클라이언트 생성.
    - proxies 같은 인자는 절대 넣지 말 것 (openai>=1.0에서 지원 안 함)
    """
    return OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url or None,
    )


def embed_text(text: str) -> List[float]:
    """
    단일 문자열을 벡터로 변환.
    """
    if not text:
        return []

    client = get_openai_client()
    resp = client.embeddings.create(
        model=settings.openai_model_embedding,
        input=text,
    )
    return resp.data[0].embedding
