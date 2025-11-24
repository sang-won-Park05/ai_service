# ai_service/llm/utils/reranker.py
from typing import List, Dict, Any


def rerank_documents(
    query: str,
    documents: List[Dict[str, Any]],
    top_k: int | None = None,
) -> List[Dict[str, Any]]:
    """
    최소 구현: OpenSearch 점수(_score) 기반으로 정렬만 수행.
    나중에 BGE/e5, LLM re-ranker 붙이고 싶으면 여기 확장하면 됨.
    """
    if not documents:
        return []

    sorted_docs = sorted(documents, key=lambda d: d.get("score", 0.0), reverse=True)
    if top_k is not None:
        sorted_docs = sorted_docs[:top_k]
    return sorted_docs
