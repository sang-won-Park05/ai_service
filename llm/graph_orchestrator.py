# ai_service/llm/utils/graph_orchestrator.py
from typing import Dict, Any, List, Optional

from .orchestrator import run_chat_rag


def run_chat_flow(
    query: str,
    user_id: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
    user_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    나중에 여러 노드(분류 → RAG → 툴콜 등)를 연결하고 싶으면
    여기서 graph-style로 orchestration.
    지금은 단일 RAG 파이프라인만 래핑.
    """
    return run_chat_rag(
        query=query,
        user_id=user_id,
        history=history,
        user_profile=user_profile,
    )
