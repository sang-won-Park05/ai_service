# ai_service/llm/utils/orchestrator.py
from typing import List, Dict, Any, Optional

from .config import settings
from .utils.preprocess import normalize_query, trim_history
from .utils.safety import check_safety
from .retriever import retrieve_documents
from .reranker import rerank_documents
from .prompts import build_system_prompt, build_messages
from .utils.formatter import build_response_payload
from .telemetry import log_info, build_debug_snapshot
from .embeddings import get_openai_client


def run_chat_rag(
    query: str,
    user_id: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
    user_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    ai_service의 핵심 엔트리 포인트.
    - 질문 전처리
    - 안전 필터
    - OpenSearch 검색 + (옵션) 리랭크
    - 프롬프트 생성
    - OpenAI Chat 호출
    - 응답 포맷팅
    """
    history = history or []

    # 1) 전처리
    normalized_query = normalize_query(query)
    trimmed_history = trim_history(history)

    # 2) 안전 필터
    safety_action, safety_meta = check_safety(normalized_query)

    # 자살/폭력 등은 여기서 바로 차단 응답 리턴 가능
    if safety_action == "block":
        answer = (
            "죄송하지만, 현재 질문 내용은 안전 정책에 따라 직접 답변해 드리기 어렵습니다.\n"
            "위급하거나 위험한 상황이라면, 즉시 119 또는 가까운 응급실/상담 기관에 연락하시기 바랍니다."
        )
        debug = build_debug_snapshot(
            normalized_query=normalized_query,
            safety_action=safety_action,
            safety_meta=safety_meta,
            retrieved_docs=[],
        )
        return build_response_payload(answer, [], debug)

    # 3) RAG 검색
    raw_docs = retrieve_documents(normalized_query, top_k=settings.retriever_top_k)
    ranked_docs = rerank_documents(normalized_query, raw_docs, top_k=settings.retriever_top_k)

    # 4) 시스템 프롬프트 & messages 구성
    system_prompt = build_system_prompt(user_profile)
    messages = build_messages(
        system_prompt=system_prompt,
        history=trimmed_history,
        documents=ranked_docs,
        user_query=normalized_query,
        safety_action=safety_action,
    )

    # 5) OpenAI Chat 호출
    client = get_openai_client()
    completion = client.chat.completions.create(
        model=settings.openai_model_chat,
        messages=messages,
        temperature=0.2,
    )
    answer = completion.choices[0].message.content

    # 6) debug 스냅샷
    debug = build_debug_snapshot(
        normalized_query=normalized_query,
        safety_action=safety_action,
        safety_meta=safety_meta,
        retrieved_count=len(raw_docs),
        top_docs=[d.get("id") for d in ranked_docs],
    )

    log_info("chat_completed", user_id=user_id, retrieved=len(raw_docs))

    return build_response_payload(answer, ranked_docs, debug)
