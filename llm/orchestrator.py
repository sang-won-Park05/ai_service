from __future__ import annotations

from typing import Any, Dict, List, Optional

from openai import OpenAI

from .config import settings
from .utils.preprocess import normalize_query, trim_history
from .utils.safety import check_safety
from .utils.formatter import build_response_payload
from .utils.user_profile import load_user_profile  # ✅ DB에서 프로필 로드
from .retriever import retrieve_documents
from .reranker import rerank_documents
from .prompts import build_system_prompt, build_messages  # ✅ 중요
from .telemetry import log_info, build_debug_snapshot
from .embeddings import get_openai_client
from .routers import route_query, RouteType

NON_MEDICAL_TAG = "[NON_MEDICAL]"


def _call_llm(messages: List[Dict[str, str]]) -> str:
    client: OpenAI = get_openai_client()
    resp = client.chat.completions.create(
        model=settings.openai_model_chat,
        messages=messages,
    )
    return resp.choices[0].message.content or ""


def run_chat_rag(
    query: str,
    user_id: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
    user_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    메인 오케스트레이터.

    1) 1차 라우터(route_query): LLM 호출 없이 코드로 분기
       - "non_medical"       → 완전 비의료 플로우 (RAG/출처 X, LLM 1번)
       - "candidate_medical" → 의료/애매 플로우 (RAG + LLM 1번)

    2) candidate_medical 루트에서:
       - LLM이 답변 앞에 [NON_MEDICAL] 태그를 붙이면,
         최종적으로 비의료로 간주하고 문서/출처를 사용하지 않는다.
    """
    history = history or []

    normalized_query = normalize_query(query)
    trimmed_history = trim_history(history)

    if not normalized_query:
        return build_response_payload("질문을 입력해 주세요.", [], debug={})

    # 0) 안전 필터
    check_safety(normalized_query)

    # 0.5) 사용자 프로필 자동 로딩 (user_profile이 없고 user_id만 들어온 경우)
    if user_profile is None and user_id:
        try:
            try:
                uid_for_db: int | str = int(user_id)
            except ValueError:
                uid_for_db = user_id  # 문자열 ID도 허용
            user_profile = load_user_profile(uid_for_db)
        except Exception as e:
            log_info("user_profile_load_failed", user_id=user_id, error=str(e))
            user_profile = None

    # 1) 1차 라우터
    route: RouteType = route_query(normalized_query)

    # ---------- A. 완전 비의료 루트 (RAG/출처 X) ----------
    if route == "non_medical":
        system_prompt = build_system_prompt(is_medical_mode=False)
        messages = build_messages(
            system_prompt=system_prompt,
            query=normalized_query,
            history=trimmed_history,
            documents=None,
            # ✅ 비의료 모드에서도 이름/기초 정보는 활용
            user_profile=user_profile,
        )
        answer = _call_llm(messages)

        debug = build_debug_snapshot(
            query=normalized_query,
            route="non_medical_router",
            router_result=route,
            is_medical_final=False,
            raw_docs=[],
            ranked_docs=[],
        )
        log_info("chat_completed", user_id=user_id, retrieved=0)
        return build_response_payload(answer, [], debug=debug)

    # ---------- B. 의료/애매 루트 (RAG + LLM) ----------
    # 여기로 오는 건 "candidate_medical" 뿐
    raw_docs = retrieve_documents(normalized_query, top_k=settings.retriever_top_k)
    ranked_docs = rerank_documents(normalized_query, raw_docs)

    system_prompt = build_system_prompt(is_medical_mode=True)
    messages = build_messages(
        system_prompt=system_prompt,
        query=normalized_query,
        history=trimmed_history,
        documents=ranked_docs,
        user_profile=user_profile,
    )
    answer_raw = _call_llm(messages)

    # 2차 판단: LLM이 [NON_MEDICAL] 태그로 비의료라고 선언했는지 확인
    is_medical_final = True
    documents_for_answer: List[Dict[str, Any]] = ranked_docs

    answer = answer_raw
    if answer_raw.startswith(NON_MEDICAL_TAG):
        is_medical_final = False
        # 태그 제거 + 앞 공백 제거
        answer = answer_raw[len(NON_MEDICAL_TAG) :].lstrip()
        documents_for_answer = []  # 최종 비의료 → RAG/출처 완전히 제거

    debug = build_debug_snapshot(
        query=normalized_query,
        route="candidate_medical",
        router_result=route,
        is_medical_final=is_medical_final,
        raw_docs=raw_docs,
        ranked_docs=ranked_docs,
    )
    log_info(
        "chat_completed",
        user_id=user_id,
        retrieved=len(ranked_docs) if is_medical_final else 0,
    )

    return build_response_payload(answer, documents_for_answer, debug=debug)
