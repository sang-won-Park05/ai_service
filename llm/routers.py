# ai_service/llm/routers.py
from __future__ import annotations

from typing import Literal

RouteType = Literal["non_medical", "candidate_medical"]

# 완전한 잡담/인사에만 쓰는 키워드 (→ RAG 스킵)
SMALLTALK_KEYWORDS = [
    "하잉", "하이", "ㅎㅇ", "안녕", "안뇽",
    "hello", "hi", "hey", "ㅎㅇㅎㅇ",
]


def route_query(query: str) -> RouteType:
    """
    1차 라우터 (LLM 호출 없음)

    - non_medical:
        완전한 인사/잡담으로만 보이는 질문.
        → 이 루트로 가면 절대 '의료'로 승격되지 않는다.
        → 임베딩/검색(RAG)도 전혀 돌리지 않는다.

    - candidate_medical:
        그 외 모든 질문 (애매하거나 의료 가능성이 있는 것들).
        → RAG + 의료 프롬프트를 태우고,
          LLM이 최종적으로 의료/비의료 여부를 판단한다.
    """
    text = (query or "").strip().lower()
    if not text:
        return "candidate_medical"

    # 진짜 인사/잡담이면 non_medical
    if any(kw in text for kw in SMALLTALK_KEYWORDS):
        return "non_medical"

    # 나머지는 다 의료 후보
    return "candidate_medical"
