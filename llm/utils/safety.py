# ai_service/llm/utils/safety.py
import re
from typing import Tuple, Literal, Dict, Any

SafetyAction = Literal["allow", "soft_warn", "block"]


SUICIDE_KEYWORDS = ["자살", "죽고싶", "목숨을 끊", "극단적 선택"]
VIOLENCE_KEYWORDS = ["죽여버려", "테러", "폭탄", "총 만드는 법"]


def _matches_any(text: str, keywords: list[str]) -> bool:
    return any(k in text for k in keywords)


def check_safety(text: str) -> Tuple[SafetyAction, Dict[str, Any]]:
    """
    매우 단순한 룰 베이스 1차 필터.
    나중에 OpenAI Moderation이나 Guardrails를 추가적으로 붙이면 됨.
    """
    t = text or ""
    t = t.lower()

    if _matches_any(t, SUICIDE_KEYWORDS):
        return "block", {"reason": "self_harm"}
    if _matches_any(t, VIOLENCE_KEYWORDS):
        return "block", {"reason": "violence"}

    # 의료 관련해서는 일단 soft_warn을 쓰고, 본문에서 디클레이머를 추가로 붙이는 방식으로 처리 가능
    if re.search(r"(처방해줘|약 추천|어떤 약|무슨 약)", t):
        return "soft_warn", {"reason": "medical_prescription"}

    return "allow", {}
