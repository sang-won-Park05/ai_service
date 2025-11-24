# ai_service/llm/utils/preprocess.py
import re
from typing import List, Dict, Any

from ..config import settings


def normalize_query(text: str) -> str:
    """
    사용자 질문 전처리: 공백/제어문자 정리 정도만 가볍게.
    """
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def trim_history(
    history: List[Dict[str, Any]], max_turns: int | None = None, max_chars: int | None = None
) -> List[Dict[str, Any]]:
    """
    대화 이력 길이 제한.
    - max_turns: 최근 N턴만 유지
    - max_chars: 전체 문자 수 기준으로 자르기
    """
    if not history:
        return []

    if max_turns is None:
        max_turns = settings.max_history_turns
    if max_chars is None:
        max_chars = settings.max_history_chars

    # 1) 최근 max_turns만 유지
    trimmed = history[-max_turns:]

    # 2) 문자 수 기준으로 뒤에서부터 잘라감
    total = 0
    kept: List[Dict[str, Any]] = []
    for msg in reversed(trimmed):
        content = msg.get("content", "")
        length = len(content)
        if total + length > max_chars:
            break
        kept.append(msg)
        total += length

    kept.reverse()
    return kept
