from __future__ import annotations

from typing import List, Dict, Any, Optional

from .guards import build_guardrails_instructions
from .config import settings


def build_system_prompt(is_medical_mode: bool) -> str:
    """
    is_medical_mode=True  : 의료/RAG 모드
    is_medical_mode=False : 비의료/잡담 모드 (RAG 사용 X)
    """
    if is_medical_mode:
        base = (
            "너는 '메디노트(MediNote)' 서비스의 건강·의약 정보 챗봇이다. "
            "질병, 증상, 약물, 검사 결과 등 건강 관련 질문에 대해 "
            "제공된 참고 문서(의약품, 질병, 상호작용 데이터)와 사용자의 건강 프로필을 바탕으로 "
            "한국어로 이해하기 쉽게 설명한다.\n\n"
            "단, 너는 의사가 아니므로 진단이나 처방을 직접 내리지 말고, "
            "언제나 '의학적 판단은 반드시 의료진과 상의해야 한다'는 톤을 유지해라.\n\n"
            "질문이 건강/의학과 전혀 상관없는 일반 대화(인사, 잡담, 게임, 생활 팁 등)라고 판단되면 "
            "참고 문서와 사용자 건강 정보를 모두 무시하고, 일반 챗봇처럼 대답해라. "
            "이 경우 답변의 맨 앞에 반드시 '[NON_MEDICAL]' 태그를 붙여라. "
            "예: '[NON_MEDICAL] 안녕! 반가워, 나는 메디노트야.'\n\n"
            "반대로 건강·의학 관련 질문이라고 판단되면 '[NON_MEDICAL]' 태그 없이 "
            "의료 정보 챗봇으로서 답변해라. "
            "이때 사용자의 만성질환, 알레르기, 생활습관 정보를 가능하면 고려해라."
        )
    else:
        base = (
            "너는 '메디노트(MediNote)' 서비스의 일반 대화용 챗봇이다. "
            "사용자가 인사하거나 잡담을 하면 가볍고 친근하게 응답해라. "
            "의학적 조언, 약 추천, 병원/의사 추천은 하지 말고, "
            "건강 상담이 필요해 보이는 질문에는 "
            "'이건 건강 관련 상담이 필요해 보인다'고만 간단히 안내해라. "
            "이 모드에서는 참고 문서(RAG)를 사용하지 않는다. "
            "대신, 제공된 사용자 프로필에 이름이 있다면 예를 들어 '홍길동님'처럼 "
            "이름을 불러주며 대답해라."
        )

    base += "\n\n" + build_guardrails_instructions()
    return base


def _build_user_profile_summary(user_profile: Dict[str, Any]) -> str:
    """
    PostgreSQL에서 읽어온 user_profile(dict)를 사람이 읽기 좋은 한국어 요약으로 변환.
    user_profile 구조 예시:
      {
        "user_id": 1,
        "basic": {...},
        "chronic_diseases": [{...}, ...],
        "allergies": [{...}, ...],
      }
    """
    if not user_profile:
        return "사용자 건강 정보가 등록되어 있지 않다."

    basic = user_profile.get("basic") or {}
    chronic = user_profile.get("chronic_diseases") or []
    allergies = user_profile.get("allergies") or []

    # ✅ 이름
    full_name = basic.get("full_name") or basic.get("name") or "미등록"

    # 기본 정보
    birth = (
        basic.get("birthdate")
        or basic.get("date_of_birth")
        or basic.get("birth_date")
        or "미등록"
    )
    gender = basic.get("gender") or basic.get("sex") or "미등록"
    blood = basic.get("blood_type") or "미등록"
    height = basic.get("height_cm")
    weight = basic.get("weight_kg")
    smoke = basic.get("smoking_status") or "미등록"
    drink = basic.get("drinking_status") or "미등록"

    hw_str = ""
    if height:
        hw_str += f"{height}cm"
    if weight:
        if hw_str:
            hw_str += ", "
        hw_str += f"{weight}kg"
    if not hw_str:
        hw_str = "미등록"

    # 만성질환 요약
    if chronic:
        chronic_list = []
        for d in chronic:
            name = d.get("disease_name") or "이름 미상 질환"
            main_med = d.get("main_medication")
            is_active = d.get("is_active")
            status = "현재 치료 중" if is_active in (True, "Y", "y", 1) else "과거 병력"
            if main_med:
                chronic_list.append(f"{name} ({status}, 주요 약: {main_med})")
            else:
                chronic_list.append(f"{name} ({status})")
        chronic_str = "; ".join(chronic_list)
    else:
        chronic_str = "없음 또는 미등록"

    # 알레르기 요약
    if allergies:
        allergy_list = []
        for a in allergies:
            allergen = a.get("allergen_name") or "원인 미상"
            a_type = a.get("allergy_type")
            severity = a.get("severity")
            piece = allergen
            extra = []
            if a_type:
                extra.append(a_type)
            if severity:
                extra.append(f"중증도: {severity}")
            if extra:
                piece += " (" + ", ".join(extra) + ")"
            allergy_list.append(piece)
        allergy_str = "; ".join(allergy_list)
    else:
        allergy_str = "없음 또는 미등록"

    summary = (
        f"- 이름: {full_name}\n"
        f"- 생년월일: {birth}\n"
        f"- 성별: {gender}\n"
        f"- 혈액형: {blood}\n"
        f"- 신체 정보: {hw_str}\n"
        f"- 흡연 여부: {smoke}\n"
        f"- 음주 여부: {drink}\n"
        f"- 만성질환: {chronic_str}\n"
        f"- 알레르기: {allergy_str}"
    )

    return summary


def build_messages(
    system_prompt: str,
    query: str,
    history: Optional[List[Dict[str, str]]] = None,
    documents: Optional[List[Dict[str, Any]]] = None,
    user_profile: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    """
    공통 messages 구성 함수.
    - history: [{"role": "...", "content": "..."}, ...] (Gradio와 동일 형식)
    - documents: 의료 모드에서만 전달. 비의료 모드면 None/[]. 
    - user_profile: PostgreSQL에서 불러온 사용자 건강 정보 dict
    """
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
    ]

    # 1) 기존 히스토리 추가
    if history:
        messages.extend(history)

    # 2) 사용자 건강 프로필을 system 메세지로 추가
    if user_profile:
        profile_summary = _build_user_profile_summary(user_profile)
        messages.append(
            {
                "role": "system",
                "content": (
                    "다음은 이 사용자의 건강 정보 요약이다. "
                    "약물에 대한 주의사항, 상호작용, 생활 습관 조언을 말할 때 "
                    "특히 이 정보를 고려해라. 단, 개인정보를 그대로 노출하지 말고 "
                    "설명에 필요한 정도로만 간접적으로 활용해라.\n\n"
                    f"{profile_summary}"
                ),
            }
        )

    # 3) RAG 컨텍스트(의약품/질병 문서) 추가
    if documents:
        context_blocks = []
        for i, doc in enumerate(documents, start=1):
            context_blocks.append(f"[{i}] {doc.get('content', '')}")
        context_text = "\n\n".join(context_blocks)
        messages.append(
            {
                "role": "system",
                "content": (
                    "다음은 참고용 의료/약 정보 문서들이다. "
                    "질문이 건강·의학 관련이라면 이 문서들을 참고하되, "
                    "그대로 복사하지 말고 이해하기 쉽게 요약해서 설명해라:\n\n"
                    f"{context_text}"
                ),
            }
        )

    # 4) 마지막에 사용자 질문
    messages.append({"role": "user", "content": query})
    return messages
