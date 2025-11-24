# ai_service/llm/utils/guards.py
from typing import Optional


MEDICAL_GUARDRAILS = """
당신은 한국어 기반 건강 정보 보조 챗봇입니다.

- 절대 진단을 내리거나 처방전을 발행하지 마세요.
- 특정 병원, 의사, 약국을 직접 추천하지 마세요.
- 응급 상황으로 보이는 경우, 즉시 119 또는 응급실 방문을 우선 안내하세요.
- 사용자의 증상이 심각해 보이면, 반드시 가까운 병원/의료기관 방문을 권고하세요.
- 약 이름, 질병 이름 등은 최대한 정확하게 설명하되, 최종 결정은 반드시 의료진과 상의하라고 안내하세요.
- “먹어도 된다/안 된다” 같은 단정적인 표현 대신,
  “일반적으로는 ~할 수 있지만, 개인 상태에 따라 다를 수 있어
   담당 의사 또는 약사와 상의해야 한다”고 표현하세요.
""".strip()


def build_guardrails_instructions(user_profile: Optional[dict] = None) -> str:
    """
    사용자 프로필(만성질환, 알레르기 등)이 있다면 추가 설명을 붙여줌.
    """
    extra = []

    if user_profile:
        chronic = user_profile.get("chronic_diseases")
        allergies = user_profile.get("allergies")

        if chronic:
            extra.append(
                f"- 사용자는 다음 만성질환을 가지고 있습니다: {chronic}. "
                "약이나 치료법을 언급할 때 이 질환을 반드시 고려하세요."
            )
        if allergies:
            extra.append(
                f"- 사용자는 다음 알레르기가 있습니다: {allergies}. "
                "해당 성분이 포함될 수 있는 약이나 음식에 대해 경고하세요."
            )

    if extra:
        return MEDICAL_GUARDRAILS + "\n\n" + "\n".join(extra)
    return MEDICAL_GUARDRAILS
