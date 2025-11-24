# ai_service/llm/utils/prompts.py
from typing import List, Dict, Any, Optional

from .guards import build_guardrails_instructions
from .config import settings


def build_system_prompt(user_profile: Optional[dict] = None) -> str:
    """
    전체 시스템 프롬프트.
    """
    guard = build_guardrails_instructions(user_profile)

    base = f"""
당신은 '메디노트(MediNote)' 서비스용 한국어 건강 정보 어시스턴트입니다.

- 사용자의 질문을 이해하고, 제공된 '지식 문서'를 우선적으로 활용하여 답변하세요.
- 지식 문서에 없는 내용은 일반적인 의학 상식을 바탕으로 답하되,
  항상 '정확한 진단/치료는 의료진의 진료가 필요하다'는 점을 상기시키세요.
- 답변은 최대한 간결하게, 3~7문장 정도로 정리하고, 필요할 때만 bullet을 사용하세요.
- 전문 용어가 나오면, 일반 사용자가 이해할 수 있도록 짧게 설명을 덧붙이세요.
- 질문이 모호할 때는, 추가로 물어봐야 할 사항을 1~3개 정도 제안하세요.

다음은 안전 가이드라인입니다:
{guard}
    """.strip()

    return base


def build_context_block(documents: List[Dict[str, Any]]) -> str:
    """
    RAG 검색 결과를 LLM에게 넘겨줄 텍스트 블록으로 변환.
    """
    if not documents:
        return "관련된 지식 문서를 찾지 못했습니다."

    chunks = []
    for idx, doc in enumerate(documents, start=1):
        title = doc.get("title") or doc.get("id") or f"doc-{idx}"
        content = doc.get("content", "")
        doc_type = doc.get("doc_type") or "unknown"

        chunks.append(
            f"[문서 {idx}] (type={doc_type}, title={title})\n{content}\n"
        )

    return "\n\n".join(chunks)


def build_messages(
    system_prompt: str,
    history: List[Dict[str, str]],
    documents: List[Dict[str, Any]],
    user_query: str,
    safety_action: str | None = None,
) -> List[Dict[str, str]]:
    """
    OpenAI ChatCompletion용 messages 생성.
    history 형식: [{"role": "user"/"assistant", "content": "..."}]
    """
    context_block = build_context_block(documents)

    rag_instruction = f"""
다음은 검색된 지식 문서들입니다. 우선적으로 참고하세요:

{context_block}

위 문서들을 기반으로, 아래의 '사용자 질문'에 답변하세요.
    """.strip()

    if safety_action == "soft_warn":
        rag_instruction += (
            "\n\n주의: 질문이 약 처방/추천을 포함하고 있습니다. "
            "직접적인 복용 지시 대신, 일반적인 정보와 의료진 상담을 강조하세요."
        )

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": rag_instruction},
    ]

    messages.extend(history)
    messages.append({"role": "user", "content": user_query})

    return messages
