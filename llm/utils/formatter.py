# ai_service/llm/utils/formatter.py
from typing import List, Dict, Any


def format_citations(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    프론트/메인 API에 내려줄 수 있는 citation 형식으로 축약.
    detail_url 도 포함.
    """
    citations = []
    for idx, doc in enumerate(documents, start=1):
        citations.append(
            {
                "rank": idx,
                "id": doc.get("id"),
                "title": doc.get("title"),
                "doc_type": doc.get("doc_type"),
                "score": doc.get("score"),
                "detail_url": doc.get("detail_url")  # 🔥 추가됨
                or doc.get("metadata", {}).get("detail_url"),
                "metadata": doc.get("metadata", {}),
            }
        )
    return citations


def build_response_payload(
    answer: str,
    documents: List[Dict[str, Any]],
    debug: Dict[str, Any] | None = None,
) -> Dict[str, Any]:

    # 🔥 top-1(가장 관련도 높은 문서) 출처 URL
    top_url = None
    if documents:
        top_url = (
            documents[0].get("detail_url")
            or documents[0].get("metadata", {}).get("detail_url")
        )

    # 🔥 answer 텍스트 끝에 출처 자동 첨부
    final_answer = answer
    if top_url:
        final_answer += f"\n\n**출처:** {top_url}"

    return {
        "answer": final_answer,
        "source_url": top_url,     # 🔥 추가됨 (프론트가 따로 쓸 수 있음)
        "citations": format_citations(documents),
        "debug": debug or {},
    }
