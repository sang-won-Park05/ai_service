# tests/test_rag_real_queries.py

"""
실제 사용자 질문을 가정한 RAG 통합 테스트:
1) 대표 질문 리스트 준비
2) 질문 임베딩 (text-embedding-3-large)
3) OpenSearch KNN 검색 (medinote_v3)
4) 검색된 상위 문서들로 컨텍스트 구성
5) GPT_API(gpt-4o-mini 등)로 최종 답변 생성
"""

import os
import json
from pathlib import Path
from pprint import pprint

from dotenv import load_dotenv
from openai import OpenAI
from llm.opensearch_client import get_opensearch_client

# =========================
# 환경 변수 및 클라이언트 설정
# =========================
ROOT = Path(__file__).resolve().parents[1]  # ai_service/
env_path = ROOT / ".env"
if env_path.exists():
    load_dotenv(env_path)

OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX", "medinote_v3")
EMBED_MODEL = "text-embedding-3-large"
CHAT_MODEL = os.getenv("OPENAI_API_MODEL", "gpt-4o-mini")

oai = OpenAI()
os_client = get_opensearch_client()


# =========================
# 유틸 함수들
# =========================
def embed(text: str):
    """질문을 벡터로 변환"""
    resp = oai.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return resp.data[0].embedding


def knn_search(query_vec, k: int = 5):
    """OpenSearch KNN 검색"""
    body = {
        "size": k,
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_vec,
                    "k": k
                }
            }
        }
    }

    resp = os_client.transport.perform_request(
        method="POST",
        url=f"/{OPENSEARCH_INDEX}/_search",
        body=json.dumps(body),
        headers={"Content-Type": "application/json"}
    )

    return resp.get("hits", {}).get("hits", [])


def build_context_from_hits(hits, max_chars: int = 2000) -> str:
    """
    검색 결과(hit)들을 하나의 큰 컨텍스트 텍스트로 정리.
    너무 길어지지 않도록 max_chars로 자름.
    """
    chunks = []
    for i, hit in enumerate(hits, start=1):
        src = hit.get("_source", {})
        score = hit.get("_score", 0.0)

        doc_id = src.get("id", "")
        title = src.get("title", "")
        # 약 / 질병 / 건강기능식품 등 이름 후보들
        drug_name = (
            src.get("drug_name_kor")
            or src.get("disease_name_kor")
            or src.get("supplement_name_kor")
            or ""
        )
        content = src.get("content", "") or src.get("raw_text", "")

        block = [
            f"[문서 {i}] (score={score})",
            f"id: {doc_id}",
            f"이름: {drug_name}",
        ]
        if title:
            block.append(f"제목: {title}")

        block.append("내용:")
        block.append(content)
        block.append("")  # 빈 줄

        chunks.append("\n".join(block))

    full_text = "\n".join(chunks)

    # 너무 길면 앞부분만 자르기
    if len(full_text) > max_chars:
        full_text = full_text[:max_chars] + "\n... (이하 생략)"

    return full_text


def call_llm(question: str, context: str) -> str:
    """
    GPT_API를 호출해서 최종 답변 생성.
    """
    system_prompt = (
        "당신은 한국어로 답변하는 의약품·건강 정보 어시스턴트입니다.\n"
        "아래 '검색된 문서'에 포함된 내용만 최대한 활용해서 답변하세요.\n"
        "사용자가 약 복용법, 효능, 주의사항을 묻는 경우:\n"
        "- 문서에 있는 정보를 기반으로 정리해서 설명합니다.\n"
        "- 정확한 진단이나 처방은 내리지 말고, 필요한 경우 의사·약사와 상담하라고 안내하세요.\n"
        "- 위험해 보이는 증상(심한 통증, 호흡곤란, 의식저하 등)이 포함된 질문이면 응급실 방문이나 긴급 진료를 권고하는 멘트를 포함합니다.\n"
        "답변은 존댓말로, 친절하고 이해하기 쉽게 작성하세요.\n"
        "문서에 정보가 없거나 불충분하면, 해당 점을 솔직하게 말하고 일반적인 주의사항 중심으로만 안내하세요."
    )

    user_content = (
        "다음은 사용자의 질문과 검색된 문서 목록입니다.\n\n"
        f"[사용자 질문]\n{question}\n\n"
        f"[검색된 문서]\n{context}\n\n"
        "위 정보를 바탕으로 사용자의 질문에 답변해 주세요."
    )

    resp = oai.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.2,
        max_tokens=800,
    )

    return resp.choices[0].message.content.strip()


# =========================
# 대표 테스트 질문 세트
# =========================

REAL_TEST_QUERIES = [
    # 1. 편의점/일반의약품 + 증상
    "속쓰림이 있을 때 편의점에서 살 수 있는 일반의약품이 뭐가 있나요?",
    "두통이 계속 있을 때 약국에서 살 수 있는 대표적인 두통약이 뭐가 있는지 알려주세요.",
    "감기 기운이 있을 때 종합감기약을 고를 때 주의할 점이 있을까요?",

    # 2. 특정 약 이름 + 복용법/주의사항
    "타이레놀 정제는 하루에 최대 몇 알까지 먹어도 되나요?",
    "리바로정 같은 콜레스테롤 약을 복용할 때 피해야 하는 음식이 있나요?",
    "고혈압 약을 아침에 먹는게 좋은지, 저녁에 먹는게 좋은지 궁금해요.",

    # 3. 약-약 상호작용
    "이부프로펜이랑 아스피린을 같이 먹어도 괜찮은가요?",
    "우울증 약을 복용 중인데, 감기약과 같이 먹어도 되는지 궁금합니다.",
    "혈액희석제를 먹고 있는 사람이 진통제를 복용할 때 주의할 점이 있나요?",

    # 4. 약-음식 / 알코올 상호작용
    "항생제를 먹는 동안 술을 마시면 안 되는 이유가 뭔가요?",
    "위장약을 복용할 때 카페인(커피)을 피해야 하는지 알려주세요.",
    "갑상선 호르몬제를 먹을 때 우유나 유제품을 같이 먹어도 되는지 궁금해요.",

    # 5. 건강기능식품/영양제
    "비타민 D 보충제를 언제, 어떻게 먹는게 흡수에 더 좋나요?",
    "오메가3와 혈액응고에 관련된 주의사항이 있나요?",
    "멀티비타민이랑 유산균을 같이 먹어도 괜찮은지 알려주세요.",

    # 6. 질병·증상 설명 + 1차 안내
    "역류성 식도염일 때 피해야 할 음식과 생활 습관을 알려주세요.",
    "당뇨병이 있는 사람이 감기약을 고를 때 주의할 점이 있나요?",
    "고지혈증 약을 복용 중인데 근육통이 심해졌다면 어떻게 해야 하나요?",

    # 7. 조금 애매한 일반 질문
    "요즘 머리가 자주 아픈데, 어떤 경우에 병원에 가봐야 할까요?",
    "감기랑 독감은 증상이 어떻게 다른지 간단히 설명해 주세요.",
]


# =========================
# 메인 테스트 함수
# =========================
def run_rag_tests(queries=None, k: int = 5):
    """
    여러 개의 실제 질문을 돌리면서
    1) 검색이 잘 되는지
    2) 컨텍스트가 이상하지 않은지
    3) 답변이 말이 되는지
    눈으로 확인하기 위한 통합 테스트 함수
    """
    if queries is None:
        queries = REAL_TEST_QUERIES

    print("\n==============================")
    print("🧪 RAG REAL QUERY TEST")
    print("==============================")
    print(f"테스트 질문 개수: {len(queries)}")
    print(f"사용 인덱스: {OPENSEARCH_INDEX}")
    print("==============================\n")

    for idx, question in enumerate(queries, start=1):
        print("\n" + "=" * 60)
        print(f"📝 질문 #{idx}")
        print("=" * 60)
        print("Q:", question)

        # 1) 임베딩
        qvec = embed(question)
        print(f"\n[1] 임베딩 벡터 길이: {len(qvec)}")

        # 2) 검색
        hits = knn_search(qvec, k=k)
        print(f"[2] 검색 결과 개수: {len(hits)}")

        print("\n[🔍 Top-{} 검색 결과 요약]".format(k))
        if not hits:
            print("검색 결과가 없습니다. 인덱스/매핑/데이터를 확인하세요.")
            continue

        for i, hit in enumerate(hits, start=1):
            src = hit.get("_source", {})
            print(f"\n--- #{i} (score={hit.get('_score')}) ---")
            print("id:", src.get("id"))
            print("drug_name_kor:", src.get("drug_name_kor"))
            print("disease_name_kor:", src.get("disease_name_kor"))
            print("supplement_name_kor:", src.get("supplement_name_kor"))
            preview_src = src.get("content") or src.get("raw_text") or ""
            preview = (preview_src[:200] + "..." if len(preview_src) > 200 else preview_src)
            print("content:", preview.replace("\n", " ")[:200])

        # 3) 컨텍스트 생성
        context = build_context_from_hits(hits)

        # 4) LLM 호출
        print("\n[3] GPT_API 호출 중...\n")
        answer = call_llm(question, context)

        print("💬 [최종 답변]\n")
        print(answer)
        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    # 1) 미리 정의한 REAL_TEST_QUERIES 전부 테스트
    run_rag_tests()

    # 2) 원하면 아래처럼 즉석에서 한 번 더 테스트할 수도 있음
    # custom_q = "여기에 직접 질문을 넣어보세요."
    # run_rag_tests([custom_q])
