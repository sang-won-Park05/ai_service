# rag/embed_documents.py
"""
merged_all.jsonl 전체를 임베딩해서
embedded_all.jsonl 로 저장하는 스크립트
"""

import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# =========================
# 경로 & 환경 변수 로드
# =========================
BASE_DIR = Path(__file__).resolve().parent          # .../ai_service/rag
DATA_DIR = BASE_DIR / "data"
INPUT_PATH = DATA_DIR / "merged_all.jsonl"
OUTPUT_PATH = DATA_DIR / "embedded_all.jsonl"       # ✅ 전체용 출력 파일

# .env 로드 (루트에 있다고 가정: .../ai_service/.env)
PROJECT_ROOT = BASE_DIR  # rag 바로 위가 ai_service 니까 이대로 써도 됨
env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    load_dotenv(env_path)

# OpenAI 클라이언트 (환경변수 OPENAI_API_KEY 사용)
client = OpenAI()

EMBED_MODEL = "text-embedding-3-large"

# MAX_DOCS = None 이면 전체 처리, 숫자를 넣으면 앞에서 그 개수만 처리
MAX_DOCS = None   # ✅ 전체 데이터 돌리려면 None, 테스트는 100 이런 식으로 바꿔도 됨


def build_text_to_embed(doc: dict) -> str:
    """
    한 문서(dict)에서 임베딩에 쓸 텍스트를 합쳐서 만든다.
    없는 필드는 무시하고, 있는 것만 이어 붙임.
    """
    parts = []

    for key in [
        "title",
        "content",
        "drug_name_kor",
        "drug_name_eng",
        "disease_name_kor",
        "disease_name_eng",
        "excipients",
        "topic",
        "departments",
        "entity_1",
        "entity_2",
    ]:
        value = doc.get(key)
        if value:
            parts.append(str(value))

    # 혹시 아무것도 없으면 id라도 넣어서 빈 문자열은 피함
    if not parts:
        parts.append(str(doc.get("id", "")))

    return "\n".join(parts)


def embed_text(text: str):
    """
    OpenAI 임베딩 호출. text-embedding-3-large 사용.
    """
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=text,
    )
    # 첫 번째 벡터만 사용
    return resp.data[0].embedding


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {INPUT_PATH}")

    print(f"📄 입력 파일: {INPUT_PATH}")
    print(f"📝 출력 파일: {OUTPUT_PATH}")
    print(f"🔢 최대 문서 수: {MAX_DOCS if MAX_DOCS is not None else '전체'}")

    processed = 0
    with INPUT_PATH.open("r", encoding="utf-8") as f_in, \
         OUTPUT_PATH.open("w", encoding="utf-8") as f_out:

        for idx, line in enumerate(f_in):
            if MAX_DOCS is not None and idx >= MAX_DOCS:
                break

            line = line.strip()
            if not line:
                continue

            doc = json.loads(line)

            text = build_text_to_embed(doc)
            print(f"[{idx+1}] 임베딩 생성 중... (길이 {len(text)} 글자)")

            embedding = embed_text(text)
            doc["embedding"] = embedding  # 벡터 필드 추가

            # 새 JSONL로 저장
            f_out.write(json.dumps(doc, ensure_ascii=False) + "\n")
            processed += 1

    print(f"✅ 완료! {processed}개 문서를 {OUTPUT_PATH.name} 에 저장했습니다.")


if __name__ == "__main__":
    main()
