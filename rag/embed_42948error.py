# rag/embed_documents.py
"""
merged_all.jsonl 전체를 임베딩해서
embedded_all.jsonl 로 저장하는 스크립트
(이미 일부 임베딩된 경우, 이어서 재개)
"""

import json
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIError, PermissionDeniedError


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

# MAX_DOCS = None 이면 전체 처리
MAX_DOCS = None   # ✅ 전체 데이터 돌리려면 None, 테스트는 100 이런 식으로


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


def safe_embed_text(text: str):
    """
    OpenAI 임베딩 호출 + 간단한 재시도 로직.
    PermissionDenied(403, 쿼터/권한 문제)는 바로 raise.
    """
    max_retries = 5

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.embeddings.create(
                model=EMBED_MODEL,
                input=text,
            )
            return resp.data[0].embedding

        except RateLimitError as e:
            wait = 5 * attempt
            print(f"[RateLimit] {attempt}/{max_retries}회째, {wait}초 대기 후 재시도: {e}")
            time.sleep(wait)

        except PermissionDeniedError as e:
            # 보통 쿼터/권한 문제라 재시도해도 의미가 없는 경우가 많음
            print("\n[PermissionDenied] 403 오류 발생 (보통 쿼터/권한 문제)")
            print("➡ OpenAI 대시보드에서 사용량/제한을 먼저 확인해야 합니다.")
            raise

        except APIError as e:
            # 일시적인 서버 에러일 수 있으니 몇 번 재시도
            wait = 5 * attempt
            print(f"[APIError] {attempt}/{max_retries}회째, {wait}초 대기 후 재시도: {e}")
            time.sleep(wait)

    raise RuntimeError("임베딩 재시도 횟수 초과")


def count_already_processed() -> int:
    """
    OUTPUT_PATH(embedded_all.jsonl)에 이미 저장된 라인 수 = 완료된 문서 수
    """
    if not OUTPUT_PATH.exists():
        return 0

    count = 0
    with OUTPUT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {INPUT_PATH}")

    already = count_already_processed()

    print(f"📄 입력 파일: {INPUT_PATH}")
    print(f"📝 출력 파일: {OUTPUT_PATH}")
    print(f"✅ 이미 임베딩 완료된 문서 수: {already}개")
    print(f"🔢 이번 실행에서 처리할 최대 문서 수: {MAX_DOCS if MAX_DOCS is not None else '제한 없음'}")

    # MAX_DOCS가 설정된 경우, 전체 중 어디까지 할지 계산
    if MAX_DOCS is not None:
        target_total = already + MAX_DOCS
    else:
        target_total = None  # 끝까지

    processed_new = 0

    # 입력은 처음부터 읽되, already 개수만큼은 건너뛴 뒤부터 처리
    with INPUT_PATH.open("r", encoding="utf-8") as f_in, \
         OUTPUT_PATH.open("a", encoding="utf-8") as f_out:   # 🔥 append 모드!

        for idx, line in enumerate(f_in):
            # 이미 끝난 부분은 스킵
            if idx < already:
                continue

            # MAX_DOCS 제한이 있으면 거기까지만
            if target_total is not None and idx >= target_total:
                break

            line = line.strip()
            if not line:
                continue

            doc = json.loads(line)

            text = build_text_to_embed(doc)
            print(f"[{idx+1}] 임베딩 생성 중... (길이 {len(text)} 글자)")

            embedding = safe_embed_text(text)
            doc["embedding"] = embedding  # 벡터 필드 추가

            # 새 JSONL로 저장 (append)
            f_out.write(json.dumps(doc, ensure_ascii=False) + "\n")
            processed_new += 1

    print(f"🎉 이번 실행에서 새로 임베딩한 문서 수: {processed_new}개")
    print(f"📦 총 임베딩 완료 문서 수 (예상): {already + processed_new}개")


if __name__ == "__main__":
    main()
