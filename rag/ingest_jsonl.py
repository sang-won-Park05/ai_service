# rag/ingest_jsonl.py
import json
from pathlib import Path
from llm.opensearch_client import get_opensearch_client

INDEX_NAME = "medinote_v3"

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# ✅ 전체 임베딩 결과 파일
INPUT_PATH = DATA_DIR / "embedded_all.jsonl"

# 한 번에 bulk로 보낼 문서 개수
BATCH_SIZE = 500  # 500~1000 선이면 적당


def bulk_ingest():
    client = get_opensearch_client()

    print(f"📄 입력 파일: {INPUT_PATH}")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"파일 없음: {INPUT_PATH}")

    with INPUT_PATH.open("r", encoding="utf-8") as f:
        batch_actions = []
        count = 0
        batch_count = 0

        for line in f:
            line = line.strip()
            if not line:
                continue

            doc = json.loads(line)

            # 서버리스: _id 사용 금지. 그냥 index만.
            action = {
                "index": {
                    "_index": INDEX_NAME
                }
            }

            batch_actions.append(json.dumps(action))
            batch_actions.append(json.dumps(doc, ensure_ascii=False))
            count += 1

            # 배치 사이즈에 도달하면 한번 전송
            if len(batch_actions) >= BATCH_SIZE * 2:
                batch_count += 1
                print(f"🚀 배치 {batch_count} 업로드 중... (누적 {count}개 문서)")

                payload = "\n".join(batch_actions) + "\n"
                resp = client.transport.perform_request(
                    method="POST",
                    url=f"/{INDEX_NAME}/_bulk",
                    body=payload,
                    headers={"Content-Type": "application/json"}
                )

                if resp.get("errors"):
                    print("⚠ 일부 문서에서 오류 발생:", resp)
                batch_actions = []

        # 남은 문서 flush
        if batch_actions:
            batch_count += 1
            print(f"🚀 마지막 배치 {batch_count} 업로드 중... (총 {count}개 문서)")
            payload = "\n".join(batch_actions) + "\n"
            resp = client.transport.perform_request(
                method="POST",
                url=f"/{INDEX_NAME}/_bulk",
                body=payload,
                headers={"Content-Type": "application/json"}
            )
            if resp.get("errors"):
                print("⚠ 일부 문서에서 오류 발생:", resp)

    print(f"✅ 전체 업로드 완료! 총 {count}개 문서 적재")


if __name__ == "__main__":
    bulk_ingest()
