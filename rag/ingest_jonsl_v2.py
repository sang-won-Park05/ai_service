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

                try:
                    resp = client.transport.perform_request(
                        method="POST",
                        url=f"/{INDEX_NAME}/_bulk",
                        body=payload,
                        headers={"Content-Type": "application/json"}
                    )
                except Exception as e:
                    # HTTP 레벨 예외
                    print(f"❌ 배치 {batch_count} 업로드 중 예외 발생: {e}")
                    fail_path = DATA_DIR / f"failed_batch_exception_{batch_count}.jsonl"
                    print(f"⚠ 예외 발생 배치 문서들을 {fail_path} 에 저장합니다.")
                    with fail_path.open("w", encoding="utf-8") as f_fail:
                        # action/doc/action/doc 구조에서 doc 라인만 저장
                        for i in range(1, len(batch_actions), 2):
                            f_fail.write(batch_actions[i] + "\n")
                    raise  # 완전히 멈추고 원인 확인할 수 있게

                if resp.get("errors"):
                    print(f"⚠ 배치 {batch_count}에서 일부 문서 오류 발생")
                    items = resp.get("items", [])
                    fail_path = DATA_DIR / f"failed_docs_batch_{batch_count}.jsonl"
                    with fail_path.open("w", encoding="utf-8") as f_fail:
                        for i, item in enumerate(items):
                            op, result = next(iter(item.items()))
                            if "error" in result:
                                err = result["error"]
                                # 어떤 에러인지 콘솔에 표시
                                print(
                                    f"  - 문서 #{i} 실패: type={err.get('type')} "
                                    f"reason={err.get('reason')}"
                                )
                                # 해당 문서 원본(JSONL) 저장
                                doc_line_index = i * 2 + 1  # action/doc/action/doc...
                                if doc_line_index < len(batch_actions):
                                    f_fail.write(batch_actions[doc_line_index] + "\n")
                    print(f"⚠ 실패 문서들은 {fail_path} 에 저장되었습니다.")

                batch_actions = []

        # 남은 문서 flush
        if batch_actions:
            batch_count += 1
            print(f"🚀 마지막 배치 {batch_count} 업로드 중... (총 {count}개 문서)")
            payload = "\n".join(batch_actions) + "\n"

            try:
                resp = client.transport.perform_request(
                    method="POST",
                    url=f"/{INDEX_NAME}/_bulk",
                    body=payload,
                    headers={"Content-Type": "application/json"}
                )
            except Exception as e:
                print(f"❌ 마지막 배치 {batch_count} 업로드 중 예외 발생: {e}")
                fail_path = DATA_DIR / f"failed_batch_exception_{batch_count}.jsonl"
                print(f"⚠ 예외 발생 배치 문서들을 {fail_path} 에 저장합니다.")
                with fail_path.open("w", encoding="utf-8") as f_fail:
                    for i in range(1, len(batch_actions), 2):
                        f_fail.write(batch_actions[i] + "\n")
                raise

            if resp.get("errors"):
                print(f"⚠ 마지막 배치 {batch_count}에서 일부 문서 오류 발생")
                items = resp.get("items", [])
                fail_path = DATA_DIR / f"failed_docs_batch_{batch_count}.jsonl"
                with fail_path.open("w", encoding="utf-8") as f_fail:
                    for i, item in enumerate(items):
                        op, result = next(iter(item.items()))
                        if "error" in result:
                            err = result["error"]
                            print(
                                f"  - 문서 #{i} 실패: type={err.get('type')} "
                                f"reason={err.get('reason')}"
                            )
                            doc_line_index = i * 2 + 1
                            if doc_line_index < len(batch_actions):
                                f_fail.write(batch_actions[doc_line_index] + "\n")
                print(f"⚠ 실패 문서들은 {fail_path} 에 저장되었습니다.")

    print(f"✅ 전체 업로드 완료! 총 {count}개 문서 적재")


if __name__ == "__main__":
    bulk_ingest()
