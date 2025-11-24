# rag/index_creator.py

import json
from pathlib import Path

from opensearchpy.exceptions import NotFoundError
from llm.opensearch_client import get_opensearch_client

SCHEMA_PATH = Path(__file__).parent / "schema" / "opensearch_schema.json"
INDEX_NAME = "medinote_v1"


def index_exists(client, index_name: str) -> bool:
    try:
        # Serverless는 항상 URL path에 "/" 필요
        client.transport.perform_request("HEAD", f"/{index_name}")
        return True
    except NotFoundError:
        return False


def create_index():
    client = get_opensearch_client()

    # BOM 허용
    with SCHEMA_PATH.open("r", encoding="utf-8-sig") as f:
        body = json.load(f)

    if index_exists(client, INDEX_NAME):
        print(f"⚠ 인덱스 이미 존재: {INDEX_NAME}")
        return

    print(f"📌 인덱스 생성 시도: {INDEX_NAME}")

    # PUT /{index_name}
    resp = client.transport.perform_request(
        "PUT",
        f"/{INDEX_NAME}",
        body=body
    )

    print("✅ 인덱스 생성 완료:", resp)


if __name__ == "__main__":
    create_index()
