# ai_service/llm/retriever.py  (또는 utils/retriever.py 실제 위치 기준)

from typing import List, Dict, Any

from .config import settings
from .embeddings import embed_text
from .opensearch_client import get_opensearch_client


def retrieve_documents(query: str, top_k: int | None = None) -> List[Dict[str, Any]]:
    """
    OpenSearch Serverless KNN 검색.
    인덱스에는 다음 필드가 있다고 가정:
      - embedding: float[] (벡터)
      - content: str (본문)
      - title, doc_type, category_top, ... (메타데이터)
      - detail_url: str (출처 URL)
    """
    if top_k is None:
        top_k = settings.retriever_top_k

    vector = embed_text(query)
    if not vector:
        return []

    client = get_opensearch_client()
    body = {
        "size": top_k,
        "query": {
            "knn": {
                settings.opensearch_vector_field: {
                    "vector": vector,
                    "k": top_k,
                }
            }
        },
        "_source": True,
    }

    resp = client.search(index=settings.opensearch_index, body=body)
    docs: List[Dict[str, Any]] = []

    for hit in resp["hits"]["hits"]:
        src = hit.get("_source", {})
        content = src.get(settings.opensearch_content_field, "")

        metadata = {
            k: v
            for k, v in src.items()
            if k not in {settings.opensearch_content_field, settings.opensearch_vector_field}
        }

        doc = {
            "id": src.get("id") or hit.get("_id"),
            "score": hit.get("_score", 0.0),
            "content": content,
            "title": src.get("title"),
            "doc_type": src.get("doc_type"),
            "metadata": metadata,
            # 🔥 출처 URL을 top 레벨로 빼서 나중에 formatter에서 바로 사용
            "detail_url": src.get("detail_url") or metadata.get("detail_url"),
        }
        docs.append(doc)

    return docs
