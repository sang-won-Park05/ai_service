# OpenSearch 연결
# llm/opensearch_client.py

import os
from pathlib import Path

from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import boto3
from dotenv import load_dotenv

# =========================
#  .env 로드
# =========================
# 이 파일 경로: ai_service/llm/opensearch_client.py
# parent      : ai_service/llm
# parent.parent : ai_service (프로젝트 루트)
BASE_DIR = Path(__file__).resolve().parent.parent
env_path = BASE_DIR / ".env"
load_dotenv(env_path)

OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST") 
OPENSEARCH_REGION = os.getenv("OPENSEARCH_REGION", "ap-northeast-2")


def get_aws_auth():
    """AWS IAM 자격증명으로 OpenSearch Serverless(aoss) 인증 객체 생성"""
    session = boto3.Session()
    credentials = session.get_credentials()

    return AWS4Auth(
        credentials.access_key,
        credentials.secret_key,
        OPENSEARCH_REGION,
        "aoss",  # Serverless OpenSearch 서비스 이름
        session_token=credentials.token,
    )


def get_opensearch_client() -> OpenSearch:
    """OpenSearch Serverless 클라이언트 생성"""
    awsauth = get_aws_auth()

    client = OpenSearch(
        hosts=[{"host": OPENSEARCH_HOST, "port": 443}],
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
    )
    return client
