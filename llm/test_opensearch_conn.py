from opensearch_client import get_opensearch_client


def main():
    client = get_opensearch_client()
    # 여기까지 예외 없이 오면 연결/인증 성공이라고 보면 됨
    print("✅ OpenSearch 클라이언트 생성 성공 (IAM 인증 OK)")


if __name__ == "__main__":
    main()
