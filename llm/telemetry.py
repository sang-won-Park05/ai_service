# ai_service/llm/utils/telemetry.py

import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, Optional

# =========================================
#  기본 Python 로거 설정
# =========================================
LOGGER_NAME = "ai_service.llm"
logger = logging.getLogger(LOGGER_NAME)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# 기본 로그 레벨 (필요하면 .env에서 조정)
logger.setLevel(logging.INFO)

# =========================================
#  LangSmith 설정 (옵션)
# =========================================
try:
    # langsmith 가 설치되어 있을 때만 사용
    from langsmith import Client
    from langsmith.run_helpers import trace as ls_trace  # context manager
    from langsmith.run_trees import configure as ls_configure
except Exception:  # ImportError 등
    Client = None
    ls_trace = None
    ls_configure = None


def _init_langsmith() -> bool:
    """
    LANGSMITH_* 환경변수를 기반으로 LangSmith 트레이싱 초기화.
    - LANGSMITH_API_KEY 가 없으면 비활성화
    - LANGSMITH_TRACING 이 'true'/'1' 일 때만 실제 전송
    """
    if Client is None or ls_configure is None:
        logger.info("LangSmith SDK not installed. Tracing disabled.")
        return False

    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        logger.info("LANGSMITH_API_KEY not set. Tracing disabled.")
        return False

    endpoint = os.getenv("LANGSMITH_ENDPOINT")  # 선택
    project = (
        os.getenv("LANGSMITH_PROJECT")
        or os.getenv("LANGCHAIN_PROJECT")
        or "medinote"
    )
    enabled_env = os.getenv("LANGSMITH_TRACING", "").lower()
    enabled = enabled_env in {"1", "true", "yes", "y"}

    client_kwargs: Dict[str, Any] = {}
    if endpoint:
        client_kwargs["api_url"] = endpoint

    client = Client(**client_kwargs)

    # 전역 설정  :contentReference[oaicite:0]{index=0}
    ls_configure(
        client=client,
        enabled=enabled,
        project_name=project,
        tags=["medinote", os.getenv("APP_ENV", "local")],
        metadata={},
    )

    logger.info(
        "LangSmith tracing initialized (enabled=%s, project=%s)",
        enabled,
        project,
    )
    return enabled


LANGSMITH_ENABLED: bool = _init_langsmith()

# =========================================
#  공통 로깅 헬퍼
# =========================================


def log_debug(event: str, **kwargs: Any) -> None:
    logger.debug("%s | %s", event, kwargs)


def log_info(event: str, **kwargs: Any) -> None:
    logger.info("%s | %s", event, kwargs)


def build_debug_snapshot(**kwargs: Any) -> Dict[str, Any]:
    """
    orchestrator 에서 응답에 같이 내려줄 디버그 정보 스냅샷.
    """
    return kwargs


# =========================================
#  LangSmith trace context
# =========================================

@contextmanager
def trace_context(
    name: str,
    run_type: str = "chain",
    inputs: Optional[Dict[str, Any]] = None,
    tags: Optional[list[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    LangSmith trace 래퍼.
    - LangSmith 활성화 시: langsmith.run_helpers.trace 를 사용
    - 비활성화/미설치 시: 그냥 패스스루

    사용 예시:

        from .telemetry import trace_context

        def handle_request(query: str):
            with trace_context("chat_request", inputs={"query": query}):
                result = run_chat_flow(query=query, ...)
                return result
    """
    if LANGSMITH_ENABLED and ls_trace is not None:
        with ls_trace(
            name=name,
            run_type=run_type,
            inputs=inputs,
            tags=tags,
            metadata=metadata,
        ):
            yield
    else:
        # LangSmith 비활성화 시에도 코드 흐름은 동일
        yield
