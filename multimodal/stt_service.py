# ai_service/llm/multimodal/stt_service.py
from io import BytesIO
from typing import Optional

from openai import OpenAI

from ..config import settings
from ..embeddings import get_openai_client  # 같은 클라이언트 재사용


def transcribe_audio_bytes(
    audio_bytes: bytes,
    filename: str = "audio.wav",
    language: Optional[str] = "ko",
) -> str:
    """
    OpenAI Whisper 기반 STT.
    - 프론트/메인 API에서 업로드 받은 파일 바이트를 넘겨주면 됨.
    """
    client: OpenAI = get_openai_client()
    file_obj = BytesIO(audio_bytes)
    file_obj.name = filename  # openai 라이브러리가 확장자로 형식 추론할 수 있게

    resp = client.audio.transcriptions.create(
        model="gpt-4o-transcribe",  # 또는 "whisper-1" 등, 실제 사용할 모델명으로 변경
        file=file_obj,
        language=language,
    )
    return resp.text
