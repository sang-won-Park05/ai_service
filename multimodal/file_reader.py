# ai_service/llm/multimodal/file_reader.py
from pathlib import Path
from typing import BinaryIO


def read_file_bytes(path_or_file: str | Path | BinaryIO) -> bytes:
    """
    로컬 경로 또는 파일 객체에서 바이너리 읽기.
    OCR, STT, 이미지분류 등에서 공통으로 사용 가능.
    """
    if hasattr(path_or_file, "read"):
        # file-like object
        return path_or_file.read()

    path = Path(path_or_file)
    with path.open("rb") as f:
        return f.read()
