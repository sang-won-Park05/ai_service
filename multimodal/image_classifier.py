# ai_service/llm/multimodal/image_classifier.py
from typing import Literal

ImageType = Literal["document", "pill", "xray", "other"]


def classify_image_type(image_bytes: bytes) -> ImageType:
    """
    현재는 더미 구현.
    나중에 GPT-4o vision이나 별도 이미지 모델 붙여서
    처방전/진단서/약/엑스레이 등 분류하고 싶을 때 이 함수를 확장.
    """
    # TODO: 실제 모델 붙이기
    return "other"
