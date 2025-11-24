# gradio_app.py
import os
from typing import List, Dict, Tuple

import gradio as gr

from llm.graph_orchestrator import run_chat_flow
from llm.telemetry import trace_context, log_info
from llm.utils.user_profile import load_user_profile   # ✅ 사용자 프로필 로더 추가


APP_TITLE = "MediNote Chat (RAG Demo)"
APP_DESCRIPTION = (
    "메디노트용 건강 챗봇 데모입니다. 텍스트로 질문하면 RAG + LLM으로 답변합니다.\n"
    "※ 실제 의료 진단/처방이 아니라, 참고용 정보만 제공합니다."
)


# =========================
#  Gradio 콜백
# =========================
def gradio_chat(
    message: str,
    history: List[Dict[str, str]],   # history: [{"role": "...", "content": "..."} ...]
    user_id: str,
) -> Tuple[str, List[Dict[str, str]]]:
    """
    Gradio Chatbot <-> run_chat_flow 연결 함수

    - message: 사용자가 방금 입력한 질문
    - history: [{"role": "user"/"assistant", "content": "..."}, ...]
    - user_id: Gradio 입력 박스에 적힌 사용자 ID (테스트 시 문자열)
    """
    if not message:
        return "", history

    # 1) Gradio 히스토리 그대로 사용 (이미 messages 형식)
    internal_history = history

    # 2) 테스트용 numeric user_id 만들기
    #    - 숫자만 들어오면 그걸 사용
    #    - 아니면 기본값 1로 고정
    try:
        numeric_user_id = int(user_id)
    except (TypeError, ValueError):
        numeric_user_id = 1

    # 3) PostgreSQL에서 사용자 프로필 로드
    #    (app_user + user_chronic_disease + user_allergy)
    user_profile = load_user_profile(numeric_user_id)

    with trace_context(
        name="gradio_chat_request",
        run_type="chain",
        inputs={"query": message, "user_id": numeric_user_id},
        tags=["gradio"],
        metadata={},
    ):
        result = run_chat_flow(
            query=message,
            user_id=str(numeric_user_id),
            history=internal_history,
            user_profile=user_profile,   # ✅ LLM 쪽으로 전달
        )

    answer = result.get("answer", "")
    log_info(
        "gradio_chat_completed",
        user_id=str(numeric_user_id),
        answer_preview=answer[:50],
    )

    # ✅ messages 형식으로 새 턴 추가
    new_history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ]

    # 입력창 비우고, 갱신된 history 반환
    return "", new_history


def clear_history():
    # Chatbot(type=messages)의 value는 list[dict]
    return "", []


# =========================
#  Gradio UI 정의
# =========================
def create_demo() -> gr.Blocks:
    with gr.Blocks(title=APP_TITLE) as demo:
        gr.Markdown(f"# {APP_TITLE}")
        gr.Markdown(APP_DESCRIPTION)

        with gr.Row():
            user_id_box = gr.Textbox(
                label="User ID",
                value="1",                 # ✅ 테스트용 기본값 1
                placeholder="사용자 ID (로그/추적용)",
            )

        chatbot = gr.Chatbot(
            label="MediNote Chat",
            height=500,
        )
        msg = gr.Textbox(
            label="질문 입력",
            placeholder="예) 타이레놀과 다른 약을 같이 먹어도 되나요?",
        )

        with gr.Row():
            send_btn = gr.Button("전송", variant="primary")
            clear_btn = gr.Button("대화 초기화")

        send_event_inputs = [msg, chatbot, user_id_box]
        send_event_outputs = [msg, chatbot]

        msg.submit(gradio_chat, send_event_inputs, send_event_outputs)
        send_btn.click(gradio_chat, send_event_inputs, send_event_outputs)
        clear_btn.click(clear_history, None, [msg, chatbot])

    return demo


if __name__ == "__main__":
    server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))

    demo = create_demo()
    demo.launch(server_name=server_name, server_port=server_port)
