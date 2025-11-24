# gradio_app.py
import os
from typing import List, Dict, Tuple

import gradio as gr

from llm.graph_orchestrator import run_chat_flow
from llm.telemetry import trace_context, log_info


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
    history: List[Dict[str, str]],   # ✅ 이제 history는 messages 형식 (dict 리스트)
    user_id: str,
) -> Tuple[str, List[Dict[str, str]]]:
    """
    Gradio Chatbot <-> run_chat_flow 연결 함수

    - message: 사용자가 방금 입력한 질문
    - history: [{"role": "user"/"assistant", "content": "..."}, ...]
    """
    if not message:
        return "", history

    # Gradio history 형식 == orchestrator history 형식과 동일하므로 그대로 사용
    internal_history = history

    with trace_context(
        name="gradio_chat_request",
        run_type="chain",
        inputs={"query": message, "user_id": user_id},
        tags=["gradio"],
        metadata={},
    ):
        result = run_chat_flow(
            query=message,
            user_id=user_id or None,
            history=internal_history,
            user_profile=None,
        )

    answer = result.get("answer", "")
    log_info("gradio_chat_completed", user_id=user_id, answer_preview=answer[:50])

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
                value="demo-user",
                placeholder="사용자 ID (로그/추적용)",
            )

        # ✅ type 지정 안 하면 기본이 messages 형식
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
