# ai_service/llm/utils/routers.py
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .graph_orchestrator import run_chat_flow

router = APIRouter(prefix="/chatbot", tags=["chatbot"])


class ChatHistoryMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    query: str
    user_id: Optional[str] = None
    history: Optional[List[ChatHistoryMessage]] = None
    user_profile: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]]
    debug: Dict[str, Any]


@router.post("/query", response_model=ChatResponse)
async def chatbot_query(payload: ChatRequest) -> ChatResponse:
    if not payload.query:
        raise HTTPException(status_code=400, detail="query is required")

    history_dicts = (
        [m.model_dump() for m in payload.history] if payload.history else []
    )

    result = run_chat_flow(
        query=payload.query,
        user_id=payload.user_id,
        history=history_dicts,
        user_profile=payload.user_profile,
    )

    return ChatResponse(**result)


@router.get("/health")
async def health_check():
    return {"status": "ok"}
