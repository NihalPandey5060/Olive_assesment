"""FastAPI application exposing chat and structured model comparison endpoints."""
from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app import config
from app.adapters.ollama_adapter import OllamaAssistant
from app.conversation.manager import ConversationManager
from app.evaluation.judge import JudgeLLM
from app.evaluation.pipeline import compare_assistants
from app.evaluation.schemas import ComparisonResponse

app = FastAPI(title="Assistant Comparison API")
conv_mgr = ConversationManager(max_history=config.MAX_HISTORY)

_oss: Optional[OllamaAssistant] = None
_frontier: Optional[OllamaAssistant] = None


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    model: str
    message: str


class ChatResponse(BaseModel):
    session_id: str
    assistant: str


class ResetRequest(BaseModel):
    session_id: str


class CompareRequest(BaseModel):
    prompt: str
    category: str = "general"
    session_id: Optional[str] = None
    model_a_name: Optional[str] = None
    model_b_name: Optional[str] = None
    model_a_label: str = "Model A"
    model_b_label: str = "Model B"


def _get_oss(model_name: Optional[str] = None) -> OllamaAssistant:
    global _oss
    model_name = model_name or config.OSS_MODEL
    if _oss is None or _oss.model_name != model_name:
        _oss = OllamaAssistant(model_name=model_name)
    return _oss


def _get_frontier(model_name: Optional[str] = None) -> OllamaAssistant:
    global _frontier
    model_name = model_name or config.FRONTIER_MODEL
    if _frontier is None or _frontier.model_name != model_name:
        _frontier = OllamaAssistant(model_name=model_name)
    return _frontier


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    sid = req.session_id or conv_mgr.create_session()
    model_key = req.model.lower()
    if model_key not in ("oss", "frontier"):
        raise HTTPException(status_code=400, detail="model must be 'oss' or 'frontier'")

    conv_mgr.add_user(sid, req.message)
    messages = conv_mgr.get_messages(sid)
    assistant = _get_oss() if model_key == "oss" else _get_frontier()
    resp = assistant.generate(messages)
    conv_mgr.add_assistant(sid, resp)
    return ChatResponse(session_id=sid, assistant=resp)


@app.post("/reset")
def reset(req: ResetRequest):
    conv_mgr.reset(req.session_id)
    return {"status": "ok", "session_id": req.session_id}


async def _compare(req: CompareRequest) -> ComparisonResponse:
    assistant_a = _get_oss(req.model_a_name)
    assistant_b = _get_frontier(req.model_b_name)
    judge = JudgeLLM()
    return await compare_assistants(
        prompt=req.prompt,
        category=req.category,
        assistant_a=assistant_a,
        assistant_b=assistant_b,
        judge=judge,
        label_a=req.model_a_label,
        label_b=req.model_b_label,
    )


@app.post("/compare", response_model=ComparisonResponse)
async def compare(req: CompareRequest):
    return await _compare(req)


@app.post("/evaluate", response_model=ComparisonResponse)
async def evaluate(req: CompareRequest):
    return await _compare(req)