"""Simple in-memory conversation manager."""
from __future__ import annotations
from typing import Dict, List
from dataclasses import dataclass, field
import uuid
from app import config
from app.prompts import SYSTEM_PROMPT


@dataclass
class Conversation:
    session_id: str
    system: str = SYSTEM_PROMPT
    messages: List[Dict[str, str]] = field(default_factory=list)


class ConversationManager:
    def __init__(self, max_history: int = config.MAX_HISTORY):
        self.sessions: Dict[str, Conversation] = {}
        self.max_history = max_history

    def create_session(self) -> str:
        sid = str(uuid.uuid4())
        self.sessions[sid] = Conversation(session_id=sid)
        return sid

    def reset(self, session_id: str) -> None:
        self.sessions[session_id] = Conversation(session_id=session_id)

    def add_user(self, session_id: str, text: str) -> None:
        conv = self.sessions.setdefault(session_id, Conversation(session_id=session_id))
        conv.messages.append({"role": "user", "content": text})
        self._trim(conv)

    def add_assistant(self, session_id: str, text: str) -> None:
        conv = self.sessions.setdefault(session_id, Conversation(session_id=session_id))
        conv.messages.append({"role": "assistant", "content": text})
        self._trim(conv)

    def _trim(self, conv: Conversation) -> None:
        # keep last N messages (user+assistant pairs)
        if len(conv.messages) > self.max_history * 2:
            conv.messages = conv.messages[-(self.max_history * 2) :]

    def get_messages(self, session_id: str) -> List[Dict[str, str]]:
        conv = self.sessions.get(session_id)
        if not conv:
            return [{"role": "system", "content": SYSTEM_PROMPT}]
        msgs = [{"role": "system", "content": conv.system}]
        msgs.extend(conv.messages)
        return msgs
