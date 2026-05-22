"""Open-source assistant adapter using Hugging Face Transformers."""
from __future__ import annotations
from typing import List, Dict, Optional
import time
from app.adapters.base import BaseAssistant
from app import config


class OSSAssistant(BaseAssistant):
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or config.GROQ_MODEL
        self._pipe = None

    def _load(self):
        if self._pipe is not None:
            return
        try:
            from transformers import pipeline
            # CPU friendly: device=-1
            self._pipe = pipeline("text-generation", model=self.model_name, device=-1)
        except Exception:
            from transformers import pipeline
            # Fallback to small GPT-2 if Qwen is not available locally
            self._pipe = pipeline("text-generation", model="gpt2", device=-1)

    def _build_prompt(self, messages: List[Dict[str, str]]) -> str:
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            parts.append(f"{role}: {content}")
        parts.append("assistant:")
        return "\n".join(parts)

    def generate(self, messages: List[Dict[str, str]]) -> str:
        self._load()
        prompt = self._build_prompt(messages)
        start = time.time()
        out = self._pipe(prompt, max_new_tokens=256, temperature=0.7)
        latency = int((time.time() - start) * 1000)
        text = out[0].get("generated_text", "")
        # Remove the echoed prompt if present
        if text.startswith(prompt):
            text = text[len(prompt) :]
        return text.strip()
