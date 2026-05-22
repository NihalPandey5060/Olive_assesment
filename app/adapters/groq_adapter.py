"""Groq adapter using the OpenAI-compatible Groq API."""
from __future__ import annotations
from typing import List, Dict, Optional
import time
import requests
from app.adapters.base import BaseAssistant
from app import config


class GroqAssistant(BaseAssistant):
    """Adapter that sends chat requests to Groq's OpenAI-compatible API.

    Uses base_url https://api.groq.com/openai/v1 and the chat completions endpoint.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or config.GROQ_MODEL
        self.base_url = getattr(config, "GROQ_BASE_URL", "https://api.groq.com/openai/v1")
        self.api_key = getattr(config, "GROQ_API_KEY", None)

    def _build_payload(self, messages: List[Dict[str, str]]):
        return {"model": self.model_name, "messages": messages}

    def generate(self, messages: List[Dict[str, str]]) -> str:
        start = time.time()
        if not self.api_key:
            return "GROQ_ERROR: GROQ_API_KEY not configured"
        payload = self._build_payload(messages)
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=60)
            r.raise_for_status()
            data = r.json()
            # Expect OpenAI-compatible response structure
            text = (data.get("choices", [])[0].get("message", {}).get("content", ""))
        except Exception as e:
            text = f"GROQ_ERROR: {e}"
        latency = time.time() - start
        return text.strip()
