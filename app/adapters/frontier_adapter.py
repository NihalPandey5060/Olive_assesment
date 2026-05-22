"""Frontier assistant adapter using OpenAI API."""
from __future__ import annotations
from typing import List, Dict, Optional
import time
from openai import OpenAI
from app.adapters.base import BaseAssistant
from app import config


class FrontierAssistant(BaseAssistant):
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or config.FRONTIER_MODEL
        self.client = OpenAI(api_key=config.OPENAI_API_KEY) if config.OPENAI_API_KEY else None

    def generate(self, messages: List[Dict[str, str]]) -> str:
        start = time.time()
        try:
            if self.client is None:
                raise RuntimeError("OPENAI_API_KEY is not set")
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=256,
            )
            text = (resp.choices[0].message.content or "").strip()
        except Exception as e:
            text = f"OPENAI_ERROR: {e}"
        return text
