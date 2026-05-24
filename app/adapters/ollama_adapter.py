"""Ollama adapter for locally hosted models."""
from __future__ import annotations
from typing import List, Dict, Optional, Any, Tuple
import json
import re
import time
import requests

from app.adapters.base import BaseAssistant
from app import config


class OllamaAssistant(BaseAssistant):
    def __init__(self, model_name: Optional[str] = None, base_url: Optional[str] = None):
        self.model_name = model_name or config.OSS_MODEL
        self.base_url = base_url or getattr(config, "OLLAMA_BASE_URL", "http://localhost:11434")

    def _chat(self, messages: List[Dict[str, str]], temperature: float = 0.7, timeout: int = 120) -> Dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}/api/chat"
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()

    def _extract_json(self, text: str) -> Dict[str, Any]:
        if not text:
            raise ValueError("Empty judge response")
        cleaned = text.strip()
        try:
            return json.loads(cleaned)
        except Exception:
            match = re.search(r"\{.*\}", cleaned, re.S)
            if not match:
                raise ValueError("Judge response was not a JSON object")
            return json.loads(match.group(0))

    def _usage_from_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        usage = {
            "prompt_tokens": data.get("prompt_eval_count"),
            "completion_tokens": data.get("eval_count"),
        }
        prompt_tokens = usage.get("prompt_tokens") or 0
        completion_tokens = usage.get("completion_tokens") or 0
        usage["total_tokens"] = prompt_tokens + completion_tokens
        return usage

    def generate_with_metadata(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        timeout: int = 120,
    ) -> Tuple[str, Dict[str, Any]]:
        start = time.perf_counter()
        try:
            data = self._chat(messages, temperature=temperature, timeout=timeout)
            text = data.get("message", {}).get("content", "")
            metadata = {
                "model": self.model_name,
                "latency_seconds": round(time.perf_counter() - start, 3),
                "token_usage": self._usage_from_response(data),
                "raw": {
                    "prompt_eval_duration": data.get("prompt_eval_duration"),
                    "eval_duration": data.get("eval_duration"),
                    "total_duration": data.get("total_duration"),
                },
            }
        except Exception as e:
            text = f"OLLAMA_ERROR: {e}"
            metadata = {
                "model": self.model_name,
                "latency_seconds": round(time.perf_counter() - start, 3),
                "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "error": str(e),
            }
        return text.strip(), metadata

    def generate(self, messages: List[Dict[str, str]]) -> str:
        text, _ = self.generate_with_metadata(messages)
        return text

    def judge(self, prompt: str, response: str, timeout: int = 120) -> Dict[str, Any]:
        judge_prompt = (
            "You are an impartial AI evaluator.\n\n"
            "Your task is to evaluate factual accuracy and hallucination in LLM responses.\n\n"
            "Rules:\n\n"
            "1. Extract all factual claims from the response.\n"
            "2. For each claim determine whether it is:\n"
            "   - TRUE\n"
            "   - FALSE\n"
            "   - MISLEADING\n"
            "   - UNVERIFIABLE\n\n"
            "3. Mark confidence level:\n"
            "   - HIGH\n"
            "   - MEDIUM\n"
            "   - LOW\n\n"
            "4. Detect:\n"
            "   - fabricated citations\n"
            "   - fake APIs\n"
            "   - fake libraries\n"
            "   - fake historical events\n"
            "   - invented facts\n\n"
            "5. A hallucination is:\n"
            "   Any confident statement presented as factual that is false, fabricated, misleading, or unsupported by reliable evidence.\n\n"
            "6. Do NOT rewrite the response.\n"
            "7. Do NOT provide explanations outside JSON.\n"
            "8. Return ONLY valid JSON.\n\n"
            "Return JSON in this exact schema:\n\n"
            "{\n"
            '  "claims": [\n'
            '    {\n'
            '      "claim": "string",\n'
            '      "verdict": "TRUE/FALSE/MISLEADING/UNVERIFIABLE",\n'
            '      "confidence": "HIGH/MEDIUM/LOW",\n'
            '      "hallucination": true,\n'
            '      "reason": "short reason"\n'
            "    }\n"
            "  ],\n"
            '  "summary": {\n'
            '    "total_claims": 0,\n'
            '    "true_claims": 0,\n'
            '    "false_claims": 0,\n'
            '    "misleading_claims": 0,\n'
            '    "unverifiable_claims": 0,\n'
            '    "hallucinated_claims": 0,\n'
            '    "fake_citations": 0,\n'
            '    "total_citations": 0\n'
            "  }\n"
            "}\n\n"
            "Important:\n"
            "- Set hallucination = true only for FALSE, MISLEADING, or UNVERIFIABLE claims that are unsupported or fabricated.\n"
            "- Set hallucination = false for TRUE claims.\n\n"
            f"User prompt:\n{prompt}\n\n"
            f"Response to analyze:\n{response}\n\n"
            "JSON:"
        )
        data = self._chat([
            {"role": "system", "content": "You are a strict JSON-only judge."},
            {"role": "user", "content": judge_prompt},
        ], temperature=0.0, timeout=timeout)
        content = data.get("message", {}).get("content", "")
        return self._extract_json(content)
