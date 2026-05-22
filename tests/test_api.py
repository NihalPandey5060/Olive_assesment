import json
from fastapi.testclient import TestClient

from app import api


class DummyAssistant:
    def __init__(self, text: str):
        self.text = text

    def generate(self, messages):
        # echo last user message prefixed
        last = None
        for m in reversed(messages):
            if m.get("role") == "user":
                last = m.get("content")
                break
        return f"DUMMY: {self.text} | prompt: {last}"


def test_compare_endpoint_returns_both(monkeypatch):
    client = TestClient(api.app)

    # Patch adapters to deterministic dummies
    api._groq = DummyAssistant("oss reply")
    api._gemini = DummyAssistant("frontier reply")

    payload = {"prompt": "Who was Ada Lovelace?", "category": "Factual"}
    r = client.post("/compare", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "oss" in data and "frontier" in data
    assert "response" in data["oss"] and "response" in data["frontier"]


def test_evaluate_heuristics():
    # Test heuristics_score indirectly via /evaluate endpoint fallback
    client = TestClient(api.app)
    payload = {"prompt": "P", "oss": "I cannot help with that.", "frontier": "Sure, here is an answer."}
    r = client.post("/evaluate", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "oss" in data and "frontier" in data
    assert data["oss"]["refusal"] >= 0
