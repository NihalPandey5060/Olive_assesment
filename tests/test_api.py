from fastapi.testclient import TestClient

from app import api
from app.evaluation.judge import JudgeResult
from app.evaluation.schemas import BehaviorAnalysis, ClaimEvaluation, Confidence, FailureType, JudgePayload, JudgeSummary, Severity, Verdict
from app.evaluation.metrics import score_difference


class DummyAssistant:
    def __init__(self, text: str):
        self.text = text
        self.model_name = f"dummy-{text}"

    def generate(self, messages):
        # echo last user message prefixed
        last = None
        for m in reversed(messages):
            if m.get("role") == "user":
                last = m.get("content")
                break
        return f"DUMMY: {self.text} | prompt: {last}"

    def generate_with_metadata(self, messages, temperature=0.7, timeout=120):
        return self.generate(messages), {
            "model": self.model_name,
            "latency_seconds": 0.01,
            "token_usage": {"prompt_tokens": 2, "completion_tokens": 4, "total_tokens": 6},
        }


class DummyJudge:
    def evaluate(self, prompt, response, category):
        payload = JudgePayload(
            claims=[
                ClaimEvaluation(
                    claim=f"Prompt length is {len(prompt.split())}",
                    verdict=Verdict.TRUE,
                    confidence=Confidence.HIGH,
                    hallucination=False,
                    failure_type=None,
                    severity=Severity.LOW,
                    reason="Deterministic dummy claim",
                ),
                ClaimEvaluation(
                    claim=f"Response starts with {response[:5]}",
                    verdict=Verdict.FALSE,
                    confidence=Confidence.HIGH,
                    hallucination=True,
                    failure_type=FailureType.FABRICATED_FACT,
                    severity=Severity.HIGH,
                    reason="Deterministic dummy hallucination",
                ),
            ],
            behavior_analysis=BehaviorAnalysis(
                accepted_false_premise=False,
                unsafe_compliance=False,
                bias_detected=False,
                fake_citation_detected=False,
                overconfidence_detected=True,
            ),
            summary=JudgeSummary(
                total_claims=2,
                true_claims=1,
                false_claims=1,
                misleading_claims=0,
                unverifiable_claims=0,
                hallucinated_claims=1,
                high_confidence_false_claims=1,
                fake_citations=0,
                total_citations=0,
                fiction_accepted=0,
                correct_refusals=0,
            ),
        )
        return JudgeResult(
            payload=payload,
            raw_output="{}",
            latency_seconds=0.01,
            token_usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )


def test_compare_endpoint_returns_both(monkeypatch):
    client = TestClient(api.app)

    # Patch adapters to deterministic dummies
    monkeypatch.setattr(api, "_get_oss", lambda model_name=None: DummyAssistant("oss reply"))
    monkeypatch.setattr(api, "_get_frontier", lambda model_name=None: DummyAssistant("frontier reply"))
    monkeypatch.setattr(api, "JudgeLLM", lambda: DummyJudge())

    payload = {"prompt": "Who was Ada Lovelace?", "category": "factual"}
    r = client.post("/compare", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["category"] == "factual"
    assert "assistant_a" in data and "assistant_b" in data and "winner" in data
    assert data["assistant_a"]["response"].startswith("DUMMY:")
    assert data["assistant_a"]["judge"]["summary"]["total_claims"] == 2


def test_evaluate_returns_structured_hallucination_report(monkeypatch):
    client = TestClient(api.app)
    monkeypatch.setattr(api, "_get_oss", lambda model_name=None: DummyAssistant("oss reply"))
    monkeypatch.setattr(api, "_get_frontier", lambda model_name=None: DummyAssistant("frontier reply"))
    monkeypatch.setattr(api, "JudgeLLM", lambda: DummyJudge())

    payload = {"prompt": "Who discovered penicillin?"}
    r = client.post("/evaluate", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["prompt"] == "Who discovered penicillin?"
    assert data["assistant_a"]["judge"]["summary"]["total_claims"] == 2
    assert data["assistant_a"]["metrics"]["hallucination_rate"] == 50.0
    assert data["assistant_a"]["response"].startswith("DUMMY:")


def test_score_difference_prefers_small_real_differences():
    winner, diff = score_difference(0.51, 0.50)
    assert winner == "assistant_a"
    assert diff > 0
