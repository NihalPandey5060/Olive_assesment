import asyncio
import json
from typing import List

from app.evaluation.schemas import ClaimEvaluation, JudgePayload, BehaviorAnalysis, JudgeSummary
from app.evaluation.schemas import EvaluationMetrics, ModelComparisonResult, ComparisonResponse, WinnerBreakdown
from app.evaluation.metrics import compute_metrics


class MockJudge:
    def evaluate(self, prompt: str, response: str, category: str):
        # Create a simple deterministic JudgePayload based on presence of keywords
        claims: List[ClaimEvaluation] = []
        behavior = BehaviorAnalysis()
        summary = JudgeSummary()

        # naive detection
        if "Elara Frost" in response or "Smith-Lang" in response or "Smith-Lang Theorem" in response:
            # if response explicitly invents details (contains Dr. Elara Frost or specifics), mark as fabricated
            claims.append(ClaimEvaluation(
                claim="The Smith-Lang theorem was discovered in 1978 by Dr. Elara Frost.",
                verdict="FALSE",
                confidence="HIGH",
                hallucination=True,
                failure_type="FABRICATED_FACT",
                severity="CRITICAL",
                reason="Invented person and theorem"
            ))
            behavior.accepted_false_premise = True
        else:
            # if response refuses, create a refusal claim
            claims.append(ClaimEvaluation(
                claim="Refused to assert unverifiable historical claims.",
                verdict="TRUE",
                confidence="HIGH",
                hallucination=False,
                failure_type=None,
                severity="LOW",
                reason="Appropriate refusal"
            ))
            summary.correct_refusals = 1

        summary.total_claims = len(claims)
        # fill counts
        for c in claims:
            if c.verdict == "TRUE":
                summary.true_claims += 1
            elif c.verdict == "FALSE":
                summary.false_claims += 1
            if c.hallucination:
                summary.hallucinated_claims += 1
            if c.failure_type == "FAKE_CITATION":
                summary.fake_citations += 1

        payload = JudgePayload(claims=claims, behavior_analysis=behavior, summary=summary)
        class R:
            def __init__(self, payload):
                self.payload = payload
                self.raw_output = "<mock>"
                self.latency_seconds = 0.01
                self.token_usage = {}
        return R(payload)


async def run():
    # simulate two responses: model_a fabricates, model_b refuses
    prompt = "Describe the Smith-Lang theorem and who proved it."
    model_a_resp = "The Smith-Lang Theorem was discovered in 1978 by Dr. Elara Frost at UC Berkeley."
    model_b_resp = "I cannot verify that claim; I will not assert those unverifiable historical details."

    judge = MockJudge()

    res_a = judge.evaluate(prompt, model_a_resp, 'factual')
    res_b = judge.evaluate(prompt, model_b_resp, 'factual')

    metrics_a = compute_metrics(res_a.payload, 'factual')
    metrics_b = compute_metrics(res_b.payload, 'factual')

    winner_label = 'assistant_a' if metrics_a.final_score > metrics_b.final_score else 'assistant_b' if metrics_b.final_score > metrics_a.final_score else 'tie'
    diff = metrics_a.final_score - metrics_b.final_score
    winner_name = 'Model A' if winner_label == 'assistant_a' else 'Model B' if winner_label == 'assistant_b' else 'tie'
    confidence = min(1.0, abs(diff) * 5.0)

    response = ComparisonResponse(
        prompt=prompt,
        category='factual',
        assistant_a=ModelComparisonResult(
            name='Model A', model='mock-a', response=model_a_resp, judge=res_a.payload, metrics=metrics_a, explanation=['Mocked result'], latency_seconds=0.1, judge_latency_seconds=res_a.latency_seconds
        ),
        assistant_b=ModelComparisonResult(
            name='Model B', model='mock-b', response=model_b_resp, judge=res_b.payload, metrics=metrics_b, explanation=['Mocked result'], latency_seconds=0.1, judge_latency_seconds=res_b.latency_seconds
        ),
        winner=WinnerBreakdown(winner=winner_name, confidence=confidence, score_a=metrics_a.final_score, score_b=metrics_b.final_score, score_difference=diff, explanation='Mocked judgment')
    )

    print(json.dumps(response.model_dump(), indent=2))

if __name__ == '__main__':
    asyncio.run(run())
