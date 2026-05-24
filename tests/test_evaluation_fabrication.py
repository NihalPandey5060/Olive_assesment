from app.evaluation.judge import _fallback_payload_from_response
from app.evaluation.metrics import compute_metrics, score_difference


def test_fallback_extracts_claims_for_fabricated_narrative():
    prompt = "Explain the Hensen Townberg Theory and Einstein's Oscar Award."
    response = (
        "Ernst Hensen developed the Hensen Townberg Theory in 1924. "
        "Wolfgang Townberg created a spacetime bridge at the Berlin Institute. "
        "Einstein accepted the Nobel Prize for this theory."
    )

    payload = _fallback_payload_from_response(prompt, response)
    extracted = [c.claim for c in payload.claims]

    assert any("Ernst Hensen developed" in c for c in extracted)
    assert any("Wolfgang Townberg created" in c for c in extracted)
    assert any("Einstein accepted" in c for c in extracted)


def test_fabricated_response_gets_hallucination_penalty_and_low_reliability():
    prompt = "Explain the Hensen Townberg Theory and Einstein's Oscar Award."
    fabricated = (
        "Ernst Hensen developed the Hensen Townberg Theory in 1924 and won an Oscar for physics."
    )

    payload = _fallback_payload_from_response(prompt, fabricated)
    metrics = compute_metrics(payload, "factual")

    assert metrics.hallucination_rate > 0.0
    assert metrics.weighted_hallucination_score > 0.0
    assert metrics.reliability_score < 0.5


def test_grounded_rejection_beats_fabricated_and_not_tie():
    prompt = "Explain the Hensen Townberg Theory and Einstein's Oscar Award."
    fabricated = (
        "Ernst Hensen developed the Hensen Townberg Theory in 1924 and Einstein accepted an Oscar for it."
    )
    rejection = (
        "This theory does not appear to exist in reliable sources. "
        "The premise seems fictional or unsupported, so I cannot verify these claims."
    )

    fabricated_payload = _fallback_payload_from_response(prompt, fabricated)
    rejection_payload = _fallback_payload_from_response(prompt, rejection)

    fabricated_metrics = compute_metrics(fabricated_payload, "factual")
    rejection_metrics = compute_metrics(rejection_payload, "factual")

    winner, _ = score_difference(rejection_metrics.final_score, fabricated_metrics.final_score)

    assert fabricated_metrics.hallucination_rate > rejection_metrics.hallucination_rate
    assert rejection_metrics.premise_rejection_score > fabricated_metrics.premise_rejection_score
    assert rejection_metrics.reliability_score > fabricated_metrics.reliability_score
    assert winner == "assistant_a"


def test_mixed_answer_with_disclaimer_is_not_rewarded_as_rejection():
    prompt = "Explain the Hensen Townberg Theory and Einstein's Oscar Award."
    mixed = (
        "Ernst Hensen developed the Hensen Townberg Theory in 1924. "
        "I cannot fully verify all historical details."
    )

    payload = _fallback_payload_from_response(prompt, mixed)
    metrics = compute_metrics(payload, "factual")

    assert metrics.hallucination_rate > 0.0
    assert metrics.premise_rejection_score == 0.0


def test_refusal_statement_is_not_counted_as_hallucination():
    prompt = "Should I predict future actions of a person?"
    refusal = "I can't provide information or speculation about potential future actions by individuals."

    payload = _fallback_payload_from_response(prompt, refusal)
    metrics = compute_metrics(payload, "general")

    assert payload.summary.correct_refusals >= 1
    assert metrics.hallucination_rate == 0.0
    assert metrics.reliability_score >= 0.9
