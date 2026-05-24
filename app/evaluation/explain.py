"""Human-readable explanation helpers for model comparison."""
from __future__ import annotations

from typing import List

from app.evaluation.schemas import JudgePayload, FailureType


def _top_failure_types(judge: JudgePayload) -> List[str]:
    counts = {}
    for claim in judge.claims:
        if claim.failure_type:
            counts[claim.failure_type.value] = counts.get(claim.failure_type.value, 0) + 1
    return [item for item, _ in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:3]]


def explain_judge(judge: JudgePayload, category: str) -> str:
    summary = judge.summary
    behavior = judge.behavior_analysis
    pieces: List[str] = []

    if summary.hallucinated_claims > 0:
        pieces.append(f"{summary.hallucinated_claims} hallucinated claim(s) were extracted.")
    if summary.false_claims > 0:
        pieces.append(f"{summary.false_claims} claim(s) were marked false.")
    if behavior.accepted_false_premise:
        pieces.append("The model accepted a false or fictional premise instead of rejecting it.")
    if behavior.unsafe_compliance:
        pieces.append("The model complied with unsafe instructions.")
    if behavior.bias_detected:
        pieces.append("Bias-related behavior was detected.")
    if behavior.fake_citation_detected or summary.fake_citations > 0:
        pieces.append("Fabricated or unsupported citations were detected.")
    if behavior.overconfidence_detected or summary.high_confidence_false_claims > 0:
        pieces.append("High-confidence false claims were present.")

    failures = _top_failure_types(judge)
    if failures:
        pieces.append("Primary failure modes: " + ", ".join(failures).replace("_", " ").lower() + ".")

    if not pieces:
        pieces.append("The response was mostly supported, cautious, and consistent with the prompt.")

    if category.lower() in {"jailbreak", "unsafe", "safety"} and summary.correct_refusals > 0:
        pieces.append("It refused unsafe requests correctly.")

    if category.lower() in {"edge_case", "fiction", "fictional"} and behavior.accepted_false_premise:
        pieces.append("It treated a fictional premise as real.")

    return " ".join(pieces)


def explain_judge_bullets(judge: JudgePayload, category: str, top_n: int = 3) -> List[str]:
    """Return short explanation bullets and top reasons the model lost points.

    Uses the judge payload to produce human-readable short bullets.
    """
    bullets: List[str] = []
    summary = judge.summary
    behavior = judge.behavior_analysis

    if summary.hallucinated_claims > 0:
        bullets.append(f"{summary.hallucinated_claims} hallucinated claim(s) were extracted.")
    if summary.false_claims > 0:
        bullets.append(f"{summary.false_claims} claim(s) were marked false.")
    if behavior.accepted_false_premise:
        bullets.append("Accepted a false or fictional premise instead of rejecting it.")
    if behavior.unsafe_compliance:
        bullets.append("Complied with unsafe or disallowed instructions.")
    if behavior.bias_detected:
        bullets.append("Bias-related behavior was detected.")
    if behavior.fake_citation_detected or summary.fake_citations > 0:
        bullets.append("Fabricated or unsupported citations were detected.")
    if behavior.overconfidence_detected or summary.high_confidence_false_claims > 0:
        bullets.append("High-confidence false claims were present.")

    failures = _top_failure_types(judge)
    for f in failures:
        bullets.append(f"Primary failure mode: {f.replace('_', ' ').lower()}.")

    if not bullets:
        bullets.append("The response was mostly supported and cautious.")

    # limit to top_n
    # Add per-claim highlights (prefer fabricated / critical first)
    claim_lines: List[str] = []
    for c in judge.claims:
        # highlight fabricated items
        if c.failure_type == FailureType.FABRICATED_FACT or c.failure_type == FailureType.FAKE_CITATION:
            claim_lines.append(f"Fabricated claim: '{c.claim}' -> {c.failure_type.value} (severity={c.severity.value})")
        elif c.verdict:
            claim_lines.append(f"Claim: '{c.claim}' -> {c.verdict.value} (severity={c.severity.value})")

    # ensure the most severe/fabricated claims appear first
    prioritized = [ln for ln in claim_lines if ln.startswith("Fabricated")] + [ln for ln in claim_lines if not ln.startswith("Fabricated")]

    return (bullets + prioritized)[:top_n]
