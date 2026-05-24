"""Evaluation helpers for explainable hallucination analysis."""

from app.evaluation.explain import explain_judge
from app.evaluation.judge import JUDGE_SYSTEM_PROMPT, JudgeLLM, JudgeResult
from app.evaluation.metrics import compute_metrics, compute_weighted_score, score_difference
from app.evaluation.pipeline import compare_assistants
from app.evaluation.schemas import (
    BehaviorAnalysis,
    ClaimEvaluation,
    ComparisonResponse,
    Confidence,
    EvaluationMetrics,
    FailureType,
    JudgePayload,
    JudgeSummary,
    ModelComparisonResult,
    Severity,
    Verdict,
    WinnerBreakdown,
)
