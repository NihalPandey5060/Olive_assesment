"""Pydantic schemas for hallucination judging and comparison results."""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Verdict(str, Enum):
    TRUE = "TRUE"
    FALSE = "FALSE"
    MISLEADING = "MISLEADING"
    UNVERIFIABLE = "UNVERIFIABLE"


class Confidence(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class FailureType(str, Enum):
    FABRICATED_FACT = "FABRICATED_FACT"
    FAKE_CITATION = "FAKE_CITATION"
    TEMPORAL_ERROR = "TEMPORAL_ERROR"
    CONTRADICTION = "CONTRADICTION"
    OVERCONFIDENT_ERROR = "OVERCONFIDENT_ERROR"
    FICTION_ACCEPTANCE = "FICTION_ACCEPTANCE"
    UNSAFE_COMPLIANCE = "UNSAFE_COMPLIANCE"
    BIAS_ISSUE = "BIAS_ISSUE"


class Severity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ClaimEvaluation(BaseModel):
    claim: str
    verdict: Verdict
    confidence: Confidence
    hallucination: bool
    failure_type: Optional[FailureType] = None
    severity: Severity = Severity.LOW
    reason: str


class BehaviorAnalysis(BaseModel):
    accepted_false_premise: bool = False
    unsafe_compliance: bool = False
    bias_detected: bool = False
    fake_citation_detected: bool = False
    overconfidence_detected: bool = False


class JudgeSummary(BaseModel):
    total_claims: int = 0
    true_claims: int = 0
    false_claims: int = 0
    misleading_claims: int = 0
    unverifiable_claims: int = 0
    hallucinated_claims: int = 0
    high_confidence_false_claims: int = 0
    fake_citations: int = 0
    total_citations: int = 0
    fiction_accepted: int = 0
    correct_refusals: int = 0


class JudgePayload(BaseModel):
    claims: List[ClaimEvaluation] = Field(default_factory=list)
    behavior_analysis: BehaviorAnalysis = Field(default_factory=BehaviorAnalysis)
    summary: JudgeSummary = Field(default_factory=JudgeSummary)


class EvaluationMetrics(BaseModel):
    hallucination_rate: float = 0.0
    truthfulness_score: float = 0.0
    reliability_score: float = 0.0
    overconfidence_score: float = 0.0
    fiction_acceptance_rate: float = 0.0
    refusal_accuracy: float = 0.0
    citation_integrity_score: float = 1.0
    weighted_score: float = 0.0
    weighted_hallucination_score: float = 0.0
    premise_rejection_score: float = 0.0
    bias_risk_score: float = 0.0
    final_score: float = 0.0


class ModelComparisonResult(BaseModel):
    name: str
    model: str
    response: str
    judge: JudgePayload
    metrics: EvaluationMetrics
    explanation: List[str]
    latency_seconds: float = 0.0
    judge_latency_seconds: float = 0.0
    token_usage: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)


class WinnerBreakdown(BaseModel):
    winner: str
    confidence: float = 0.0
    score_a: float = 0.0
    score_b: float = 0.0
    score_difference: float = 0.0
    explanation: str = ""


class ComparisonResponse(BaseModel):
    prompt: str
    category: str = "general"
    assistant_a: ModelComparisonResult
    assistant_b: ModelComparisonResult
    winner: WinnerBreakdown
