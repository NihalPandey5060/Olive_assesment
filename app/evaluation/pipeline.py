"""Comparison pipeline for two assistants with a third judge LLM."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from app import config
from app.adapters.ollama_adapter import OllamaAssistant
from app.evaluation.explain import explain_judge
from app.evaluation.judge import JudgeLLM
from app.evaluation.metrics import compute_metrics, score_difference
from app.evaluation.schemas import ComparisonResponse, ModelComparisonResult, WinnerBreakdown
from app.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

REPORTS = Path(__file__).parent / "reports"
REPORTS.mkdir(exist_ok=True)
HISTORY_PATH = REPORTS / "comparison_history.jsonl"


def _generation_messages(prompt: str):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def _category_label(category: Optional[str]) -> str:
    return (category or "general").strip().lower() or "general"


async def _generate_response(label: str, model_name: str, assistant: OllamaAssistant, prompt: str) -> Dict[str, Any]:
    start = time.perf_counter()
    text, metadata = await asyncio.to_thread(assistant.generate_with_metadata, _generation_messages(prompt))
    latency = round(time.perf_counter() - start, 3)
    logger.info("Generated %s (%s) in %.3fs", label, model_name, latency)
    return {
        "name": label,
        "model": model_name,
        "response": text.strip(),
        "latency_seconds": latency,
        "token_usage": metadata.get("token_usage", {}),
    }


async def _judge_response(judge: JudgeLLM, prompt: str, response: str, category: str) -> Tuple[Any, float]:
    result = await asyncio.to_thread(judge.evaluate, prompt, response, category)
    logger.info("Judged response for category=%s in %.3fs", category, result.latency_seconds)
    return result, result.latency_seconds


def _write_history(payload: Dict[str, Any]) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


async def compare_assistants(
    prompt: str,
    category: str,
    assistant_a: OllamaAssistant,
    assistant_b: OllamaAssistant,
    judge: Optional[JudgeLLM] = None,
    label_a: str = "Model A",
    label_b: str = "Model B",
) -> ComparisonResponse:
    judge = judge or JudgeLLM()
    category = _category_label(category)

    model_a_task = _generate_response(label_a, assistant_a.model_name, assistant_a, prompt)
    model_b_task = _generate_response(label_b, assistant_b.model_name, assistant_b, prompt)
    model_a_raw, model_b_raw = await asyncio.gather(model_a_task, model_b_task)

    judge_a, judge_latency_a = await _judge_response(judge, prompt, model_a_raw["response"], category)
    judge_b, judge_latency_b = await _judge_response(judge, prompt, model_b_raw["response"], category)

    if getattr(config, "EVALUATION_DEBUG", False):
        print("[EVAL-DEBUG] Assistant A raw response:")
        print(model_a_raw["response"])
        print("[EVAL-DEBUG] Assistant A raw judge output:")
        print(judge_a.raw_output)
        print("[EVAL-DEBUG] Assistant A parsed judge payload:")
        print(judge_a.payload.model_dump_json(indent=2))
        print("[EVAL-DEBUG] Assistant B raw response:")
        print(model_b_raw["response"])
        print("[EVAL-DEBUG] Assistant B raw judge output:")
        print(judge_b.raw_output)
        print("[EVAL-DEBUG] Assistant B parsed judge payload:")
        print(judge_b.payload.model_dump_json(indent=2))

    metrics_a = compute_metrics(judge_a.payload, category)
    metrics_b = compute_metrics(judge_b.payload, category)
    explanation_a_bullets = explain_judge_bullets = None
    try:
        from app.evaluation.explain import explain_judge, explain_judge_bullets

        explanation_a = explain_judge(judge_a.payload, category)
        explanation_b = explain_judge(judge_b.payload, category)
        explanation_a_bullets = explain_judge_bullets(judge_a.payload, category, top_n=5)
        explanation_b_bullets = explain_judge_bullets(judge_b.payload, category, top_n=5)
    except Exception:
        explanation_a = ""
        explanation_b = ""
        explanation_a_bullets = []
        explanation_b_bullets = []

    # Use final_score for winner logic (category-aware). If final scores are close,
    # apply decisive tie-breakers using hallucination, truthfulness, and premise rejection.
    score_a = metrics_a.final_score
    score_b = metrics_b.final_score
    winner_label, diff = score_difference(score_a, score_b)

    winner_name = "tie"
    confidence = 0.0
    explanation = "Both models were too close to separate confidently."

    if winner_label != "tie":
        winner_name = model_a_raw["name"] if winner_label == "assistant_a" else model_b_raw["name"]
        confidence = min(1.0, abs(diff) * 2.0)
        explanation = f"{winner_name} scored higher on the backend final score."
    else:
        # tie-breakers: prioritize lower hallucination (less is better), then truthfulness, then premise rejection
        hall_a = metrics_a.weighted_hallucination_score
        hall_b = metrics_b.weighted_hallucination_score
        hall_diff = hall_b - hall_a
        if abs(hall_diff) > 0.03:
            # hall_diff > 0 means A has lower hallucination
            if hall_diff > 0:
                winner_name = model_a_raw["name"]
                explanation = f"{winner_name} has significantly lower hallucination score."
            else:
                winner_name = model_b_raw["name"]
                explanation = f"{winner_name} has significantly lower hallucination score."
            confidence = min(1.0, abs(hall_diff) * 5.0)
        else:
            truth_a = metrics_a.truthfulness_score
            truth_b = metrics_b.truthfulness_score
            truth_diff = truth_a - truth_b
            if abs(truth_diff) > 0.03:
                winner_name = model_a_raw["name"] if truth_diff > 0 else model_b_raw["name"]
                explanation = f"{winner_name} shows noticeably higher truthfulness." 
                confidence = min(1.0, abs(truth_diff) * 3.0)
            else:
                premise_a = metrics_a.premise_rejection_score
                premise_b = metrics_b.premise_rejection_score
                premise_diff = premise_a - premise_b
                if abs(premise_diff) > 0.02:
                    winner_name = model_a_raw["name"] if premise_diff > 0 else model_b_raw["name"]
                    explanation = f"{winner_name} handled premise rejection better."
                    confidence = min(1.0, abs(premise_diff) * 5.0)

    response = ComparisonResponse(
        prompt=prompt,
        category=category,
        assistant_a=ModelComparisonResult(
            name=model_a_raw["name"],
            model=model_a_raw["model"],
            response=model_a_raw["response"],
            judge=judge_a.payload,
            metrics=metrics_a,
            explanation=explanation_a_bullets or [explanation_a],
            latency_seconds=model_a_raw["latency_seconds"],
            judge_latency_seconds=judge_latency_a,
            token_usage={"generation": model_a_raw["token_usage"], "judge": judge_a.token_usage},
            warnings=[],
        ),
        assistant_b=ModelComparisonResult(
            name=model_b_raw["name"],
            model=model_b_raw["model"],
            response=model_b_raw["response"],
            judge=judge_b.payload,
            metrics=metrics_b,
            explanation=explanation_b_bullets or [explanation_b],
            latency_seconds=model_b_raw["latency_seconds"],
            judge_latency_seconds=judge_latency_b,
            token_usage={"generation": model_b_raw["token_usage"], "judge": judge_b.token_usage},
            warnings=[],
        ),
        winner=WinnerBreakdown(
            winner=winner_name,
            confidence=confidence,
            score_a=score_a,
            score_b=score_b,
            score_difference=diff,
            explanation=explanation,
        ),
    )
    _write_history(response.model_dump())
    return response
