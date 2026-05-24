"""Batch evaluation helpers for the explainable comparison pipeline."""
from __future__ import annotations

import asyncio
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt

from app import config
from app.adapters.ollama_adapter import OllamaAssistant
from app.evaluation.judge import JudgeLLM
from app.evaluation.pipeline import compare_assistants

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)


def load_dataset(name: str) -> List[Dict[str, Any]]:
    path = ROOT / "datasets" / name
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


async def evaluate_dataset(name: str, dataset: List[Dict[str, Any]], judge: JudgeLLM) -> Dict[str, Any]:
    assistant_a = OllamaAssistant(model_name=config.OSS_MODEL)
    assistant_b = OllamaAssistant(model_name=config.FRONTIER_MODEL)
    results = []

    for item in dataset:
        prompt = item["prompt"]
        category = item.get("category", name)
        result = await compare_assistants(
            prompt=prompt,
            category=category,
            assistant_a=assistant_a,
            assistant_b=assistant_b,
            judge=judge,
            label_a="Model A",
            label_b="Model B",
        )
        payload = result.model_dump()
        payload["dataset_item"] = item
        results.append(payload)

    return {"count": len(results), "results": results}


def _average(values: List[float]) -> float:
    return sum(values) / max(1, len(values))


def _summarize_model(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    if not rows:
        return {
            "hallucination_rate": 0.0,
            "truthfulness_score": 0.0,
            "reliability_score": 0.0,
            "overconfidence_score": 0.0,
            "fiction_acceptance_rate": 0.0,
            "refusal_accuracy": 0.0,
            "citation_integrity_score": 0.0,
            "weighted_score": 0.0,
        }

    metrics_rows = [row["metrics"] for row in rows]
    return {
        "hallucination_rate": _average([float(row["hallucination_rate"]) for row in metrics_rows]),
        "truthfulness_score": _average([float(row["truthfulness_score"]) for row in metrics_rows]),
        "reliability_score": _average([float(row["reliability_score"]) for row in metrics_rows]),
        "overconfidence_score": _average([float(row["overconfidence_score"]) for row in metrics_rows]),
        "fiction_acceptance_rate": _average([float(row["fiction_acceptance_rate"]) for row in metrics_rows]),
        "refusal_accuracy": _average([float(row["refusal_accuracy"]) for row in metrics_rows]),
        "citation_integrity_score": _average([float(row["citation_integrity_score"]) for row in metrics_rows]),
        "weighted_score": _average([float(row["weighted_score"]) for row in metrics_rows]),
    }


def _write_report(name: str, payload: Dict[str, Any]) -> None:
    with open(REPORTS / f"{name}_comparison.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def run_all() -> Dict[str, Any]:
    async def _run() -> Dict[str, Any]:
        judge = JudgeLLM()
        datasets = {
            "factual": load_dataset("factual.json"),
            "jailbreak": load_dataset("jailbreak.json"),
            "bias": load_dataset("bias.json"),
        }

        summary: Dict[str, Any] = {}
        for name, dataset in datasets.items():
            print(f"Evaluating dataset: {name}")
            payload = await evaluate_dataset(name, dataset, judge)
            _write_report(name, payload)

            model_a_rows = [item["assistant_a"] for item in payload["results"]]
            model_b_rows = [item["assistant_b"] for item in payload["results"]]

            summary[name] = {
                "model_a": _summarize_model(model_a_rows),
                "model_b": _summarize_model(model_b_rows),
            }

            plt.figure()
            plt.bar(
                [0, 1],
                [summary[name]["model_a"]["weighted_score"], summary[name]["model_b"]["weighted_score"]],
                tick_label=["Model A", "Model B"],
            )
            plt.ylabel("Weighted score")
            plt.title(f"Weighted comparison ({name})")
            plt.savefig(REPORTS / f"{name}_weighted_score.png")
            plt.close()

        with open(REPORTS / "summary.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "dataset",
                "model",
                "weighted_score",
                "hallucination_rate",
                "truthfulness_score",
                "reliability_score",
                "overconfidence_score",
                "fiction_acceptance_rate",
                "refusal_accuracy",
                "citation_integrity_score",
            ])
            for dataset_name, models in summary.items():
                for model_name, metrics in models.items():
                    writer.writerow([
                        dataset_name,
                        model_name,
                        metrics["weighted_score"],
                        metrics["hallucination_rate"],
                        metrics["truthfulness_score"],
                        metrics["reliability_score"],
                        metrics["overconfidence_score"],
                        metrics["fiction_acceptance_rate"],
                        metrics["refusal_accuracy"],
                        metrics["citation_integrity_score"],
                    ])

        with open(REPORTS / "summary.json", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)

        print("Evaluation complete. Reports saved to:", REPORTS)
        return summary

    return asyncio.run(_run())


if __name__ == "__main__":
    run_all()
