"""Evaluation runner and helpers.

This module provides a batch evaluator that runs two local Ollama adapters
over provided datasets and writes structured reports and plots to the
`reports/` folder.
"""
from __future__ import annotations
import time
import json
import csv
from pathlib import Path
from typing import List, Dict
import matplotlib.pyplot as plt
from app.adapters.ollama_adapter import OllamaAssistant
from app.prompts import REFUSAL_KEYWORDS
from app import config

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)


def load_dataset(name: str) -> List[Dict]:
    path = ROOT / "datasets" / name
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_refusal(text: str) -> bool:
    txt = (text or "").lower()
    return any(k.lower() in txt for k in REFUSAL_KEYWORDS)


def evaluate_model(adapter, dataset: List[Dict]) -> Dict:
    results = []
    for item in dataset:
        prompt = item["prompt"]
        start = time.time()
        out = adapter.generate([{"role": "system", "content": ""}, {"role": "user", "content": prompt}])
        latency = time.time() - start
        refusal = is_refusal(out)
        expected = item.get("expected")
        hallucinated = False
        if expected:
            hallucinated = expected.lower() not in (out or "").lower()
        results.append({
            "id": item.get("id"),
            "prompt": prompt,
            "response": out,
            "latency": latency,
            "refusal": refusal,
            "hallucinated": hallucinated,
        })
    return {"count": len(results), "results": results}


def run_all():
    oss = OllamaAssistant(model_name=config.OSS_MODEL)
    frontier = OllamaAssistant(model_name=config.FRONTIER_MODEL)

    factual = load_dataset("factual.json")
    jailbreak = load_dataset("jailbreak.json")
    bias = load_dataset("bias.json")

    datasets = {"factual": factual, "jailbreak": jailbreak, "bias": bias}
    summary = {}

    for name, ds in datasets.items():
        print(f"Evaluating dataset: {name}")
        r1 = evaluate_model(oss, ds)
        r2 = evaluate_model(frontier, ds)

        # Save JSON
        with open(REPORTS / f"{name}_oss.json", "w", encoding="utf-8") as f:
            json.dump(r1, f, indent=2)
        with open(REPORTS / f"{name}_frontier.json", "w", encoding="utf-8") as f:
            json.dump(r2, f, indent=2)

        # Compute summaries
        def summarize(res):
            lat = [r["latency"] for r in res["results"]]
            refusal = sum(1 for r in res["results"] if r["refusal"]) / max(1, res["count"]) * 100
            halluc = sum(1 for r in res["results"] if r["hallucinated"]) / max(1, res["count"]) * 100
            return {"avg_latency": sum(lat) / max(1, len(lat)), "refusal_rate": refusal, "hallucination_rate": halluc}

        summary[name] = {"oss": summarize(r1), "frontier": summarize(r2)}

        # Plot latency comparison
        oss_lats = [r["latency"] for r in r1["results"]]
        fr_lats = [r["latency"] for r in r2["results"]]
        plt.figure()
        plt.bar([0, 1], [sum(oss_lats) / max(1, len(oss_lats)), sum(fr_lats) / max(1, len(fr_lats))], tick_label=["OSS", "Frontier"])
        plt.title(f"Average latency ({name})")
        plt.savefig(REPORTS / f"{name}_latency.png")

    # Save summary CSV
    csv_path = REPORTS / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)
        writer.writerow(["dataset", "model", "avg_latency", "refusal_rate", "hallucination_rate"])
        for ds, vals in summary.items():
            for model, s in vals.items():
                writer.writerow([ds, model, s["avg_latency"], s["refusal_rate"], s["hallucination_rate"]])

    with open(REPORTS / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Evaluation complete. Reports saved to:", REPORTS)


if __name__ == "__main__":
    run_all()
