"""Streamlit frontend that talks to the FastAPI backend."""
from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import plotly.graph_objects as go
import importlib
from dotenv import load_dotenv
import requests
import streamlit as st

from app import config

API_URL = f"http://localhost:{config.API_PORT}"
REQUEST_TIMEOUT_SECONDS = int(getattr(config, "REQUEST_TIMEOUT_SECONDS", 120))


def post_compare(prompt: str, category: str) -> Dict[str, Any]:
    payload = {"prompt": prompt, "category": category}
    r = requests.post(f"{API_URL}/compare", json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
    r.raise_for_status()
    return r.json()


def _metric_value(metrics: Dict[str, Any], name: str, default: float = 0.0) -> float:
    value = metrics.get(name, default)
    try:
        return float(value)
    except Exception:
        return default


def render_radar(metrics: Dict[str, Any]):
    categories = [
        "Truthfulness",
        "Reliability",
        "Citation Integrity",
        "1 - Hallucination",
        "1 - Fiction Acceptance",
        "Refusal Accuracy",
    ]
    values = [
        _metric_value(metrics, "truthfulness_score"),
        _metric_value(metrics, "reliability_score"),
        _metric_value(metrics, "citation_integrity_score"),
        max(0.0, 1.0 - _metric_value(metrics, "hallucination_rate") / 100.0),
        max(0.0, 1.0 - _metric_value(metrics, "fiction_acceptance_rate") / 100.0),
        _metric_value(metrics, "refusal_accuracy"),
    ]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=values + [values[0]], theta=categories + [categories[0]], fill="toself"))
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20), polar=dict(radialaxis=dict(range=[0, 1])), showlegend=False)
    st.plotly_chart(fig, use_container_width=True, key=f"radar-{hash(tuple(values))}")


def render_claim_table(claims: List[Dict[str, Any]]):
    if not claims:
        st.caption("No claims were extracted by the judge.")
        return
    frame = pd.DataFrame(claims)
    desired = ["claim", "verdict", "confidence", "hallucination", "failure_type", "severity", "reason"]
    frame = frame[[column for column in desired if column in frame.columns]]
    st.dataframe(frame, use_container_width=True, hide_index=True)


def _response_block(title: str, result: Dict[str, Any]):
    st.subheader(title)
    st.caption(result.get("model", ""))
    st.write(result.get("response", "(no response)"))
    st.write(result.get("explanation", ""))


def main():
    st.set_page_config(page_title="Assistant Comparison", layout="wide")
    st.title("Assistant Comparison Arena")
    st.markdown("Compare two local models, judge them independently, and explain which one is better.")

    categories = ["general", "factual", "jailbreak", "bias", "edge_case"]
    with st.sidebar:
        st.header("Settings")
        if st.button("Reload config"):
            load_dotenv(override=True)
            importlib.reload(config)
            st.rerun()
        category = st.selectbox("Prompt category", categories, index=0)
        st.caption(f"OSS model: {config.OSS_MODEL}")
        st.caption(f"Frontier model: {config.FRONTIER_MODEL}")
        st.caption(f"Judge provider: {config.JUDGE_PROVIDER}")
        st.caption(f"Judge model: {config.JUDGE_MODEL}")

    prompt = st.text_area("Enter a prompt to send to both assistants", height=140)

    if st.button("Run comparison"):
        if not prompt.strip():
            st.warning("Please enter a prompt.")
            return

        with st.spinner("Generating both responses, judging them, and computing backend metrics..."):
            try:
                result = post_compare(prompt, category)
            except Exception as exc:
                st.error(f"Error contacting backend: {exc}")
                return

        a = result.get("assistant_a", {})
        b = result.get("assistant_b", {})
        winner = result.get("winner", {})

        top_cols = st.columns(3)
        top_cols[0].metric("Winner", winner.get("winner", "tie"))
        top_cols[1].metric("Score A", f"{winner.get('score_a', 0):.3f}")
        top_cols[2].metric("Score B", f"{winner.get('score_b', 0):.3f}")

        st.info(winner.get("explanation", ""))

        left, right = st.columns(2)
        with left:
            _response_block(a.get("name", "Model A"), a)
            st.metric("Hallucination rate", f"{_metric_value(a.get('metrics', {}), 'hallucination_rate'):.2f}%")
            st.metric("Truthfulness", f"{_metric_value(a.get('metrics', {}), 'truthfulness_score'):.2f}")
            st.metric("Reliability", f"{_metric_value(a.get('metrics', {}), 'reliability_score'):.2f}")
            st.metric("Refusal accuracy", f"{_metric_value(a.get('metrics', {}), 'refusal_accuracy'):.2f}")

        with right:
            _response_block(b.get("name", "Model B"), b)
            st.metric("Hallucination rate", f"{_metric_value(b.get('metrics', {}), 'hallucination_rate'):.2f}%")
            st.metric("Truthfulness", f"{_metric_value(b.get('metrics', {}), 'truthfulness_score'):.2f}")
            st.metric("Reliability", f"{_metric_value(b.get('metrics', {}), 'reliability_score'):.2f}")
            st.metric("Refusal accuracy", f"{_metric_value(b.get('metrics', {}), 'refusal_accuracy'):.2f}")

        st.divider()
        comparison = pd.DataFrame([
            {"metric": "Hallucination rate", "Assistant A": _metric_value(a.get("metrics", {}), "hallucination_rate"), "Assistant B": _metric_value(b.get("metrics", {}), "hallucination_rate")},
            {"metric": "Truthfulness score", "Assistant A": _metric_value(a.get("metrics", {}), "truthfulness_score"), "Assistant B": _metric_value(b.get("metrics", {}), "truthfulness_score")},
            {"metric": "Reliability score", "Assistant A": _metric_value(a.get("metrics", {}), "reliability_score"), "Assistant B": _metric_value(b.get("metrics", {}), "reliability_score")},
            {"metric": "Overconfidence score", "Assistant A": _metric_value(a.get("metrics", {}), "overconfidence_score"), "Assistant B": _metric_value(b.get("metrics", {}), "overconfidence_score")},
            {"metric": "Fiction acceptance rate", "Assistant A": _metric_value(a.get("metrics", {}), "fiction_acceptance_rate"), "Assistant B": _metric_value(b.get("metrics", {}), "fiction_acceptance_rate")},
            {"metric": "Refusal accuracy", "Assistant A": _metric_value(a.get("metrics", {}), "refusal_accuracy"), "Assistant B": _metric_value(b.get("metrics", {}), "refusal_accuracy")},
        ])
        st.subheader("Comparison table")
        st.dataframe(comparison, use_container_width=True, hide_index=True)

        chart_left, chart_right = st.columns(2)
        with chart_left:
            st.subheader("Assistant A radar")
            render_radar(a.get("metrics", {}))
        with chart_right:
            st.subheader("Assistant B radar")
            render_radar(b.get("metrics", {}))

        st.subheader("Extracted claims")
        claim_left, claim_right = st.columns(2)
        with claim_left:
            st.markdown("**Assistant A**")
            render_claim_table(a.get("judge", {}).get("claims", []))
        with claim_right:
            st.markdown("**Assistant B**")
            render_claim_table(b.get("judge", {}).get("claims", []))

        st.subheader("Judgment summary")
        summary_cols = st.columns(4)
        summary_cols[0].metric("Assistant A judge latency", f"{a.get('judge_latency_seconds', 0):.2f}s")
        summary_cols[1].metric("Assistant B judge latency", f"{b.get('judge_latency_seconds', 0):.2f}s")
        summary_cols[2].metric("Assistant A generation latency", f"{a.get('latency_seconds', 0):.2f}s")
        summary_cols[3].metric("Assistant B generation latency", f"{b.get('latency_seconds', 0):.2f}s")

        st.write("Assistant A explanation:")
        expl_a = a.get("explanation", [])
        if isinstance(expl_a, list):
            for item in expl_a:
                st.markdown(f"- {item}")
        else:
            st.write(expl_a)

        st.write("Assistant B explanation:")
        expl_b = b.get("explanation", [])
        if isinstance(expl_b, list):
            for item in expl_b:
                st.markdown(f"- {item}")
        else:
            st.write(expl_b)

        with st.expander("Raw JSON"):
            st.json(result)


if __name__ == "__main__":
    main()