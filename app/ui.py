"""Streamlit frontend that talks to the FastAPI backend."""
from __future__ import annotations
from typing import Dict, Any

import streamlit as st
import requests
import time
from app import config
import plotly.graph_objects as go

API_URL = f"http://localhost:{config.API_PORT}"
REQUEST_TIMEOUT_SECONDS = int(getattr(config, "REQUEST_TIMEOUT_SECONDS", 120))


def post_compare(prompt: str) -> Dict[str, Any]:
    payload = {"prompt": prompt}
    r = requests.post(f"{API_URL}/compare", json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
    r.raise_for_status()
    return r.json()


def post_evaluate(oss_resp: str, frontier_resp: str, prompt: str) -> Dict[str, Any]:
    payload = {"prompt": prompt, "oss": oss_resp, "frontier": frontier_resp}
    r = requests.post(f"{API_URL}/evaluate", json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
    r.raise_for_status()
    return r.json()


def render_radar(scores: Dict[str, float]):
    categories = list(scores.keys())
    values = [scores[k] for k in categories]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=values + [values[0]], theta=categories + [categories[0]], fill='toself', name='Score'))
    fig.update_layout(margin=dict(l=20, r=20, t=30, b=20), polar=dict(radialaxis=dict(range=[0, 10])))
    st.plotly_chart(fig, use_container_width=True)


def main():
    st.set_page_config(page_title="Assistant Comparison", layout="wide")
    st.title("Assistant Comparison Arena")
    st.markdown("Compare two local Ollama assistants side-by-side with no API keys or token spend.")

    with st.sidebar:
        st.header("Settings")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
        st.caption(f"OSS model: {config.OSS_MODEL}")
        st.caption(f"Frontier model: {config.FRONTIER_MODEL}")

    prompt = st.text_area("Enter a prompt to send to both assistants", height=120)
    col1, col2 = st.columns([1, 1])

    if st.button("Send to both assistants"):
        if not prompt.strip():
            st.warning("Please enter a prompt.")
        else:
            with st.spinner("Sending prompt and waiting for responses..."):
                try:
                    result = post_compare(prompt)
                except Exception as e:
                    st.error(f"Error contacting backend: {e}")
                    return

            oss = result.get("oss", {})
            frontier = result.get("frontier", {})

            # Display responses
            with col1:
                st.subheader("Local OSS Assistant")
                st.write(f"**Model:** {oss.get('model')}  ")
                st.write(oss.get("response", "(no response)"))
                st.write(f"**Latency:** {oss.get('latency', None)}s  ")
                st.write(f"**Tokens (est):** {oss.get('tokens', 'n/a')}  ")

            with col2:
                st.subheader("Local Frontier Assistant")
                st.write(f"**Model:** {frontier.get('model')}  ")
                st.write(frontier.get("response", "(no response)"))
                st.write(f"**Latency:** {frontier.get('latency', None)}s  ")
                st.write(f"**Tokens (est):** {frontier.get('tokens', 'n/a')}  ")

            # Evaluate
            with st.expander("Live evaluation and comparison", expanded=True):
                try:
                    eval_result = post_evaluate(oss.get('response',''), frontier.get('response',''), prompt)
                except Exception as e:
                    st.error(f"Evaluation error: {e}")
                    return

                st.subheader("Scorecards")
                cols = st.columns(6)
                metrics = ["hallucination", "safety", "bias", "refusal", "helpfulness", "latency"]
                for i, m in enumerate(metrics):
                    score_oss = eval_result.get('oss', {}).get(m, None)
                    score_front = eval_result.get('frontier', {}).get(m, None)
                    cols[i].metric(label=m.title(), value=f"OSS: {score_oss}\nFR: {score_front}")

                st.subheader("Radar - OSS")
                render_radar(eval_result.get('oss', {}))
                st.subheader("Radar - Frontier")
                render_radar(eval_result.get('frontier', {}))


if __name__ == "__main__":
    main()
