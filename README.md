# Assistant Comparison — Open Source vs Frontier

This repository implements a lightweight, internship-quality benchmarking platform comparing two AI assistants that share the same interface and behavior pipeline.

Assistants
- Open Source Assistant: Groq-hosted `llama-3.1-8b-instant`
- Frontier Assistant: Google Gemini `gemini-2.5-flash`

Core goals
- Same pipeline for both assistants
- Multi-turn conversations with short-term memory
- Simple Gradio frontend + FastAPI backend
- Reproducible evaluation with CSV / JSON outputs and charts

Quick start

1. Copy `.env.example` → `.env`.
2. Make sure Ollama is running locally and has the models you want, for example `gemma3:1b` and `llama3.2:1b`.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Start the FastAPI backend:

```bash
uvicorn app.api:app --host 127.0.0.1 --port 8000 --reload
```

5. In a separate terminal, launch the Streamlit frontend:

```bash
streamlit run app/ui.py
```

6. Open the Streamlit UI in your browser (it will print the local URL, typically `http://localhost:8501`).

To pull the smaller frontier model locally, run:

```powershell
ollama pull llama3.2:1b
```

Project layout

See the `app/` folder for `api.py`, adapters, conversation manager, evaluation tools and the Streamlit frontend. The project is deliberately modular so each piece can be tested independently.

Evaluation

- Use `app.evaluation.runner` or `app.evaluation.evaluator` to run batched tests over provided prompt datasets (`factual.json`, `jailbreak.json`, `bias.json`).
- Outputs: JSON, CSV and charts saved to `app/evaluation/reports/`.

Design tradeoffs

- Simplicity and reproducibility chosen over advanced optimizations.
- Groq and Gemini are API-backed, so results depend on network availability and provider limits.

Notes for deployment

- Designed to run on Hugging Face Spaces CPU tier. Keep `requirements.txt` minimal and avoid GPU-specific configs.

If you want, I can now scaffold the remaining project files (API, adapters, conversation manager, evaluation scripts). Which part should I create first?'