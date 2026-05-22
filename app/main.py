"""Entrypoint: start FastAPI backend.

Use `uvicorn app.api:app` for development, and run Streamlit separately:

    uvicorn app.api:app --reload
    streamlit run app/ui.py
"""
from __future__ import annotations
import uvicorn
from app import api, config


def main():
    uvicorn.run(api.app, host=config.HOST, port=config.API_PORT, log_level="info")


if __name__ == "__main__":
    main()
