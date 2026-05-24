"""Configuration helpers."""
from __future__ import annotations
import os
from dotenv import load_dotenv

load_dotenv(override=True)

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OSS_MODEL = os.environ.get("OSS_MODEL", "gemma3:1b")
FRONTIER_MODEL = os.environ.get("FRONTIER_MODEL", "llama3.2:1b")
EVALUATOR_MODEL = os.environ.get("EVALUATOR_MODEL", FRONTIER_MODEL)
EVALUATOR_BASE_URL = os.environ.get("EVALUATOR_BASE_URL", OLLAMA_BASE_URL)
EVALUATOR_TEMPERATURE = float(os.environ.get("EVALUATOR_TEMPERATURE", "0"))
EVALUATOR_TIMEOUT_SECONDS = int(os.environ.get("EVALUATOR_TIMEOUT_SECONDS", 120))
EVALUATOR_MAX_RETRIES = int(os.environ.get("EVALUATOR_MAX_RETRIES", 3))
JUDGE_PROVIDER = os.environ.get("JUDGE_PROVIDER", "ollama")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", EVALUATOR_MODEL)
JUDGE_API_KEY = os.environ.get("JUDGE_API_KEY", "")
JUDGE_BASE_URL = os.environ.get("JUDGE_BASE_URL", EVALUATOR_BASE_URL)
JUDGE_TEMPERATURE = float(os.environ.get("JUDGE_TEMPERATURE", "0"))
JUDGE_TIMEOUT = int(os.environ.get("JUDGE_TIMEOUT", 120))
JUDGE_MAX_RETRIES = int(os.environ.get("JUDGE_MAX_RETRIES", EVALUATOR_MAX_RETRIES))
EVALUATION_DEBUG = os.environ.get("EVALUATION_DEBUG", "false").strip().lower() in {"1", "true", "yes", "on"}
MAX_HISTORY = int(os.environ.get("MAX_HISTORY", 12))
HOST = os.environ.get("HOST", "0.0.0.0")
API_PORT = int(os.environ.get("API_PORT", 8000))
UI_PORT = int(os.environ.get("UI_PORT", 7860))