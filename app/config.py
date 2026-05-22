"""Configuration helpers."""
from __future__ import annotations
import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OSS_MODEL = os.environ.get("OSS_MODEL", "gemma3:1b")
FRONTIER_MODEL = os.environ.get("FRONTIER_MODEL", "llama3.2:1b")
MAX_HISTORY = int(os.environ.get("MAX_HISTORY", 12))
HOST = os.environ.get("HOST", "0.0.0.0")
API_PORT = int(os.environ.get("API_PORT", 8000))
UI_PORT = int(os.environ.get("UI_PORT", 7860))