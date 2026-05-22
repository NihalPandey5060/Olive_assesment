from __future__ import annotations
from typing import List, Dict


class BaseAssistant:
    """Abstract assistant interface. Subclasses must implement `generate`."""

    def generate(self, messages: List[Dict[str, str]]) -> str:
        raise NotImplementedError()
