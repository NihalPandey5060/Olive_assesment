"""Command-line entrypoint for batch benchmarking."""
from __future__ import annotations

from app.evaluation.evaluator import run_all


def main() -> None:
    print("Starting batch evaluation...")
    run_all()


if __name__ == "__main__":
    main()
