"""Small runner script to execute batched evaluation and save reports."""
from __future__ import annotations
from app.evaluation.evaluator import run_all


def main():
    print("Starting batch evaluation...")
    run_all()


if __name__ == "__main__":
    main()
