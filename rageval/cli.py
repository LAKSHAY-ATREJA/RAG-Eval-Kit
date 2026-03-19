"""Command-line interface: evaluate a retriever against a JSON eval set.

Usage:
    python -m rageval.cli examples/eval_set.json --k 3
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List

from .evaluator import EvalCase, evaluate
from .retriever import TfidfRetriever


def load_dataset(path: str):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        print(f"Error: dataset file not found: {path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as exc:
        print(f"Error: invalid JSON in {path}: {exc}", file=sys.stderr)
        sys.exit(1)

    missing = [key for key in ("corpus", "queries") if key not in data]
    if missing:
        print(f"Error: dataset is missing required keys: {missing}", file=sys.stderr)
        sys.exit(1)

    corpus: Dict[str, str] = data["corpus"]
    if not corpus:
        print("Error: corpus is empty", file=sys.stderr)
        sys.exit(1)

    cases: List[EvalCase] = [
        EvalCase(query=item["query"], relevant=item["relevant"])
        for item in data["queries"]
        if item.get("query") and item.get("relevant")
    ]
    if not cases:
        print("Error: no valid queries found in dataset", file=sys.stderr)
        sys.exit(1)

    return corpus, cases


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a RAG retriever against a labelled dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n  rag-eval examples/eval_set.json --k 3",
    )
    parser.add_argument("dataset", help="Path to a JSON eval set.")
    parser.add_argument("--k", type=int, default=5, help="Cutoff rank (default 5).")
    parser.add_argument("--json", action="store_true", help="Emit JSON scores.")
    parser.add_argument("--per-case", action="store_true", help="Show per-query scores.")
    args = parser.parse_args(argv)

    if args.k < 1:
        print("Error: --k must be a positive integer", file=sys.stderr)
        return 1

    corpus, cases = load_dataset(args.dataset)
    retriever = TfidfRetriever().index(corpus)

    report = evaluate(retriever.ranked_ids, cases, k=args.k)

    if args.json:
        output = {"summary": report.scores}
        if args.per_case:
            output["per_case"] = report.per_case
        print(json.dumps(output, indent=2))
    else:
        print(report.pretty())
        if args.per_case:
            print()
            print("Per-query breakdown:")
            print("-" * 44)
            for row in report.per_case:
                print(f"  {row['query'][:40]!r}")
                for k, v in row.items():
                    if k != "query":
                        print(f"    {k:<16} {v:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
