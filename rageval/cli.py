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
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    corpus: Dict[str, str] = data["corpus"]
    cases: List[EvalCase] = [
        EvalCase(query=item["query"], relevant=item["relevant"])
        for item in data["queries"]
    ]
    return corpus, cases


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a RAG retriever.")
    parser.add_argument("dataset", help="Path to a JSON eval set.")
    parser.add_argument("--k", type=int, default=5, help="Cutoff rank (default 5).")
    parser.add_argument("--json", action="store_true", help="Emit JSON scores.")
    args = parser.parse_args(argv)

    corpus, cases = load_dataset(args.dataset)
    retriever = TfidfRetriever().index(corpus)

    report = evaluate(retriever.ranked_ids, cases, k=args.k)

    if args.json:
        print(json.dumps(report.scores, indent=2))
    else:
        print(report.pretty())
    return 0


if __name__ == "__main__":
    sys.exit(main())
