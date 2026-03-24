"""Command-line interface: evaluate a retriever against a JSON eval set.

Usage
-----
    python -m rageval.cli examples/eval_set.json --k 3
    rag-eval examples/eval_set.json --k 5 --json
    rag-eval examples/eval_set.json --k 5 --per-case --output results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from .evaluator import EvalCase, EvalReport, evaluate
from .retriever import TfidfRetriever


def load_dataset(path: str) -> Tuple[Dict[str, str], List[EvalCase]]:
    """Parse a JSON eval set file and return (corpus, cases).

    The expected JSON schema is::

        {
          "corpus": {"doc_id": "document text", ...},
          "queries": [{"query": "...", "relevant": ["doc_id", ...]}, ...]
        }
    """
    p = Path(path)
    if not p.exists():
        print(f"Error: dataset file not found: {path}", file=sys.stderr)
        sys.exit(1)
    if not p.is_file():
        print(f"Error: {path} is not a file", file=sys.stderr)
        sys.exit(1)

    try:
        with p.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        print(f"Error: invalid JSON in {path}: {exc}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(data, dict):
        print("Error: dataset JSON must be an object at the top level", file=sys.stderr)
        sys.exit(1)

    missing = [key for key in ("corpus", "queries") if key not in data]
    if missing:
        print(f"Error: dataset is missing required keys: {missing}", file=sys.stderr)
        sys.exit(1)

    corpus: Dict[str, str] = data["corpus"]
    if not corpus:
        print("Error: corpus is empty", file=sys.stderr)
        sys.exit(1)

    raw_queries = data.get("queries", [])
    if not isinstance(raw_queries, list):
        print("Error: 'queries' must be a JSON array", file=sys.stderr)
        sys.exit(1)

    cases: List[EvalCase] = []
    for i, item in enumerate(raw_queries):
        query = item.get("query", "").strip()
        relevant = item.get("relevant", [])
        if not query:
            print(f"Warning: skipping query #{i} — empty 'query' field", file=sys.stderr)
            continue
        if not relevant:
            print(
                f"Warning: skipping query #{i} ({query!r}) — empty 'relevant' list",
                file=sys.stderr,
            )
            continue
        cases.append(EvalCase(query=query, relevant=relevant))

    if not cases:
        print("Error: no valid queries found in dataset", file=sys.stderr)
        sys.exit(1)

    return corpus, cases


def _write_output(content: str, path: str) -> None:
    """Write *content* to *path*, creating parent directories as needed."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8")
    print(f"Results written to {out}", file=sys.stderr)


def _format_text(report: EvalReport, per_case: bool) -> str:
    """Format a plain-text report string."""
    lines = [report.pretty()]
    if per_case:
        lines.append("")
        lines.append("Per-query breakdown:")
        lines.append("-" * 52)
        for row in report.per_case:
            lines.append(f"  {row['query'][:50]!r}")
            for key, val in row.items():
                if key != "query":
                    lines.append(f"    {key:<16} {val:.4f}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="rag-eval",
        description="Evaluate a RAG retriever against a labelled JSON dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  rag-eval examples/eval_set.json --k 3\n"
            "  rag-eval examples/eval_set.json --k 5 --json\n"
            "  rag-eval examples/eval_set.json --k 5 --per-case --output results.json\n"
        ),
    )
    parser.add_argument("dataset", help="Path to a JSON eval set file.")
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        metavar="K",
        help="Cutoff rank — only the top-K results are considered (default: 5).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit results as JSON instead of plain text.",
    )
    parser.add_argument(
        "--per-case",
        action="store_true",
        help="Include per-query score breakdown.",
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        help="Write results to FILE instead of stdout.",
    )
    args = parser.parse_args(argv)

    if args.k < 1:
        print("Error: --k must be a positive integer", file=sys.stderr)
        return 1

    corpus, cases = load_dataset(args.dataset)
    retriever = TfidfRetriever().index(corpus)

    report = evaluate(retriever.ranked_ids, cases, k=args.k, retriever_name="TF-IDF")

    if args.json:
        output_data = report.to_dict()
        if not args.per_case:
            output_data.pop("per_case", None)
        result = json.dumps(output_data, indent=2)
    else:
        result = _format_text(report, per_case=args.per_case)

    if args.output:
        _write_output(result, args.output)
    else:
        print(result)

    return 0


if __name__ == "__main__":
    sys.exit(main())
