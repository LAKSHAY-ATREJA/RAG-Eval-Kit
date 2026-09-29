"""CI quality gate for the bundled retrieval evaluation set."""

import json
from pathlib import Path

from rageval import EvalCase, TfidfRetriever, evaluate

DATASET = Path("examples/eval_set.json")
MIN_MRR = 0.70
MIN_RECALL = 0.70
K = 3


def main() -> None:
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    retriever = TfidfRetriever().index(data["corpus"])
    cases = [EvalCase(query=item["query"], relevant=item["relevant"]) for item in data["queries"]]
    report = evaluate(retriever.ranked_ids, cases, k=K, retriever_name="tfidf-baseline")

    print(report.pretty())

    failures = []
    if report.scores["mrr"] < MIN_MRR:
        failures.append(f"MRR {report.scores['mrr']:.4f} < {MIN_MRR:.2f}")
    if report.scores["recall@k"] < MIN_RECALL:
        failures.append(f"Recall@{K} {report.scores['recall@k']:.4f} < {MIN_RECALL:.2f}")

    if failures:
        raise SystemExit("Retrieval quality gate failed: " + "; ".join(failures))

    print("Retrieval quality gate passed.")


if __name__ == "__main__":
    main()
