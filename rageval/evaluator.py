"""Aggregate retrieval metrics over a labelled evaluation set."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Callable, Dict, List, Sequence

from . import metrics


@dataclass
class EvalCase:
    """One evaluation example: a query and its ground-truth relevant doc ids."""

    query: str
    relevant: List[str]


@dataclass
class EvalReport:
    k: int
    n_cases: int
    scores: Dict[str, float]
    per_case: List[Dict[str, float]] = field(default_factory=list)

    def pretty(self) -> str:
        lines = [
            f"RAG retrieval evaluation  (n={self.n_cases}, k={self.k})",
            "-" * 44,
        ]
        for name, value in self.scores.items():
            lines.append(f"{name:<16} {value:.4f}")
        return "\n".join(lines)


# A retriever is anything that maps (query, k) -> ranked list of doc ids.
Retriever = Callable[[str, int], Sequence[str]]


def evaluate(
    retriever: Retriever,
    cases: Sequence[EvalCase],
    k: int = 5,
) -> EvalReport:
    """Run ``retriever`` over every case and average the core metrics."""
    if not cases:
        raise ValueError("cannot evaluate an empty set of cases")

    agg: Dict[str, List[float]] = {
        "precision@k": [],
        "recall@k": [],
        "hit@k": [],
        "mrr": [],
        "map": [],
        "ndcg@k": [],
    }
    per_case: List[Dict[str, float]] = []

    for case in cases:
        ranked = list(retriever(case.query, k))
        row = {
            "precision@k": metrics.precision_at_k(ranked, case.relevant, k),
            "recall@k": metrics.recall_at_k(ranked, case.relevant, k),
            "hit@k": metrics.hit_at_k(ranked, case.relevant, k),
            "mrr": metrics.reciprocal_rank(ranked, case.relevant),
            "map": metrics.average_precision(ranked, case.relevant),
            "ndcg@k": metrics.ndcg_at_k(ranked, case.relevant, k),
        }
        for key, value in row.items():
            agg[key].append(value)
        per_case.append({"query": case.query, **row})

    scores = {name: mean(values) for name, values in agg.items()}
    return EvalReport(k=k, n_cases=len(cases), scores=scores, per_case=per_case)
