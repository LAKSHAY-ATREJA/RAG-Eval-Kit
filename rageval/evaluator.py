"""Aggregate retrieval metrics over a labelled evaluation set."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from statistics import mean
from typing import Callable, Dict, List, Optional, Sequence

from . import metrics


@dataclass
class EvalCase:
    """One evaluation example: a query and its ground-truth relevant doc ids."""

    query: str
    relevant: List[str]

    def __post_init__(self) -> None:
        if not self.query or not self.query.strip():
            raise ValueError("EvalCase.query must be a non-empty string")
        if not self.relevant:
            raise ValueError("EvalCase.relevant must contain at least one doc id")


@dataclass
class EvalReport:
    """Results of evaluating a retriever over a set of cases.

    Attributes
    ----------
    k:
        The cutoff rank used during evaluation.
    n_cases:
        Number of evaluation cases.
    scores:
        Dict mapping metric name to mean score across all cases.
    per_case:
        List of per-query score dicts (populated when evaluate() is called).
    retriever_name:
        Optional label for the retriever, shown in the pretty-print header.
    """

    k: int
    n_cases: int
    scores: Dict[str, float]
    per_case: List[Dict] = field(default_factory=list)
    retriever_name: Optional[str] = None

    def pretty(self) -> str:
        """Return a human-readable multi-line summary of the evaluation."""
        label = self.retriever_name or "retriever"
        lines = [
            f"RAG retrieval evaluation — {label}  (n={self.n_cases}, k={self.k})",
            "-" * 52,
        ]
        for name, value in self.scores.items():
            lines.append(f"{name:<16} {value:.4f}")
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Return a JSON-serialisable dictionary of the full report."""
        return {
            "retriever": self.retriever_name,
            "k": self.k,
            "n_cases": self.n_cases,
            "summary": self.scores,
            "per_case": self.per_case,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialise the report to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_csv(self) -> str:
        """Return a CSV string with one row per evaluation case.

        The first row is a header. The last row is the aggregated mean.
        """
        if not self.per_case:
            raise ValueError("per_case is empty; no per-query data to export")

        buf = io.StringIO()
        metric_names = [k for k in self.per_case[0].keys() if k != "query"]
        writer = csv.DictWriter(buf, fieldnames=["query"] + metric_names)
        writer.writeheader()
        for row in self.per_case:
            writer.writerow(row)
        # Aggregate row
        agg_row: Dict = {"query": "__mean__"}
        for name in metric_names:
            agg_row[name] = f"{self.scores[name]:.4f}"
        writer.writerow(agg_row)
        return buf.getvalue()


# A retriever is anything that maps (query, k) -> ranked list of doc ids.
Retriever = Callable[[str, int], Sequence[str]]


def evaluate(
    retriever: Retriever,
    cases: Sequence[EvalCase],
    k: int = 5,
    retriever_name: Optional[str] = None,
) -> EvalReport:
    """Run *retriever* over every case and average the core metrics.

    Parameters
    ----------
    retriever:
        Callable with signature ``(query: str, k: int) -> Sequence[str]``.
        It should return a ranked list of document ids (best first).
    cases:
        Non-empty sequence of :class:`EvalCase` objects.
    k:
        Cutoff rank. Only the top-k results from the retriever are considered.
    retriever_name:
        Optional human-readable label for the retriever (shown in output).

    Returns
    -------
    EvalReport
        Aggregated and per-case scores.
    """
    if not cases:
        raise ValueError("cannot evaluate an empty set of cases")
    if k <= 0:
        raise ValueError(f"k must be a positive integer, got {k!r}")

    agg: Dict[str, List[float]] = {
        "precision@k": [],
        "recall@k": [],
        "f1@k": [],
        "hit@k": [],
        "mrr": [],
        "map": [],
        "ndcg@k": [],
    }
    per_case: List[Dict] = []

    for case in cases:
        ranked = list(retriever(case.query, k))
        row: Dict = {
            "query": case.query,
            "precision@k": metrics.precision_at_k(ranked, case.relevant, k),
            "recall@k": metrics.recall_at_k(ranked, case.relevant, k),
            "f1@k": metrics.f1_at_k(ranked, case.relevant, k),
            "hit@k": metrics.hit_at_k(ranked, case.relevant, k),
            "mrr": metrics.reciprocal_rank(ranked, case.relevant),
            "map": metrics.average_precision(ranked, case.relevant),
            "ndcg@k": metrics.ndcg_at_k(ranked, case.relevant, k),
        }
        for key in agg:
            agg[key].append(row[key])
        per_case.append(row)

    scores = {name: mean(values) for name, values in agg.items()}
    return EvalReport(
        k=k,
        n_cases=len(cases),
        scores=scores,
        per_case=per_case,
        retriever_name=retriever_name,
    )


def compare(
    retrievers: Dict[str, Retriever],
    cases: Sequence[EvalCase],
    k: int = 5,
) -> Dict[str, EvalReport]:
    """Evaluate multiple retrievers over the same case set.

    Parameters
    ----------
    retrievers:
        Mapping of retriever name -> retriever callable.
    cases:
        Non-empty sequence of :class:`EvalCase` objects.
    k:
        Cutoff rank applied to all retrievers.

    Returns
    -------
    Dict mapping retriever name -> :class:`EvalReport`.

    Example::

        reports = compare(
            {"tfidf": retriever_a.ranked_ids, "bm25": retriever_b.ranked_ids},
            cases,
            k=5,
        )
        for name, report in reports.items():
            print(report.pretty())
    """
    if not retrievers:
        raise ValueError("retrievers dict must contain at least one entry")
    return {
        name: evaluate(fn, cases, k=k, retriever_name=name)
        for name, fn in retrievers.items()
    }
