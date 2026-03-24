"""Retrieval quality metrics for RAG evaluation.

All functions operate on *ranked* result lists and a set of relevant ids.
Pure Python, no dependencies, so they are trivial to test and audit.

Conventions
-----------
- ``retrieved``: an ordered list of document ids, best-ranked first.
- ``relevant``:  a set (or iterable) of document ids that are truly relevant.
- ``k``:         cutoff rank. Metrics consider only the top-k retrieved items.

All metric functions are stateless and safe to call from multiple threads.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence


def _top_k(retrieved: Sequence[str], k: int) -> Sequence[str]:
    if k <= 0:
        raise ValueError(f"k must be a positive integer, got {k!r}")
    return retrieved[:k]


def precision_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of the top-k retrieved items that are relevant.

    Returns 0.0 when the retrieved list is shorter than k (uses actual length).
    """
    relevant_set = set(relevant)
    top = _top_k(retrieved, k)
    if not top:
        return 0.0
    hits = sum(1 for doc in top if doc in relevant_set)
    return hits / len(top)


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of all relevant items that appear in the top-k.

    Returns 0.0 when the relevant set is empty (undefined recall treated as 0).
    """
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    top = _top_k(retrieved, k)
    hits = sum(1 for doc in top if doc in relevant_set)
    return hits / len(relevant_set)


def f1_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Harmonic mean of precision@k and recall@k.

    Returns 0.0 when both precision and recall are zero.
    """
    p = precision_at_k(retrieved, relevant, k)
    r = recall_at_k(retrieved, relevant, k)
    if p + r == 0.0:
        return 0.0
    return 2.0 * p * r / (p + r)


def hit_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """1.0 if at least one relevant item is in the top-k, else 0.0."""
    relevant_set = set(relevant)
    top = _top_k(retrieved, k)
    return 1.0 if any(doc in relevant_set for doc in top) else 0.0


def reciprocal_rank(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    """Reciprocal of the rank of the first relevant item (0.0 if none found)."""
    relevant_set = set(relevant)
    for idx, doc in enumerate(retrieved, start=1):
        if doc in relevant_set:
            return 1.0 / idx
    return 0.0


def average_precision(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    """Average precision: mean of precision@k at each relevant hit position.

    This is the area under the precision-recall curve for a single query.
    Returns 0.0 when the relevant set is empty.
    """
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    hits = 0
    score = 0.0
    for idx, doc in enumerate(retrieved, start=1):
        if doc in relevant_set:
            hits += 1
            score += hits / idx
    return score / len(relevant_set)


def ndcg_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Normalised Discounted Cumulative Gain with binary relevance.

    Rewards retrievers that place relevant items at higher ranks. The score is
    normalised against the ideal ranking (all relevant items ranked first), so
    it always lies in [0, 1].
    """
    relevant_set = set(relevant)
    top = _top_k(retrieved, k)

    dcg = 0.0
    for idx, doc in enumerate(top, start=1):
        if doc in relevant_set:
            dcg += 1.0 / math.log2(idx + 1)

    # Ideal DCG: all relevant items ranked first.
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))

    return dcg / idcg if idcg > 0 else 0.0
