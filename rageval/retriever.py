"""A small, dependency-free TF-IDF retriever.

This exists so the evaluation harness has something concrete to score out of
the box. It is intentionally simple and readable rather than optimised; in a
real system you would swap this for FAISS, a vector DB, or a hybrid retriever
and evaluate it with the exact same :mod:`rageval.metrics` functions.

The retriever uses smoothed IDF (sklearn-style) and cosine normalisation. For
corpora up to a few thousand documents it runs fast enough for interactive use.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Sequence, Tuple

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Lowercase and extract alphanumeric tokens from *text*."""
    return _TOKEN_RE.findall(text.lower())


class TfidfRetriever:
    """Index a corpus of ``{doc_id: text}`` and rank docs for a query.

    Usage::

        retriever = TfidfRetriever().index(corpus)
        results = retriever.search("my query", k=5)   # list of (doc_id, score)
        ids     = retriever.ranked_ids("my query", k=5)  # list of doc_ids

    The retriever must be indexed before searching. Calling :meth:`search` or
    :meth:`ranked_ids` on an un-indexed instance raises ``RuntimeError``.
    """

    def __init__(self) -> None:
        self._doc_ids: List[str] = []
        self._tf: List[Counter] = []
        self._idf: Dict[str, float] = {}
        self._norms: List[float] = []
        self._indexed: bool = False

    # ---------------------------------------------------------------------- #
    # Public API                                                               #
    # ---------------------------------------------------------------------- #

    def index(self, corpus: Dict[str, str]) -> "TfidfRetriever":
        """Build the inverted index from *corpus*.

        Parameters
        ----------
        corpus:
            Mapping of document id -> document text. Must be non-empty.

        Returns
        -------
        self
            Returns the retriever so calls can be chained:
            ``TfidfRetriever().index(corpus)``.
        """
        if not corpus:
            raise ValueError("corpus must contain at least one document")

        self._doc_ids = list(corpus.keys())
        self._tf = [Counter(tokenize(corpus[doc_id])) for doc_id in self._doc_ids]

        n_docs = len(self._doc_ids)
        df: Counter = Counter()
        for tf in self._tf:
            df.update(tf.keys())

        # Smoothed IDF (sklearn convention): log((1+N)/(1+df)) + 1
        self._idf = {
            term: math.log((1 + n_docs) / (1 + freq)) + 1.0
            for term, freq in df.items()
        }

        self._norms = []
        for tf in self._tf:
            weight_sq = sum(
                (count * self._idf.get(term, 0.0)) ** 2
                for term, count in tf.items()
            )
            self._norms.append(math.sqrt(weight_sq) or 1.0)

        self._indexed = True
        return self

    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Return the top-k ``(doc_id, score)`` pairs by cosine similarity.

        Parameters
        ----------
        query:
            Free-text query string.
        k:
            Number of results to return.

        Returns
        -------
        List of ``(doc_id, cosine_score)`` tuples, sorted descending by score.
        """
        self._require_indexed()
        if k <= 0:
            raise ValueError(f"k must be a positive integer, got {k!r}")

        q_tf = Counter(tokenize(query))
        q_weights = {
            term: count * self._idf.get(term, 0.0) for term, count in q_tf.items()
        }
        q_norm = math.sqrt(sum(w * w for w in q_weights.values())) or 1.0

        scored: List[Tuple[str, float]] = []
        for doc_id, tf, norm in zip(self._doc_ids, self._tf, self._norms):
            dot = sum(
                weight * tf.get(term, 0) * self._idf.get(term, 0.0)
                for term, weight in q_weights.items()
            )
            scored.append((doc_id, dot / (q_norm * norm)))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]

    def ranked_ids(self, query: str, k: int = 5) -> Sequence[str]:
        """Convenience wrapper: returns only the document ids from :meth:`search`."""
        return [doc_id for doc_id, _ in self.search(query, k)]

    @property
    def n_docs(self) -> int:
        """Number of documents in the index."""
        return len(self._doc_ids)

    @property
    def vocabulary_size(self) -> int:
        """Number of unique tokens in the index vocabulary."""
        return len(self._idf)

    # ---------------------------------------------------------------------- #
    # Internal helpers                                                         #
    # ---------------------------------------------------------------------- #

    def _require_indexed(self) -> None:
        if not self._indexed:
            raise RuntimeError(
                "TfidfRetriever has not been indexed yet. "
                "Call .index(corpus) before searching."
            )

    def __repr__(self) -> str:  # pragma: no cover
        if self._indexed:
            return (
                f"TfidfRetriever(n_docs={self.n_docs}, "
                f"vocab_size={self.vocabulary_size})"
            )
        return "TfidfRetriever(not indexed)"
