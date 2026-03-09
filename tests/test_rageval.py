"""Test suite for rag-eval-kit. Run with: pytest"""

import math

import pytest

from rageval import metrics
from rageval.evaluator import EvalCase, evaluate
from rageval.retriever import TfidfRetriever


# --------------------------------------------------------------------------- #
# Metrics                                                                      #
# --------------------------------------------------------------------------- #

class TestPrecisionRecall:
    def test_perfect_precision(self):
        assert metrics.precision_at_k(["a", "b"], {"a", "b"}, k=2) == 1.0

    def test_half_precision(self):
        assert metrics.precision_at_k(["a", "x"], {"a"}, k=2) == 0.5

    def test_recall_partial(self):
        # one of two relevant docs retrieved in top-2
        assert metrics.recall_at_k(["a", "x"], {"a", "b"}, k=2) == 0.5

    def test_recall_full(self):
        assert metrics.recall_at_k(["a", "b"], {"a", "b"}, k=2) == 1.0

    def test_empty_relevant_set(self):
        assert metrics.recall_at_k(["a"], set(), k=1) == 0.0

    def test_k_must_be_positive(self):
        with pytest.raises(ValueError):
            metrics.precision_at_k(["a"], {"a"}, k=0)


class TestHitAndRank:
    def test_hit_true(self):
        assert metrics.hit_at_k(["x", "a"], {"a"}, k=2) == 1.0

    def test_hit_false(self):
        assert metrics.hit_at_k(["x", "y"], {"a"}, k=2) == 0.0

    def test_reciprocal_rank_first(self):
        assert metrics.reciprocal_rank(["a", "x"], {"a"}) == 1.0

    def test_reciprocal_rank_third(self):
        assert metrics.reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)

    def test_reciprocal_rank_none(self):
        assert metrics.reciprocal_rank(["x", "y"], {"a"}) == 0.0


class TestAveragePrecision:
    def test_all_relevant_first(self):
        # relevant at ranks 1 and 2 -> AP = (1/1 + 2/2) / 2 = 1.0
        assert metrics.average_precision(["a", "b", "x"], {"a", "b"}) == 1.0

    def test_interleaved(self):
        # relevant at ranks 1 and 3 -> AP = (1/1 + 2/3) / 2
        expected = (1.0 + (2 / 3)) / 2
        assert metrics.average_precision(["a", "x", "b"], {"a", "b"}) == pytest.approx(expected)


class TestNdcg:
    def test_perfect_order_is_one(self):
        assert metrics.ndcg_at_k(["a", "b"], {"a", "b"}, k=2) == pytest.approx(1.0)

    def test_known_value(self):
        # relevant only at rank 2: DCG = 1/log2(3); IDCG = 1/log2(2) = 1
        expected = (1.0 / math.log2(3)) / 1.0
        assert metrics.ndcg_at_k(["x", "a"], {"a"}, k=2) == pytest.approx(expected)

    def test_no_relevant_is_zero(self):
        assert metrics.ndcg_at_k(["x", "y"], {"a"}, k=2) == 0.0


# --------------------------------------------------------------------------- #
# Retriever                                                                    #
# --------------------------------------------------------------------------- #

class TestTfidfRetriever:
    @pytest.fixture
    def retriever(self):
        corpus = {
            "d1": "cats and dogs are common household pets",
            "d2": "the stock market rallied on strong earnings",
            "d3": "kittens are baby cats that love to play",
        }
        return TfidfRetriever().index(corpus)

    def test_relevant_doc_ranks_first(self, retriever):
        ranked = retriever.ranked_ids("baby cats kittens", k=3)
        assert ranked[0] == "d3"

    def test_returns_at_most_k(self, retriever):
        assert len(retriever.ranked_ids("cats", k=2)) == 2

    def test_unrelated_query_still_returns(self, retriever):
        # should not crash on out-of-vocabulary terms
        assert len(retriever.ranked_ids("quantum chromodynamics", k=3)) == 3


# --------------------------------------------------------------------------- #
# Evaluator                                                                    #
# --------------------------------------------------------------------------- #

class TestEvaluate:
    def test_perfect_retriever_scores_one(self):
        cases = [EvalCase(query="q1", relevant=["a"])]

        def oracle(query, k):
            return ["a", "b", "c"][:k]

        report = evaluate(oracle, cases, k=3)
        assert report.scores["hit@k"] == 1.0
        assert report.scores["mrr"] == 1.0

    def test_empty_cases_raises(self):
        with pytest.raises(ValueError):
            evaluate(lambda q, k: [], [], k=3)

    def test_report_has_all_metrics(self):
        cases = [EvalCase(query="q", relevant=["a"])]
        report = evaluate(lambda q, k: ["a"], cases, k=1)
        for key in ["precision@k", "recall@k", "hit@k", "mrr", "map", "ndcg@k"]:
            assert key in report.scores
