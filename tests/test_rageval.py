"""Test suite for rag-eval-kit.

Run with:
    pytest                       # all tests
    pytest --cov=rageval         # with coverage report
"""

import json
import math

import pytest

from rageval import metrics
from rageval.evaluator import EvalCase, compare, evaluate
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

    def test_empty_retrieved_list(self):
        assert metrics.precision_at_k([], {"a"}, k=5) == 0.0

    def test_k_must_be_positive(self):
        with pytest.raises(ValueError, match="k must be a positive integer"):
            metrics.precision_at_k(["a"], {"a"}, k=0)

    def test_k_negative_raises(self):
        with pytest.raises(ValueError):
            metrics.recall_at_k(["a"], {"a"}, k=-1)


class TestF1AtK:
    def test_perfect_f1(self):
        # precision=1.0, recall=1.0 -> f1=1.0
        assert metrics.f1_at_k(["a", "b"], {"a", "b"}, k=2) == pytest.approx(1.0)

    def test_zero_precision_zero_f1(self):
        assert metrics.f1_at_k(["x", "y"], {"a"}, k=2) == 0.0

    def test_f1_is_harmonic_mean(self):
        # precision@2 = 0.5 (one of two relevant), recall@2 = 0.5 (one of two relevant)
        p = metrics.precision_at_k(["a", "x"], {"a", "b"}, k=2)
        r = metrics.recall_at_k(["a", "x"], {"a", "b"}, k=2)
        expected = 2 * p * r / (p + r)
        assert metrics.f1_at_k(["a", "x"], {"a", "b"}, k=2) == pytest.approx(expected)


class TestHitAndRank:
    def test_hit_true(self):
        assert metrics.hit_at_k(["x", "a"], {"a"}, k=2) == 1.0

    def test_hit_false(self):
        assert metrics.hit_at_k(["x", "y"], {"a"}, k=2) == 0.0

    def test_hit_respects_k_cutoff(self):
        # relevant item at position 3, but k=2 should miss it
        assert metrics.hit_at_k(["x", "y", "a"], {"a"}, k=2) == 0.0

    def test_reciprocal_rank_first(self):
        assert metrics.reciprocal_rank(["a", "x"], {"a"}) == 1.0

    def test_reciprocal_rank_third(self):
        assert metrics.reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)

    def test_reciprocal_rank_none(self):
        assert metrics.reciprocal_rank(["x", "y"], {"a"}) == 0.0

    def test_reciprocal_rank_empty_retrieved(self):
        assert metrics.reciprocal_rank([], {"a"}) == 0.0


class TestAveragePrecision:
    def test_all_relevant_first(self):
        # relevant at ranks 1 and 2 -> AP = (1/1 + 2/2) / 2 = 1.0
        assert metrics.average_precision(["a", "b", "x"], {"a", "b"}) == 1.0

    def test_interleaved(self):
        # relevant at ranks 1 and 3 -> AP = (1/1 + 2/3) / 2
        expected = (1.0 + (2 / 3)) / 2
        assert metrics.average_precision(["a", "x", "b"], {"a", "b"}) == pytest.approx(expected)

    def test_empty_relevant_returns_zero(self):
        assert metrics.average_precision(["a", "b"], set()) == 0.0

    def test_no_hit_returns_zero(self):
        assert metrics.average_precision(["x", "y"], {"a"}) == 0.0


class TestNdcg:
    def test_perfect_order_is_one(self):
        assert metrics.ndcg_at_k(["a", "b"], {"a", "b"}, k=2) == pytest.approx(1.0)

    def test_known_value(self):
        # relevant only at rank 2: DCG = 1/log2(3); IDCG = 1/log2(2) = 1
        expected = (1.0 / math.log2(3)) / 1.0
        assert metrics.ndcg_at_k(["x", "a"], {"a"}, k=2) == pytest.approx(expected)

    def test_no_relevant_is_zero(self):
        assert metrics.ndcg_at_k(["x", "y"], {"a"}, k=2) == 0.0

    def test_more_relevant_than_k(self):
        # 4 relevant docs, k=2: ideal is top 2, so IDCG = 1/log2(2) + 1/log2(3)
        result = metrics.ndcg_at_k(["a", "b"], {"a", "b", "c", "d"}, k=2)
        assert 0.0 < result <= 1.0


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

    def test_search_returns_tuples(self, retriever):
        results = retriever.search("cats", k=2)
        assert len(results) == 2
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)

    def test_scores_are_non_negative(self, retriever):
        results = retriever.search("cats", k=3)
        assert all(score >= 0 for _, score in results)

    def test_n_docs_property(self, retriever):
        assert retriever.n_docs == 3

    def test_vocabulary_size_positive(self, retriever):
        assert retriever.vocabulary_size > 0

    def test_empty_corpus_raises(self):
        with pytest.raises(ValueError, match="at least one document"):
            TfidfRetriever().index({})

    def test_not_indexed_raises_on_search(self):
        r = TfidfRetriever()
        with pytest.raises(RuntimeError, match="not been indexed"):
            r.search("query")

    def test_invalid_k_raises(self, retriever):
        with pytest.raises(ValueError):
            retriever.search("cats", k=0)

    def test_index_returns_self(self):
        corpus = {"d1": "hello world"}
        r = TfidfRetriever()
        assert r.index(corpus) is r


# --------------------------------------------------------------------------- #
# EvalCase validation                                                          #
# --------------------------------------------------------------------------- #


class TestEvalCase:
    def test_valid_case(self):
        case = EvalCase(query="test query", relevant=["doc1"])
        assert case.query == "test query"

    def test_empty_query_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            EvalCase(query="", relevant=["doc1"])

    def test_whitespace_only_query_raises(self):
        with pytest.raises(ValueError):
            EvalCase(query="   ", relevant=["doc1"])

    def test_empty_relevant_raises(self):
        with pytest.raises(ValueError, match="at least one doc id"):
            EvalCase(query="test", relevant=[])


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
        with pytest.raises(ValueError, match="empty set"):
            evaluate(lambda q, k: [], [], k=3)

    def test_invalid_k_raises(self):
        cases = [EvalCase(query="q", relevant=["a"])]
        with pytest.raises(ValueError):
            evaluate(lambda q, k: ["a"], cases, k=0)

    def test_report_has_all_metrics(self):
        cases = [EvalCase(query="q", relevant=["a"])]
        report = evaluate(lambda q, k: ["a"], cases, k=1)
        for key in ["precision@k", "recall@k", "f1@k", "hit@k", "mrr", "map", "ndcg@k"]:
            assert key in report.scores

    def test_per_case_populated(self):
        cases = [
            EvalCase(query="q1", relevant=["a"]),
            EvalCase(query="q2", relevant=["b"]),
        ]
        report = evaluate(lambda q, k: ["a", "b"][:k], cases, k=2)
        assert len(report.per_case) == 2
        assert all("query" in row for row in report.per_case)

    def test_retriever_name_stored(self):
        cases = [EvalCase(query="q", relevant=["a"])]
        report = evaluate(lambda q, k: ["a"], cases, k=1, retriever_name="test-model")
        assert report.retriever_name == "test-model"

    def test_n_cases_matches(self):
        cases = [EvalCase(query=f"q{i}", relevant=["a"]) for i in range(5)]
        report = evaluate(lambda q, k: ["a"], cases, k=1)
        assert report.n_cases == 5


class TestEvalReport:
    @pytest.fixture
    def report(self):
        cases = [
            EvalCase(query="what is RAG", relevant=["doc1"]),
            EvalCase(query="vector search", relevant=["doc2", "doc3"]),
        ]
        return evaluate(lambda q, k: ["doc1", "doc2", "doc3"][:k], cases, k=3)

    def test_pretty_contains_metric_names(self, report):
        text = report.pretty()
        for name in ["precision@k", "recall@k", "hit@k", "mrr", "map", "ndcg@k"]:
            assert name in text

    def test_to_json_is_valid_json(self, report):
        j = report.to_json()
        parsed = json.loads(j)
        assert "summary" in parsed
        assert "per_case" in parsed

    def test_to_csv_contains_header(self, report):
        csv_str = report.to_csv()
        lines = csv_str.strip().splitlines()
        assert lines[0].startswith("query")

    def test_to_csv_row_count(self, report):
        csv_str = report.to_csv()
        lines = csv_str.strip().splitlines()
        # header + 2 cases + 1 aggregate mean row
        assert len(lines) == 4

    def test_to_dict_structure(self, report):
        d = report.to_dict()
        assert "summary" in d
        assert "per_case" in d
        assert "k" in d
        assert "n_cases" in d


class TestCompare:
    def test_compare_returns_all_names(self):
        cases = [EvalCase(query="q", relevant=["a"])]
        reports = compare(
            {"r1": lambda q, k: ["a"], "r2": lambda q, k: ["b"]},
            cases,
            k=1,
        )
        assert set(reports.keys()) == {"r1", "r2"}

    def test_compare_scores_differ(self):
        cases = [EvalCase(query="q", relevant=["a"])]
        reports = compare(
            {"good": lambda q, k: ["a"], "bad": lambda q, k: ["b"]},
            cases,
            k=1,
        )
        assert reports["good"].scores["hit@k"] == 1.0
        assert reports["bad"].scores["hit@k"] == 0.0

    def test_compare_empty_raises(self):
        cases = [EvalCase(query="q", relevant=["a"])]
        with pytest.raises(ValueError, match="at least one entry"):
            compare({}, cases, k=1)
