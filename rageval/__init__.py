"""rag-eval-kit: dependency-light evaluation toolkit for RAG retrieval."""

from . import metrics
from .evaluator import EvalCase, EvalReport, compare, evaluate
from .retriever import TfidfRetriever

__version__ = "0.2.0"
__all__ = [
    "EvalCase",
    "EvalReport",
    "evaluate",
    "compare",
    "TfidfRetriever",
    "metrics",
]
