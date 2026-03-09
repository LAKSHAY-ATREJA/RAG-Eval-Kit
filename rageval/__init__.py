"""rag-eval-kit: dependency-light evaluation toolkit for RAG retrieval."""

from .evaluator import EvalCase, EvalReport, evaluate
from .retriever import TfidfRetriever
from . import metrics

__version__ = "0.1.0"
__all__ = ["EvalCase", "EvalReport", "evaluate", "TfidfRetriever", "metrics"]
