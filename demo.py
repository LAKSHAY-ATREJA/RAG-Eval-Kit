"""
demo.py - end-to-end demonstration of rag-eval-kit

Run:
    python demo.py

No external dependencies are required. The script exercises the library
API directly, then calls the CLI against the bundled example dataset.
"""

import subprocess
import sys

from rageval import TfidfRetriever, EvalCase, evaluate


def demo_library_api():
    print("=" * 60)
    print("Demo 1: Library API")
    print("=" * 60)

    corpus = {
        "doc1": "retrieval augmented generation grounds answers in documents",
        "doc2": "faiss enables fast similarity search over embeddings",
        "doc3": "cosine similarity measures the angle between two vectors",
        "doc4": "precision and recall are classic information retrieval metrics",
        "doc5": "langchain provides building blocks for chaining llm calls",
        "doc6": "a vector database stores embeddings for nearest neighbour search",
    }

    retriever = TfidfRetriever().index(corpus)

    cases = [
        EvalCase(query="how does retrieval augmented generation work", relevant=["doc1", "doc5"]),
        EvalCase(query="fast similarity search over embeddings", relevant=["doc2", "doc3", "doc6"]),
        EvalCase(query="metrics for evaluating ranked search results", relevant=["doc4", "doc3"]),
    ]

    report = evaluate(retriever.ranked_ids, cases, k=3)
    print(report.pretty())
    print()
    print("Raw scores dict:")
    for name, value in report.scores.items():
        print(f"  {name}: {value:.4f}")

    print()
    print("Per-query detail:")
    for row in report.per_case:
        q = row["query"][:50]
        hit = row["hit@k"]
        rr = row["mrr"]
        print(f"  {q!r:55s}  hit@3={hit:.1f}  mrr={rr:.3f}")

    print()


def demo_custom_retriever():
    print("=" * 60)
    print("Demo 2: Custom retriever (always returns doc1 first)")
    print("=" * 60)

    def my_oracle_retriever(query: str, k: int):
        return ["doc1", "doc2", "doc3"][:k]

    cases = [EvalCase(query="anything", relevant=["doc1"])]
    report = evaluate(my_oracle_retriever, cases, k=3)
    print(report.pretty())
    print()


def demo_cli():
    print("=" * 60)
    print("Demo 3: CLI against bundled example dataset")
    print("=" * 60)
    result = subprocess.run(
        [sys.executable, "-m", "rageval.cli", "examples/eval_set.json", "--k", "3"],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print("stderr:", result.stderr)

    print("CLI with --json and --per-case flags:")
    result2 = subprocess.run(
        [sys.executable, "-m", "rageval.cli", "examples/eval_set.json", "--k", "3", "--json", "--per-case"],
        capture_output=True,
        text=True,
    )
    print(result2.stdout[:800])
    print()


if __name__ == "__main__":
    demo_library_api()
    demo_custom_retriever()
    demo_cli()
    print("All demos complete.")
