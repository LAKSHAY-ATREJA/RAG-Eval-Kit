"""
demo.py - end-to-end demonstration of rag-eval-kit

Run:
    python demo.py

No external dependencies required. The script exercises the library API,
demonstrates retriever comparison, shows CSV and JSON export, and finally
calls the CLI against the bundled example dataset.
"""

import subprocess
import sys

from rageval import EvalCase, TfidfRetriever, compare, evaluate

# Sample corpus used across multiple demos
CORPUS = {
    "doc1": "retrieval augmented generation grounds language model answers in source documents",
    "doc2": "faiss enables fast approximate nearest neighbour search over dense vector embeddings",
    "doc3": "cosine similarity measures the angle between two vectors and ranks text relevance",
    "doc4": "precision recall and F1 are classic information retrieval metrics for ranked results",
    "doc5": "langchain provides composable building blocks for chaining language model calls",
    "doc6": "a vector database stores embeddings and supports fast nearest neighbour lookup at scale",
    "doc7": "sparse retrieval with BM25 uses term frequency and inverse document frequency scores",
    "doc8": "hybrid search combines dense vector search with sparse keyword matching for better coverage",
}

CASES = [
    EvalCase(
        query="how does retrieval augmented generation work",
        relevant=["doc1", "doc5"],
    ),
    EvalCase(
        query="fast similarity search over embeddings",
        relevant=["doc2", "doc3", "doc6"],
    ),
    EvalCase(
        query="metrics for evaluating ranked search results",
        relevant=["doc4", "doc3"],
    ),
    EvalCase(
        query="hybrid dense and sparse retrieval",
        relevant=["doc7", "doc8", "doc2"],
    ),
]


def separator(title: str) -> None:
    width = 64
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def demo_library_api() -> None:
    separator("Demo 1: Library API — basic evaluation")

    retriever = TfidfRetriever().index(CORPUS)
    print(f"Indexed {retriever.n_docs} documents, vocabulary size: {retriever.vocabulary_size}")
    print()

    report = evaluate(retriever.ranked_ids, CASES, k=3, retriever_name="TF-IDF baseline")
    print(report.pretty())

    print()
    print("Per-query detail (hit@3 and MRR):")
    for row in report.per_case:
        q = row["query"][:55]
        hit = row["hit@k"]
        rr = row["mrr"]
        f1 = row["f1@k"]
        print(f"  {q!r:58s}  hit@3={hit:.0f}  mrr={rr:.3f}  f1@3={f1:.3f}")


def demo_custom_retriever() -> None:
    separator("Demo 2: Custom / oracle retriever")

    # Simulates a retriever that always returns the perfect answer for every query.
    def perfect_oracle(query: str, k: int):
        mapping = {
            "how does retrieval augmented generation work": ["doc1", "doc5"],
            "fast similarity search over embeddings": ["doc2", "doc6", "doc3"],
            "metrics for evaluating ranked search results": ["doc4", "doc3"],
            "hybrid dense and sparse retrieval": ["doc7", "doc8", "doc2"],
        }
        return mapping.get(query, [])[:k]

    # Simulates a weaker retriever that only ever returns one fixed document.
    def naive_retriever(query: str, k: int):
        return ["doc1"] * min(k, 1)

    report = evaluate(perfect_oracle, CASES, k=3, retriever_name="perfect oracle")
    print(report.pretty())
    print()

    report_naive = evaluate(naive_retriever, CASES, k=3, retriever_name="naive (always doc1)")
    print(report_naive.pretty())


def demo_compare() -> None:
    separator("Demo 3: Comparing two retrievers side-by-side")

    retriever = TfidfRetriever().index(CORPUS)

    # A second variant that uses a smaller k internally — simulates a weaker retriever
    def restricted_retriever(query: str, k: int):
        return retriever.ranked_ids(query, k=max(1, k // 2))

    reports = compare(
        retrievers={
            "tfidf-full": retriever.ranked_ids,
            "tfidf-restricted": restricted_retriever,
        },
        cases=CASES,
        k=5,
    )

    print(f"{'Metric':<18}", end="")
    for name in reports:
        print(f"  {name:>20}", end="")
    print()
    print("-" * 60)

    metric_names = list(next(iter(reports.values())).scores.keys())
    for metric in metric_names:
        print(f"  {metric:<16}", end="")
        for report in reports.values():
            val = report.scores[metric]
            print(f"  {val:>20.4f}", end="")
        print()


def demo_export() -> None:
    separator("Demo 4: Exporting results to JSON and CSV")

    retriever = TfidfRetriever().index(CORPUS)
    report = evaluate(retriever.ranked_ids, CASES, k=3, retriever_name="TF-IDF")

    print("JSON output (first 500 chars):")
    print(report.to_json()[:500])
    print("...")

    print()
    print("CSV output:")
    print(report.to_csv())


def demo_cli() -> None:
    separator("Demo 5: CLI against bundled example dataset")

    print("$ rag-eval examples/eval_set.json --k 3")
    result = subprocess.run(
        [sys.executable, "-m", "rageval.cli", "examples/eval_set.json", "--k", "3"],
        capture_output=True,
        text=True,
        cwd="/Users/lakshay/RAG-Eval-Kit",
    )
    print(result.stdout)
    if result.returncode != 0:
        print("stderr:", result.stderr, file=sys.stderr)

    print()
    print("$ rag-eval examples/eval_set.json --k 3 --per-case")
    result2 = subprocess.run(
        [
            sys.executable,
            "-m",
            "rageval.cli",
            "examples/eval_set.json",
            "--k",
            "3",
            "--per-case",
        ],
        capture_output=True,
        text=True,
        cwd="/Users/lakshay/RAG-Eval-Kit",
    )
    print(result2.stdout)
    if result2.returncode != 0:
        print("stderr:", result2.stderr, file=sys.stderr)

    print()
    print("$ rag-eval examples/eval_set.json --k 3 --json (abbreviated)")
    result3 = subprocess.run(
        [
            sys.executable,
            "-m",
            "rageval.cli",
            "examples/eval_set.json",
            "--k",
            "3",
            "--json",
        ],
        capture_output=True,
        text=True,
        cwd="/Users/lakshay/RAG-Eval-Kit",
    )
    print(result3.stdout[:600])
    if result3.returncode != 0:
        print("stderr:", result3.stderr, file=sys.stderr)


if __name__ == "__main__":
    demo_library_api()
    demo_custom_retriever()
    demo_compare()
    demo_export()
    demo_cli()

    print()
    print("All demos complete.")
