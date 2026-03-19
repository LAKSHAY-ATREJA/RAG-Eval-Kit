# RAG Eval Kit

A dependency-light toolkit for evaluating RAG retrieval quality. Most RAG demos stop at "it returns an answer." This measures whether the retriever actually surfaces the right documents — the part that determines whether a RAG system is trustworthy in production.

The core library has zero runtime dependencies. Every metric is pure, auditable Python.

## Why this exists

A retrieval-augmented generation system is only as good as its retriever. If the right chunks never make it into the context window, no amount of prompt engineering will save the answer. Yet most projects ship without ever measuring retrieval quality. RAG Eval Kit provides standard information-retrieval metrics, a reference TF-IDF retriever to score, and a CLI to run evaluations over a labelled dataset — so retrieval quality becomes a number you can track and regression-test against.

## Metrics

| Metric      | What it answers                                            |
|-------------|------------------------------------------------------------|
| Precision@k | Of the top-k retrieved, how many were relevant?            |
| Recall@k    | Of all relevant docs, how many did we retrieve in top-k?   |
| Hit@k       | Did any relevant doc make the top-k?                       |
| MRR         | How highly ranked was the first relevant doc?              |
| MAP         | Mean average precision across all relevant hits            |
| nDCG@k      | Rank-weighted relevance (rewards putting good docs higher) |

## Installation

```bash
git clone https://github.com/LAKSHAY-ATREJA/RAG-Eval-Kit.git
cd RAG-Eval-Kit
pip install -e .
```

## Quick start — CLI

```bash
rag-eval examples/eval_set.json --k 3
```

Output:

```
RAG retrieval evaluation  (n=4, k=3)
--------------------------------------------
precision@k      0.5000
recall@k         0.7500
hit@k            1.0000
mrr              1.0000
map              0.7500
ndcg@k           0.8066
```

Additional flags:

```bash
rag-eval examples/eval_set.json --k 5 --json             # emit JSON
rag-eval examples/eval_set.json --k 5 --per-case         # show per-query breakdown
```

## Quick start — library

Evaluate any retriever by passing a `(query, k) -> ranked doc ids` callable:

```python
from rageval import TfidfRetriever, EvalCase, evaluate

corpus = {
    "doc1": "retrieval augmented generation grounds answers in documents",
    "doc2": "faiss enables fast similarity search over embeddings",
}
retriever = TfidfRetriever().index(corpus)

cases = [EvalCase(query="how does RAG work", relevant=["doc1"])]

report = evaluate(retriever.ranked_ids, cases, k=3)
print(report.pretty())
print(report.scores)   # dict of metric -> score
```

To benchmark your production retriever, wrap it:

```python
def my_retriever(query: str, k: int):
    return [hit.id for hit in my_vector_db.search(query, top_k=k)]

report = evaluate(my_retriever, cases, k=5)
```

## Demo

Run the full end-to-end demonstration (no dependencies beyond the package):

```bash
python demo.py
```

The demo covers library API usage, a custom retriever, and the CLI against the bundled example dataset.

## Dataset format

```json
{
  "corpus": { "doc1": "text...", "doc2": "text..." },
  "queries": [
    { "query": "a question", "relevant": ["doc1"] }
  ]
}
```

## Development

```bash
pip install -e . -r requirements-dev.txt
pytest --cov=rageval     # run tests with coverage
ruff check .             # lint
```

## Project layout

```
rageval/
    metrics.py      pure-Python IR metrics (precision, recall, MRR, MAP, nDCG)
    retriever.py    reference TF-IDF retriever
    evaluator.py    aggregates metrics over a labelled eval set
    cli.py          command-line entry point
tests/
    test_rageval.py full pytest suite
examples/
    eval_set.json   sample dataset with 8-doc corpus and 4 queries
demo.py             runnable end-to-end demonstration
```

## License

MIT. See LICENSE.

Built by Lakshay Atreja.
