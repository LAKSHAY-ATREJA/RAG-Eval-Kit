# RAG Eval Kit

A dependency-light toolkit for evaluating RAG retrieval quality. Most RAG projects stop at "the system returns an answer." RAG Eval Kit measures whether the retriever actually surfaces the right documents — the step that determines whether a RAG system is trustworthy in production.

The core library has zero runtime dependencies. Every metric is pure, auditable Python that runs anywhere Python 3.9 or later is available.

---

## Why this exists

A retrieval-augmented generation system is only as good as its retriever. If the right chunks never make it into the context window, no amount of prompt engineering will save the final answer. Yet most projects ship without measuring retrieval quality at all.

RAG Eval Kit gives you:

- Standard information-retrieval metrics computed over a labelled eval set
- A reference TF-IDF retriever to score out of the box
- A CLI for running evaluations against a JSON dataset without writing any code
- A `compare()` function for benchmarking multiple retrievers against the same cases
- JSON and CSV export so results plug into dashboards, CI pipelines, or spreadsheets

---

## Metrics

| Metric      | What it measures                                              |
|-------------|---------------------------------------------------------------|
| Precision@k | Of the top-k retrieved documents, what fraction were relevant |
| Recall@k    | Of all relevant documents, what fraction appeared in the top-k |
| F1@k        | Harmonic mean of precision@k and recall@k                    |
| Hit@k       | Whether at least one relevant document appeared in the top-k  |
| MRR         | Reciprocal rank of the first relevant document                |
| MAP         | Mean average precision across all relevant hits               |
| nDCG@k      | Rank-weighted relevance (rewards placing good documents higher) |

---

## Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/LAKSHAY-ATREJA/RAG-Eval-Kit.git
cd RAG-Eval-Kit
pip install -e .
```

No other packages are needed to run the library or CLI. Development tools (pytest, ruff, coverage) are in `requirements-dev.txt`:

```bash
pip install -r requirements-dev.txt
```

---

## How to run locally

### Command-line interface

Evaluate the bundled TF-IDF retriever against the included example dataset:

```bash
rag-eval examples/eval_set.json --k 3
```

Output:

```
RAG retrieval evaluation — TF-IDF  (n=4, k=3)
----------------------------------------------------
precision@k      0.5000
recall@k         0.7500
f1@k             0.5750
hit@k            1.0000
mrr              1.0000
map              0.7500
ndcg@k           0.8066
```

Show per-query scores:

```bash
rag-eval examples/eval_set.json --k 5 --per-case
```

Emit results as JSON:

```bash
rag-eval examples/eval_set.json --k 5 --json
```

Save results to a file instead of stdout:

```bash
rag-eval examples/eval_set.json --k 5 --json --output results.json
```

### Library API

Evaluate any retriever by wrapping it in a callable that takes `(query, k)` and returns a ranked list of document ids:

```python
from rageval import TfidfRetriever, EvalCase, evaluate

corpus = {
    "doc1": "retrieval augmented generation grounds answers in documents",
    "doc2": "faiss enables fast similarity search over embeddings",
    "doc3": "cosine similarity measures the angle between two vectors",
}

retriever = TfidfRetriever().index(corpus)

cases = [
    EvalCase(query="how does RAG work", relevant=["doc1"]),
    EvalCase(query="fast similarity search", relevant=["doc2", "doc3"]),
]

report = evaluate(retriever.ranked_ids, cases, k=3)
print(report.pretty())
print(report.scores)    # dict of metric -> float
```

To benchmark your production retriever, wrap it in the same interface:

```python
def my_retriever(query: str, k: int) -> list[str]:
    return [hit.id for hit in my_vector_db.search(query, top_k=k)]

report = evaluate(my_retriever, cases, k=5)
```

### Comparing multiple retrievers

```python
from rageval import compare

reports = compare(
    retrievers={
        "tfidf": tfidf_retriever.ranked_ids,
        "bm25": bm25_retriever.ranked_ids,
    },
    cases=cases,
    k=5,
)

for name, report in reports.items():
    print(report.pretty())
```

### Exporting results

```python
# JSON string
print(report.to_json())

# CSV string (one row per query, last row is the aggregate mean)
print(report.to_csv())

# Plain dict for custom serialisation
data = report.to_dict()
```

---

## Running the demo

The `demo.py` script exercises every part of the toolkit with no setup required:

```bash
python demo.py
```

The demo covers:
1. Basic library API evaluation over a sample corpus
2. Custom oracle retriever vs naive retriever
3. Side-by-side retriever comparison using `compare()`
4. JSON and CSV export
5. CLI invocation against the bundled dataset

---

## Dataset format

The JSON eval set file must follow this schema:

```json
{
  "corpus": {
    "doc1": "full text of document one",
    "doc2": "full text of document two"
  },
  "queries": [
    {
      "query": "a natural language question",
      "relevant": ["doc1"]
    },
    {
      "query": "another question",
      "relevant": ["doc1", "doc2"]
    }
  ]
}
```

The `corpus` is used to build the retriever index. Each query in `queries` specifies the ground-truth document ids that a good retriever should return. See `examples/eval_set.json` for a complete example.

---

## Environment variables

The core library requires no API keys or environment variables. If you extend the toolkit to call an external vector database or embedding service, add credentials to a `.env` file. A template is provided in `.env.example`.

---

## Development

Run the test suite:

```bash
pytest --cov=rageval
```

Run the linter:

```bash
ruff check .
```

All 56 tests pass on Python 3.9 through 3.14.

---

## Project layout

```
rageval/
    __init__.py     public API surface (EvalCase, EvalReport, evaluate, compare, TfidfRetriever)
    metrics.py      pure-Python IR metrics: precision, recall, F1, hit, MRR, MAP, nDCG
    retriever.py    reference TF-IDF retriever with cosine normalisation
    evaluator.py    aggregates metrics over a labelled eval set, compare() helper
    cli.py          rag-eval command-line entry point
tests/
    test_rageval.py 56-case pytest suite covering all metrics, retriever, and evaluator
examples/
    eval_set.json   sample dataset with 8-doc corpus and 4 labelled queries
demo.py             runnable end-to-end demonstration
.env.example        template for environment variables
```

---

## License

MIT. See LICENSE.

Built by Lakshay Atreja.
