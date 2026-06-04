# Tool Retrieval Benchmarking

Evaluate `toolz` BM25 tool retrieval quality against the [ToolRet](https://github.com/mangopy/tool-retrieval-benchmark) benchmark datasets.

## Quick Start

```bash
# Install benchmark dependencies
uv pip install -e ".[benchmark]"

# List available tasks
python -m benchmark --list-tasks

# Run a single-task benchmark (3 strategies: BM25-only, BM25+cosine, BM25+MMR)
python -m benchmark --task gorilla-pytorch

# Run with custom options
python -m benchmark \
    --task apibank \
    --strategies bm25 bm25+cosine bm25+mmr \
    --top-k 20 \
    --k-values 5 10 20 \
    --output results.json \
    --verbose
```

## Available Tasks

35 tasks across 3 categories (2147 total queries):

| Category | Tasks | Description |
|----------|-------|-------------|
| **code** | 7 | Programming/ML tools (gorilla-*, craft-*, toolink) |
| **web** | 19 | API/web service tools (apibank, toolbench, tooleyes, etc.) |
| **customized** | 9 | Domain-specific tools (gpt4tools, toolalpaca, etc.) |

Full list: `python -m benchmark --list-tasks`

## Strategies

| Strategy | Description |
|----------|-------------|
| `bm25` | BM25 retrieval only, no reranking |
| `bm25+cosine` | BM25 + cosine similarity reranking |
| `bm25+mmr` | BM25 + Maximal Marginal Relevance reranking |

## Metrics

Computed via [pytrec_eval](https://github.com/aperraan/pytrec_eval):

| Metric | Meaning |
|--------|---------|
| `recall@K` | Fraction of relevant tools found in top-K |
| `precision@K` | Fraction of top-K results that are relevant |
| `ndcg_cut@K` | NDCG at K (accounts for ranking position) |
| `map` | Mean Average Precision (across all K) |
| `recip_rank` | Mean Reciprocal Rank (MRR) |
| `comprehensiveness@K` | 1.0 if recall@K == 1.0, else 0.0 |

K values: 5, 10, 20 (configurable via `--k-values`).

## Programmatic Usage

```python
from benchmark import load_task, run_benchmark

# Load a single task
task = load_task("gorilla-pytorch")
print(f"{task.name}: {len(task.queries)} queries, category={task.category}")

# Run benchmark
results = run_benchmark(
    task_name="gorilla-pytorch",
    strategies=["bm25", "bm25+cosine"],
    top_k=20,
    output_file="results.json",
)

# results is a dict: {strategy_name: {metric@K: value}}
for strategy, metrics in results.items():
    print(f"\n{strategy}:")
    for metric, value in sorted(metrics.items()):
        print(f"  {metric}: {value:.4f}")
```

## How It Works

1. **Dataset loading**: Queries + ground-truth tools are downloaded from HuggingFace (`mangopy/ToolRet-Queries`). Each query has a list of relevant tools with name, description, and parameters.

2. **Registry construction**: A temporary `ToolRegistry` is built from the ground-truth tools in the task. This ensures the BM25 index contains exactly the tools that appear in the ground truth.

3. **Search**: For each strategy, `ToolzSearch` builds a BM25 index and runs all queries through the search pipeline (optional LLM query expansion, BM25 retrieval, optional reranking).

4. **Metrics**: Retrieved results are compared against ground truth using `pytrec_eval`. Tool names are used as document identifiers to align with `ToolDiscoveryResult.name`.

## Notes

- **LLM configuration**: Create a `.env` file in the project root with your OpenAI credentials:

  ```env
  OPENAI_API_KEY=sk-...
  OPENAI_MODEL_NAME=gpt-4o-mini
  OPENAI_BASE_URL=https://api.openai.com/v1  # optional, for custom endpoints
  ```

  The benchmark auto-loads `.env` and configures DSPy when credentials are present.

- **No LLM?** The benchmark still works without an LLM — it falls back to raw tokenization
  of tool metadata (name, description, parameters) for BM25 indexing. Results will be less
  precise since BM25 won't have LLM-generated keyword boosts, but the pipeline completes
  without errors.

- **First-run download**: ToolRet datasets are downloaded on first use and cached by the
  HuggingFace `datasets` library.

- **Reproducibility**: Results may vary between runs due to LLM-based keyword generation.
  Seed the LLM or use deterministic settings for reproducible benchmarks.
