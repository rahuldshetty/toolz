"""Orchestrate benchmark runs across strategies."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Sequence

from toolregistry import Tool, ToolRegistry

from .dataset import BenchmarkTask, Query, load_task
from .metrics import compute_metrics, _results_to_trec_format

logger = logging.getLogger(__name__)

# Built-in strategy configurations.
STRATEGY_CONFIGS: dict[str, dict[str, Any]] = {
    "bm25": {"rerank": "none", "use_llm_query_expansion": False},
    "bm25+cosine": {"rerank": "cosine", "use_llm_query_expansion": False},
    "bm25+mmr": {"rerank": "mmr", "use_llm_query_expansion": False},
}


def _build_registry_from_task(task: BenchmarkTask) -> ToolRegistry:
    """Build a temporary ToolRegistry from the ground-truth tools in a task.

    This ensures the BM25 index contains exactly the tools that appear in
    the ground-truth labels, so retrieval can match them.
    """
    registry = ToolRegistry()

    # Collect unique tools (by name) across all queries.
    seen: set[str] = set()
    for query in task.queries:
        for gt_tool in query.ground_truth:
            if gt_tool.name not in seen:
                # Register a minimal Tool — the BM25 index only needs
                # name, description, and parameters.
                registry.register(Tool(
                    name=gt_tool.name,
                    description=gt_tool.description or "",
                    parameters={"properties": {}},
                    callable=lambda: None,
                    metadata={"tags": [], "search_hint": gt_tool.parameters},
                ))
                seen.add(gt_tool.name)

    return registry


def _make_search_fn(
    registry: ToolRegistry,
    strategy_kwargs: dict[str, Any],
) -> Callable[[str, int], list[Any]]:
    """Create a search function bound to a specific strategy configuration.

    Returns a callable that matches the signature expected by
    :func:`_results_to_trec_format`.
    """
    from toolz.search import ToolzSearch  # lazy import

    search = ToolzSearch(registry, **strategy_kwargs)
    search.build_index()

    def search_fn(query: str, top_k: int):
        return search.search(query, top_k=top_k)

    return search_fn


def run_benchmark(
    task_name: str,
    strategies: Sequence[str] | None = None,
    top_k: int = 20,
    output_file: str | Path | None = None,
    k_values: Sequence[int] = (5, 10, 20),
) -> dict[str, dict[str, float]]:
    """Run the full benchmark on a single task across multiple strategies.

    Args:
        task_name: Task config name (e.g. ``"gorilla-pytorch"``).
        strategies: Strategy names to evaluate.  Defaults to all built-in
            strategies (``bm25``, ``bm25+cosine``, ``bm25+mmr``).
        top_k: Maximum number of results to retrieve per query.
        output_file: Optional path to serialize raw results as JSON.
        k_values: K values for metric computation.

    Returns:
        Nested dict: ``{strategy_name: {metric@K: value, ...}}``.
    """
    if strategies is None:
        strategies = list(STRATEGY_CONFIGS.keys())

    task = load_task(task_name)
    logger.info(
        "Benchmarking task=%s (%d queries, category=%s) across %s strategies",
        task.name,
        len(task.queries),
        task.category,
        len(strategies),
    )

    registry = _build_registry_from_task(task)
    results: dict[str, dict[str, float]] = {}

    for strategy_name in strategies:
        config = STRATEGY_CONFIGS.get(strategy_name)
        if config is None:
            logger.warning("Unknown strategy '%s', skipping", strategy_name)
            continue

        logger.info("Running strategy: %s", strategy_name)
        search_fn = _make_search_fn(registry, config)

        qrels, trec_results = _results_to_trec_format(
            task.queries, search_fn, top_k
        )
        metrics = compute_metrics(qrels, trec_results, k_values=k_values)
        results[strategy_name] = metrics
        logger.info(
            "  %s: recall@5=%.4f  mrr@5=%.4f  ndcg@5=%.4f",
            strategy_name,
            metrics.get("recall@5", 0),
            metrics.get("map@5", 0),
            metrics.get("ndcg_cut@5", 0),
        )

    if output_file is not None:
        _serialize_results(results, task_name, strategies, top_k, output_file)

    return results


def _serialize_results(
    results: dict[str, dict[str, float]],
    task_name: str,
    strategies: Sequence[str],
    top_k: int,
    output_file: str | Path,
) -> None:
    """Write benchmark results to a JSON file."""
    payload = {
        "task": task_name,
        "top_k": top_k,
        "strategies": list(strategies),
        "results": results,
    }
    Path(output_file).write_text(json.dumps(payload, indent=2))
    logger.info("Results written to %s", output_file)
