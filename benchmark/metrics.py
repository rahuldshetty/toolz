"""IR metric computation using pytrec_eval."""

from __future__ import annotations

from typing import Any, Sequence

import pytrec_eval

# Default K values used by the ToolRet benchmark protocol.
DEFAULT_K_VALUES: tuple[int, ...] = (5, 10, 20)

# pytrec_eval measure names with K-parameterized variants.
# Format: (pytrec_measure, base_name_for_output)
_MEASURES = [
    "recall",       # recall.K
    "P",            # P.K (precision)
    "ndcg_cut",     # ndcg_cut.K
    "map",          # map (no K param)
    "recip_rank",   # recip_rank (MRR)
]


def _build_measure_strings(k_values: Sequence[int]) -> list[str]:
    """Build the list of measure strings for pytrec_eval.

    pytrec_eval uses the format ``measure.K`` for parameterized measures.
    """
    measures: list[str] = []
    for k in k_values:
        for base in ("recall", "P", "ndcg_cut"):
            measures.append(f"{base}.{k}")
    # Non-parameterized measures
    measures.extend(("map", "recip_rank"))
    return measures


def compute_metrics(
    qrels: dict[str, dict[str, int]],
    results: dict[str, dict[str, float]],
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> dict[str, float]:
    """Compute IR metrics from ground-truth and retrieval results.

    Args:
        qrels: ``{query_id: {tool_id: relevance_score, ...}, ...}``
        results: ``{query_id: {tool_id: score, ...}, ...}``
            Only the top-K tools per query should be included.
        k_values: Which K values to compute metrics for.

    Returns:
        Flat dictionary mapping ``{metric@K: value}``.  Also includes
        ``comprehensiveness@K`` (1.0 iff ``recall@K == 1.0``).
    """
    measure_strings = _build_measure_strings(k_values)

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, measure_strings)
    scores = evaluator.evaluate(results)

    output: dict[str, float] = {}

    # Parameterized metrics (recall, precision, ndcg_cut)
    for base in ("recall", "P", "ndcg_cut"):
        for k in k_values:
            key = f"{base}@{k}"
            # pytrec_eval key format: "base.K"
            measure_key = f"{base}.{k}"
            values = []
            for query_scores in scores.values():
                values.append(query_scores.get(measure_key, 0.0))
            avg = sum(values) / len(values) if values else 0.0
            output[key] = round(avg, 6)

    # Non-parameterized metrics
    output["map"] = round(
        sum(s.get("map", 0.0) for s in scores.values()) / len(scores)
        if scores else 0.0,
        6,
    )
    output["recip_rank"] = round(
        sum(s.get("recip_rank", 0.0) for s in scores.values()) / len(scores)
        if scores else 0.0,
        6,
    )

    # Comprehensiveness: 1.0 iff recall@K == 1.0
    for k in k_values:
        recall_val = output.get(f"recall@{k}", 0.0)
        output[f"comprehensiveness@{k}"] = 1.0 if recall_val >= 1.0 else 0.0

    return output


def _results_to_trec_format(
    queries: list[Any],
    search_fn,
    top_k: int,
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, float]]]:
    """Convert benchmark queries and a search function into trec_eval format.

    Uses tool **names** as document identifiers so they align with the
    ``ToolDiscoveryResult.name`` returned by :class:`~toolz.search.ToolzSearch`.

    Args:
        queries: List of :class:`Query` objects.
        search_fn: Callable ``(query_text: str, top_k: int) -> list[ToolDiscoveryResult]``.
        top_k: Maximum number of results to retrieve per query.

    Returns:
        ``(qrels, results)`` suitable for :func:`compute_metrics`.
    """
    qrels: dict[str, dict[str, int]] = {}
    trec_results: dict[str, dict[str, float]] = {}

    for query in queries:
        qrels[query.id] = {
            gt_tool.name: gt_tool.metadata.get("relevance", 1)
            for gt_tool in query.ground_truth
        }

        retrieved = search_fn(query.text, top_k)
        trec_results[query.id] = {
            result.name: float(result.score) for result in retrieved[:top_k]
        }

    return qrels, trec_results
