"""Load ToolRet benchmark datasets from HuggingFace."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

_AVAILABLE_TASKS: list[str] | None = None


@dataclass(frozen=True)
class GroundTruthTool:
    """A single tool from the ground-truth labels."""

    id: str
    name: str
    description: str
    parameters: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Query:
    """A single benchmark query with its ground-truth tools."""

    id: str
    text: str
    instruction: str = ""
    ground_truth: list[GroundTruthTool] = field(default_factory=list)


@dataclass(frozen=True)
class BenchmarkTask:
    """A complete benchmark task (one ToolRet task family)."""

    name: str
    category: str
    queries: list[Query]


def _resolve_doc(doc: dict[str, Any], relevance: int = 1) -> GroundTruthTool:
    """Normalize a ``doc`` dict from ToolRet into a ``GroundTruthTool``.

    ToolRet uses different ``doc`` schemas across task families:
    - ``apibank``-style: ``{name, description, parameters, ...}``
    - ``gorilla``-style: ``{name, description, api_arguments, ...}``

    This function extracts the common fields and passes the rest through
    as ``metadata`` so callers can inspect task-specific details.

    Args:
        doc: The ``doc`` sub-dict from a ToolRet label entry.
        relevance: The relevance score from the label (passed separately
            because it sits at the label level, not inside ``doc``).
    """
    tool_id = doc.get("id", "")
    name = doc.get("name", "")
    description = doc.get("description", "")
    # Prefer the ``parameters`` field (apibank), fall back to api_arguments (gorilla)
    params = doc.get("parameters", "")
    if not params and "api_arguments" in doc:
        params = json.dumps(doc["api_arguments"])
    metadata = {k: v for k, v in doc.items() if k not in ("id", "name", "description", "parameters", "api_arguments")}
    metadata["relevance"] = relevance
    return GroundTruthTool(
        id=tool_id,
        name=name,
        description=description,
        parameters=params,
        metadata=metadata,
    )


def load_task(task_name: str) -> BenchmarkTask:
    """Load a single ToolRet task from HuggingFace.

    Args:
        task_name: One of the available task configs (e.g. ``"gorilla-pytorch"``,
            ``"apibank"``).  Use :func:`available_tasks` to list them.

    Returns:
        A :class:`BenchmarkTask` with all queries and their ground-truth tools.

    Raises:
        ValueError: If the task name is not found.
    """
    from datasets import load_dataset  # lazy import

    ds = load_dataset("mangopy/ToolRet-Queries", task_name, split="queries")
    category = ds[0].get("category", "") if ds else ""
    queries: list[Query] = []

    for row in ds:
        labels_raw = row.get("labels", "[]")
        labels: list[dict[str, Any]] = json.loads(labels_raw) if isinstance(labels_raw, str) else labels_raw

        ground_truth = [
            _resolve_doc(label["doc"], label.get("relevance", 1))
            for label in labels
            if "doc" in label
        ]
        queries.append(Query(
            id=row["id"],
            text=row["query"],
            instruction=row.get("instruction", ""),
            ground_truth=ground_truth,
        ))

    return BenchmarkTask(name=task_name, category=category, queries=queries)


def available_tasks() -> list[str]:
    """Return the list of available task names (configs)."""
    global _AVAILABLE_TASKS
    if _AVAILABLE_TASKS is None:
        from datasets import get_dataset_config_names

        _AVAILABLE_TASKS = get_dataset_config_names("mangopy/ToolRet-Queries")
    return _AVAILABLE_TASKS
