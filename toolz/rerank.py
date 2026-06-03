"""Reranking strategies for ToolzSearch results."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from toolregistry import ToolRegistry


class Reranker(ABC):
    """Base class for reranking strategies."""

    @abstractmethod
    def rerank(
        self,
        results: list[tuple[dict[str, Any], float]],
        query: str,
        registry: ToolRegistry,
    ) -> list[tuple[dict[str, Any], float]]:
        """Rerank (doc, score) pairs and return new ordered list."""


class CosineReranker(Reranker):
    """Rerank by cosine similarity on tool descriptions + name-match bonus."""

    def __init__(self, name_bonus: float = 0.05) -> None:
        self._name_bonus = name_bonus

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [t for t in text.lower().split() if t.isalpha() and len(t) >= 2]

    def _cosine(self, a: dict[str, int], b: dict[str, int]) -> float:
        if not a or not b:
            return 0.0
        common = set(a.keys()) & set(b.keys())
        if not common:
            return 0.0
        dot = sum(a[t] * b[t] for t in common)
        na = sum(v * v for v in a.values()) ** 0.5
        nb = sum(v * v for v in b.values()) ** 0.5
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def rerank(
        self,
        results: list[tuple[dict[str, Any], float]],
        query: str,
        registry: ToolRegistry,
    ) -> list[tuple[dict[str, Any], float]]:
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return results

        query_freq: dict[str, int] = {}
        for t in query_tokens:
            query_freq[t] = query_freq.get(t, 0) + 1

        reranked: list[tuple[dict[str, Any], float]] = []
        for doc, bm25_score in results:
            tool_name = doc.get("tool_name", "")
            tool = registry.get_tool(tool_name)
            description = tool.description if tool and tool.description else ""

            desc_tokens = self._tokenize(description)
            desc_freq: dict[str, int] = {}
            for t in desc_tokens:
                desc_freq[t] = desc_freq.get(t, 0) + 1

            cos_sim = self._cosine(query_freq, desc_freq)
            combined = bm25_score + cos_sim

            name_lower = tool_name.lower().replace("_", " ")
            for qt in query_tokens:
                if qt in name_lower:
                    combined += self._name_bonus
                    break

            reranked.append((doc, combined))

        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked


class MMRReranker(Reranker):
    """Maximal Marginal Relevance — balances relevance and diversity.

    Iteratively selects results that are highly relevant to the query
    but also diverse with respect to already-selected results.

    Args:
        lambda_mult: Balance between relevance (1) and diversity (0).
            Default 0.7 means 70% relevance, 30% diversity.
    """

    def __init__(self, lambda_mult: float = 0.7) -> None:
        if not 0 <= lambda_mult <= 1:
            raise ValueError("lambda_mult must be between 0 and 1")
        self._lambda_mult = lambda_mult

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [t for t in text.lower().split() if t.isalpha() and len(t) >= 2]

    def _cosine(self, a: dict[str, int], b: dict[str, int]) -> float:
        if not a or not b:
            return 0.0
        common = set(a.keys()) & set(b.keys())
        if not common:
            return 0.0
        dot = sum(a[t] * b[t] for t in common)
        na = sum(v * v for v in a.values()) ** 0.5
        nb = sum(v * v for v in b.values()) ** 0.5
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def rerank(
        self,
        results: list[tuple[dict[str, Any], float]],
        query: str,
        registry: ToolRegistry,
    ) -> list[tuple[dict[str, Any], float]]:
        query_tokens = self._tokenize(query)
        if not query_tokens or len(results) <= 1:
            return results

        query_freq: dict[str, int] = {}
        for t in query_tokens:
            query_freq[t] = query_freq.get(t, 0) + 1

        # Pre-compute description frequencies and BM25 relevance scores
        candidates: list[tuple[dict[str, Any], float, dict[str, int]]] = []
        for doc, bm25_score in results:
            tool_name = doc.get("tool_name", "")
            tool = registry.get_tool(tool_name)
            description = tool.description if tool and tool.description else ""
            desc_tokens = self._tokenize(description)
            desc_freq: dict[str, int] = {}
            for t in desc_tokens:
                desc_freq[t] = desc_freq.get(t, 0) + 1
            candidates.append((doc, bm25_score, desc_freq))

        selected: list[tuple[dict[str, Any], float, dict[str, int]]] = []
        remaining = list(candidates)

        while remaining:
            best_score = -1.0
            best_idx = 0
            best_candidate = None

            for i, (doc, rel, desc_freq) in enumerate(remaining):
                # Relevance term
                relevance = rel
                # Diversity term: max similarity to already selected
                max_sim = 0.0
                for _, _, sel_desc in selected:
                    s = self._cosine(desc_freq, sel_desc)
                    if s > max_sim:
                        max_sim = s

                mmr_score = (
                    self._lambda_mult * relevance
                    - (1 - self._lambda_mult) * max_sim
                )
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
                    best_candidate = (doc, rel, desc_freq)

            remaining.pop(best_idx)
            selected.append(best_candidate)

        return [(doc, score) for doc, score, _ in selected]


class NoopReranker(Reranker):
    """Pass-through reranker — returns results unchanged."""

    def rerank(
        self,
        results: list[tuple[dict[str, Any], float]],
        query: str,
        registry: ToolRegistry,
    ) -> list[tuple[dict[str, Any], float]]:
        return results


# Registry of available rerankers
_RERANKERS: dict[str, type[Reranker]] = {
    "cosine": CosineReranker,
    "mmr": MMRReranker,
    "none": NoopReranker,
    "noop": NoopReranker,
}


def get_reranker(name: str, **kwargs: Any) -> Reranker:
    """Create a reranker by name.

    Args:
        name: Reranker identifier. One of ``cosine``, ``mmr``, ``none``.
        **kwargs: Passed to the reranker constructor.
            For ``mmr``, use ``lambda_mult`` (default 0.7).

    Returns:
        A :class:`Reranker` instance.

    Raises:
        ValueError: If ``name`` is not recognized.

    Examples:
        >>> get_reranker("cosine")
        >>> get_reranker("mmr", lambda_mult=0.8)
        >>> get_reranker("none")
    """
    cls = _RERANKERS.get(name.lower())
    if cls is None:
        available = ", ".join(sorted(_RERANKERS.keys()))
        raise ValueError(f"Unknown reranker '{name}'. Available: {available}")
    return cls(**kwargs)
