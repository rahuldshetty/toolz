"""LLM-enhanced BM25 tool discovery over a ToolRegistry."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import bm25s
import dspy
import numpy as np

from toolregistry import ToolRegistry, Tool
from toolregistry.tool_registry import TOOL_DISCOVERY_NAME

from .keywords import BM25KeywordSignature, QueryExpansionSignature
from .rerank import get_reranker, Reranker
from .results import ToolDiscoveryResult

logger = logging.getLogger(__name__)

__all__ = ["ToolzSearch"]


class ToolzSearch:
    """BM25 search over a :class:`~toolregistry.ToolRegistry` with LLM-generated keywords.

    At index-build time, an LLM (via DSPy) generates BM25-optimal keywords
    from each tool's metadata. These keywords are concatenated with raw
    metadata and indexed using bm25s.

    Optionally rerank results using pluggable strategies (cosine similarity,
    MMR, or none).

    Example::

        from toolregistry import ToolRegistry
        from toolz.search import ToolzSearch

        registry = ToolRegistry()
        registry.register(read_file)

        search = ToolzSearch(registry, rerank="cosine")
        search.build_index()
        results = search.search("read a file", top_k=5)
    """

    _KEYWORD_BOOST = 3
    _BM25_PARAMS = dict(k1=1.5, b=0.75, delta=0.5, method="lucene")

    def __init__(
        self,
        registry: ToolRegistry,
        use_llm_query_expansion: bool = False,
        lm: dspy.LM | None = None,
        keyword_min: int = 10,
        keyword_max: int = 100,
        rerank: str = "none",
        reranker_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.registry = registry
        self.use_llm_query_expansion = use_llm_query_expansion
        self._bm25 = bm25s.BM25(corpus=[], **self._BM25_PARAMS)
        if lm is not None:
            dspy.configure(lm=lm)
        self._keyword_predictor = dspy.Predict(BM25KeywordSignature)
        self._query_predictor = dspy.Predict(QueryExpansionSignature)
        self._corpora: list[dict[str, Any]] = []
        self._keyword_min = keyword_min
        self._keyword_max = keyword_max
        self._reranker: Reranker = get_reranker(rerank, **(reranker_kwargs or {}))

    @staticmethod
    def _clean_token(token: str) -> str | None:
        """Clean a token into a keyword, or return None if invalid."""
        clean = token.lower().strip().strip(".,;:!?\"'()[]{}")
        if clean and clean.isalpha() and 2 <= len(clean) <= 50:
            return clean
        return None

    def build_index(self) -> None:
        """Build the BM25 index from all tools in the registry."""
        self._corpora = []

        for name, tool in self.registry._tools.items():
            if name == TOOL_DISCOVERY_NAME:
                continue

            tool_name = tool.name
            description = tool.description or ""
            param_str = self._extract_param_info(tool)
            tags_str = " ".join(tool.metadata.all_tags) if tool.metadata else ""
            search_hint = (
                tool.metadata.search_hint if tool.metadata and tool.metadata.search_hint else ""
            )

            llm_keywords = self._generate_keywords(
                tool_name=tool_name,
                description=description,
                parameters=param_str,
                tags=tags_str,
                search_hint=search_hint,
            )

            self._corpora.append({
                "tool_name": tool_name,
                "keywords": " ".join(llm_keywords),
                "raw": f"{tool_name} {description} {param_str} {tags_str}",
                "tags": list(tool.metadata.all_tags) if tool.metadata else [],
                "namespace": tool.namespace,
            })

        if not self._corpora:
            self._bm25 = bm25s.BM25(corpus=[], **self._BM25_PARAMS)
            return

        indexed_texts = []
        for doc in self._corpora:
            kw = doc["keywords"]
            raw = doc["raw"]
            boosted_kw = " ".join(kw.split()) if kw else ""
            indexed_texts.append(
                " ".join([boosted_kw] * self._KEYWORD_BOOST + [raw]) if boosted_kw else raw
            )

        tokenized = bm25s.tokenize(indexed_texts, return_ids=False, show_progress=False)
        self._bm25 = bm25s.BM25(corpus=self._corpora, **self._BM25_PARAMS)
        self._bm25.index(tokenized)

    def search(
        self,
        query: str,
        top_k: int = 10,
        tags: Sequence[str] | None = None,
        include_schema: bool = False,
        api_format: str = "openai-chat",
        use_llm_query_expansion: bool | None = None,
        rerank: str | None = None,
        reranker_kwargs: dict[str, Any] | None = None,
    ) -> list[ToolDiscoveryResult]:
        """Search for tools matching the query.

        Args:
            query: Natural language query string.
            top_k: Maximum number of results.
            tags: Optional tags to filter results (all must match).
            include_schema: Include full tool schema in results.
            api_format: API format for tool schema generation.
            use_llm_query_expansion: Enable LLM-based query expansion for this
                query only.  Defaults to ``self.use_llm_query_expansion`` if
                ``None``.
            rerank: Reranker name for this query only (e.g. ``"cosine"``,
                ``"mmr"``).  Defaults to ``self._reranker`` if ``None``.
            reranker_kwargs: Passed to the reranker constructor for this
                query only (e.g. ``{"lambda_mult": 0.8}`` for MMR).

        Returns:
            List of :class:`ToolDiscoveryResult` models.
        """
        self.build_index()

        expand = use_llm_query_expansion if use_llm_query_expansion is not None else self.use_llm_query_expansion
        query_texts = [self._generate_keywords_for_query(query)] if expand else [query]

        query_tokens = bm25s.tokenize(query_texts, return_ids=False, show_progress=False)
        weight_mask = self._build_tag_mask(set(tags)) if tags else None

        corpus_size = self._bm25.scores["num_docs"]
        k = min(top_k, corpus_size) if corpus_size else 0

        results = self._bm25.retrieve(
            query_tokens,
            corpus=self._corpora,
            k=k,
            show_progress=False,
            weight_mask=weight_mask,
        )

        pairs = list(zip(results.documents.ravel(), results.scores.ravel()))

        if rerank is not None:
            reranker = get_reranker(rerank, **(reranker_kwargs or {}))
            pairs = reranker.rerank(pairs, query, self.registry)
        else:
            pairs = self._reranker.rerank(pairs, query, self.registry)

        out: list[ToolDiscoveryResult] = []
        for doc, score in pairs:
            out.append(self._doc_to_result(
                dict(doc), float(score), include_schema=include_schema, api_format=api_format,
            ))
        return out

    def inspect_index(self) -> list[dict[str, Any]]:
        """Return the full indexed corpus for debugging.

        Each entry is a dict with:

        - **tool_name**: Registered name of the tool.
        - **namespace**: Optional namespace.
        - **keywords**: Space-separated LLM-generated keywords.
        - **raw**: Concatenated raw metadata text.
        - **tags**: List of tags.

        Returns:
            List of dicts, one per indexed tool.
        """
        self.build_index()
        return [dict(doc) for doc in self._corpora]

    def save(self, path: str | Path) -> None:
        """Save the BM25 index and corpus to disk."""
        self._bm25.save(str(path), corpus=self._corpora, show_progress=False)

    @classmethod
    def load(cls, path: str | Path, registry: ToolRegistry) -> "ToolzSearch":
        """Load a previously saved BM25 index."""
        bm25_obj = bm25s.BM25.load(str(path), load_corpus=True)
        instance = cls(registry=registry)
        instance._bm25 = bm25_obj
        instance._corpora = list(bm25_obj.corpus) if bm25_obj.corpus else []
        return instance

    def _extract_param_info(self, tool: Tool) -> str:
        props = tool.parameters.get("properties", {})
        parts = []
        for name, schema in props.items():
            if name == "toolcall_reason":
                continue
            desc = schema.get("description", "")
            dtype = schema.get("type", "")
            parts.append(f"{name} ({dtype}) {desc}".strip())
        return " ".join(parts)

    def _generate_keywords(
        self,
        tool_name: str,
        description: str,
        parameters: str,
        tags: str,
        search_hint: str,
    ) -> list[str]:
        try:
            result = self._keyword_predictor(
                tool_name=tool_name, description=description, parameters=parameters,
                tags=tags, search_hint=search_hint,
            )
            keywords = result.keywords if hasattr(result, "keywords") else []
            logger.info("Generated %d raw keywords for tool '%s'", len(keywords), tool_name)
        except Exception:
            logger.warning("LLM keyword generation failed for tool '%s'", tool_name, exc_info=True)
            keywords = []

        # Deduplicate and clean
        seen: set[str] = set()
        filtered: list[str] = []
        for kw in keywords:
            clean = self._clean_token(kw)
            if clean and clean not in seen:
                seen.add(clean)
                filtered.append(clean)

        # Trim to max
        if len(filtered) > self._keyword_max:
            filtered = filtered[: self._keyword_max]

        # Backfill from raw metadata if under minimum
        if len(filtered) < self._keyword_min:
            filtered = self._expand_to_min(filtered, tool_name, description, parameters)

        return filtered

    def _expand_to_min(
        self,
        keywords: list[str],
        tool_name: str,
        description: str,
        parameters: str,
    ) -> list[str]:
        """Fill in keyword gap from raw metadata tokens."""
        seen = set(keywords)
        for token in f"{tool_name} {description} {parameters}".split():
            if len(keywords) >= self._keyword_min:
                break
            clean = self._clean_token(token)
            if clean and clean not in seen:
                seen.add(clean)
                keywords.append(clean)
        return keywords

    def _generate_keywords_for_query(self, query: str) -> str:
        try:
            result = self._query_predictor(user_query=query)
            return result.keywords if hasattr(result, "keywords") else query
        except Exception:
            return query

    def _build_tag_mask(self, required_tags: set[str]) -> Any:
        if not self._corpora:
            return None
        mask = [0.0 if not required_tags.issubset(set(doc.get("tags", []))) else 1.0 for doc in self._corpora]
        return np.array(mask, dtype="float32")

    def _doc_to_result(
        self,
        doc: dict[str, Any],
        score: float,
        include_schema: bool,
        api_format: str,
    ) -> ToolDiscoveryResult:
        tool_name = doc.get("tool_name", "")
        result = ToolDiscoveryResult(
            name=tool_name,
            description="",
            score=score,
            namespace=doc.get("namespace"),
            keywords=doc.get("keywords", "").split() if doc.get("keywords") else [],
            tags=doc.get("tags", []),
        )
        tool = self.registry.get_tool(tool_name)
        if tool is not None:
            result.description = tool.description or ""
            if include_schema:
                result.tool_schema = tool.get_schema(api_format)
        return result
