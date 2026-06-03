"""LLM-enhanced BM25 tool discovery over a ToolRegistry."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import bm25s
import dspy

from toolregistry import ToolRegistry, Tool
from toolregistry.tool_registry import TOOL_DISCOVERY_NAME

from .keywords import BM25KeywordSignature, QueryExpansionSignature
from .results import ToolDiscoveryResult

__all__ = ["ToolzSearch"]


class ToolzSearch:
    """BM25 search over a :class:`~toolregistry.ToolRegistry` with LLM-generated keywords.

    At index-build time, an LLM (via DSPy) generates BM25-optimal keywords
    from each tool's metadata. These keywords are concatenated with raw
    metadata and indexed using bm25s.

    Example::

        from toolregistry import ToolRegistry
        from toolz.search import ToolzSearch

        registry = ToolRegistry()
        registry.register(read_file)

        search = ToolzSearch(registry)
        search.build_index()
        results = search.search("read a file", top_k=5)
    """

    _KEYWORD_BOOST = 3

    def __init__(
        self,
        registry: ToolRegistry,
        use_llm_query_expansion: bool = False,
        lm: dspy.LM | None = None,
    ) -> None:
        self.registry = registry
        self.use_llm_query_expansion = use_llm_query_expansion
        self._bm25 = bm25s.BM25(k1=1.5, b=0.75, delta=0.5, method="lucene", corpus=[])
        self._lm = lm
        self._keyword_predictor = dspy.Predict(BM25KeywordSignature, lm=lm)
        self._query_predictor = dspy.Predict(QueryExpansionSignature, lm=lm)
        self._tokenize = bm25s.tokenize
        self._corpora: list[dict[str, Any]] = []
        self._indexed: bool = False

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
                "deferred": bool(tool.metadata.defer) if tool.metadata else False,
            })

        if not self._corpora:
            self._bm25 = bm25s.BM25(k1=1.5, b=0.75, delta=0.5, method="lucene", corpus=[])
            self._indexed = True
            return

        indexed_texts = []
        for doc in self._corpora:
            kw = doc["keywords"]
            raw = doc["raw"]
            boosted_kw = " ".join(kw.split()) if kw else ""
            indexed_texts.append(
                " ".join([boosted_kw] * self._KEYWORD_BOOST + [raw]) if boosted_kw else raw
            )

        tokenized = self._tokenize(indexed_texts, return_ids=False, show_progress=False)

        self._bm25 = bm25s.BM25(k1=1.5, b=0.75, delta=0.5, method="lucene", corpus=self._corpora)
        self._bm25.index(tokenized)
        self._indexed = True

    def search(
        self,
        query: str,
        top_k: int = 10,
        tags: Sequence[str] | None = None,
        include_schema: bool = False,
        api_format: str = "openai-chat",
    ) -> list[ToolDiscoveryResult]:
        """Search for tools matching the query.

        Args:
            query: Natural language query string.
            top_k: Maximum number of results.
            tags: Optional tags to filter results (all must match).
            include_schema: Include full tool schema in results.
            api_format: API format for tool schema generation.

        Returns:
            List of :class:`ToolDiscoveryResult` models.
        """
        if not self._indexed:
            self.build_index()

        if self.use_llm_query_expansion:
            expanded = self._generate_keywords_for_query(query)
            query_texts = [expanded]
        else:
            query_texts = [query]

        query_tokens = self._tokenize(query_texts, return_ids=False, show_progress=False)

        weight_mask = self._build_tag_mask(set(tags)) if tags else None

        corpus_size = self._bm25.scores["num_docs"]
        k = min(top_k, corpus_size) if corpus_size > 0 else 0

        results = self._bm25.retrieve(
            query_tokens,
            corpus=self._corpora,
            k=k,
            show_progress=False,
            weight_mask=weight_mask,
        )

        out: list[ToolDiscoveryResult] = []
        for doc, score in zip(results.documents.ravel(), results.scores.ravel()):
            out.append(self._doc_to_result(
                dict(doc), float(score), include_schema=include_schema, api_format=api_format,
            ))
        return out

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
        instance._indexed = True
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

    def _generate_keywords(self, tool_name: str, description: str, parameters: str, tags: str, search_hint: str) -> list[str]:
        try:
            result = self._keyword_predictor(
                tool_name=tool_name, description=description, parameters=parameters,
                tags=tags, search_hint=search_hint,
            )
            keywords = result.keywords if hasattr(result, "keywords") else []
        except Exception:
            keywords = []

        seen: set[str] = set()
        filtered: list[str] = []
        for kw in keywords:
            clean = kw.lower().strip().strip(".,;:!?\"'()[]{}")
            if clean and clean.isalpha() and len(clean) >= 2 and clean not in seen:
                seen.add(clean)
                filtered.append(clean)
        return filtered

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
        import numpy as np
        return np.array(mask, dtype="float32")

    def _doc_to_result(self, doc: dict[str, Any], score: float, include_schema: bool, api_format: str) -> ToolDiscoveryResult:
        tool_name = doc.get("tool_name", "")
        result = ToolDiscoveryResult(
            name=tool_name,
            description="",
            score=score,
            namespace=doc.get("namespace"),
            deferred=doc.get("deferred", False),
            keywords=doc.get("keywords", "").split() if doc.get("keywords") else [],
            tags=doc.get("tags", []),
        )
        tool = self.registry.get_tool(tool_name)
        if tool is not None:
            result.description = tool.description or ""
            if include_schema:
                result.tool_schema = tool.get_schema(api_format)
        return result
