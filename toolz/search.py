"""LLM-enhanced BM25 tool discovery over a ToolRegistry.

Uses DSPy to generate BM25-optimal keywords from tool metadata at index
build time, then uses bm25s for fast lexical retrieval at query time.
"""

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
    """LLM-enhanced BM25 search over a :class:`~toolregistry.ToolRegistry`.

    At index-build time, an LLM (via DSPy) generates optimal BM25 keywords
    from each tool's metadata.  These keywords are concatenated with raw
    metadata into a single BM25 index (bm25s), with the LLM keywords
    repeated to receive higher term-frequency weight.

    At query time, the user prompt can optionally be expanded through the
    same LLM before tokenization (``use_llm_query_expansion=True``),
    improving recall for natural-language queries.

    Example::

        from toolregistry import ToolRegistry
        from toolz.search import ToolzSearch

        registry = ToolRegistry()
        registry.register(read_file)
        registry.register(write_file)

        search = ToolzSearch(registry)
        search.build_index()
        results = search.search("read a file", top_k=5)
        for r in results:
            print(r.name, r.score)

    Args:
        registry: The ToolRegistry to search over.
        use_llm_query_expansion: If True, the user query is passed through
            the LLM to generate expanded BM25 keywords before tokenization.
            Default is False (raw tokenization).
        lm: A ``dspy.LM`` instance to use for DSPy predictions. If None,
            uses the globally configured DSPy LM.
    """

    _KEYWORD_BOOST = 3  # repeat LLM keywords this many times for BM25 weight

    def __init__(
        self,
        registry: ToolRegistry,
        use_llm_query_expansion: bool = False,
        lm: dspy.LM | None = None,
    ) -> None:
        self.registry = registry
        self.use_llm_query_expansion = use_llm_query_expansion

        # BM25 index with corpus storage for per-tool metadata
        self._bm25 = bm25s.BM25(
            k1=1.5,
            b=0.75,
            delta=0.5,
            method="lucene",
            corpus=[],
        )

        self._lm = lm

        # DSPy keyword generator — dspy.Predict for direct one-shot output
        self._keyword_predictor = dspy.Predict(BM25KeywordSignature, lm=lm)

        # DSPy query expansion predictor
        self._query_predictor = dspy.Predict(QueryExpansionSignature, lm=lm)

        # Module-level tokenizer — returns List[List[str]] when return_ids=False
        self._tokenize = bm25s.tokenize

        # Internal state
        self._corpora: list[dict[str, Any]] = []
        self._indexed: bool = False

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def build_index(self) -> None:
        """Build the BM25 index from all tools in the registry.

        For each tool:
        1. Extract metadata fields (name, description, parameters, tags, hint).
        2. Call the DSPy keyword generator to produce BM25-optimal keywords.
        3. Store a corpus entry with ``keywords`` (boosted) and ``raw``
           (baseline) fields.
        4. Index the full corpus.
        """
        self._corpora = []

        for name, tool in self.registry._tools.items():
            if name == TOOL_DISCOVERY_NAME:
                continue

            # --- Extract metadata ---
            tool_name = tool.name
            description = tool.description or ""
            param_str = self._extract_param_info(tool)
            tags_str = " ".join(tool.metadata.all_tags) if tool.metadata else ""
            search_hint = (
                tool.metadata.search_hint if tool.metadata and tool.metadata.search_hint
                else ""
            )

            # --- Generate LLM keywords ---
            llm_keywords = self._generate_keywords(
                tool_name=tool_name,
                description=description,
                parameters=param_str,
                tags=tags_str,
                search_hint=search_hint,
            )

            # --- Build corpus entry ---
            corpus_entry: dict[str, Any] = {
                "tool_name": tool_name,
                "keywords": " ".join(llm_keywords),
                "raw": f"{tool_name} {description} {param_str} {tags_str}",
                "tags": list(tool.metadata.all_tags) if tool.metadata else [],
                "namespace": tool.namespace,
                "deferred": bool(tool.metadata.defer) if tool.metadata else False,
            }
            self._corpora.append(corpus_entry)

        # --- Tokenize and index ---
        if not self._corpora:
            self._bm25 = bm25s.BM25(
                k1=1.5, b=0.75, delta=0.5, method="lucene", corpus=[]
            )
            self._indexed = True
            return

        # bm25s does not support multi-field BM25F, so we concatenate into
        # a single text field.  We repeat the LLM keywords N times to give
        # them higher term-frequency weight (a standard BM25 boosting trick).
        indexed_texts: list[str] = []
        for doc in self._corpora:
            kw = doc["keywords"]
            raw = doc["raw"]
            boosted_kw = " ".join(kw.split()) if kw else ""
            if boosted_kw:
                boosted = " ".join([boosted_kw] * self._KEYWORD_BOOST + [raw])
            else:
                boosted = raw
            indexed_texts.append(boosted)

        # Tokenize (module-level function, returns List[List[str]])
        tokenized = self._tokenize(indexed_texts, return_ids=False, show_progress=False)

        # Re-initialize BM25 with corpus for retrieval
        self._bm25 = bm25s.BM25(
            k1=1.5,
            b=0.75,
            delta=0.5,
            method="lucene",
            corpus=self._corpora,
        )
        self._bm25.index(tokenized)
        self._indexed = True

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

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
            tags: Optional sequence of tags to filter results. Only tools
                matching *all* specified tags are returned.
            include_schema: If True, fetches and includes the full tool
                schema (via :meth:`~toolregistry.Tool.get_schema`) for
                each result, matching the output format of
                :meth:`~toolregistry.llm.discovery.ToolDiscoveryTool.discover`.
            api_format: API format to use when generating tool schemas.
                Passed through to :meth:`~toolregistry.Tool.get_schema`.

        Returns:
            List of :class:`ToolDiscoveryResult` models with keys:

            - ``name`` (str): Tool name.
            - ``description`` (str): Tool description.
            - ``score`` (float): BM25 relevance score.
            - ``namespace`` (str | None): Tool namespace.
            - ``deferred`` (bool): Whether the tool is deferred.
            - ``keywords`` (list[str]): LLM-generated keywords.
            - ``tags`` (list[str]): Tool tags.
            - ``tool_schema`` (dict | None): Full tool schema (only when
              ``include_schema=True``).
        """
        if not self._indexed:
            self.build_index()

        # --- Tokenize query ---
        if self.use_llm_query_expansion:
            expanded = self._generate_keywords_for_query(query)
            query_texts = [expanded]
        else:
            query_texts = [query]

        query_tokens = self._tokenize(query_texts, return_ids=False, show_progress=False)

        # --- Build tag filter mask ---
        weight_mask: Any = None
        if tags:
            weight_mask = self._build_tag_mask(set(tags))

        # --- Retrieve ---
        corpus_size = self._bm25.scores["num_docs"]
        k = min(top_k, corpus_size) if corpus_size > 0 else 0

        results = self._bm25.retrieve(
            query_tokens,
            corpus=self._corpora,
            k=k,
            show_progress=False,
            weight_mask=weight_mask,
        )

      # --- Format output ---
        docs = results.documents.ravel()
        scores = results.scores.ravel()
        out: list[ToolDiscoveryResult] = []
        for doc, score in zip(docs, scores):
            result = self._doc_to_result(
                dict(doc),
                float(score),
                include_schema=include_schema,
                api_format=api_format,
            )
            out.append(result)

        return out

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save the BM25 index and corpus to disk.

        Args:
            path: Directory path to save the index to.
        """
        self._bm25.save(
            str(path),
            corpus=self._corpora,
            show_progress=False,
        )

    @classmethod
    def load(cls, path: str | Path, registry: ToolRegistry) -> "ToolzSearch":
        """Load a previously saved BM25 index.

        Args:
            path: Directory path where the index was saved.
            registry: The ToolRegistry this search belongs to
                (used for metadata lookups).

        Returns:
            A new ToolzSearch instance with the loaded index.
        """
        bm25_obj = bm25s.BM25.load(
            str(path),
            load_corpus=True,
        )

        instance = cls(registry=registry)
        instance._bm25 = bm25_obj
        instance._corpora = list(bm25_obj.corpus) if bm25_obj.corpus else []
        instance._indexed = True
        return instance

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_param_info(self, tool: Tool) -> str:
        """Extract a human-readable parameter description from a tool's JSON schema."""
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
        """Call the DSPy keyword generator for a single tool."""
        try:
            result = self._keyword_predictor(
                tool_name=tool_name,
                description=description,
                parameters=parameters,
                tags=tags,
                search_hint=search_hint,
            )
            keywords = result.keywords if hasattr(result, "keywords") else []
        except Exception:
            keywords = []

        # Deduplicate while preserving order, filter to alpha-only tokens >= 2 chars
        seen: set[str] = set()
        filtered: list[str] = []
        for kw in keywords:
            clean = kw.lower().strip().strip(".,;:!?\"'()[]{}")
            if clean and clean.isalpha() and len(clean) >= 2 and clean not in seen:
                seen.add(clean)
                filtered.append(clean)
        return filtered

    def _generate_keywords_for_query(self, query: str) -> str:
        """Expand a user query into BM25-matching keywords via DSPy."""
        try:
            result = self._query_predictor(user_query=query)
            expanded = result.keywords if hasattr(result, "keywords") else query
        except Exception:
            expanded = query

        return expanded

    def _build_tag_mask(self, required_tags: set[str]) -> Any:
        """Build a weight mask that excludes tools missing any required tag."""
        if not self._corpora:
            return None

        n = len(self._corpora)
        mask = [1.0] * n
        for i, doc in enumerate(self._corpora):
            doc_tags = set(doc.get("tags", []))
            if not required_tags.issubset(doc_tags):
                mask[i] = 0.0

        import numpy as np

        return np.array(mask, dtype="float32")

    def _doc_to_result(
        self,
        doc: dict[str, Any],
        score: float,
        include_schema: bool = False,
        api_format: str = "openai-chat",
    ) -> ToolDiscoveryResult:
        """Convert a corpus doc dict + score into a ToolDiscoveryResult.

        If ``include_schema`` is True, fetches the full tool schema from
        the registry (matching :meth:`~toolregistry.llm.discovery.ToolDiscoveryTool.discover`).
        """
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

        # Fetch description and optional schema from the registry
        tool = self.registry.get_tool(tool_name)
        if tool is not None:
            result.description = tool.description or ""
            if include_schema:
                result.tool_schema = tool.get_schema(api_format)

        return result
