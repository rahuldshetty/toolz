"""Result models for ToolzSearch.

Pydantic models that mirror the structure of
:class:`~toolregistry.llm.discovery.ToolDiscoveryTool.discover()`
output, with optional tool schemas.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


class ToolDiscoveryResult(BaseModel):
    """A single tool discovery result.

    Mirrors the dict structure returned by
    :meth:`~toolregistry.llm.discovery.ToolDiscoveryTool.discover`.

    Attributes:
        name: Tool name.
        description: Tool description.
        score: BM25 relevance score.
        namespace: Tool namespace, if any.
        deferred: Whether the tool is deferred (hidden from initial prompt).
        keywords: LLM-generated BM25 keywords for this tool.
        tags: Tool tags (predefined + custom).
        tool_schema: Full tool schema in the requested API format
            (e.g. ``"openai-chat"``). Only populated when
            ``include_schema=True`` is passed to :meth:`ToolzSearch.search`.
    """

    name: str
    description: str
    score: float
    namespace: Optional[str] = None
    deferred: bool = False
    keywords: list[str] = []
    tags: list[str] = []
    tool_schema: Optional[dict[str, Any]] = None

    model_config = {"populate_by_name": True}
