"""Result models for ToolzSearch."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


class ToolDiscoveryResult(BaseModel):
    """A single tool discovery result."""

    name: str
    description: str
    score: float
    namespace: Optional[str] = None
    keywords: list[str] = []
    tags: list[str] = []
    tool_schema: Optional[dict[str, Any]] = None

    model_config = {"populate_by_name": True}
