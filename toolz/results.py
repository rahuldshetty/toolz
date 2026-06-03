"""Result models for ToolzSearch."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict


class ToolDiscoveryResult(BaseModel):
    """A single tool discovery result."""

    model_config = ConfigDict(populate_by_name=True)

    name: str
    description: str
    score: float
    namespace: Optional[str] = None
    keywords: list[str] = []
    tags: list[str] = []
    tool_schema: Optional[dict[str, Any]] = None
