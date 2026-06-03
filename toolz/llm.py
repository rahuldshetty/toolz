"""LLM configuration for DSPy.

Factory to create a ``dspy.LM`` from environment variables.

Environment variables:
    OPENAI_API_KEY   - API key (required).
    OPENAI_BASE_URL  - Base URL (optional, e.g. ``http://localhost:1234/v1``).
    OPENAI_MODEL_NAME - Model name (required).
"""

from __future__ import annotations

import os

import dspy


def make_lm(
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> dspy.LM:
    """Create a ``dspy.LM`` from environment variables or explicit args."""
    api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
    base_url = base_url or os.environ.get("OPENAI_BASE_URL", "")
    model = model or os.environ.get("OPENAI_MODEL_NAME", "")

    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set.")
    if not model:
        raise ValueError("OPENAI_MODEL_NAME is not set.")

    return dspy.LM(model=model, api_base=base_url or None, api_key=api_key)
