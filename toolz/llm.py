"""LLM configuration for DSPy."""

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

    if not api_key or not model:
        missing = []
        if not api_key:
            missing.append("OPENAI_API_KEY")
        if not model:
            missing.append("OPENAI_MODEL_NAME")
        raise ValueError(f"{', '.join(missing)} is not set.")

    return dspy.LM(model=model, api_base=base_url or None, api_key=api_key)
