"""LLM configuration for DSPy.

Provides a factory function to create a ``dspy.LM`` from environment
variables, following the OpenAI-compatible API convention.

Environment variables
---------------------
OPENAI_API_KEY   – API key (required).
OPENAI_BASE_URL  – Base URL for the API (optional, e.g.
                   ``http://localhost:1234/v1`` for a local server).
OPENAI_MODEL_NAME – Model name (required).

Example
-------
>>> from toolz.llm import make_lm
>>> lm = make_lm()
>>> import dspy
>>> dspy.configure(lm=lm)
"""

from __future__ import annotations

import os

import dspy


def make_lm(
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> dspy.LM:
    """Create a ``dspy.LM`` from environment variables.

    Falls back to the defaults:
    - ``api_key`` → ``OPENAI_API_KEY`` (no default; raises if missing)
    - ``base_url`` → ``OPENAI_BASE_URL`` (empty string → uses OpenAI
      upstream)
    - ``model`` → ``OPENAI_MODEL_NAME`` (no default; raises if missing)

    Args:
        api_key: Override for ``OPENAI_API_KEY``.
        base_url: Override for ``OPENAI_BASE_URL``.
        model: Override for ``OPENAI_MODEL_NAME``.

    Returns:
        A configured ``dspy.LM`` instance.

    Raises:
        ValueError: If *api_key* or *model* is not provided after
            checking environment variables.
    """
    api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
    base_url = base_url or os.environ.get("OPENAI_BASE_URL", "")
    model = model or os.environ.get("OPENAI_MODEL_NAME", "")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. "
            "Set it in your environment or pass api_key= explicitly."
        )
    if not model:
        raise ValueError(
            "OPENAI_MODEL_NAME is not set. "
            "Set it in your environment or pass model= explicitly."
        )

    return dspy.LM(
        model=model,
        api_base=base_url or None,
        api_key=api_key,
    )
