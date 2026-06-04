"""Allow running as: python -m benchmark."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (one level up from benchmark/).
_project_root = Path(__file__).resolve().parent.parent
load_dotenv(_project_root / ".env")

# Configure DSPy LM if credentials are available.
_openai_key = os.environ.get("OPENAI_API_KEY")
_openai_model = os.environ.get("OPENAI_MODEL_NAME")
_openai_base = os.environ.get("OPENAI_BASE_URL")

if _openai_key and _openai_model:
    import dspy

    dspy.configure(lm=dspy.LM(model=_openai_model, api_base=_openai_base or None, api_key=_openai_key))

from .cli import main

main()
