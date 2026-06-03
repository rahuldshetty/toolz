"""DSPy signatures for BM25 keyword generation.

Signatures used by :class:`~toolz.search.ToolzSearch` to extract
BM25-optimal keywords from tool metadata and user queries.
"""

from __future__ import annotations

import dspy


class BM25KeywordSignature(dspy.Signature):
    """Generate BM25-optimal keywords/tokens from tool metadata.

    Takes the full metadata profile of a tool and produces a list of
    lowercase, deduplicated, alpha-only keywords suitable for indexing
    in a BM25 sparse index.
    """

    tool_name: str = dspy.InputField(desc="The registered name of the tool")
    description: str = dspy.InputField(
        desc="A detailed description of what the tool does"
    )
    parameters: str = dspy.InputField(
        desc="Parameter names, types, and descriptions from the JSON schema"
    )
    tags: str = dspy.InputField(
        desc="Tool tags — predefined (e.g. file_system, read_only) and custom string tags"
    )
    search_hint: str = dspy.InputField(
        desc="Existing search hint or free-form keywords, if provided"
    )
    keywords: list[str] = dspy.OutputField(
        desc=(
            "BM25-relevant keywords and synonyms for tool discovery. "
            "Include the tool name split into words, action verbs, domain terms, "
            "parameter-related terms, and common user phrasings. Lowercase only, "
            "no duplicates, no punctuation."
        )
    )


class QueryExpansionSignature(dspy.Signature):
    """Expand a natural-language query into BM25-matching keywords.

    Used at search time when ``use_llm_query_expansion=True`` to
    transform a raw user prompt into a space-separated list of
    BM25-friendly tokens before tokenization.
    """

    user_query: str = dspy.InputField(
        desc="The user's natural language query for finding tools"
    )
    keywords: str = dspy.OutputField(
        desc=(
            "Space-separated BM25-relevant keywords and synonyms that "
            "capture the intent of the query. Include action verbs, "
            "domain terms, and common user phrasings. Lowercase only."
        )
    )
