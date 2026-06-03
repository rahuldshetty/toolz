"""DSPy signatures for BM25 keyword generation."""

from __future__ import annotations

import dspy


class BM25KeywordSignature(dspy.Signature):
    """Generate BM25-optimal keywords from tool metadata."""

    tool_name: str = dspy.InputField(desc="The registered name of the tool")
    description: str = dspy.InputField(desc="A detailed description of what the tool does")
    parameters: str = dspy.InputField(desc="Parameter names, types, and descriptions")
    tags: str = dspy.InputField(desc="Tool tags — predefined and custom string tags")
    search_hint: str = dspy.InputField(desc="Existing search hint or free-form keywords")
    keywords: list[str] = dspy.OutputField(
        desc=(
            "BM25-relevant keywords and synonyms. Include the tool name split "
            "into words, action verbs, domain terms, parameter-related terms, "
            "and common user phrasings. Lowercase only, no duplicates, no "
            "punctuation. Generate a rich, diverse set of keywords."
        )
    )


class QueryExpansionSignature(dspy.Signature):
    """Expand a natural-language query into BM25-matching keywords."""

    user_query: str = dspy.InputField(desc="The user's natural language query for finding tools")
    keywords: str = dspy.OutputField(
        desc=(
            "Space-separated BM25-relevant keywords and synonyms that capture the intent "
            "of the query. Include action verbs, domain terms, and common user phrasings. "
            "Lowercase only."
        )
    )
