"""DSPy signatures for BM25 keyword generation."""

from __future__ import annotations

from pydantic import BaseModel, Field

import dspy


# 1. Pydantic model with strict length constraints on the keyword list
class KeywordList(BaseModel):
    """A constrained list of BM25 keywords."""

    items: list[str] = Field(
        ...,
        min_length=10,
        max_length=100,
        description=(
            "BM25-relevant keywords and synonyms. Include the tool name split "
            "into words, action verbs, domain terms, parameter-related terms, "
            "and common user phrasings. Lowercase only, no duplicates, no "
            "punctuation."
        ),
    )


class BM25KeywordSignature(dspy.Signature):
    """Generate BM25-optimal keywords from tool metadata."""

    tool_name: str = dspy.InputField(desc="The registered name of the tool")
    description: str = dspy.InputField(desc="A detailed description of what the tool does")
    parameters: str = dspy.InputField(desc="Parameter names, types, and descriptions")
    tags: str = dspy.InputField(desc="Tool tags — predefined and custom string tags")
    search_hint: str = dspy.InputField(desc="Existing search hint or free-form keywords")
    extra_keywords: str = dspy.InputField(
        desc="Additional keywords provided by the caller, space-separated, lowercase",
    )
    keywords: KeywordList = dspy.OutputField()


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
