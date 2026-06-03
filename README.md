# toolz

LLM-enhanced BM25 tool discovery over a [ToolRegistry](https://github.com/polynote/toolregistry).

Uses DSPy to generate BM25-optimal keywords from tool metadata at index-build time, then uses [bm25s](https://github.com/xhluca/bm25s) for fast lexical retrieval at query time.

## Installation

```bash
pip install toolz
```

## Quick Start

```python
from toolregistry import ToolRegistry
from toolz import ToolzSearch

registry = ToolRegistry()
registry.register(read_file, name="read_file")
registry.register(write_file, name="write_file")

search = ToolzSearch(registry)
search.build_index()

results = search.search("read a file from disk", top_k=5)
for r in results:
    print(r["tool_name"], r["score"])
```

## Configuration

### LLM Setup

Set the following environment variables before using `ToolzSearch`:

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | API key for the LLM provider |
| `OPENAI_MODEL_NAME` | Yes | Model name (e.g. `gpt-4o`, `claude-3.5-sonnet`) |
| `OPENAI_BASE_URL` | No | Custom API base URL (for local servers or proxies) |

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL_NAME="gpt-4o"
```

### Custom LLM

Pass a `dspy.LM` instance directly to override the environment-based configuration:

```python
import dspy
from toolz import ToolzSearch, make_lm

lm = make_lm(model="gpt-4o", api_key="sk-...")
search = ToolzSearch(registry, lm=lm)
```

### Query Expansion

Enable LLM-based query expansion for higher recall at the cost of latency:

```python
search = ToolzSearch(registry, use_llm_query_expansion=True)
```

## API Reference

### `ToolzSearch`

```python
class ToolzSearch:
    def __init__(
        self,
        registry: ToolRegistry,
        use_llm_query_expansion: bool = False,
        lm: dspy.LM | None = None,
    )

    def build_index(self) -> None
        """Build the BM25 index from all tools in the registry."""

    def search(
        self,
        query: str,
        top_k: int = 10,
        tags: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]
        """Search for tools matching the query.

        Returns a list of dicts with keys:
        - ``tool_name`` (str): Tool name.
        - ``description`` (str): Tool description.
        - ``score`` (float): BM25 relevance score.
        - ``namespace`` (str | None): Tool namespace.
        - ``deferred`` (bool): Whether the tool is deferred.
        - ``keywords`` (list[str]): LLM-generated keywords.
        - ``tags`` (list[str]): Tool tags.
        """

    def save(self, path: str | Path) -> None
        """Save the BM25 index and corpus to disk."""

    @classmethod
    def load(cls, path: str | Path, registry: ToolRegistry) -> "ToolzSearch"
        """Load a previously saved BM25 index."""
```

### `make_lm`

```python
def make_lm(
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> dspy.LM
    """Create a dspy.LM from environment variables or explicit arguments."""
```

## Architecture

```
toolz/
├── __init__.py      # Public API exports
├── llm.py           # LLM factory (make_lm)
├── keywords.py      # DSPy signatures (BM25KeywordSignature, QueryExpansionSignature)
└── search.py        # ToolzSearch implementation
```

### How it works

1. **Indexing**: For each tool in the registry, DSPy generates BM25-optimal keywords from the tool's metadata (name, description, parameters, tags, search hint). These keywords are concatenated with raw metadata and indexed using bm25s.

2. **Keyword Boosting**: Since bm25s uses single-field BM25 (no BM25F multi-field support), LLM-generated keywords are repeated 3× to give them higher term-frequency weight — a standard BM25 boosting technique.

3. **Search**: The user query is tokenized directly (default) or expanded through DSPy first (`use_llm_query_expansion=True`), then matched against the BM25 index. Optional tag filtering is applied via bm25s weight masks.

## License

MIT
