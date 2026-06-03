"""Basic usage example for toolz."""

from dotenv import load_dotenv
from toolregistry import ToolRegistry
from toolz import ToolzSearch
from toolz.llm import make_lm

load_dotenv()

try:
    lm = make_lm()
    print(f"LLM configured: {lm.model if lm else 'None'}")
except ValueError:
    lm = None
    print("LLM not configured — BM25-only fallback")


def read_file(path: str, encoding: str = "utf-8") -> str:
    """Read the contents of a file from the filesystem."""
    return ""


def write_file(path: str, content: str, encoding: str = "utf-8") -> None:
    """Write content to a file on the filesystem."""
    pass


def get_weather(location: str) -> str:
    """Fetch current weather conditions for a given location."""
    return "sunny"


def calculate_expression(expression: str) -> float:
    """Evaluate a mathematical expression string."""
    return 0.0


def main() -> None:
    registry = ToolRegistry()
    registry.register(read_file, name="read_file")
    registry.register(write_file, name="write_file")
    registry.register(get_weather, name="get_weather")
    registry.register(calculate_expression, name="calculate_expression")
    print("No. of Tools in Registry:", len(registry._tools))

    search = ToolzSearch(registry, keyword_min=10, keyword_max=30, lm=lm, rerank="cosine")
    search.build_index()

    print("\nIndexed corpus:\n")
    for doc in search.inspect_index():
        keywords = doc["keywords"].split()
        print(f"  {doc['tool_name']:20s}  keywords={len(keywords):3d}  tags={doc['tags']}")
        print(f"    first 15 keywords: {keywords[:15]}")

    search_queries = [
        "read a file from disk",
        "weather forecast",
        "what's the temperature in Antarctica",
        "Is it raining in UK",
        "store results into disk drive",
    ]

    for query in search_queries:
        results = search.search(query, top_k=5, include_schema=True, use_llm_query_expansion=True)
        print("\nScores\n===========")
        print("Search Query:", query)
        for r in results:
            print(f"  {r.name:20s} score={r.score:.3f}  keywords={r.keywords[:5]}")

    print("\n\n--- Without LLM query expansion ---")
    results = search.search("store results into disk drive", top_k=5, use_llm_query_expansion=False)
    print("Search Query: store results into disk drive")
    for r in results:
        print(f"  {r.name:20s} score={r.score:.3f}")

    print("\n\n--- With reranking (cosine similarity + name match) ---")
    results = search.search("store results into disk drive", top_k=5, rerank="cosine")
    print("Search Query: store results into disk drive")
    for r in results:
        print(f"  {r.name:20s} score={r.score:.3f}")

    print("\n\nTool Schema Output:\n")
    print(results[0].tool_schema)


if __name__ == "__main__":
    main()
