"""Basic usage example for toolz.

This example demonstrates how to use ToolzSearch to discover tools
in a ToolRegistry using LLM-enhanced BM25 search.
"""

from toolregistry import ToolRegistry
from toolz import ToolzSearch


# --- Define some example tools ---

search_queies = [
    "read a file from disk",
    "weather forecast",
    "what's the temperature in Antartica"
]

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
    # 1. Create a registry and register tools
    registry = ToolRegistry()
    registry.register(read_file, name="read_file")
    registry.register(write_file, name="write_file")
    registry.register(get_weather, name="get_weather")
    registry.register(calculate_expression, name="calculate_expression")
    print("No. of Tools in Registry:", len(registry._tools))

    # 2. Create a search instance
    search = ToolzSearch(registry)

    # 3. Build the BM25 index (calls LLM to generate keywords per tool)
    search.build_index()

    # 4. Search for tools
    for i, query in enumerate(search_queies):
        results = search.search(query, top_k=5, include_schema=True)

        print("\nScores\n===========")
        print("Search Query:", query)
        for r in results:
            print(f"  {r.name:20s} score={r.score:.3f}")

        if i == len(search_queies) - 1:
            print("\n\nTool Schema Output:\n")
            print(results[0].tool_schema)


if __name__ == "__main__":
    main()
