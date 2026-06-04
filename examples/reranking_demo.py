"""Demo: BM25-only vs BM25 + cosine-sim reranking.

Shows how different reranking strategies change scores and ordering
by comparing them side-by-side on the same query.

Run without an LLM configured — falls back to keyword backfilling.
"""

import logging

# Suppress LLM fallback warnings for cleaner demo output
logging.getLogger("toolz.search").setLevel(logging.ERROR)
logging.getLogger("LiteLLM").setLevel(logging.ERROR)

from toolregistry import ToolRegistry
from toolz import ToolzSearch

registry = ToolRegistry()

registry.register(
    lambda path: "",
    name="read_file",
    description="Read the contents of a file from the filesystem and return it as a string.",
)
registry.register(
    lambda path, content: None,
    name="write_file",
    description="Write content to a file on the filesystem, creating it if it does not exist.",
)
registry.register(
    lambda location: "",
    name="get_weather",
    description="Fetch current weather conditions and forecasts for a given geographic location.",
)
registry.register(
    lambda expr: 0.0,
    name="calculate_expression",
    description="Evaluate a mathematical expression string and return the numeric result.",
)
registry.register(
    lambda query: [],
    name="search_files",
    description="Search for files on disk by name pattern, size, or modification date.",
)
registry.register(
    lambda url: "",
    name="fetch_webpage",
    description="Download and return the HTML content of a web page at the given URL.",
)

search = ToolzSearch(registry)
search.build_index()

print("\nIndexed tools and their generated keywords:")
for doc in search.inspect_index():
    keywords = doc["keywords"].split()
    print(f"\n  {doc['tool_name']:20s} ({len(keywords):3d} keywords): {', '.join(keywords[:20])}")

queries = [
    "read a file from disk",
    "store results into disk drive",
    "get weather for a location",
]

for query in queries:
    print(f"\n{'='*60}")
    print(f"Query: {query!r}")
    print(f"{'='*60}")

    # BM25 only (noop reranker)
    results_bm25 = search.search(query, top_k=6, rerank="none")
    print("\nBM25 only (rerank=none):")
    for i, r in enumerate(results_bm25, 1):
        print(f"  {i:2d}. {r.name:20s} score={r.score:.4f}")

    # Cosine similarity reranking
    results_cosine = search.search(query, top_k=6, rerank="cosine")
    print("\nBM25 + cosine similarity:")
    for i, r in enumerate(results_cosine, 1):
        print(f"  {i:2d}. {r.name:20s} score={r.score:.4f}")

    # MMR reranking (lambda=0.7 relevance, 0.3 diversity)
    results_mmr = search.search(query, top_k=6, rerank="mmr", reranker_kwargs={"lambda_mult": 0.7})
    print("\nBM25 + MMR (lambda=0.7):")
    for i, r in enumerate(results_mmr, 1):
        print(f"  {i:2d}. {r.name:20s} score={r.score:.4f}")

    # Compare ordering
    bm25_names = [r.name for r in results_bm25]
    cosine_names = [r.name for r in results_cosine]
    mmr_names = [r.name for r in results_mmr]

    if bm25_names != cosine_names:
        print("\n  >>> Cosine changed the ordering!")
    if bm25_names != mmr_names:
        print("  >>> MMR changed the ordering!")
    if bm25_names == cosine_names == mmr_names:
        print("\n  (All strategies agree on ordering)")
