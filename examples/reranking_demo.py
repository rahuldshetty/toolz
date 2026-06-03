"""Demo: BM25-only vs BM25 + cosine-sim reranking.

Shows how reranking changes scores and ordering by comparing both
modes side-by-side on the same query.

Run without an LLM configured — falls back to keyword backfilling.
"""
from dotenv import load_dotenv
from toolz.llm import make_lm

import logging

from toolregistry import ToolRegistry
from toolz import ToolzSearch

# Suppress LLM fallback warnings for cleaner demo output
logging.getLogger("toolz.search").setLevel(logging.ERROR)
logging.getLogger("LiteLLM").setLevel(logging.ERROR)

load_dotenv()

try:
    lm = make_lm()
    print(f"LLM configured: {lm.model if lm else 'None'}")
except ValueError:
    lm = None
    print("LLM not configured — BM25-only fallback")


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

search = ToolzSearch(registry, keyword_min=5, keyword_max=30, lm=lm)
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

    # BM25-only
    results_bm25 = search.search(query, top_k=6, use_llm_query_expansion=False)
    print("\nBM25 only:")
    for i, r in enumerate(results_bm25, 1):
        print(f"  {i:2d}. {r.name:20s} score={r.score:.4f}")

    # BM25 + reranking
    results_rerank = search.search(query, top_k=6, use_llm_query_expansion=False, rerank=True)
    print("\nBM25 + cosine similarity reranking:")
    for i, r in enumerate(results_rerank, 1):
        print(f"  {i:2d}. {r.name:20s} score={r.score:.4f}")

    # Show which results changed position
    bm25_names = [r.name for r in results_bm25]
    rerank_names = [r.name for r in results_rerank]
    if bm25_names != rerank_names:
        print("\n  >>> Ordering changed by reranking!")
    else:
        print("\n  (Ordering unchanged — BM25 already ranked correctly)")
