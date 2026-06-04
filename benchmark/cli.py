"""CLI entry point for running toolz benchmarks."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .dataset import available_tasks
from .run import STRATEGY_CONFIGS, run_benchmark


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="toolz-benchmark",
        description="Benchmark toolz tool retrieval against ToolRet datasets",
    )
    parser.add_argument(
        "--task",
        required=True,
        help="Task config name (e.g. 'gorilla-pytorch', 'apibank'). "
             "Use 'python -m toolz.benchmark --list-tasks to see all.",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=list(STRATEGY_CONFIGS.keys()),
        choices=list(STRATEGY_CONFIGS.keys()),
        help="Strategies to evaluate (default: all built-in).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Maximum results per query (default: 20).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write JSON results file.",
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[5, 10, 20],
        help="K values for metrics (default: 5 10 20).",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List all available task names and exit.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    # Handle --list-tasks before argparse validates required args.
    check_args = argv if argv is not None else sys.argv[1:]
    if "--list-tasks" in check_args:
        tasks = available_tasks()
        print(f"Available tasks ({len(tasks)}):")
        for t in tasks:
            print(f"  {t}")
        return

    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.list_tasks:
        tasks = available_tasks()
        print(f"Available tasks ({len(tasks)}):")
        for t in tasks:
            print(f"  {t}")
        return

    run_benchmark(
        task_name=args.task,
        strategies=args.strategies,
        top_k=args.top_k,
        output_file=args.output,
        k_values=args.k_values,
    )


if __name__ == "__main__":
    main()
