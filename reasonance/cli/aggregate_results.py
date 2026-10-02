#!/usr/bin/env python3
"""Aggregate benchmark result JSONs into a single CSV.

Usage::

    python -m reasonance.cli.aggregate_results --results-dir RESULTS --save-path OUT.csv [--tasks ACSIncome ...]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from reasonance.utils import build_aggregate, report_seed_coverage


def setup_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser for the aggregation script.
    """
    parser = argparse.ArgumentParser(description="Aggregate benchmark results into a CSV.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Root results directory to search for result JSONs.",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        required=True,
        help="Output CSV file path.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["ACSIncome"],
        help="Task name(s) to include (default: ACSIncome).",
    )
    return parser


def main() -> None:
    """CLI entry point: build the aggregate CSV and report seed coverage."""
    args = setup_arg_parser().parse_args()
    df = build_aggregate(
        results_dir=args.results_dir,
        tasks=args.tasks,
        save_path=args.save_path,
    )
    print(f"Saved {len(df)} rows to {args.save_path}")

    # Seed coverage is the thing that quietly goes wrong: a cell executed twice is
    # de-duplicated downstream, and if the kept execution's files have moved the seed
    # vanishes from the comparison with no error. Print it on every run.
    report_seed_coverage(df)


if __name__ == "__main__":
    main()
