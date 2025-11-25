"""Command-line entry point for DriftGuard."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from driftguard import load_engine


def _add_common_flags(parser: argparse.ArgumentParser) -> None:
    """Attach shared CLI flags to a subparser."""

    parser.add_argument(
        "--scope",
        default="repo",
        help="Policy scope to evaluate (e.g. repo or staged).",
    )
    parser.add_argument(
        "--mode",
        default="full",
        help="Evaluation mode hint (full or fast).",
    )


def build_parser() -> argparse.ArgumentParser:
    """Construct the DriftGuard argument parser."""

    parser = argparse.ArgumentParser(prog="driftguard")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Path to the repository root. Defaults to the current directory.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser("check", help="Run policy checks.")
    _add_common_flags(check_parser)

    fix_parser = subparsers.add_parser("fix", help="Apply safe policy fixes.")
    _add_common_flags(fix_parser)

    metrics_parser = subparsers.add_parser(
        "metrics", help="Compute drift metrics without enforcement."
    )
    _add_common_flags(metrics_parser)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point for ``driftguard``.

    The current implementation focuses on argument parsing and engine wiring so
    rule execution can be layered in later without changing the public surface.
    """

    parser = build_parser()
    args = parser.parse_args(argv)

    engine = load_engine(repo_root=args.repo_root)

    if args.command == "check":
        result = engine.check(scope=args.scope, mode=args.mode)
        _ = result
    elif args.command == "fix":
        result = engine.fix(scope=args.scope, mode=args.mode)
        _ = result
    else:
        metrics = engine.metrics(scope=args.scope, mode=args.mode)
        for metric in metrics:
            threshold = (
                f" (threshold {metric.threshold})"
                if metric.threshold is not None
                else ""
            )
            path = f" [{metric.path}]" if metric.path is not None else ""
            print(f"{metric.name}: {metric.value}{threshold}{path}")

    # TODO: compute exit codes once rules emit hard violations.
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via console entry
    sys.exit(main())
