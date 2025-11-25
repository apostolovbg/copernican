# Last Updated: 2025-11-25
"""Command line entry point for DriftGuard."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

from driftguard import load_engine
from driftguard.rules import Metric, Violation


def _format_violation(violation: Violation) -> str:
    location = f" ({violation.path})" if violation.path else ""
    return (
        f"[{violation.severity.upper()}] {violation.rule_id}: "
        f"{violation.message}{location}"
    )


def _print_metrics(metrics: List[Metric]) -> None:
    for metric in metrics:
        details = f" ({metric.details})" if metric.details else ""
        print(f"{metric.name}: {metric.value}{details}")


def _json_metrics(metrics: List[Metric]) -> str:
    serialisable = [
        {"name": metric.name, "value": metric.value, "details": metric.details}
        for metric in metrics
    ]
    return json.dumps(serialisable, indent=2)


def _cmd_check(args: argparse.Namespace) -> int:
    engine = load_engine(repo_root=args.repo_root)
    violations, metrics = engine.check(scope=args.scope, mode=args.mode)
    for violation in violations:
        print(_format_violation(violation))
    for metric in metrics:
        print(f"METRIC {metric.name}: {metric.value} {metric.details}")
    if any(violation.severity == "hard" for violation in violations):
        return 1
    return 0


def _cmd_fix(args: argparse.Namespace) -> int:
    engine = load_engine(repo_root=args.repo_root)
    messages = engine.fix(scope=args.scope, safe_only=args.safe_only)
    for message in messages:
        print(message)
    return 0


def _cmd_metrics(args: argparse.Namespace) -> int:
    engine = load_engine(repo_root=args.repo_root)
    _, metrics = engine.check(scope="repo", mode="full")
    if args.format == "json":
        print(_json_metrics(metrics))
    else:
        _print_metrics(metrics)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run DriftGuard policy checks."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Optional repository root containing driftguard.yml.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser("check", help="Run policy checks.")
    check_parser.add_argument(
        "--scope",
        choices=["repo", "staged"],
        default="repo",
        help="Scope to evaluate.",
    )
    check_parser.add_argument(
        "--mode",
        choices=["fast", "full"],
        default="fast",
        help="Check mode to execute.",
    )
    check_parser.set_defaults(func=_cmd_check)

    fix_parser = subparsers.add_parser("fix", help="Apply safe auto-fixes.")
    fix_parser.add_argument(
        "--scope",
        choices=["repo", "staged"],
        default="staged",
        help="Scope to evaluate for fixes.",
    )
    fix_parser.add_argument(
        "--safe-only",
        action="store_true",
        default=True,
        help="Limit execution to safe fixes.",
    )
    fix_parser.add_argument(
        "--allow-unsafe",
        action="store_false",
        dest="safe_only",
        help="Permit rules that require non-trivial changes.",
    )
    fix_parser.set_defaults(func=_cmd_fix)

    metrics_parser = subparsers.add_parser(
        "metrics", help="Report drift metrics without failing."
    )
    metrics_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format for metrics.",
    )
    metrics_parser.set_defaults(func=_cmd_metrics)

    return parser


def main(argv: List[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
