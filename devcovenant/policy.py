#!/usr/bin/env python3
"""CLI entry point for namespaced policy-born commands."""

from __future__ import annotations

if __package__ in {None, ""}:  # pragma: no cover
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import sys
from pathlib import Path
from typing import Sequence

import devcovenant.core.cli_support as cli_args_module


def _build_parser() -> argparse.ArgumentParser:
    """Build the parser for the namespaced policy command surface."""
    parser = cli_args_module.build_command_parser(
        "policy",
        "Run one explicit policy-born command declared by an enabled policy.",
    )
    parser.add_argument("policy_id", help="Policy id to target")
    parser.add_argument("policy_command", help="Policy command to run")
    parser.add_argument(
        "command_args",
        nargs=argparse.REMAINDER,
        help="Arguments for the declared policy command",
    )
    return parser


def _render_generic_result(result: object) -> None:
    """Print a stable human-readable summary from one runtime result."""
    from devcovenant.core.execution import runtime_print

    if isinstance(result, dict):
        message = str(result.get("message", "")).strip()
        if message:
            runtime_print(message, file=sys.stdout)
        lines = result.get("lines")
        if isinstance(lines, list):
            for entry in lines:
                token = str(entry).rstrip()
                if token:
                    runtime_print(token, file=sys.stdout)
        lock_results = result.get("lock_results")
        if isinstance(lock_results, list):
            runtime_print("Dependency refresh results:", file=sys.stdout)
            for entry in lock_results:
                if not isinstance(entry, dict):
                    continue
                lock_file = str(entry.get("lock_file", "")).strip()
                detail = str(entry.get("message", "")).strip()
                if lock_file and detail:
                    runtime_print(f"- {lock_file}: {detail}", file=sys.stdout)
        refreshed = result.get("refreshed_artifacts")
        if isinstance(refreshed, list) and refreshed:
            runtime_print(
                "Refreshed artifacts: "
                + ", ".join(str(item) for item in refreshed),
                file=sys.stdout,
            )
        return
    if result is not None:
        runtime_print(str(result), file=sys.stdout)


def run(args: argparse.Namespace) -> int:
    """Execute one declared policy-born command."""
    import devcovenant.core.policy_commands as policy_commands_service
    from devcovenant.core.execution import resolve_repo_root, runtime_print
    from devcovenant.core.policy_runtime import run_policy_runtime_action

    repo_root = resolve_repo_root(require_install=True)
    command = policy_commands_service.find_policy_command(
        repo_root,
        policy_id=args.policy_id,
        command_name=args.policy_command,
    )
    if command is None:
        runtime_print(
            (
                "Policy command not found: "
                f"`{args.policy_id} {args.policy_command}`. "
                "Run `devcovenant refresh` if command metadata changed."
            ),
            file=sys.stderr,
        )
        return 1
    payload = policy_commands_service.parse_policy_command_payload(
        args.policy_id,
        command,
        list(args.command_args or []),
    )
    result = run_policy_runtime_action(
        repo_root,
        policy_id=args.policy_id,
        action=command.runtime_action,
        payload=payload,
    )
    _render_generic_result(result)
    return 0


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    cli_args_module.apply_output_mode_override_from_namespace(args)
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
