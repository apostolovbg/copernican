#!/usr/bin/env python3
"""Refresh command entrypoint for DevCovenant."""

from __future__ import annotations

if __package__ in {None, ""}:  # pragma: no cover
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import devcovenant.core.cli_support as cli_args_module


def _build_parser() -> argparse.ArgumentParser:
    """Build parser for refresh command."""
    return cli_args_module.build_command_parser(
        "refresh",
        "Run a full refresh.",
    )


def run(args: argparse.Namespace) -> int:
    """Execute refresh command."""
    from devcovenant.core.execution import (
        devcovenant_banner_title,
        print_banner,
        print_step,
        resolve_repo_root,
    )
    from devcovenant.core.refresh_runtime import refresh_repo

    del args
    repo_root = resolve_repo_root(require_install=True)

    print_banner(devcovenant_banner_title(), "🚀")
    print_step("Command: refresh", "🧭")
    print_banner("Full refresh", "🔄")

    return refresh_repo(repo_root)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    cli_args_module.apply_output_mode_override_from_namespace(args)
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
