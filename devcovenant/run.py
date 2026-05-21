"""Workflow-run command implementation for DevCovenant."""

from __future__ import annotations

if __package__ in {None, ""}:  # pragma: no cover
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import devcovenant.core.cli_support as cli_args_module


def _build_parser() -> argparse.ArgumentParser:
    """Build parser for the run command."""
    return cli_args_module.build_command_parser(
        "run",
        "Run all declared DevCovenant workflow runs.",
    )


def run(args: argparse.Namespace) -> int:
    """Execute the workflow-run command from parsed arguments."""
    from devcovenant.core.execution import (
        devcovenant_banner_title,
        print_banner,
        print_step,
        resolve_repo_root,
        run_bootstrap_registry_refresh,
        run_workflow_runs,
        warn_version_mismatch,
    )

    del args
    repo_root = resolve_repo_root(require_install=True)

    print_banner(devcovenant_banner_title(), "🚀")
    print_step("Command: run", "🧭")
    run_bootstrap_registry_refresh(repo_root)
    warn_version_mismatch(repo_root)

    print_banner("DevCovenant workflow run", "🏃")
    print_step("Running workflow runs", "▶️")
    return run_workflow_runs(repo_root, notes="")


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    cli_args_module.apply_output_mode_override_from_namespace(args)
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
