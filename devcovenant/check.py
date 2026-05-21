#!/usr/bin/env python3
"""Check command implementation for DevCovenant."""

from __future__ import annotations

if __package__ in {None, ""}:  # pragma: no cover
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import os
from pathlib import Path

import devcovenant.core.cli_support as cli_args_module

_CHECK_APPLY_FIXES_ENV = "DEVCOV_CHECK_APPLY_FIXES"
_CHECK_RUN_REFRESH_ENV = "DEVCOV_CHECK_RUN_REFRESH"
_CHECK_CLEAN_BYTECODE_ENV = "DEVCOV_CHECK_CLEAN_BYTECODE"


def _env_flag(name: str) -> bool:
    """Return True when the given environment variable is truthy."""
    raw = os.environ.get(name, "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _build_parser() -> argparse.ArgumentParser:
    """Build parser for check command."""
    return cli_args_module.build_command_parser(
        "check",
        "Run read-only DevCovenant audit checks.",
    )


def _run_check(
    repo_root: Path,
    *,
    apply_fixes: bool,
    run_refresh: bool,
    cleanup_bytecode: bool,
) -> int:
    """Run policy checks through the engine."""
    from devcovenant.core.execution import (
        cleanup_repo_bytecode_artifacts,
        devcovenant_banner_title,
        print_banner,
        print_step,
        warn_version_mismatch,
    )
    from devcovenant.core.policy_runtime import DevCovenantEngine
    from devcovenant.core.refresh_runtime import refresh_repo

    print_banner(devcovenant_banner_title(), "🚀")
    print_step("Command: check", "🧭")
    if apply_fixes or run_refresh or cleanup_bytecode:
        print_step("Mode: gate-orchestrated", "🛡️")
    else:
        print_step("Mode: read-only audit", "🔎")
    print_step(f"Auto-fix: {'enabled' if apply_fixes else 'disabled'}", "🛠️")

    if run_refresh:
        print_step("Running full refresh", "🔄")
        refresh_exit = refresh_repo(repo_root)
        if refresh_exit != 0:
            print_step("Full refresh failed", "🚫")
            return refresh_exit
        print_step("Full refresh complete", "✅")
    else:
        print_step("Startup refresh skipped (audit mode)", "⏭️")

    if cleanup_bytecode:
        cleanup_repo_bytecode_artifacts(repo_root)
        print_step("Bytecode cleanup complete", "🧹")
    else:
        print_step("Bytecode cleanup skipped (audit mode)", "⏭️")
    warn_version_mismatch(repo_root)

    print_step("Initializing engine", "🧠")
    engine = DevCovenantEngine(repo_root=repo_root)
    print_step("Engine ready", "✅")

    print_banner("Policy checks", "🔍")
    print_step("Running policy checks", "▶️")
    result = engine.check(apply_fixes=apply_fixes)
    print_step("Policy checks complete", "🏁")

    if result.should_block:
        return 1
    if result.has_sync_issues():
        return 1
    return 0


def run(_args: argparse.Namespace) -> int:
    """Execute check command."""
    from devcovenant.core.execution import resolve_repo_root

    repo_root = resolve_repo_root(require_install=True)
    # `check` is the read-only audit command. Gate orchestration can opt into
    # refresh/autofix/cleanup for the same checking routine via environment.
    apply_fixes = _env_flag(_CHECK_APPLY_FIXES_ENV)
    run_refresh = _env_flag(_CHECK_RUN_REFRESH_ENV)
    cleanup_bytecode = _env_flag(_CHECK_CLEAN_BYTECODE_ENV)
    return _run_check(
        repo_root,
        apply_fixes=apply_fixes,
        run_refresh=run_refresh,
        cleanup_bytecode=cleanup_bytecode,
    )


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    cli_args_module.apply_output_mode_override_from_namespace(args)
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
