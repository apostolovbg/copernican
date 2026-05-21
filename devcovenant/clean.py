"""Clean command implementation for DevCovenant."""

from __future__ import annotations

if __package__ in {None, ""}:  # pragma: no cover
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import devcovenant.core.cli_support as cli_args_module


def _build_parser() -> argparse.ArgumentParser:
    """Build parser for the clean command."""
    parser = cli_args_module.build_command_parser(
        "clean",
        (
            "Remove disposable build, cache, runtime-registry, "
            "or log artifacts safely."
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Remove both build/package and cache/test-output artifacts.",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Remove build/package artifacts only.",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Remove cache/test-output artifacts only.",
    )
    parser.add_argument(
        "--registry",
        action="store_true",
        help="Remove runtime registry artifacts only.",
    )
    parser.add_argument(
        "--logs",
        action="store_true",
        help="Remove run-log artifacts only.",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    """Execute clean command from parsed arguments."""
    from devcovenant.core.cleanup import clean_repo
    from devcovenant.core.execution import (
        devcovenant_banner_title,
        print_banner,
        print_step,
        resolve_repo_root,
    )

    repo_root = resolve_repo_root(require_install=True)

    print_banner(devcovenant_banner_title(), "🚀")
    print_step("Command: clean", "🧭")
    print_banner("Cleanup", "🧹")

    return clean_repo(
        repo_root,
        include_all=bool(args.all),
        include_build=bool(args.build),
        include_cache=bool(args.cache),
        include_registry=bool(args.registry),
        include_logs=bool(args.logs),
    )


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    cli_args_module.apply_output_mode_override_from_namespace(args)
    if args.all and (args.build or args.cache or args.registry or args.logs):
        parser.error(
            "`--all` cannot be combined with `--build`, `--cache`, "
            "`--registry`, or `--logs`."
        )
    if not any((args.all, args.build, args.cache, args.registry, args.logs)):
        parser.error(
            "select at least one cleanup scope: `--all`, `--build`, "
            "`--cache`, `--registry`, or `--logs`."
        )
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
