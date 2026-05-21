"""Materialize one reusable asset or managed doc for operator reuse."""

from __future__ import annotations

if __package__ in {None, ""}:  # pragma: no cover
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import devcovenant.core.cli_support as cli_args_module


def _build_parser() -> argparse.ArgumentParser:
    """Build parser for the asset command."""
    parser = cli_args_module.build_command_parser(
        "asset",
        "Write one Desktop copy of a shipped profile asset or managed doc.",
    )
    parser.add_argument(
        "asset_name",
        help=("Asset target filename or exact asset target path to render."),
    )
    parser.add_argument(
        "output_name",
        nargs="?",
        help=(
            "Optional Desktop filename override. Must be a filename only; "
            "defaults to the asset's original filename."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the destination when it already exists.",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    """Execute the asset command."""
    import devcovenant.core.asset_materialization as asset_service
    from devcovenant.core.execution import (
        print_banner,
        print_step,
        resolve_repo_root,
        warn_version_mismatch,
    )

    repo_root = resolve_repo_root(require_install=True)

    print_banner("DevCovenant asset", "🧰")
    print_step("Command: asset", "🧭")
    warn_version_mismatch(repo_root)

    result = asset_service.materialize_named_asset(
        repo_root,
        str(args.asset_name or ""),
        output_name=str(args.output_name or "").strip() or None,
        overwrite=bool(args.overwrite),
    )
    selected = result.candidate
    print_step(
        f"Selected {selected.kind} `{selected.target_path}` from profile "
        f"`{selected.profile_name}`",
        "🧩",
    )
    print_step(
        f"Materialized to {result.output_path}",
        "✅",
    )
    return 0


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    cli_args_module.apply_output_mode_override_from_namespace(args)
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
