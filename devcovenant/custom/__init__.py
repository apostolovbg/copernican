"""Promote or retract builtin policy/profile customizations."""

from __future__ import annotations

if __package__ in {None, ""}:  # pragma: no cover
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
from pathlib import Path

import devcovenant.core.cli_support as cli_args_module


def _build_parser() -> argparse.ArgumentParser:
    """Build the parser for the custom command."""
    parser = cli_args_module.build_command_parser(
        "custom",
        (
            "Promote or retract builtin policy/profile custom copies and "
            "any mirrored tests."
        ),
    )
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--policy",
        dest="policy_id",
        help="Builtin policy id to shadow with a repo-owned custom copy.",
    )
    target_group.add_argument(
        "--profile",
        dest="profile_name",
        help="Builtin profile name to shadow with a repo-owned custom copy.",
    )
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument(
        "--do",
        action="store_true",
        help=(
            "Copy the builtin source tree and materialize any shipped test "
            "mirrors."
        ),
    )
    action_group.add_argument(
        "--undo",
        action="store_true",
        help="Remove the repo-owned copy and any mirrored tests.",
    )
    return parser


def _target_kind_and_name(args: argparse.Namespace) -> tuple[str, str]:
    """Return the selected customization kind and identifier."""
    policy_id = str(getattr(args, "policy_id", "") or "").strip()
    profile_name = str(getattr(args, "profile_name", "") or "").strip()
    if policy_id:
        return "policy", policy_id
    if profile_name:
        return "profile", profile_name
    raise SystemExit("Custom requires either `--policy` or `--profile`.")


def _describe_target(paths) -> str:
    """Return a stable display name for one customization target."""
    return f"{paths.kind} `{paths.name}`"


def run(args: argparse.Namespace) -> int:
    """Execute the custom command."""
    import devcovenant.core.customization as customization_service
    from devcovenant.core.execution import (
        devcovenant_banner_title,
        print_banner,
        print_step,
        resolve_repo_root,
    )
    from devcovenant.core.refresh_runtime import refresh_repo

    repo_root = resolve_repo_root(require_install=True)
    kind, name = _target_kind_and_name(args)
    paths = customization_service.resolve_customization_paths(
        repo_root,
        kind=kind,
        name=name,
    )

    print_banner(devcovenant_banner_title(), "🚀")
    print_step("Command: custom", "🧭")
    print_banner("Builtin customization", "🧩")
    print_step(
        f"Selected {_describe_target(paths)}",
        "📘",
    )

    if bool(getattr(args, "do", False)):
        customization_service.copy_builtin_customization(paths)
        materialized = customization_service.materialize_custom_tests(paths)
        print_step(
            f"Copied builtin source tree to {paths.custom_source_root}",
            "✅",
        )
        if materialized:
            print_step(
                f"Materialized mirrored tests under {paths.custom_test_root}",
                "✅",
            )
        else:
            print_step(
                "No shipped test blueprints were available to materialize.",
                "ℹ️",
            )
    else:
        removed_copy = customization_service.remove_customization(paths)
        removed_tests = customization_service.remove_custom_tests(paths)
        if removed_copy:
            print_step(
                f"Removed custom source tree at {paths.custom_source_root}",
                "🧹",
            )
        if removed_tests:
            print_step(
                f"Removed mirrored tests at {paths.custom_test_root}",
                "🧹",
            )

    print_step("Running refresh after customization", "🔄")
    return refresh_repo(repo_root)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    cli_args_module.apply_output_mode_override_from_namespace(args)
    raise SystemExit(run(args))
