#!/usr/bin/env python3
"""Print the fixed DevCovenant quickstart reminder."""

from __future__ import annotations

if __package__ in {None, ""}:  # pragma: no cover
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import devcovenant.core.cli_support as cli_args_module


def _build_parser() -> argparse.ArgumentParser:
    """Build parser for the quickstart command."""
    return cli_args_module.build_command_parser(
        "quickstart",
        "Print a terse static reminder.",
    )


def _render_quickstart() -> str:
    """Return the fixed quickstart reminder text."""
    lines = [
        "DevCovenant quickstart reminder",
        "",
        "1. Install the CLI in an isolated machine environment.",
        "   pipx install devcovenant",
        "   devcovenant --version",
        "",
        "2. Install DevCovenant into the repository you want to govern.",
        "   cd your-repo",
        "   devcovenant install",
        "",
        "3. Review devcovenant/config.yaml, then deploy the reviewed setup.",
        "   Start with project-governance, developer_mode, and",
        "   profiles.active.",
        "   Builtin devcovuser is the always-active baseline for every",
        "   DevCovenant user repo.",
        "   In most downstream repositories, keep devcovuser active and add",
        "   a repo-owned userproject custom profile when the repository needs",
        "   its own starter layer.",
        "   devcovenant deploy",
        "",
        "4. If you want the real evaluation path, run the demo instead.",
        "   devcovenant demo",
        "",
        "5. If you are already in a governed repository, run the normal",
        "   gate cycle.",
        "   devcovenant gate --open",
        "   devcovenant gate --verify",
        "   devcovenant run",
        "   devcovenant gate --close",
        "",
        "This is a reminder, not the primary onboarding path.",
    ]
    return "\n".join(lines) + "\n"


def run(_args: argparse.Namespace) -> int:
    """Print the fixed quickstart reminder."""
    from devcovenant.core.cli_support import write_console_text

    del _args
    write_console_text(_render_quickstart(), flush=True)
    return 0


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    cli_args_module.apply_output_mode_override_from_namespace(args)
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
