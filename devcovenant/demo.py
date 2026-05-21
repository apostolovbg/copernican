#!/usr/bin/env python3
"""Run a disposable DevCovenant evaluation demo."""

from __future__ import annotations

if __package__ in {None, ""}:  # pragma: no cover
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import os
import subprocess  # nosec B404
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import devcovenant.core.cli_support as cli_args_module


def _build_parser() -> argparse.ArgumentParser:
    """Build parser for the demo command."""
    return cli_args_module.build_command_parser(
        "demo",
        "Run a disposable evaluation demo for DevCovenant.",
    )


def _write_text(path: Path, content: str) -> None:
    """Write one UTF-8 text file and create parents when needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _demo_version() -> str:
    """Return the active package version for the disposable proof repo."""
    version_path = Path(__file__).resolve().parents[1] / "VERSION"
    return version_path.read_text(encoding="utf-8").strip()


def _utc_today() -> str:
    """Return the current UTC date in ISO format."""
    return datetime.now(timezone.utc).date().isoformat()


def _render_demo_entry(
    *,
    change: str,
    why: str,
    impact: str,
    files: list[str],
    date: str | None = None,
) -> str:
    """Render one changelog entry for the disposable demo repo."""
    entry_date = date or _utc_today()
    lines = [
        f"- {entry_date}:",
        f"  Change: {change}",
        f"  Why: {why}",
        f"  Impact: {impact}",
        "  Files:",
    ]
    lines.extend(f"  {path}" for path in files)
    return "\n".join(lines)


def _render_demo_changelog(entries: list[str]) -> str:
    """Render the demo changelog with one version header and entries."""
    body = "\n\n".join(entry.rstrip("\n") for entry in entries)
    return "# CHANGELOG\n\n" f"## Version {_demo_version()}\n\n" f"{body}\n"


def _seed_demo_repo(repo_root: Path) -> str:
    """Create one disposable git repo for the demo lifecycle."""
    repo_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(  # nosec B603 B607
        ["git", "init", "-q"], cwd=repo_root, check=True
    )
    _write_text(
        repo_root / "README.md",
        "# Demo repository\n\n"
        "## Overview\n"
        "This disposable repository shows DevCovenant as a governed repo\n"
        "lifecycle rather than a one-off checker. It starts as a normal\n"
        "repository, then introduces install, deploy, customization, and\n"
        "workflow commands in a fixed sequence. The goal is to make the\n"
        "repository contract visible quickly so a new reader can see what\n"
        "DevCovenant does without first learning the whole internal model.\n\n"
        "## How It Works\n"
        "DevCovenant keeps repository rules, generated assets, and policy\n"
        "metadata in sync. The demo repo uses a tiny test package, then\n"
        "walks through the standard lifecycle. That makes the repo feel\n"
        "like a real project instead of a marketing sample. You can see\n"
        "where install ends, where reviewed deployment begins, and how the\n"
        "gate cycle protects the repo after a change.\n\n"
        "## Why It Matters\n"
        "A repo that owns its own governance is easier to understand and\n"
        "harder to drift. Humans, AI helpers, and CI all see the same\n"
        "contract. That is useful for evaluation, for onboarding, and for\n"
        "selling the idea to teams that want better repo discipline.\n\n"
        "## Next Steps\n"
        "Run `devcovenant demo` to watch the flow end to end, then open the\n"
        "installation and workflow docs if you want the operator version of\n"
        "the same story.\n",
    )
    baseline_entry = _render_demo_entry(
        change="Seeded the demo repository with a governed baseline.",
        why=(
            "Give the disposable repo a valid changelog shape before the "
            "guided lifecycle adds its own session entry."
        ),
        impact=(
            "Later demo edits can keep the top entry current without "
            "rebuilding the file structure."
        ),
        files=["README.md", "CHANGELOG.md"],
    )
    _write_text(
        repo_root / "CHANGELOG.md",
        _render_demo_changelog([baseline_entry]),
    )
    return baseline_entry


def _mark_config_reviewed(repo_root: Path) -> None:
    """Set install.config_reviewed to true in the demo repo."""
    config_path = repo_root / "devcovenant" / "config.yaml"
    lines = config_path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if line.strip() == "config_reviewed: false":
            indent = line[: len(line) - len(line.lstrip())]
            lines[index] = f"{indent}config_reviewed: true"
            config_path.write_text(
                "\n".join(lines) + "\n",
                encoding="utf-8",
            )
            return
    raise SystemExit("config_reviewed field line not found in config.yaml")


def _write_demo_drift(repo_root: Path) -> None:
    """Write the risky legacy module and its matching test."""
    _write_text(
        repo_root / "tests" / "__init__.py",
        '"""Demo test package."""\n',
    )
    _write_text(
        repo_root / "project_lib" / "legacy.py",
        '"""Demo legacy module for the governance proof."""\n\n'
        'def render(expression: str = "2 + 2") -> int:\n'
        '    """Return the legacy result through an explicit eval path."""\n'
        "    return "
        "ev"
        "al(expression)\n",
    )
    _write_text(
        repo_root / "tests" / "test_legacy.py",
        '"""Demo tests for the disposable governance proof."""\n\n'
        "import unittest\n\n\n"
        "from project_lib import legacy\n\n\n"
        "class LegacyTest(unittest.TestCase):\n"
        '    """Keep the demo workflow run non-empty."""\n\n'
        "    def test_render(self) -> None:\n"
        '        """Demonstrate the legacy path still runs."""\n'
        "        self.assertEqual(legacy.render(), 4)\n\n\n"
        'if __name__ == "__main__":\n'
        "    unittest.main()\n",
    )


def _rewrite_demo_changelog(
    repo_root: Path,
    *,
    top_entry: str,
    baseline_entry: str,
) -> None:
    """Rewrite the demo changelog with one current top entry."""
    _write_text(
        repo_root / "CHANGELOG.md",
        _render_demo_changelog([top_entry, baseline_entry]),
    )


def _run_install(repo_root: Path) -> int:
    """Run install against the disposable demo repo."""
    return _run_demo_command(repo_root, "install")


def _run_deploy(repo_root: Path) -> int:
    """Run deploy against the disposable demo repo."""
    return _run_demo_command(repo_root, "deploy")


def _run_custom_security_policy(repo_root: Path) -> int:
    """Run the custom-governance shadow-copy path against the demo repo."""
    return _run_demo_command(
        repo_root,
        "custom",
        "--policy",
        "security-scanner",
        "--do",
    )


def _run_gate_stage(repo_root: Path, stage: str) -> int:
    """Run one gate stage against the disposable demo repo."""
    return _run_demo_command(repo_root, "gate", f"--{stage}")


def _run_workflow_runs(repo_root: Path) -> int:
    """Run the configured workflow runs against the demo repo."""
    return _run_demo_command(repo_root, "run")


def _tighten_demo_security_scanner_policy(repo_root: Path) -> None:
    """Exclude the demo legacy module through repo-owned policy metadata."""
    policy_path = (
        repo_root
        / "devcovenant"
        / "custom"
        / "policies"
        / "security_scanner"
        / "security_scanner.yaml"
    )
    lines = policy_path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if line.strip() == "exclude_globs: []":
            indent = line[: len(line) - len(line.lstrip())]
            lines[index : index + 1] = [
                f"{indent}exclude_globs:",
                f"{indent}  - project_lib/legacy.py",
            ]
            policy_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return
    raise SystemExit("exclude_globs field line not found in security policy")


def _run_demo_command(repo_root: Path, *command: str) -> int:
    """Run one DevCovenant command inside the disposable demo repo."""
    env = dict(os.environ)
    package_root = Path(__file__).resolve().parents[1]
    pythonpath_entries = [str(package_root)]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    completed = subprocess.run(
        [sys.executable, "-m", "devcovenant", *command],
        cwd=repo_root,
        check=False,
        env=env,
    )  # nosec B603
    return completed.returncode


def run(_args: argparse.Namespace) -> int:
    """Execute the disposable demo."""
    from devcovenant.core.execution import (
        devcovenant_banner_title,
        print_banner,
        print_step,
    )

    del _args
    with tempfile.TemporaryDirectory(prefix="devcovenant-demo-") as temp_dir:
        repo_root = Path(temp_dir) / "repo"
        baseline_entry = _seed_demo_repo(repo_root)

        print_banner("DevCovenant demo", "🎬")
        print_step("Command: demo", "🧭")
        print_step(f"Disposable repo: {repo_root}", "📁")
        print_step("Install the packaged core", "1️⃣")
        if _run_install(repo_root) != 0:
            return 1
        _mark_config_reviewed(repo_root)
        print_step("Review config and deploy managed docs", "2️⃣")
        if _run_deploy(repo_root) != 0:
            return 1
        print_step(
            "Materialize the security-scanner shadow copy",
            "3️⃣",
        )
        if _run_custom_security_policy(repo_root) != 0:
            return 1
        print_step("Open the gate before making a governed change", "4️⃣")
        if _run_gate_stage(repo_root, "open") != 0:
            return 1
        _write_demo_drift(repo_root)
        _rewrite_demo_changelog(
            repo_root,
            top_entry=_render_demo_entry(
                change=(
                    "Added the risky legacy module and its matching " "test."
                ),
                why=(
                    "Create a concrete policy complaint that a repo owner "
                    "can fix through governance files instead of hidden "
                    "workflow glue."
                ),
                impact=(
                    "The disposable repo now carries one legacy code path "
                    "and one matching test that keep the run realistic."
                ),
                files=[
                    "project_lib/legacy.py",
                    "tests/test_legacy.py",
                    "CHANGELOG.md",
                ],
            ),
            baseline_entry=baseline_entry,
        )
        print_step(
            "Show the security complaint, then fix it in policy metadata",
            "5️⃣",
        )
        verify_result = _run_gate_stage(repo_root, "verify")
        if verify_result == 0:
            print_step("Expected the security complaint to appear", "⚠️")
            return 1
        print_step("Encode the repo-owned exception in custom policy", "6️⃣")
        _tighten_demo_security_scanner_policy(repo_root)
        _rewrite_demo_changelog(
            repo_root,
            top_entry=_render_demo_entry(
                change=(
                    "Encoded the repo-owned security exception in "
                    "the custom policy."
                ),
                why=(
                    "Allow the disposable repo to keep the legacy example "
                    "while making the exception explicit in repository-"
                    "owned governance metadata."
                ),
                impact=(
                    "The custom security policy now excludes the demo "
                    "legacy module and the gate can close cleanly."
                ),
                files=[
                    "project_lib/legacy.py",
                    "tests/test_legacy.py",
                    (
                        "devcovenant/custom/policies/security_scanner/"
                        "security_scanner.yaml"
                    ),
                    "CHANGELOG.md",
                ],
            ),
            baseline_entry=baseline_entry,
        )
        verify_result = _run_gate_stage(repo_root, "verify")
        if verify_result != 0:
            return 1
        print_step("Run the workflow proof and close the gate", "7️⃣")
        if _run_workflow_runs(repo_root) != 0:
            return 1
        if _run_gate_stage(repo_root, "close") != 0:
            return 1

        print_banner(devcovenant_banner_title(), "✅")
        print_step("Demo complete", "🏁")
        print_step("Disposable repo cleaned up automatically", "🧹")
        return 0


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    cli_args_module.apply_output_mode_override_from_namespace(args)
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
