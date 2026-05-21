#!/usr/bin/env python3
"""Remove deployed DevCovenant generated artifacts while keeping core."""

from __future__ import annotations

if __package__ in {None, ""}:  # pragma: no cover
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import shutil
import time
from pathlib import Path

import yaml

import devcovenant.core.cli_support as cli_args_module
import devcovenant.core.repository_paths as yaml_cache_service

BLOCK_BEGIN = "<!-- DEVCOV:BEGIN -->"
BLOCK_END = "<!-- DEVCOV:END -->"
WORKFLOW_BEGIN = "<!-- DEVCOV-WORKFLOW:BEGIN -->"
WORKFLOW_END = "<!-- DEVCOV-WORKFLOW:END -->"
POLICY_BEGIN = "<!-- DEVCOV-POLICIES:BEGIN -->"
POLICY_END = "<!-- DEVCOV-POLICIES:END -->"
USER_GITIGNORE_BEGIN = "# --- User entries (preserved) ---"
USER_GITIGNORE_END = "# --- End user entries ---"
_MANAGED_MARKERS = (
    BLOCK_BEGIN,
    BLOCK_END,
    WORKFLOW_BEGIN,
    WORKFLOW_END,
    POLICY_BEGIN,
    POLICY_END,
)
_RECOVERY_MANAGED_DOCS = (
    "AGENTS.md",
    "README.md",
    "CONTRIBUTING.md",
    "SPEC.md",
    "PLAN.md",
    "CHANGELOG.md",
    "devcovenant/README.md",
)


def _read_yaml(path: Path) -> dict[str, object]:
    """Load YAML mapping payload from disk."""
    if not path.exists():
        raise ValueError(
            f"Undeploy config issue: missing required config file: {path}."
        )
    try:
        payload = yaml_cache_service.load_yaml(path)
    except yaml.YAMLError as exc:
        raise ValueError(
            f"Undeploy config issue: invalid YAML in {path}: {exc}"
        ) from exc
    except OSError as exc:
        raise ValueError(
            f"Undeploy config issue: unable to read {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(
            f"Undeploy config issue: {path} must contain a YAML mapping."
        )
    return payload


def _normalize_doc_name(name: str) -> str:
    """Normalize configured doc names to canonical markdown paths."""
    mapping = {
        "AGENTS": "AGENTS.md",
        "README": "README.md",
        "CONTRIBUTING": "CONTRIBUTING.md",
        "SPEC": "SPEC.md",
        "PLAN": "PLAN.md",
        "CHANGELOG": "CHANGELOG.md",
    }
    token = str(name or "").strip()
    if not token:
        return ""
    upper = token.upper()
    if upper in mapping:
        return mapping[upper]
    if upper.endswith(".MD") and upper[:-3] in mapping:
        return mapping[upper[:-3]]
    return token


def _managed_docs_from_config(repo_root: Path) -> list[str]:
    """Resolve managed docs from config doc_assets."""
    config = _read_yaml(repo_root / "devcovenant" / "config.yaml")
    doc_assets = config.get("doc_assets")
    if not isinstance(doc_assets, dict):
        raise ValueError(
            "Undeploy config issue: `doc_assets` must be a mapping in "
            "devcovenant/config.yaml."
        )

    autogen_raw = doc_assets.get("autogen")
    user_raw = doc_assets.get("user")

    if not isinstance(autogen_raw, list):
        raise ValueError(
            "Undeploy config issue: `doc_assets.autogen` must be a list."
        )
    if not isinstance(user_raw, list):
        raise ValueError(
            "Undeploy config issue: `doc_assets.user` must be a list."
        )

    autogen = [_normalize_doc_name(item) for item in autogen_raw]

    user_docs = {_normalize_doc_name(item) for item in user_raw if item}

    selected = [doc for doc in autogen if doc and doc not in user_docs]
    if not selected:
        raise ValueError(
            "Undeploy config issue: "
            "`doc_assets.autogen` resolved to no managed documents "
            "after removing `doc_assets.user` entries."
        )

    ordered = []
    for doc in selected:
        if doc not in ordered:
            ordered.append(doc)
    return ordered


def _managed_docs_from_registry(repo_root: Path) -> list[str]:
    """Resolve enabled managed docs from the tracked registry when present."""
    registry_path = repo_root / "devcovenant" / "registry" / "registry.yaml"
    if not registry_path.exists():
        return []
    try:
        payload = yaml_cache_service.load_yaml(registry_path)
    except (yaml.YAMLError, OSError):
        return []
    if not isinstance(payload, dict):
        return []
    managed_docs = payload.get("managed-docs")
    if not isinstance(managed_docs, dict):
        return []
    enabled_docs = managed_docs.get("enabled_docs")
    if not isinstance(enabled_docs, list):
        return []
    ordered: list[str] = []
    for entry in enabled_docs:
        doc_name = _normalize_doc_name(str(entry or "").strip())
        if doc_name and doc_name not in ordered:
            ordered.append(doc_name)
    return ordered


def _discover_docs_with_managed_markers(repo_root: Path) -> list[str]:
    """Return text files that contain any known managed marker."""
    discovered: list[str] = []
    for file_path in repo_root.rglob("*"):
        if not file_path.is_file():
            continue
        if ".git" in file_path.parts:
            continue
        if file_path.suffix.lower() != ".md":
            continue
        try:
            text = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if any(marker in text for marker in _MANAGED_MARKERS):
            discovered.append(file_path.relative_to(repo_root).as_posix())
    return discovered


def _strip_blocks(text: str, begin: str, end: str) -> str:
    """Remove all begin/end managed block ranges from text."""
    updated = text
    while begin in updated and end in updated:
        start = updated.find(begin)
        stop = updated.find(end, start)
        if stop < 0:
            break
        updated = updated[:start] + updated[stop + len(end) :]
    return updated


def _remove_generated_gitignore(repo_root: Path) -> bool:
    """Remove generated .gitignore content and preserve user entries only."""
    gitignore_path = repo_root / ".gitignore"
    if not gitignore_path.exists():
        return False

    existing = gitignore_path.read_text(encoding="utf-8")
    begin_index = existing.find(USER_GITIGNORE_BEGIN)
    end_index = existing.find(USER_GITIGNORE_END)
    if begin_index < 0 or end_index < 0 or end_index < begin_index:
        return False

    body_start = begin_index + len(USER_GITIGNORE_BEGIN)
    preserved_lines = existing[body_start:end_index].splitlines()
    while preserved_lines and not preserved_lines[0].strip():
        preserved_lines.pop(0)
    while preserved_lines and not preserved_lines[-1].strip():
        preserved_lines.pop()

    preserved_text = "\n".join(
        line.rstrip() for line in preserved_lines
    ).strip()
    if not preserved_text:
        gitignore_path.unlink(missing_ok=True)
        return True

    updated = preserved_text + "\n"
    if updated == existing:
        return False
    gitignore_path.write_text(updated, encoding="utf-8")
    return True


def undeploy_repo(repo_root: Path) -> int:
    """Remove managed blocks and generated registry state."""
    from devcovenant.core.execution import (
        merge_active_run_phase_timings,
        print_step,
    )

    phase_timings: list[dict[str, object]] = []
    docs: list[str]
    registry_docs: list[str] = []
    recovery_scan_required = False
    doc_selection_started = time.perf_counter()
    try:
        docs = _managed_docs_from_config(repo_root)
    except ValueError as exc:
        docs = []
        recovery_scan_required = True
        print_step(f"{exc} Continuing with recovery teardown.", "⚠️")
    registry_docs = _managed_docs_from_registry(repo_root)
    doc_candidates = set(docs)
    doc_candidates.update(registry_docs)
    doc_candidates.update(_RECOVERY_MANAGED_DOCS)
    phase_timings.append(
        {
            "phase": "doc_selection",
            "duration_seconds": round(
                time.perf_counter() - doc_selection_started, 6
            ),
            "changed": bool(doc_candidates),
        }
    )
    recovery_scan_started = time.perf_counter()
    if recovery_scan_required:
        doc_candidates.update(_discover_docs_with_managed_markers(repo_root))
    phase_timings.append(
        {
            "phase": "recovery_doc_scan",
            "duration_seconds": round(
                time.perf_counter() - recovery_scan_started, 6
            ),
            "changed": recovery_scan_required,
            "skipped": not recovery_scan_required,
        }
    )
    stripped_docs = []

    strip_docs_started = time.perf_counter()
    for doc_name in sorted(doc_candidates):
        path = repo_root / doc_name
        if not path.exists():
            continue

        try:
            original = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            print_step(
                f"Skipping unreadable file during undeploy: {doc_name}: {exc}",
                "⚠️",
            )
            continue
        updated = _strip_blocks(original, BLOCK_BEGIN, BLOCK_END)
        updated = _strip_blocks(updated, WORKFLOW_BEGIN, WORKFLOW_END)
        updated = _strip_blocks(updated, POLICY_BEGIN, POLICY_END)
        if updated == original:
            continue

        try:
            path.write_text(updated.strip() + "\n", encoding="utf-8")
        except OSError as exc:
            print_step(
                f"Unable to write managed-block cleanup for {doc_name}: {exc}",
                "⚠️",
            )
            continue
        stripped_docs.append(doc_name)
    phase_timings.append(
        {
            "phase": "strip_managed_docs",
            "duration_seconds": round(
                time.perf_counter() - strip_docs_started, 6
            ),
            "changed": bool(stripped_docs),
        }
    )

    cleanup_started = time.perf_counter()
    registry_runtime = repo_root / "devcovenant" / "registry" / "runtime"
    if registry_runtime.exists():
        shutil.rmtree(registry_runtime)

    tracked_registry = repo_root / "devcovenant" / "registry" / "registry.yaml"
    tracked_registry.unlink(missing_ok=True)

    if _remove_generated_gitignore(repo_root):
        print_step("Removed generated .gitignore fragments", "✅")

    if stripped_docs:
        print_step(
            f"Removed managed blocks from: {', '.join(stripped_docs)}",
            "✅",
        )
    phase_timings.append(
        {
            "phase": "registry_cleanup",
            "duration_seconds": round(
                time.perf_counter() - cleanup_started, 6
            ),
            "changed": True,
        }
    )
    merge_active_run_phase_timings("undeploy", phase_timings)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    """Build parser for undeploy command."""
    return cli_args_module.build_command_parser(
        "undeploy",
        "Remove deployed managed artifacts and keep core files.",
    )


def run(args: argparse.Namespace) -> int:
    """Execute undeploy command."""
    from devcovenant.core.execution import (
        devcovenant_banner_title,
        print_banner,
        print_step,
        resolve_repo_root,
    )

    del args
    repo_root = resolve_repo_root(require_install=True)

    print_banner(devcovenant_banner_title(), "🚀")
    print_step("Command: undeploy", "🧭")
    print_banner("Undeploy", "📤")

    return undeploy_repo(repo_root)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    cli_args_module.apply_output_mode_override_from_namespace(args)
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
