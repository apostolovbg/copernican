"""Helpers for loading and materializing test blueprint YAML files."""

from __future__ import annotations

import base64
import shutil
from dataclasses import dataclass
from pathlib import Path
from textwrap import wrap
from typing import Iterable

import yaml

import devcovenant.core.repository_paths as repository_paths


@dataclass(frozen=True)
class TestBlueprintFile:
    """One blueprint-rendered test file."""

    path: str
    content: str


@dataclass(frozen=True)
class TestBlueprintSyncIssue:
    """One mismatch between a blueprint file and a materialized tree."""

    issue_type: str
    path: str
    details: str


class _BlueprintDumper(yaml.SafeDumper):
    """YAML dumper that keeps multiline content in literal block scalars."""


def _represent_str(dumper: yaml.SafeDumper, text: str):
    """Represent multiline strings as literal block scalars."""
    style = "|" if "\n" in text else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", text, style=style)


_BlueprintDumper.add_representer(str, _represent_str)


def _encode_blueprint_content(content: str) -> str:
    """Encode one test file body for blueprint storage."""
    encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
    return "\n".join(wrap(encoded, width=72))


def _decode_blueprint_content(content: str) -> str:
    """Decode one stored blueprint body back into source text."""
    cleaned = "".join(str(content).split())
    return base64.b64decode(cleaned.encode("ascii"), validate=True).decode(
        "utf-8"
    )


def _normalize_relative_path(raw_path: object) -> str:
    """Normalize one blueprint path into a safe repository-relative path."""
    token = str(raw_path or "").strip().replace("\\", "/")
    if not token:
        raise ValueError("Blueprint path cannot be empty.")
    relative = Path(token)
    if relative.is_absolute():
        raise ValueError(f"Blueprint path must be relative, got `{token}`.")
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(
            f"Blueprint path must be a plain relative path, got `{token}`."
        )
    return relative.as_posix()


def _should_skip_path(path: Path) -> bool:
    """Return True when a candidate file should be excluded from blueprints."""
    if "__pycache__" in path.parts:
        return True
    if path.suffix in {".pyc", ".pyo", ".pyd"}:
        return True
    return False


def collect_test_blueprints_from_tree(root: Path) -> list[TestBlueprintFile]:
    """Collect blueprint files from an on-disk test tree."""
    if not root.exists():
        return []
    entries: list[TestBlueprintFile] = []
    for file_path in sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda candidate: candidate.relative_to(root).as_posix(),
    ):
        if _should_skip_path(file_path):
            continue
        entries.append(
            TestBlueprintFile(
                path=file_path.relative_to(root).as_posix(),
                content=file_path.read_text(encoding="utf-8"),
            )
        )
    return entries


def dump_test_blueprints_yaml(entries: Iterable[TestBlueprintFile]) -> str:
    """Render one blueprint collection as YAML text."""
    payload = {
        "test_blueprints": [
            {
                "path": entry.path,
                "encoding": "base64",
                "content": _encode_blueprint_content(entry.content),
            }
            for entry in entries
        ]
    }
    return yaml.dump(payload, Dumper=_BlueprintDumper, sort_keys=False)


def write_test_blueprints_yaml(
    path: Path,
    entries: Iterable[TestBlueprintFile],
) -> bool:
    """Write one blueprint YAML file and return True when it changed."""
    rendered = dump_test_blueprints_yaml(entries)
    current = ""
    if path.exists() and path.is_file():
        current = path.read_text(encoding="utf-8")
    if current == rendered:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")
    return True


def load_test_blueprints_yaml(path: Path) -> list[TestBlueprintFile]:
    """Load one blueprint YAML file from disk."""
    if not path.exists():
        return []
    try:
        payload = repository_paths.load_yaml(path)
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(
            f"Invalid test blueprint YAML in {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Test blueprint YAML must contain a mapping: {path}")
    raw_entries = payload.get("test_blueprints", [])
    if not isinstance(raw_entries, list):
        raise ValueError(
            "Test blueprint YAML must define `test_blueprints` as a list: "
            f"{path}"
        )
    entries: list[TestBlueprintFile] = []
    for index, raw_entry in enumerate(raw_entries):
        if not isinstance(raw_entry, dict):
            raise ValueError(
                f"Test blueprint entry #{index + 1} must be a mapping: {path}"
            )
        blueprint_path = _normalize_relative_path(raw_entry.get("path"))
        content = str(raw_entry.get("content", ""))
        encoding = str(raw_entry.get("encoding", "")).strip().lower()
        if encoding == "base64":
            try:
                content = _decode_blueprint_content(content)
            except (ValueError, UnicodeDecodeError) as exc:
                raise ValueError(
                    f"Invalid base64 blueprint content in {path}: {exc}"
                ) from exc
        entries.append(
            TestBlueprintFile(
                path=blueprint_path,
                content=content,
            )
        )
    return entries


def load_test_blueprints_for_tree(
    blueprint_path: Path,
    fallback_root: Path,
) -> list[TestBlueprintFile]:
    """Load blueprint entries from YAML, falling back to a source tree."""
    if blueprint_path.exists():
        return load_test_blueprints_yaml(blueprint_path)
    return collect_test_blueprints_from_tree(fallback_root)


def materialize_test_blueprints(
    target_root: Path,
    entries: Iterable[TestBlueprintFile],
    *,
    overwrite: bool = True,
) -> list[Path]:
    """Write blueprint entries into a target test tree."""
    normalized_entries = [
        TestBlueprintFile(
            path=_normalize_relative_path(entry.path),
            content=entry.content,
        )
        for entry in entries
    ]
    if target_root.exists() and overwrite:
        shutil.rmtree(target_root)
    written: list[Path] = []
    for entry in normalized_entries:
        output_path = target_root / entry.path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(entry.content, encoding="utf-8")
        written.append(output_path)
    return written


def compare_test_tree_to_blueprints(
    root: Path,
    entries: Iterable[TestBlueprintFile],
) -> list[TestBlueprintSyncIssue]:
    """Compare one test tree to its blueprint definition."""
    expected = {
        _normalize_relative_path(entry.path): entry.content
        for entry in entries
    }
    actual: dict[str, str] = {}
    if root.exists():
        for file_path in sorted(
            (path for path in root.rglob("*") if path.is_file()),
            key=lambda candidate: candidate.relative_to(root).as_posix(),
        ):
            if _should_skip_path(file_path):
                continue
            actual[file_path.relative_to(root).as_posix()] = (
                file_path.read_text(encoding="utf-8")
            )

    issues: list[TestBlueprintSyncIssue] = []
    for path in sorted(expected):
        if path not in actual:
            issues.append(
                TestBlueprintSyncIssue(
                    issue_type="missing_file",
                    path=path,
                    details="Blueprint file is missing from the test tree.",
                )
            )
            continue
        if actual[path] != expected[path]:
            issues.append(
                TestBlueprintSyncIssue(
                    issue_type="content_mismatch",
                    path=path,
                    details="Test file content differs from the blueprint.",
                )
            )
    for path in sorted(actual):
        if path not in expected:
            issues.append(
                TestBlueprintSyncIssue(
                    issue_type="extra_file",
                    path=path,
                    details="Test file exists without a matching blueprint.",
                )
            )
    return issues
