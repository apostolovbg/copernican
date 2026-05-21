"""Gate lifecycle, session snapshots, and gate-status helpers."""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
import shlex
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import yaml

import devcovenant.core.execution as execution_runtime_module
import devcovenant.core.profile_registry as profile_registry_service
import devcovenant.core.project_governance as project_governance_service
import devcovenant.core.repository_paths as yaml_cache_service
import devcovenant.core.tracked_registry as tracked_registry_module
import devcovenant.core.workflow_support as registry_runtime
import devcovenant.core.workflow_support as registry_runtime_module
import devcovenant.core.workflow_support as workflow_contract_module
from devcovenant.core.document_exemptions import (
    EMPTY_MANAGED_MARKER_SIGNATURE as _EMPTY_MANAGED_MARKER_SIGNATURE,
)
from devcovenant.core.document_exemptions import (
    document_exemption_fingerprint_for_path,
)
from devcovenant.core.repository_paths import display_path


def load_gate_status_payload(path: Path) -> dict[str, object]:
    """Load one gate-status payload, returning empty mapping when missing."""
    rendered = display_path(path)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid gate status JSON in {rendered}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Gate status payload must be a mapping: {rendered}")
    return payload


def validate_gate_status_payload(path: Path) -> dict[str, object]:
    """Raise `ValueError` when one gate-status payload is malformed."""
    if not path.exists():
        raise ValueError(
            f"Gate status payload is missing: {display_path(path)}"
        )

    payload = load_gate_status_payload(path)
    last_run_utc = str(payload.get("last_run_utc", "")).strip()
    try:
        datetime.fromisoformat(last_run_utc.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            "Field 'last_run_utc' must be an ISO-8601 timestamp."
        ) from exc

    commands = payload.get("commands")
    if not isinstance(commands, list):
        raise ValueError(
            "Field 'commands' must record the executed workflow command "
            "list."
        )
    normalized_commands = [
        str(entry or "").strip()
        for entry in commands
        if str(entry or "").strip()
    ]
    if not normalized_commands:
        raise ValueError(
            "Field 'commands' must record at least one executed workflow "
            "command."
        )
    return payload


def _load_status(path: Path) -> dict:
    """Load the current status payload."""
    return load_gate_status_payload(path)


def _gate_status_summary_lines(repo_root: Path) -> list[str]:
    """Return short, deterministic status lines for `gate --status`."""
    repo_root = Path(repo_root)
    status_path = registry_runtime_module.gate_status_path(repo_root)
    status_rel = _repo_relative(repo_root, status_path)
    latest_pointer = _resolve_latest_relevant_run_pointer(repo_root)
    latest_line = _latest_pointer_summary_line(latest_pointer)
    if not status_path.exists():
        lines = [
            "Gate Status: missing",
            f"Status File: {status_rel}",
        ]
        if latest_line:
            lines.append(latest_line)
        return lines

    try:
        payload = _load_status(status_path)
    except ValueError as error:
        lines = [
            "Gate Status: malformed",
            f"Status File: {status_rel}",
            f"Error: {error}",
        ]
        if latest_line:
            lines.append(latest_line)
        return lines
    workflow_payload = _load_workflow_session_payload(repo_root)

    session_state = str(payload.get("session_state", "")).strip().lower()
    state_label = session_state or "unknown"
    lines = [f"Gate Status: {state_label}"]
    session_id = str(payload.get("session_id", "")).strip()
    if session_id:
        lines.append(f"Session ID: {session_id}")
    last_stage = _infer_last_gate_stage(payload, workflow_payload)
    if last_stage:
        lines.append(f"Last Stage: {last_stage}")

    session_open = _status_time_token(
        payload, "session_open_utc"
    ) or _status_time_token(payload, "pre_commit_open_utc")
    if session_open:
        lines.append(f"Session Open: {session_open}")
    session_close = _status_time_token(payload, "session_close_utc")
    if session_close:
        lines.append(f"Session Close: {session_close}")
    last_workflow_run = _latest_workflow_run_utc(payload, workflow_payload)
    if last_workflow_run:
        lines.append(f"Last Workflow Run: {last_workflow_run}")
    if latest_line:
        lines.append(latest_line)
    return lines


def _infer_last_gate_stage(
    payload: dict[str, object],
    workflow_payload: dict[str, object] | None = None,
) -> str:
    """Infer the latest completed public workflow stage."""

    stage_epochs = _stage_epochs(payload, workflow_payload)
    resolved = [
        (index, stage, epoch)
        for index, (stage, epoch) in enumerate(stage_epochs)
        if epoch > 0.0
    ]
    if not resolved:
        return ""
    _, stage, _ = max(resolved, key=lambda item: (item[2], item[0]))
    return stage


def _load_workflow_session_payload(repo_root: Path) -> dict[str, object]:
    """Load workflow-session payload for status rendering when available."""

    try:
        return load_workflow_session(repo_root)
    except ValueError:
        return {}


def _anchor_epoch(
    workflow_payload: dict[str, object] | None,
    stage: str,
) -> float:
    """Return one workflow-session anchor epoch when present."""

    if not isinstance(workflow_payload, dict):
        return 0.0
    anchors = workflow_payload.get("anchors")
    if not isinstance(anchors, dict):
        return 0.0
    anchor = anchors.get(stage)
    if not isinstance(anchor, dict):
        return 0.0
    return _status_epoch(anchor, "last_run_epoch")


def _runs_epoch(workflow_payload: dict[str, object] | None) -> float:
    """Return the latest workflow-run epoch from session state."""

    if not isinstance(workflow_payload, dict):
        return 0.0
    runs = workflow_payload.get("runs")
    if not isinstance(runs, dict):
        return 0.0
    latest = 0.0
    for entry in runs.values():
        if not isinstance(entry, dict):
            continue
        latest = max(latest, _status_epoch(entry, "last_run_epoch"))
    return latest


def _runs_last_run_utc(workflow_payload: dict[str, object] | None) -> str:
    """Return the latest workflow-run UTC token from session state."""

    if not isinstance(workflow_payload, dict):
        return ""
    runs = workflow_payload.get("runs")
    if not isinstance(runs, dict):
        return ""
    latest_epoch = 0.0
    latest_token = ""
    for entry in runs.values():
        if not isinstance(entry, dict):
            continue
        epoch = _status_epoch(entry, "last_run_epoch")
        token = _status_time_token(entry, "last_run_utc")
        if epoch > latest_epoch and token:
            latest_epoch = epoch
            latest_token = token
    return latest_token


def _stage_epochs(
    payload: dict[str, object],
    workflow_payload: dict[str, object] | None = None,
) -> list[tuple[str, float]]:
    """Return ordered public workflow stages with their latest epochs."""

    return [
        (
            "open",
            max(
                _status_epoch(payload, "pre_commit_open_epoch"),
                _anchor_epoch(workflow_payload, "open"),
            ),
        ),
        ("verify", _anchor_epoch(workflow_payload, "verify")),
        (
            "run",
            max(
                _status_epoch(payload, "last_run_epoch"),
                _runs_epoch(workflow_payload),
            ),
        ),
        (
            "close",
            max(
                _status_epoch(payload, "pre_commit_close_epoch"),
                _anchor_epoch(workflow_payload, "close"),
            ),
        ),
    ]


def _latest_workflow_run_utc(
    payload: dict[str, object],
    workflow_payload: dict[str, object] | None = None,
) -> str:
    """Return the latest workflow-run UTC token across both ledgers."""

    gate_epoch = _status_epoch(payload, "last_run_epoch")
    gate_token = _status_time_token(payload, "last_run_utc")
    session_epoch = _runs_epoch(workflow_payload)
    session_token = _runs_last_run_utc(workflow_payload)
    if session_epoch > gate_epoch and session_token:
        return session_token
    return gate_token


def _status_epoch(payload: dict[str, object], key: str) -> float:
    """Return one epoch-like numeric field from status payload or `0.0`."""
    try:
        value = float(payload.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0
    if value < 0.0:
        return 0.0
    return value


def _status_time_token(payload: dict[str, object], key: str) -> str:
    """Return one trimmed time field from status payload when present."""
    return str(payload.get(key, "")).strip()


def _resolve_latest_relevant_run_pointer(
    repo_root: Path,
) -> dict[str, str] | None:
    """Return the latest relevant run pointer for status output."""
    pointer_payload = _resolve_latest_pointer_impl(repo_root)
    if not pointer_payload:
        return None
    active_context = execution_runtime_module.get_active_run_log_context()
    active_run_id = active_context.run_id if active_context is not None else ""
    pointer_run_id = str(pointer_payload.get("run_id", "")).strip()
    if pointer_run_id and pointer_run_id != active_run_id:
        return pointer_payload
    return None


def _resolve_latest_pointer_impl(repo_root: Path) -> dict[str, str]:
    """Load and normalize the latest-run pointer payload for status use."""
    run_logging = execution_runtime_module.run_logging_runtime_module
    pointer_path = run_logging.latest_run_pointer_path(repo_root)
    return _normalize_latest_pointer_payload(_load_json_mapping(pointer_path))


def _normalize_latest_pointer_payload(
    payload: dict[str, object],
) -> dict[str, str]:
    """Normalize `latest.json` payload fields used by status output."""
    return {
        "run_id": str(payload.get("run_id", "")).strip(),
        "command_name": str(payload.get("command_name", "")).strip(),
        "status": str(payload.get("status", "")).strip(),
        "run_dir": str(payload.get("run_dir", "")).strip(),
        "summary_txt": str(payload.get("summary_txt", "")).strip(),
        "summary_json": str(payload.get("summary_json", "")).strip(),
    }


def _latest_pointer_summary_line(
    pointer: dict[str, str] | None,
) -> str:
    """Render one short latest-run line for `gate --status` output."""
    if not pointer:
        return ""
    run_dir = str(pointer.get("run_dir", "")).strip()
    summary_txt = str(pointer.get("summary_txt", "")).strip()
    command_name = str(pointer.get("command_name", "")).strip()
    status = str(pointer.get("status", "")).strip()
    if not run_dir:
        return ""
    suffix_parts = []
    if command_name:
        suffix_parts.append(f"command: {command_name}")
    if status:
        suffix_parts.append(f"status: {status}")
    if summary_txt:
        suffix_parts.append(f"summary: {summary_txt}")
    suffix = f" ({', '.join(suffix_parts)})" if suffix_parts else ""
    return f"Latest Relevant Logs: {run_dir}{suffix}"


def _load_json_mapping(path: Path) -> dict[str, object]:
    """Read one JSON file into a mapping, returning empty mapping on errors."""
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _repo_relative(repo_root: Path, path: Path) -> str:
    """Return repo-relative path text when possible."""
    return display_path(path, repo_root=repo_root)


_DATE_ENTRY_PATTERN = re.compile(r"^\s*-\s*\d{4}-\d{2}-\d{2}\b")
_MANAGED_BEGIN = "<!-- DEVCOV:BEGIN -->"
_MANAGED_END = "<!-- DEVCOV:END -->"
_LOG_MARKER = "## Log changes here"


def _visible_changelog_lines(changelog_text: str) -> list[str]:
    """Return changelog lines outside managed blocks and fenced examples."""
    start = changelog_text.find(_LOG_MARKER)
    content = changelog_text[start:] if start >= 0 else changelog_text
    visible: list[str] = []
    in_managed = False
    in_fence = False
    for line in content.splitlines():
        stripped = line.strip()
        if stripped == _MANAGED_BEGIN:
            in_managed = True
            continue
        if stripped == _MANAGED_END:
            in_managed = False
            continue
        if in_managed:
            continue
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        visible.append(line)
    return visible


def _latest_changelog_entry(repo_root: Path) -> str:
    """Return the topmost changelog entry from the latest version section."""
    changelog_path = repo_root / _resolve_main_changelog(repo_root)
    if not changelog_path.exists():
        return ""
    lines = _visible_changelog_lines(
        changelog_path.read_text(encoding="utf-8")
    )
    release_headings = _resolve_release_headings(repo_root)

    version_start: int | None = None
    for index, line in enumerate(lines):
        if _line_matches_release_heading(line, release_headings):
            version_start = index
            break
    if version_start is None:
        return ""

    entry_start: int | None = None
    for index in range(version_start + 1, len(lines)):
        if _DATE_ENTRY_PATTERN.match(lines[index]):
            entry_start = index
            break
    if entry_start is None:
        return ""

    entry_end = len(lines)
    for index in range(entry_start + 1, len(lines)):
        if _DATE_ENTRY_PATTERN.match(
            lines[index]
        ) or _line_matches_release_heading(lines[index], release_headings):
            entry_end = index
            break

    return "\n".join(lines[entry_start:entry_end]).strip()


def _latest_changelog_version(repo_root: Path) -> str:
    """Return the topmost changelog version label from the latest section."""
    changelog_path = repo_root / _resolve_main_changelog(repo_root)
    if not changelog_path.exists():
        return ""
    lines = _visible_changelog_lines(
        changelog_path.read_text(encoding="utf-8")
    )
    release_headings = _resolve_release_headings(repo_root)
    for line in lines:
        stripped = line.strip()
        for heading in release_headings:
            if stripped.startswith(heading):
                return stripped[len(heading) :].strip()
    return ""


def _resolve_main_changelog(repo_root: Path) -> Path:
    """Resolve main changelog path from changelog-coverage metadata."""
    metadata = _load_changelog_metadata(repo_root)
    raw_target = metadata.get("main_changelog", "")
    if isinstance(raw_target, list):
        target = ""
        for entry in raw_target:
            token = str(entry).strip()
            if token:
                target = token
                break
    else:
        target = str(raw_target).strip()
    if not target:
        raise ValueError(
            (
                "`changelog-coverage.main_changelog` is missing in "
                "policy metadata."
            )
        )
    return Path(target)


def _load_changelog_metadata(repo_root: Path) -> dict[str, object]:
    """Return changelog-coverage metadata mapping from policy registry."""
    registry_path = tracked_registry_module.policy_registry_path(repo_root)
    rendered = display_path(registry_path, repo_root=repo_root)
    if not registry_path.exists():
        raise ValueError(
            f"Missing policy registry file: {rendered}. "
            "Run `devcovenant refresh`."
        )
    try:
        payload = yaml_cache_service.load_yaml(registry_path)
    except yaml.YAMLError as exc:
        raise ValueError(
            f"Invalid YAML in policy registry {rendered}: {exc}"
        ) from exc
    except OSError as exc:
        raise ValueError(
            f"Unable to read policy registry {rendered}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(
            f"Invalid policy registry payload in {rendered}: "
            "expected a mapping."
        )
    policies = payload.get("policies", {})
    if not isinstance(policies, dict):
        raise ValueError(
            f"Invalid policy registry payload in {rendered}: "
            "`policies` must be a mapping."
        )
    changelog_coverage = policies.get("changelog-coverage", {})
    if not isinstance(changelog_coverage, dict):
        raise ValueError(
            "Missing `changelog-coverage` policy entry in policy registry."
        )
    metadata = changelog_coverage.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError(
            "Invalid `changelog-coverage.metadata` payload in policy registry."
        )
    return metadata


def _normalize_list_option(
    value: object,
    default: list[str],
) -> list[str]:
    """Normalize metadata value into non-empty string list."""
    if value is None:
        source: list[str] = default
    elif isinstance(value, str):
        source = [entry.strip() for entry in value.split(",") if entry.strip()]
    elif isinstance(value, list):
        source = [str(entry).strip() for entry in value if str(entry).strip()]
    else:
        source = [str(value).strip()]
    normalized = [entry for entry in source if entry]
    return normalized or default


def _resolve_doc_exemption_options(
    repo_root: Path,
) -> tuple[list[str], list[str], int]:
    """Resolve doc allowlist metadata from changelog-coverage descriptor."""
    metadata = _load_changelog_metadata(repo_root)
    suffixes = _normalize_list_option(
        metadata.get("header_doc_suffixes"),
        [".md", ".rst", ".txt"],
    )
    header_keys = _normalize_list_option(
        metadata.get("header_keys"),
        ["Last Updated", "Project Version", "DevCovenant Version"],
    )
    raw_scan = metadata.get("header_scan_lines", 4)
    try:
        scan_lines = int(raw_scan)
    except (TypeError, ValueError):
        scan_lines = 4
    if scan_lines < 0:
        scan_lines = 0
    return suffixes, header_keys, scan_lines


def _resolve_release_headings(repo_root: Path) -> list[str]:
    """Return release-section headings active for this repository."""
    return project_governance_service.resolve_release_headings(repo_root)


def _line_matches_release_heading(
    line: str,
    headings: list[str],
) -> bool:
    """Return True when one changelog heading matches active release heads."""
    stripped = line.strip()
    return any(stripped.startswith(heading) for heading in headings)


def _entry_fingerprint(entry_text: str) -> str:
    """Return a deterministic hash for an entry block."""
    if not entry_text.strip():
        return ""
    normalized = "\n".join(
        line.rstrip() for line in entry_text.strip().splitlines()
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


SCHEMA_VERSION = 1
_RUN_SNAPSHOTS_KEY = "workflow_run_snapshots"


def _normalize_commands(raw_value: object) -> list[str]:
    """Normalize workflow entry commands into a trimmed ordered list."""

    if isinstance(raw_value, list):
        values = raw_value
    elif isinstance(raw_value, str):
        values = [part.strip() for part in raw_value.split("&&")]
    else:
        values = []
    commands: list[str] = []
    for entry in values:
        token = str(entry or "").strip()
        if token and token not in commands:
            commands.append(token)
    return commands


def _normalize_entry_payload(entry_raw: object) -> dict[str, object]:
    """Normalize one anchor/run payload into the current session shape."""

    entry = dict(entry_raw) if isinstance(entry_raw, Mapping) else {}
    last_run_utc = str(entry.get("last_run_utc") or "").strip()
    if last_run_utc:
        entry["last_run_utc"] = last_run_utc
    else:
        entry.pop("last_run_utc", None)
    entry.pop("last_run", None)
    commands = _normalize_commands(entry.get("commands"))
    if commands:
        entry["commands"] = commands
    else:
        entry.pop("commands", None)
    entry.pop("command", None)
    return entry


def _normalize_entry_mapping(raw_entries: object) -> dict[str, object]:
    """Normalize stored anchor/run entry mappings."""

    if not isinstance(raw_entries, dict):
        return {}
    normalized: dict[str, object] = {}
    for key, value in raw_entries.items():
        token = str(key or "").strip()
        if not token:
            continue
        normalized[token] = _normalize_entry_payload(value)
    return normalized


def workflow_session_path(repo_root: Path) -> Path:
    """Return the runtime workflow-session path for a repository."""

    return registry_runtime.workflow_session_path(repo_root)


def _base_payload() -> dict[str, object]:
    """Return an empty workflow-session payload."""

    return {
        "schema_version": SCHEMA_VERSION,
        "session_id": "",
        "session_state": "",
        "anchors": {},
        "runs": {},
        "run_ids": [],
    }


def load_workflow_session(repo_root: Path) -> dict[str, object]:
    """Load workflow-session payload, defaulting to an empty structure."""

    path = workflow_session_path(repo_root)
    if not path.exists():
        return _base_payload()
    rendered = display_path(path, repo_root=repo_root)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid workflow session JSON in {rendered}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(
            f"Workflow session payload must be a mapping: {rendered}"
        )
    normalized = _base_payload()
    normalized.update(payload)
    normalized.pop("required_run_ids", None)
    anchors = payload.get("anchors")
    normalized["anchors"] = _normalize_entry_mapping(anchors)
    runs = payload.get("runs")
    normalized["runs"] = _normalize_entry_mapping(runs)
    run_ids = payload.get("run_ids")
    normalized["run_ids"] = list(run_ids) if isinstance(run_ids, list) else []
    return normalized


def write_workflow_session(
    repo_root: Path,
    payload: Mapping[str, object],
) -> Path:
    """Persist workflow-session payload to the runtime registry."""

    path = workflow_session_path(repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = _base_payload()
    normalized.update(dict(payload))
    normalized.pop("required_run_ids", None)
    normalized["anchors"] = _normalize_entry_mapping(normalized.get("anchors"))
    normalized["runs"] = _normalize_entry_mapping(normalized.get("runs"))
    path.write_text(
        json.dumps(normalized, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def resolve_run_snapshot(
    repo_root: Path,
    payload: Mapping[str, object],
    run_id: str,
) -> dict[str, str] | None:
    """Return the stored verification snapshot for one workflow run."""

    snapshot_payload = load_session_snapshot_payload(
        repo_root,
        payload,
    )
    raw_snapshots = snapshot_payload.get(_RUN_SNAPSHOTS_KEY)
    if not isinstance(raw_snapshots, dict):
        return None
    raw_snapshot = raw_snapshots.get(str(run_id or "").strip())
    if not isinstance(raw_snapshot, dict):
        return None
    return normalize_snapshot_rows(
        raw_snapshot,
        field_name=f"{_RUN_SNAPSHOTS_KEY}.{run_id}",
    )


def merge_run_snapshot(
    repo_root: Path,
    payload: Mapping[str, object],
    run_id: str,
    snapshot: Mapping[str, str],
) -> tuple[str, dict[str, object]]:
    """Merge one run snapshot into the shared session-snapshot file."""

    snapshot_payload = load_session_snapshot_payload(
        repo_root,
        payload,
    )
    run_snapshots = snapshot_payload.get(_RUN_SNAPSHOTS_KEY)
    normalized_snapshots = (
        dict(run_snapshots) if isinstance(run_snapshots, dict) else {}
    )
    normalized_snapshots[str(run_id or "").strip()] = dict(snapshot)
    return merge_session_snapshot_payload(
        repo_root,
        dict(payload),
        updates={_RUN_SNAPSHOTS_KEY: normalized_snapshots},
    )


_SNAPSHOT_BASE_IGNORED_DIRS = frozenset(
    {
        ".git",
        ".venv",
        ".python",
        ".gha-pycache",
        "output",
        "logs",
        "build",
        "dist",
        "node_modules",
        "__pycache__",
        ".cache",
        ".ruff_cache",
        ".pytest_cache",
        ".mypy_cache",
        ".tox",
        ".nox",
        ".hypothesis",
        ".venv.lock",
    }
)

_SNAPSHOT_IGNORED_FILES = frozenset(
    {
        "devcovenant/registry/runtime/gate_status.json",
        "devcovenant/registry/runtime/latest.json",
        "devcovenant/registry/runtime/session_snapshot.json",
    }
)

_SNAPSHOT_IGNORED_PREFIXES = ("devcovenant/registry/runtime/",)
_SNAPSHOT_IGNORED_SUFFIXES = frozenset({".pyc", ".pyo", ".pyd"})
_AGENTS_WORKFLOW_BEGIN = "<!-- DEVCOV-WORKFLOW:BEGIN -->"
_AGENTS_WORKFLOW_END = "<!-- DEVCOV-WORKFLOW:END -->"
SESSION_SNAPSHOT_POINTER_KEY = "session_snapshot_file"
SESSION_SNAPSHOT_UPDATED_UTC_KEY = "session_snapshot_updated_utc"
SESSION_SNAPSHOT_UPDATED_EPOCH_KEY = "session_snapshot_updated_epoch"
SESSION_SNAPSHOT_BULKY_KEYS = (
    "document_exemption_baseline",
    "last_run_snapshot",
    "session_baseline_snapshot",
    "session_close_snapshot",
    "session_open_snapshot",
    "run_events",
)


def _normalize_snapshot_payload(
    payload_raw: object,
) -> dict[str, object]:
    """Normalize snapshot payload keys."""

    return dict(payload_raw) if isinstance(payload_raw, dict) else {}


def capture_current_numstat_snapshot(repo_root: Path) -> dict[str, str]:
    """
    Return a filesystem snapshot mapping keyed by relative path.

    Snapshot rows are deterministic `sha256<TAB>path` strings. This avoids
    HEAD/working-tree diff logic and lets session policies compare one baseline
    snapshot against the current filesystem state directly.
    """
    rows: dict[str, str] = {}
    ignored_dirs = _snapshot_ignored_dirs(repo_root)
    files = _snapshot_files(repo_root, ignored_dirs)
    for file_path in files:
        rel = file_path.relative_to(repo_root).as_posix()
        if rel in _SNAPSHOT_IGNORED_FILES:
            continue
        if any(
            rel == prefix.rstrip("/") or rel.startswith(prefix)
            for prefix in _SNAPSHOT_IGNORED_PREFIXES
        ):
            continue
        digest = _sha256_file(file_path)
        rows[rel] = f"{digest}\t{rel}"
    return rows


def default_session_snapshot_relative_path(repo_root: Path) -> str:
    """Return the canonical repo-relative session snapshot path."""
    return (
        registry_runtime_module.session_snapshot_path(repo_root)
        .relative_to(repo_root)
        .as_posix()
    )


def resolve_session_snapshot_path(
    repo_root: Path,
    gate_status: Mapping[str, object] | None = None,
    *,
    require_pointer: bool = False,
) -> Path:
    """Resolve the companion session snapshot path from gate status."""
    raw_pointer = str(
        (gate_status or {}).get(SESSION_SNAPSHOT_POINTER_KEY, "")
    ).strip()
    if not raw_pointer:
        if require_pointer:
            raise ValueError(
                "Invalid gate status payload: "
                "`session_snapshot_file` is required for session checks."
            )
        return registry_runtime_module.session_snapshot_path(repo_root)
    pointer = Path(raw_pointer)
    if pointer.is_absolute() or ".." in pointer.parts:
        raise ValueError(
            "Invalid gate status payload: `session_snapshot_file` must be "
            "a repo-relative path inside devcovenant/registry/runtime/."
        )
    return repo_root / pointer


def load_session_snapshot_payload(
    repo_root: Path,
    gate_status: Mapping[str, object] | None,
    *,
    require: bool = False,
) -> dict[str, object]:
    """Load the companion session snapshot payload for one gate status."""
    path = resolve_session_snapshot_path(
        repo_root,
        gate_status,
        require_pointer=require,
    )
    if not path.exists():
        if require:
            rel = display_path(path, repo_root=repo_root)
            raise ValueError(f"Session snapshot file is missing: {rel}.")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            "Invalid session snapshot JSON in "
            f"{display_path(path, repo_root=repo_root)}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(
            "Session snapshot payload must be a mapping: "
            f"{display_path(path, repo_root=repo_root)}"
        )
    return _normalize_snapshot_payload(payload)


def merge_session_snapshot_payload(
    repo_root: Path,
    gate_status: Mapping[str, object] | None,
    *,
    updates: Mapping[str, object] | None = None,
    remove_keys: Sequence[str] = (),
) -> tuple[str, dict[str, object]]:
    """Merge updates into the companion snapshot payload and write it."""
    path = resolve_session_snapshot_path(repo_root, gate_status)
    payload = load_session_snapshot_payload(repo_root, gate_status)
    payload = _normalize_snapshot_payload(payload)
    for key in remove_keys:
        payload.pop(str(key), None)
    for key, value in dict(updates or {}).items():
        payload[str(key)] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path.relative_to(repo_root).as_posix(), payload


def prune_inline_session_snapshot_fields(
    gate_status: dict[str, object],
) -> None:
    """Remove bulky snapshot/session fields from gate status payloads."""
    for key in SESSION_SNAPSHOT_BULKY_KEYS:
        gate_status.pop(key, None)


def capture_current_snapshot_paths(repo_root: Path) -> list[str]:
    """Return deterministic repo-relative path list from filesystem scan."""
    ignored_dirs = _snapshot_ignored_dirs(repo_root)
    files = _snapshot_files(repo_root, ignored_dirs)
    return [path.relative_to(repo_root).as_posix() for path in files]


def changed_numstat_paths(
    before: dict[str, str], after: dict[str, str]
) -> set[str]:
    """Return changed paths present in the current snapshot."""
    changed: set[str] = set()
    for path, row in after.items():
        if before.get(path) != row:
            changed.add(path)
    return changed


def diff_snapshot_paths(
    before: dict[str, str], after: dict[str, str]
) -> set[str]:
    """
    Return changed paths across both snapshots, including deletions.

    This helper is used for gate/session drift detection where deletions must
    be treated as real changes, not silently ignored.
    """
    changed: set[str] = set()
    for path in set(before).union(after):
        if before.get(path) != after.get(path):
            changed.add(path)
    return changed


def snapshot_signature(snapshot: dict[str, str]) -> str:
    """
    Return a deterministic signature for one normalized snapshot mapping.

    This is the canonical session-signature API for gate/runtime checks.
    """
    rows = [snapshot[path] for path in sorted(snapshot)]
    payload = "\n".join(rows)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_snapshot_rows(
    raw: object, *, field_name: str = "snapshot"
) -> dict[str, str]:
    """Validate and normalize snapshot payload mappings into strings."""
    if not isinstance(raw, dict):
        raise ValueError(
            "Invalid gate status payload: "
            f"`{field_name}` must be a mapping."
        )
    snapshot: dict[str, str] = {}
    for key, value in raw.items():
        path = str(key).strip()
        row = str(value).strip()
        if not path or not row:
            raise ValueError(
                "Invalid gate status payload: "
                f"`{field_name}` contains empty keys or rows."
            )
        snapshot[path] = row
    return snapshot


def snapshot_row_style(snapshot: dict[str, str]) -> str:
    """Classify snapshot row style for current-format validation."""
    if not snapshot:
        return "empty"
    tab_counts: list[int] = []
    for row in snapshot.values():
        text = str(row).strip()
        if not text:
            continue
        tab_counts.append(text.count("\t"))
    if not tab_counts:
        return "empty"
    if all(count >= 2 for count in tab_counts):
        return "unsupported_legacy"
    if all(count == 1 for count in tab_counts):
        return "filesystem_hash"
    return "mixed"


def session_delta_paths(
    repo_root: Path,
    open_snapshot: dict[str, str],
    current_snapshot: dict[str, str],
    *,
    session_open_epoch: float | None = None,
) -> set[str]:
    """
    Return session delta paths using shared snapshot comparison semantics.
    """
    open_style = snapshot_row_style(open_snapshot)
    if open_style == "unsupported_legacy":
        raise ValueError(
            "Invalid gate status payload: legacy snapshot rows are no longer "
            "supported. Run `devcovenant gate --open` to record a fresh "
            "session with the current snapshot format."
        )
    if open_style == "mixed":
        raise ValueError(
            "Invalid gate status payload: mixed snapshot row formats are not "
            "supported. Run `devcovenant gate --open` to record a fresh "
            "session with the current snapshot format."
        )
    current_style = snapshot_row_style(current_snapshot)
    if current_style in {"unsupported_legacy", "mixed"}:
        raise ValueError(
            "Invalid current snapshot state: unsupported snapshot row format "
            "encountered during session comparison."
        )
    return changed_numstat_paths(open_snapshot, current_snapshot)


def snapshot_paths_changed_since(repo_root: Path, epoch: float) -> set[str]:
    """Return snapshot paths whose mtime is after the given epoch."""
    if epoch < 0:
        raise ValueError("Snapshot epoch must be non-negative.")
    # Gate/session epochs are persisted with datetime microsecond precision.
    # Compare at the same precision to avoid boundary false-positives caused by
    # float representation drift when values are reloaded from JSON.
    cutoff_micros = int(round(float(epoch) * 1_000_000))
    ignored_dirs = _snapshot_ignored_dirs(repo_root)
    files = _snapshot_files(repo_root, ignored_dirs)
    changed: set[str] = set()
    for file_path in files:
        rel = file_path.relative_to(repo_root).as_posix()
        if rel in _SNAPSHOT_IGNORED_FILES:
            continue
        try:
            mtime_micros = file_path.stat().st_mtime_ns // 1000
        except OSError as exc:
            raise ValueError(
                f"Unable to stat snapshot file {file_path}: {exc}"
            ) from exc
        if mtime_micros > cutoff_micros:
            changed.add(rel)
    return changed


def _snapshot_ignored_dirs(repo_root: Path) -> set[str]:
    """Return snapshot ignored directories from config and active profiles."""
    ignored = set(_SNAPSHOT_BASE_IGNORED_DIRS)
    config_path = repo_root / "devcovenant" / "config.yaml"
    if not config_path.exists():
        return ignored
    rendered_config = display_path(config_path, repo_root=repo_root)
    try:
        payload = yaml_cache_service.load_yaml(config_path)
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(
            f"Unable to read snapshot ignore settings from {rendered_config}: "
            f"{exc}"
        ) from exc
    if not isinstance(payload, dict):
        return ignored
    engine_cfg = payload.get("engine", {})
    if not isinstance(engine_cfg, dict):
        engine_cfg = {}
    extra = engine_cfg.get("ignore_dirs", [])
    if isinstance(extra, str):
        extra_dirs = [extra]
    elif isinstance(extra, list):
        extra_dirs = extra
    else:
        extra_dirs = []
    for entry in extra_dirs:
        name = str(entry).strip()
        if name:
            ignored.add(name)
    active_profiles = profile_registry_service.parse_active_profiles(
        payload,
        include_global=True,
    )
    profile_registry = profile_registry_service.load_profile_registry(
        repo_root
    )
    profile_ignored_dirs = (
        profile_registry_service.resolve_profile_ignore_dirs(
            profile_registry,
            active_profiles,
        )
    )
    for entry in profile_ignored_dirs:
        name = str(entry).strip()
        if name:
            ignored.add(name)
    return ignored


def _snapshot_files(repo_root: Path, ignored_dirs: set[str]) -> list[Path]:
    """Collect snapshot files under the repository root using ignore-dir
    filtering."""
    files: list[Path] = []
    for root, dirs, names in os.walk(repo_root):
        root_path = Path(root)
        dirs[:] = [name for name in dirs if name not in ignored_dirs]
        for name in names:
            file_path = root_path / name
            try:
                rel = file_path.relative_to(repo_root).as_posix()
            except ValueError:
                continue
            if rel in _SNAPSHOT_IGNORED_FILES:
                continue
            if any(
                rel == prefix.rstrip("/") or rel.startswith(prefix)
                for prefix in _SNAPSHOT_IGNORED_PREFIXES
            ):
                continue
            if any(part in ignored_dirs for part in file_path.parts):
                continue
            if file_path.suffix in _SNAPSHOT_IGNORED_SUFFIXES:
                continue
            if not file_path.is_file():
                continue
            files.append(file_path)
    files.sort(key=lambda path: path.relative_to(repo_root).as_posix())
    return files


def _sha256_file(path: Path) -> str:
    """Return SHA-256 digest for one file path."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as file_obj:
            while True:
                chunk = file_obj.read(65536)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as exc:
        raise ValueError(
            f"Unable to read snapshot file {display_path(path)}: {exc}"
        ) from exc
    return digest.hexdigest()


def _hash_lines(lines: list[str]) -> str:
    """Return deterministic SHA-256 digest for normalized text lines."""
    normalized = "\n".join(line.rstrip() for line in lines)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _split_agents_workflow_lines(content: str) -> tuple[list[str], list[str]]:
    """Split AGENTS text into workflow-block lines and non-workflow lines."""
    workflow_lines: list[str] = []
    non_workflow_lines: list[str] = []
    in_workflow = False
    for line in content.splitlines():
        stripped = line.strip()
        if stripped == _AGENTS_WORKFLOW_BEGIN:
            in_workflow = True
            workflow_lines.append(line)
            continue
        if stripped == _AGENTS_WORKFLOW_END:
            workflow_lines.append(line)
            in_workflow = False
            continue
        if in_workflow:
            workflow_lines.append(line)
        else:
            non_workflow_lines.append(line)
    return workflow_lines, non_workflow_lines


def capture_agents_section_hashes(repo_root: Path) -> dict[str, str]:
    """Capture deterministic AGENTS full/workflow/non-workflow hashes."""
    payload = {
        "agents_file": "AGENTS.md",
        "agents_full_sha256": "",
        "agents_workflow_sha256": "",
        "agents_non_workflow_sha256": "",
    }
    agents_path = repo_root / "AGENTS.md"
    if not agents_path.exists() or not agents_path.is_file():
        return payload
    try:
        content = agents_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return payload

    payload["agents_full_sha256"] = hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()
    workflow_lines, non_workflow_lines = _split_agents_workflow_lines(content)
    payload["agents_workflow_sha256"] = _hash_lines(workflow_lines)
    payload["agents_non_workflow_sha256"] = _hash_lines(non_workflow_lines)
    return payload


def capture_document_exemption_baseline(
    repo_root: Path,
    *,
    header_doc_suffixes: list[str],
    header_keys: list[str],
    header_scan_lines: int,
) -> dict[str, dict[str, str]]:
    """Capture a baseline for header exemptions and managed regenerations."""
    suffixes = {
        entry.strip().lower() for entry in header_doc_suffixes if entry
    }
    keys = {entry.strip().lower() for entry in header_keys if entry}
    scan_lines = max(int(header_scan_lines), 0)

    ignored_dirs = _snapshot_ignored_dirs(repo_root)
    baseline: dict[str, dict[str, str]] = {}
    for path in _snapshot_files(repo_root, ignored_dirs):
        rel = path.relative_to(repo_root).as_posix()
        entry = document_exemption_fingerprint_for_path(
            repo_root,
            rel,
            header_doc_suffixes=suffixes,
            header_keys=keys,
            header_scan_lines=scan_lines,
        )
        if entry is not None:
            is_header_doc = path.suffix.lower() in suffixes
            has_managed_markers = (
                entry.get("managed_marker_signature", "")
                != _EMPTY_MANAGED_MARKER_SIGNATURE
            )
            if not is_header_doc and not has_managed_markers:
                continue
            baseline[rel] = entry
    return baseline


__all__ = [
    "capture_current_numstat_snapshot",
    "capture_current_snapshot_paths",
    "session_delta_paths",
    "snapshot_paths_changed_since",
    "changed_numstat_paths",
    "snapshot_signature",
    "normalize_snapshot_rows",
    "snapshot_row_style",
]


runtime_print = execution_runtime_module.runtime_print
_CHECK_APPLY_FIXES_ENV = "DEVCOV_CHECK_APPLY_FIXES"
_CHECK_RUN_REFRESH_ENV = "DEVCOV_CHECK_RUN_REFRESH"
_CHECK_CLEAN_BYTECODE_ENV = "DEVCOV_CHECK_CLEAN_BYTECODE"
_DEFAULT_PRE_COMMIT_COMMAND = (
    workflow_contract_module.DEFAULT_PRE_COMMIT_COMMAND
)
_PRE_COMMIT_EXECUTABLE_TOKENS = frozenset(
    {"pre-commit", "pre-commit.exe", "pre_commit", "pre_commit.exe"}
)


def _utc_now() -> _dt.datetime:
    """Return the current UTC time."""
    return _dt.datetime.now(tz=_dt.timezone.utc)


def show_gate_status(repo_root: Path) -> int:
    """Print a short, read-only gate status summary."""
    for line in _gate_status_summary_lines(repo_root):
        runtime_print(line)
    return 0


def _is_pre_commit_run_command(tokens: list[str]) -> bool:
    """Return True when tokens describe a `pre-commit run` invocation."""
    if not tokens or "run" not in tokens:
        return False
    first = Path(tokens[0]).name.lower()
    if first in _PRE_COMMIT_EXECUTABLE_TOKENS:
        return True
    for index, token in enumerate(tokens[:-1]):
        if token == "-m" and tokens[index + 1] == "pre_commit":
            return True
    return False


def _resolve_hook_command(repo_root: Path, command: str) -> str:
    """Resolve the effective pre-commit command for gate hook execution."""
    tokens = shlex.split(command)
    if not tokens:
        raise SystemExit("Pre-commit command is empty.")
    if "--all-files" not in tokens:
        return command
    if "--files" in tokens:
        return command
    if not _is_pre_commit_run_command(tokens):
        return command
    snapshot_paths = execution_runtime_module.capture_current_snapshot_paths(
        repo_root
    )
    if not snapshot_paths:
        return command
    resolved: list[str] = []
    replaced = False
    for token in tokens:
        if token == "--all-files" and not replaced:
            resolved.append("--files")
            resolved.extend(snapshot_paths)
            replaced = True
            continue
        if token == "--all-files":
            continue
        resolved.append(token)
    if not replaced:
        return command
    return shlex.join(resolved)


def _resolve_gate_execution_command(
    command: str,
    *,
    env: Mapping[str, str],
    managed_python: str | None,
) -> str:
    """Resolve managed pre-commit execution through the managed interpreter."""
    tokens = shlex.split(command)
    if not tokens:
        raise SystemExit("Pre-commit command is empty.")
    if managed_python is None or not _is_pre_commit_run_command(tokens):
        return command
    executable = tokens[0]
    first_name = Path(executable).name.lower()
    if first_name not in _PRE_COMMIT_EXECUTABLE_TOKENS:
        path_value = str(env.get("PATH", "")).strip() or None
        if shutil.which(executable, path=path_value) is not None:
            return command
        return command
    resolved_tokens = [managed_python, "-m", "pre_commit", *tokens[1:]]
    return shlex.join(resolved_tokens)


_TEST_IRRELEVANT_FILES = {"changelog.md"}
_DEVCOV_POLICY_HOOK_TOKEN = "enforce repository policies (DevCovenant)"
_DEVCOV_BLOCKING_MARKERS = (
    "Status: 🚫 BLOCKED",
    "Status: BLOCKED",
    "critical violations must be fixed",
    "violations >= error threshold",
)
_HOOK_MODIFIED_FILES_MARKER = "files were modified by this hook"
_SUPPRESSED_FAILURE_TAIL_MAX_LINES = 40
_SUPPRESSED_FAILURE_TAIL_MAX_CHARS = 6000
_START_GATE_DRIFT_PATH_LIMIT = 12


def _emit_suppressed_failure_tail(command_output: str) -> None:
    """Emit a bounded tail when normal mode suppresses gate child output."""
    output = str(command_output or "").strip()
    if not output:
        return
    lines = output.splitlines()
    tail_lines = lines[-_SUPPRESSED_FAILURE_TAIL_MAX_LINES:]
    tail_text = "\n".join(tail_lines)
    if len(tail_text) > _SUPPRESSED_FAILURE_TAIL_MAX_CHARS:
        tail_text = tail_text[-_SUPPRESSED_FAILURE_TAIL_MAX_CHARS:]
    runtime_print(
        "Pre-commit output tail " "(normal mode child output suppressed):",
        file=sys.stderr,
    )
    for line in tail_text.splitlines():
        runtime_print(line, file=sys.stderr)


def _format_changed_path_tail(changed_paths: set[str]) -> str:
    """Render a bounded changed-path summary for gate failure messages."""
    ordered = sorted(path for path in changed_paths if str(path).strip())
    if not ordered:
        return ""
    visible = ordered[:_START_GATE_DRIFT_PATH_LIMIT]
    rendered = ", ".join(visible)
    hidden_count = len(ordered) - len(visible)
    if hidden_count > 0:
        rendered += f", ... (+{hidden_count} more)"
    return rendered


def _is_devcov_hook_modified_failure(command_output: str) -> bool:
    """Return whether DevCovenant changed files during pre-commit."""
    output = str(command_output or "").strip()
    if not output:
        return False
    return (
        _DEVCOV_POLICY_HOOK_TOKEN in output
        and _HOOK_MODIFIED_FILES_MARKER in output
    )


def _emit_open_gate_drift_failure(
    command: str,
    *,
    exit_code: int,
    command_output: str,
    changed_paths: set[str],
) -> None:
    """Explain why open gate rejected hook-induced baseline drift."""
    runtime_print(
        "Open gate detected hook-induced baseline drift and did not "
        "record a usable session.",
        file=sys.stderr,
    )
    rendered_paths = _format_changed_path_tail(changed_paths)
    if rendered_paths:
        runtime_print(
            f"Hook-changed paths: {rendered_paths}",
            file=sys.stderr,
        )
    if _is_devcov_hook_modified_failure(command_output):
        runtime_print(
            "The DevCovenant hook refreshed managed files during "
            "`devcovenant gate --open`. Settle those managed updates "
            "first, then rerun `devcovenant gate --open`.",
            file=sys.stderr,
        )
        return
    rendered = " ".join(shlex.split(command)) or command
    if exit_code != 0:
        runtime_print(
            "Pre-commit command failed with exit code "
            f"{exit_code}: {rendered}",
            file=sys.stderr,
        )
    runtime_print(
        "Clear the hook-induced edits and rerun " "`devcovenant gate --open`.",
        file=sys.stderr,
    )


def _restore_status_file(path: Path, previous_bytes: bytes | None) -> None:
    """Restore gate status file from prior bytes, or remove when absent."""
    if previous_bytes is None:
        if path.exists():
            path.unlink()
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(previous_bytes)


def _run_command(
    command: str,
    env: dict[str, str] | None = None,
    *,
    strict: bool = True,
) -> int:
    """Execute a shell command string and optionally fail on non-zero exit."""
    exit_code, _ = _run_command_with_output(command, env=env)
    if strict and exit_code != 0:
        parts = shlex.split(command)
        rendered = " ".join(parts) if parts else command
        runtime_print(
            f"Pre-commit command failed with exit code {exit_code}:"
            f" {rendered}",
            file=sys.stderr,
        )
        raise SystemExit(exit_code)
    return exit_code


def _run_command_with_output(
    command: str,
    env: dict[str, str] | None = None,
) -> tuple[int, str]:
    """Execute a shell command string and return exit code with output."""
    parts = shlex.split(command)
    if not parts:
        raise SystemExit("Pre-commit command is empty.")
    result, combined_output = (
        execution_runtime_module.run_child_command_with_output_policy(
            parts,
            channel="gate_child",
            env=env,
            capture_combined_output=True,
        )
    )
    exit_code = int(result.returncode)
    output_plan = (
        execution_runtime_module.resolve_child_output_plan_for_channel(
            "gate_child"
        )
    )
    if exit_code != 0 and output_plan.child_output_suppressed:
        _emit_suppressed_failure_tail(combined_output)
    return exit_code, combined_output


def _is_blocking_devcov_failure(
    exit_code: int,
    command_output: str,
) -> bool:
    """Return whether command output reflects blocking DevCovenant failures."""
    if exit_code == 0:
        return False
    output = command_output.strip()
    if not output:
        return False
    if _DEVCOV_POLICY_HOOK_TOKEN not in output:
        return False
    return any(marker in output for marker in _DEVCOV_BLOCKING_MARKERS)


def _is_test_relevant_path(path: str) -> bool:
    """Return True when a changed path should trigger test reruns."""
    leaf = path.replace("\\", "/").rsplit("/", 1)[-1].lower()
    return leaf not in _TEST_IRRELEVANT_FILES


def _format_run_rerun_instructions(
    run_ids: list[str],
) -> str:
    """Render the canonical rerun instruction."""
    del run_ids
    return "`devcovenant run`"


def _stale_run_ids(
    repo_root: Path,
    contract: Mapping[str, object],
    workflow_payload: Mapping[str, object],
    current_snapshot: Mapping[str, str],
    *,
    session_id: str,
) -> list[str]:
    """Return configured runs whose latest evidence is missing or stale."""
    runs_raw = workflow_payload.get("runs")
    run_map = dict(runs_raw) if isinstance(runs_raw, dict) else {}
    stale: list[str] = []
    for run_id in workflow_contract_module.run_ids(contract):
        run = workflow_contract_module.resolve_run(contract, run_id)
        if run is None:
            stale.append(run_id)
            continue
        entry = run_map.get(run_id)
        if not isinstance(entry, dict):
            stale.append(run_id)
            continue
        if session_id:
            last_run_session_id = str(
                entry.get("last_run_session_id", "")
            ).strip()
            if last_run_session_id != session_id:
                stale.append(run_id)
                continue
        try:
            run_snapshot = resolve_run_snapshot(
                repo_root,
                workflow_payload,
                run_id,
            )
        except ValueError:
            stale.append(run_id)
            continue
        if not run_snapshot:
            stale.append(run_id)
            continue
        changed_paths = _changed_paths_between(run_snapshot, current_snapshot)
        if workflow_contract_module.run_relevant_paths_changed(
            run,
            sorted(changed_paths),
        ):
            stale.append(run_id)
    return stale


def _record_workflow_anchor(
    repo_root: Path,
    *,
    contract: Mapping[str, object],
    stage: str,
    command: str,
    notes: str,
    when: _dt.datetime,
    session_id: str,
    session_state: str,
    reset_runs: bool = False,
    session_snapshot_file: str = "",
    session_snapshot_updated_utc: str = "",
    session_snapshot_updated_epoch: float = 0.0,
) -> None:
    """Persist workflow-session anchor state for one gate stage."""
    try:
        workflow_payload = load_workflow_session(repo_root)
    except ValueError:
        workflow_payload = {
            "schema_version": SCHEMA_VERSION,
            "session_id": "",
            "session_state": "",
            "anchors": {},
            "runs": {},
            "run_ids": [],
        }
    anchors_raw = workflow_payload.get("anchors")
    anchors = dict(anchors_raw) if isinstance(anchors_raw, dict) else {}
    anchor_entry = dict(anchors.get(stage) or {})
    anchor_entry.update(
        {
            "id": stage,
            "status": "passed",
            "last_run_utc": when.isoformat(),
            "last_run_epoch": when.timestamp(),
            "commands": [command.strip()] if command.strip() else [],
            "command_name": f"gate --{stage}",
            "notes": notes.strip(),
        }
    )
    anchor_entry.pop("last_run", None)
    anchor_entry.pop("command", None)
    anchors[stage] = anchor_entry
    workflow_payload["schema_version"] = SCHEMA_VERSION
    workflow_payload["workflow_contract_schema_version"] = contract.get(
        "schema_version",
        workflow_contract_module.SCHEMA_VERSION,
    )
    workflow_payload["run_ids"] = workflow_contract_module.run_ids(contract)
    workflow_payload["session_id"] = session_id
    workflow_payload["session_state"] = session_state
    workflow_payload["anchors"] = anchors
    if reset_runs:
        workflow_payload["runs"] = {}
    if session_snapshot_file:
        workflow_payload["session_snapshot_file"] = session_snapshot_file
    if session_snapshot_updated_utc:
        workflow_payload["session_snapshot_updated_utc"] = (
            session_snapshot_updated_utc
        )
    if session_snapshot_updated_epoch > 0.0:
        workflow_payload["session_snapshot_updated_epoch"] = (
            session_snapshot_updated_epoch
        )
    write_workflow_session(
        repo_root,
        workflow_payload,
    )


def _current_numstat_snapshot(repo_root: Path) -> dict[str, str]:
    """Return deterministic filesystem-hash snapshot rows keyed by path."""
    return execution_runtime_module.capture_current_numstat_snapshot(repo_root)


def _changed_paths_between(
    before: dict[str, str], after: dict[str, str]
) -> set[str]:
    """Return changed paths across two snapshots, including deletions."""
    return execution_runtime_module.diff_snapshot_paths(before, after)


def run_pre_commit_gate(
    repo_root: Path,
    stage: str,
    *,
    command: str | None = None,
    notes: str = "",
) -> int:
    """Run one gate pre-commit stage (`open`, `verify`, or `close`)."""
    if stage not in {"open", "verify", "close"}:
        raise SystemExit("stage must be 'open', 'verify', or 'close'.")
    is_open = stage == "open"
    is_verify = stage == "verify"
    is_close = stage == "close"
    resolved_command = str(command or "").strip()
    if not resolved_command:
        resolved_command = workflow_contract_module.resolve_pre_commit_command(
            repo_root
        )
    command = resolved_command

    status_path = registry_runtime_module.gate_status_path(repo_root)
    status_path.parent.mkdir(parents=True, exist_ok=True)

    if is_close or is_verify:
        try:
            pre_payload = _load_status(status_path)
        except ValueError as error:
            runtime_print(str(error), file=sys.stderr)
            return 1
        session_id = str(pre_payload.get("session_id", "")).strip()
        session_state = (
            str(pre_payload.get("session_state", "")).strip().lower()
        )
        if not session_id or session_state != "open":
            runtime_print(
                f"Cannot run {stage} gate without an active open session. "
                "Run `devcovenant gate --open` first.",
                file=sys.stderr,
            )
            return 1

    open_ts = _utc_now() if is_open else None
    run_ids_pending: list[str] = []
    recovery_run_ids: list[str] = []
    recovery_status_active = False
    recovery_status_previous: bytes | None = None
    managed_env_stage = "managed" if is_verify else stage
    try:
        managed_env, managed_python = (
            execution_runtime_module.resolve_managed_environment_for_stage(
                repo_root,
                managed_env_stage,
            )
        )
    except ValueError as error:
        runtime_print(str(error), file=sys.stderr)
        return 1
    effective_command = (
        execution_runtime_module.rewrite_command_string_for_managed_python(
            resolved_command,
            managed_python,
        )
    )
    try:
        workflow_contract = workflow_contract_module.load_workflow_contract(
            repo_root
        )
    except ValueError as error:
        runtime_print(str(error), file=sys.stderr)
        return 1

    while True:
        env = dict(managed_env or os.environ)
        env["DEVCOV_DEVFLOW_STAGE"] = "" if is_verify else stage
        hook_env = dict(env)
        auto_fix_enabled = (
            execution_runtime_module.resolve_engine_auto_fix_enabled(repo_root)
        )
        # Gate owns refresh/autofix/lifecycle orchestration; the local
        # pre-commit `devcovenant check` hook reads these to enable mutating
        # behavior while public `check` stays read-only by default.
        hook_env[_CHECK_APPLY_FIXES_ENV] = "1" if auto_fix_enabled else "0"
        hook_env[_CHECK_RUN_REFRESH_ENV] = "1"
        hook_env[_CHECK_CLEAN_BYTECODE_ENV] = "1"
        hook_command = _resolve_hook_command(repo_root, effective_command)
        hook_command = _resolve_gate_execution_command(
            hook_command,
            env=hook_env,
            managed_python=managed_python,
        )
        try:
            diff_before = _current_numstat_snapshot(repo_root)
        except ValueError as error:
            runtime_print(str(error), file=sys.stderr)
            return 1
        if is_close:
            session_id = str(pre_payload.get("session_id", "")).strip()
            try:
                workflow_payload = load_workflow_session(repo_root)
            except ValueError as error:
                runtime_print(str(error), file=sys.stderr)
                return 1
            run_ids_pending = _stale_run_ids(
                repo_root,
                workflow_contract,
                workflow_payload,
                diff_before,
                session_id=session_id,
            )
        if is_open:
            status_exists = status_path.exists()
            status_payload: dict[str, object] = {}
            status_parse_error = ""
            status_snapshot_payload: dict[str, object] = {}
            status_snapshot_error = ""
            workflow_payload: dict[str, object] = {}
            workflow_status_error = ""
            if status_exists:
                try:
                    recovery_status_previous = status_path.read_bytes()
                except OSError as error:
                    runtime_print(str(error), file=sys.stderr)
                    return 1
                try:
                    status_payload = _load_status(status_path)
                except ValueError as error:
                    status_parse_error = str(error)
                else:
                    try:
                        status_snapshot_payload = (
                            load_session_snapshot_payload(
                                repo_root,
                                status_payload,
                            )
                        )
                    except ValueError as error:
                        status_snapshot_error = str(error)
            try:
                workflow_payload = load_workflow_session(repo_root)
            except ValueError as error:
                workflow_status_error = str(error)

            session_state = (
                str(status_payload.get("session_state", "")).strip().lower()
            )
            recovery_reason = ""
            recovery_baseline_snapshot: dict[str, str] | None = None
            if status_parse_error:
                recovery_reason = (
                    "Gate status is malformed; opening a recovery session "
                    "from the current baseline."
                )
            elif session_state == "open":
                runtime_print(
                    "Cannot open a new session while another session is "
                    "open. Complete it with `devcovenant gate --close`.",
                    file=sys.stderr,
                )
                return 1
            elif session_state and session_state != "closed":
                recovery_reason = (
                    "Gate status has an invalid `session_state`; opening "
                    "a recovery session from the current baseline."
                )
            elif session_state == "closed":
                if status_snapshot_error:
                    recovery_reason = (
                        "Closed gate session snapshot is unusable; opening "
                        "a recovery session from the current baseline."
                    )
                raw_close_snapshot = status_snapshot_payload.get(
                    "session_close_snapshot"
                )
                if not isinstance(raw_close_snapshot, dict):
                    recovery_reason = (
                        "Closed gate status is missing "
                        "`session_close_snapshot`; "
                        "opening a recovery session from the current "
                        "baseline."
                    )
                else:
                    try:
                        close_snapshot = (
                            execution_runtime_module.normalize_snapshot_rows(
                                raw_close_snapshot,
                                field_name="session_close_snapshot",
                            )
                        )
                    except ValueError as error:
                        runtime_print(str(error), file=sys.stderr)
                        return 1
                    changed_since_close = _changed_paths_between(
                        close_snapshot,
                        diff_before,
                    )
                    if changed_since_close:
                        recovery_baseline_snapshot = dict(close_snapshot)
                        recovery_reason = (
                            "Detected edits after the previous "
                            "`devcovenant gate --close`; opening a recovery "
                            "session that includes those unsessioned edits."
                        )
                        recovery_session_id = str(
                            status_payload.get("session_id", "")
                        ).strip()
                        if workflow_status_error:
                            recovery_run_ids = list(
                                workflow_contract_module.run_ids(
                                    workflow_contract
                                )
                            )
                        else:
                            recovery_run_ids = _stale_run_ids(
                                repo_root,
                                workflow_contract,
                                workflow_payload,
                                diff_before,
                                session_id=recovery_session_id,
                            )
            elif status_exists:
                recovery_reason = (
                    "Gate status is missing session metadata; opening a "
                    "recovery session from the current baseline."
                )

            if recovery_reason:
                recovery_payload: dict[str, object] = (
                    dict(status_payload) if status_payload else {}
                )
                try:
                    top_entry = _latest_changelog_entry(repo_root)
                except ValueError as error:
                    runtime_print(str(error), file=sys.stderr)
                    return 1
                recovery_payload["session_id"] = str(
                    int(open_ts.timestamp() * 1000000)
                )
                recovery_payload["session_state"] = "open"
                recovery_payload["session_open_utc"] = open_ts.isoformat()
                recovery_payload["session_open_epoch"] = open_ts.timestamp()
                recovery_payload.pop("changelog_baseline_reset", None)
                recovery_payload.pop("changelog_baseline_reset_utc", None)
                recovery_payload.pop("changelog_baseline_reset_epoch", None)
                recovery_payload["changelog_open_top_entry_fingerprint"] = (
                    _entry_fingerprint(top_entry)
                )
                recovery_payload["changelog_open_top_entry_present"] = bool(
                    top_entry
                )
                recovery_payload["changelog_open_top_version"] = (
                    _latest_changelog_version(repo_root)
                )
                recovery_remove_keys = [
                    "session_close_snapshot",
                    "last_run_snapshot",
                    "run_events",
                ]
                recovery_updates: dict[str, object] = {}
                if recovery_baseline_snapshot is not None:
                    recovery_updates["session_baseline_snapshot"] = dict(
                        recovery_baseline_snapshot
                    )
                else:
                    recovery_remove_keys.append("session_baseline_snapshot")
                (
                    snapshot_rel_path,
                    _recovery_snapshot_payload,
                ) = merge_session_snapshot_payload(
                    repo_root,
                    recovery_payload,
                    updates=recovery_updates,
                    remove_keys=tuple(recovery_remove_keys),
                )
                recovery_payload["session_snapshot_file"] = snapshot_rel_path
                recovery_payload["session_snapshot_updated_utc"] = (
                    open_ts.isoformat()
                )
                recovery_payload["session_snapshot_updated_epoch"] = (
                    open_ts.timestamp()
                )
                recovery_payload.pop("run_events_count", None)
                prune_inline_session_snapshot_fields(recovery_payload)
                recovery_payload["recovery_open_reason"] = recovery_reason
                status_path.parent.mkdir(parents=True, exist_ok=True)
                status_path.write_text(
                    json.dumps(recovery_payload, indent=2) + "\n",
                    encoding="utf-8",
                )
                recovery_status_active = True
            elif status_exists:
                recovery_status_active = False
                recovery_status_previous = None

        command_output = ""
        if is_open or is_close or is_verify:
            exit_code, command_output = _run_command_with_output(
                hook_command,
                env=hook_env,
            )
        else:
            exit_code = _run_command(
                hook_command,
                env=hook_env,
                strict=False,
            )
        try:
            diff_after_hooks = _current_numstat_snapshot(repo_root)
        except ValueError as error:
            if stage == "open" and recovery_status_active:
                _restore_status_file(status_path, recovery_status_previous)
                recovery_status_active = False
            runtime_print(str(error), file=sys.stderr)
            return 1
        hook_changed_paths = _changed_paths_between(
            diff_before, diff_after_hooks
        )
        hooks_changed = bool(hook_changed_paths)
        if is_open and hooks_changed:
            if recovery_status_active:
                _restore_status_file(status_path, recovery_status_previous)
                recovery_status_active = False
            _emit_open_gate_drift_failure(
                command,
                exit_code=exit_code,
                command_output=command_output,
                changed_paths=hook_changed_paths,
            )
            return 1
        if is_open and exit_code != 0:
            if recovery_status_active:
                _restore_status_file(status_path, recovery_status_previous)
                recovery_status_active = False
            rendered = " ".join(shlex.split(command)) or command
            runtime_print(
                "Pre-commit command failed with exit code "
                f"{exit_code}: {rendered}",
                file=sys.stderr,
            )
            runtime_print(
                "Open gate failed. Clear pre-commit violations and rerun "
                "`devcovenant gate --open`.",
                file=sys.stderr,
            )
            return exit_code

        if stage == "open" and recovery_status_active and recovery_run_ids:
            _restore_status_file(status_path, recovery_status_previous)
            recovery_status_active = False
            recovery_rerun = _format_run_rerun_instructions(
                recovery_run_ids,
            )
            runtime_print(
                "Recovery open detected unsessioned edits and requires "
                "fresh workflow runs before recording a new "
                "baseline.",
                file=sys.stderr,
            )
            runtime_print(
                "Run "
                f"{recovery_rerun},"
                " "
                "then rerun `devcovenant gate --open`. Open gate performs "
                "no internal workflow-run runs.",
                file=sys.stderr,
            )
            return 1

        if is_close and _is_blocking_devcov_failure(
            exit_code,
            command_output,
        ):
            rendered = " ".join(shlex.split(command)) or command
            runtime_print(
                "Pre-commit command failed with exit code "
                f"{exit_code}: {rendered}",
                file=sys.stderr,
            )
            runtime_print(
                "Close gate found blocking non-autofixed DevCovenant "
                "violations. Fix violations and rerun "
                "`devcovenant gate --close`.",
                file=sys.stderr,
            )
            return exit_code
        if is_verify and _is_blocking_devcov_failure(
            exit_code,
            command_output,
        ):
            rendered = " ".join(shlex.split(command)) or command
            runtime_print(
                "Pre-commit command failed with exit code "
                f"{exit_code}: {rendered}",
                file=sys.stderr,
            )
            runtime_print(
                "Verify gate found blocking non-autofixed DevCovenant "
                "violations. Fix violations and rerun `devcovenant gate "
                "--verify` before `devcovenant run`.",
                file=sys.stderr,
            )
            return exit_code

        if is_verify and exit_code == 0 and hooks_changed:
            runtime_print(
                "Verify gate detected hook-induced file changes. "
                "Rerun `devcovenant gate --verify` until hooks converge, then "
                "run `devcovenant run`.",
                file=sys.stderr,
            )
            return 1
        if is_verify and exit_code != 0:
            rendered = " ".join(shlex.split(command)) or command
            runtime_print(
                "Pre-commit command failed with exit code "
                f"{exit_code}: {rendered}",
                file=sys.stderr,
            )
            runtime_print(
                "Verify gate failed. Clear pre-commit violations and rerun "
                "`devcovenant gate --verify` before `devcovenant run`.",
                file=sys.stderr,
            )
            return exit_code
        if is_close and exit_code == 0 and hooks_changed:
            runtime_print(
                "Close gate detected hook-induced file changes. "
                "Run `devcovenant run`, then rerun "
                "`devcovenant gate --close`.",
                file=sys.stderr,
            )
            return 1
        if is_close and exit_code == 0 and run_ids_pending:
            rerun_runs = _format_run_rerun_instructions(
                run_ids_pending,
            )
            runtime_print(
                "Close gate requires fresh workflow runs before "
                "closure. Run "
                f"{rerun_runs},"
                " "
                "then rerun `devcovenant gate --close`.",
                file=sys.stderr,
            )
            return 1
        if is_close and exit_code != 0:
            rendered = " ".join(shlex.split(command)) or command
            runtime_print(
                "Pre-commit command failed with exit code "
                f"{exit_code}: {rendered}",
                file=sys.stderr,
            )
            return exit_code
        break

    if is_verify:
        _record_workflow_anchor(
            repo_root,
            contract=workflow_contract,
            stage="verify",
            command=command,
            notes=notes,
            when=_utc_now(),
            session_id=session_id,
            session_state="open",
        )
        runtime_print(
            "Completed verify gate pre-commit sweep without changing gate "
            "session lifecycle state."
        )
        return 0

    try:
        payload = _load_status(status_path)
    except ValueError as error:
        runtime_print(str(error), file=sys.stderr)
        return 1
    now = _utc_now()
    prefix = f"pre_commit_{stage}"
    if open_ts is not None:
        payload[f"{prefix}_utc"] = open_ts.isoformat()
        payload[f"{prefix}_epoch"] = open_ts.timestamp()
    else:
        payload[f"{prefix}_utc"] = now.isoformat()
        payload[f"{prefix}_epoch"] = now.timestamp()
    payload[f"{prefix}_command"] = command.strip()
    payload[f"{prefix}_notes"] = notes.strip()
    payload.pop(f"{prefix}_cache_enabled", None)
    payload.pop(f"{prefix}_cache_control_env", None)
    if is_open:
        # Purge legacy keys so old payload shape cannot silently persist.
        payload.pop("sha", None)
        payload.pop("tests_coverage_evidence", None)
        payload.pop("changelog_open_diff_numstat", None)
        payload.pop("changelog_open_exemption_fingerprints", None)
        payload.pop("session_open_signature", None)
        # Clear stale close-stage evidence so ordering checks stay
        # session-bound.
        for key in (
            "pre_commit_close_utc",
            "pre_commit_close_epoch",
            "pre_commit_close_command",
            "pre_commit_close_notes",
            "pre_commit_close_cache_enabled",
            "pre_commit_close_cache_control_env",
        ):
            payload.pop(key, None)
        session_id = str(int(open_ts.timestamp() * 1000000))
        payload["session_id"] = session_id
        payload["session_state"] = "open"
        payload["session_open_utc"] = open_ts.isoformat()
        payload["session_open_epoch"] = open_ts.timestamp()
        payload.pop("session_baseline_epoch", None)
        payload.pop("changelog_baseline_reset", None)
        payload.pop("changelog_baseline_reset_utc", None)
        payload.pop("changelog_baseline_reset_epoch", None)
        try:
            header_doc_suffixes, header_keys, header_scan_lines = (
                _resolve_doc_exemption_options(repo_root)
            )
        except ValueError as error:
            if recovery_status_active:
                _restore_status_file(status_path, recovery_status_previous)
                recovery_status_active = False
            runtime_print(str(error), file=sys.stderr)
            return 1
        snapshot_remove_keys = [
            "session_close_snapshot",
            "last_run_snapshot",
            "run_events",
        ]
        snapshot_updates: dict[str, object] = {
            # Persist the gate-open filesystem snapshot so policies can scope
            # deleted-file coverage to this session instead of HEAD-wide
            # history.
            "session_open_snapshot": dict(diff_before),
            "document_exemption_baseline": (
                execution_runtime_module.capture_document_exemption_baseline(
                    repo_root,
                    header_doc_suffixes=header_doc_suffixes,
                    header_keys=header_keys,
                    header_scan_lines=header_scan_lines,
                )
            ),
        }
        if recovery_status_active and recovery_baseline_snapshot is not None:
            snapshot_updates["session_baseline_snapshot"] = dict(
                recovery_baseline_snapshot
            )
        else:
            # Normal opens must not carry a stale recovery baseline forward.
            snapshot_remove_keys.append("session_baseline_snapshot")
        (
            snapshot_rel_path,
            _snapshot_payload,
        ) = merge_session_snapshot_payload(
            repo_root,
            payload,
            updates=snapshot_updates,
            remove_keys=tuple(snapshot_remove_keys),
        )
        payload["session_snapshot_file"] = snapshot_rel_path
        payload["session_snapshot_updated_utc"] = open_ts.isoformat()
        payload["session_snapshot_updated_epoch"] = open_ts.timestamp()
        payload.pop("run_events_count", None)
        payload.update(
            execution_runtime_module.capture_agents_section_hashes(repo_root)
        )
        payload.pop("session_close_utc", None)
        payload.pop("session_close_epoch", None)
        payload.pop("session_close_signature", None)
        payload.pop("recovery_open_reason", None)
        try:
            top_entry = _latest_changelog_entry(repo_root)
        except ValueError as error:
            if recovery_status_active:
                _restore_status_file(status_path, recovery_status_previous)
                recovery_status_active = False
            runtime_print(str(error), file=sys.stderr)
            return 1
        payload["changelog_open_top_entry_fingerprint"] = _entry_fingerprint(
            top_entry
        )
        payload["changelog_open_top_entry_present"] = bool(top_entry)
        payload["changelog_open_top_version"] = _latest_changelog_version(
            repo_root
        )
    else:
        session_id = str(payload.get("session_id", "")).strip()
        session_state = str(payload.get("session_state", "")).strip().lower()
        if not session_id or session_state != "open":
            runtime_print(
                "Cannot close gate without an active open session. "
                "Run `devcovenant gate --open` first.",
                file=sys.stderr,
            )
            return 1
        payload["session_state"] = "closed"
        payload["session_close_utc"] = now.isoformat()
        payload["session_close_epoch"] = now.timestamp()
        payload.pop("changelog_baseline_reset", None)
        payload.pop("changelog_baseline_reset_utc", None)
        payload.pop("changelog_baseline_reset_epoch", None)
        (
            snapshot_rel_path,
            _snapshot_payload,
        ) = merge_session_snapshot_payload(
            repo_root,
            payload,
            updates={"session_close_snapshot": dict(diff_after_hooks)},
        )
        payload["session_snapshot_file"] = snapshot_rel_path
        payload["session_snapshot_updated_utc"] = now.isoformat()
        payload["session_snapshot_updated_epoch"] = now.timestamp()
    prune_inline_session_snapshot_fields(payload)
    status_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    runtime_print(
        f"Recorded {prefix} at {payload[f'{prefix}_utc']} "
        f"for command `{payload[f'{prefix}_command']}`."
    )
    _record_workflow_anchor(
        repo_root,
        contract=workflow_contract,
        stage=stage,
        command=command,
        notes=notes,
        when=open_ts if open_ts is not None else now,
        session_id=str(payload.get("session_id", "")).strip(),
        session_state=str(payload.get("session_state", "")).strip().lower(),
        reset_runs=is_open,
        session_snapshot_file=str(payload.get("session_snapshot_file", "")),
        session_snapshot_updated_utc=str(
            payload.get("session_snapshot_updated_utc", "")
        ),
        session_snapshot_updated_epoch=float(
            payload.get("session_snapshot_updated_epoch") or 0.0
        ),
    )
    return 0
