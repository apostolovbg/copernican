"""
Workflow foundations: runtime registry, contract resolution, and validation.
"""

from __future__ import annotations

import fnmatch
import json
import os
import shlex
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import yaml

import devcovenant.core.profile_registry as profile_registry_service
import devcovenant.core.repository_paths as repository_paths
import devcovenant.core.tracked_registry as tracked_registry
from devcovenant.core.policy_contract import CheckContext, Violation

RUNTIME_REGISTRY_DIR = f"{tracked_registry.REGISTRY_DIR}/runtime"
GATE_STATUS_FILENAME = "gate_status.json"
WORKFLOW_SESSION_FILENAME = "workflow_session.json"
LATEST_RUNTIME_FILENAME = "latest.json"
SESSION_SNAPSHOT_FILENAME = "session_snapshot.json"


def runtime_registry_root(repo_root: Path) -> Path:
    """Return the path to the runtime registry directory."""
    return repo_root / RUNTIME_REGISTRY_DIR


def _default_runtime_evidence_relative_path(filename: str) -> Path:
    """Return the canonical repo-relative runtime evidence path."""
    return Path(RUNTIME_REGISTRY_DIR) / str(filename or "").strip()


def _load_config_payload_or_empty(repo_root: Path) -> dict[str, object]:
    """Load config when present, otherwise return an empty payload."""
    config_path = repo_root / "devcovenant" / "config.yaml"
    if not config_path.exists():
        return {}
    try:
        payload = repository_paths.load_yaml(config_path)
    except (OSError, yaml.YAMLError):
        return {}
    if isinstance(payload, dict):
        return payload
    return {}


def _config_runtime_path_override(
    repo_root: Path,
    *,
    option_name: str,
    config_payload: dict[str, object] | None = None,
) -> object | None:
    """Return one configured runtime evidence override when available."""
    payload = config_payload
    if not isinstance(payload, dict):
        payload = _load_config_payload_or_empty(repo_root)
    paths = payload.get("paths") if isinstance(payload, dict) else None
    if not isinstance(paths, dict):
        return None
    return paths.get(option_name)


def _resolve_runtime_evidence_path(
    repo_root: Path,
    *,
    option_name: str,
    default_filename: str,
    override_value: object | None = None,
    config_payload: dict[str, object] | None = None,
) -> Path:
    """Resolve one configurable runtime path under the runtime root."""
    default_relative = _default_runtime_evidence_relative_path(
        default_filename
    )
    raw_value = override_value
    if raw_value is None:
        raw_value = _config_runtime_path_override(
            repo_root,
            option_name=option_name,
            config_payload=config_payload,
        )
    token = str(raw_value or "").strip()
    relative_path = Path(token) if token else default_relative
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(
            f"`{option_name}` must be a repo-relative path inside "
            "`devcovenant/registry/runtime/`."
        )
    candidate = repo_root / relative_path
    runtime_root = runtime_registry_root(repo_root).resolve()
    resolved = candidate.resolve()
    if resolved != runtime_root and runtime_root not in resolved.parents:
        raise ValueError(
            f"`{option_name}` must stay inside "
            "`devcovenant/registry/runtime/`."
        )
    return candidate


def latest_runtime_path(repo_root: Path) -> Path:
    """Return the runtime latest-run pointer path."""
    return runtime_registry_root(repo_root) / LATEST_RUNTIME_FILENAME


def session_snapshot_path(repo_root: Path) -> Path:
    """Return the runtime session snapshot companion path."""
    return runtime_registry_root(repo_root) / SESSION_SNAPSHOT_FILENAME


def gate_status_path(
    repo_root: Path,
    config_payload: dict[str, object] | None = None,
) -> Path:
    """Return the configured gate-status file path."""
    return _resolve_runtime_evidence_path(
        repo_root,
        option_name="gate_status_file",
        default_filename=GATE_STATUS_FILENAME,
        config_payload=config_payload,
    )


def gate_status_path_from_option(
    repo_root: Path,
    raw_value: object | None,
) -> Path:
    """Resolve one gate-status path from an explicit option value."""
    return _resolve_runtime_evidence_path(
        repo_root,
        option_name="gate_status_file",
        default_filename=GATE_STATUS_FILENAME,
        override_value=raw_value,
    )


def workflow_session_path(
    repo_root: Path,
    config_payload: dict[str, object] | None = None,
) -> Path:
    """Return the configured workflow-session file path."""
    return _resolve_runtime_evidence_path(
        repo_root,
        option_name="workflow_session_file",
        default_filename=WORKFLOW_SESSION_FILENAME,
        config_payload=config_payload,
    )


def workflow_session_path_from_option(
    repo_root: Path,
    raw_value: object | None,
) -> Path:
    """Resolve one workflow-session path from an explicit option value."""
    return _resolve_runtime_evidence_path(
        repo_root,
        option_name="workflow_session_file",
        default_filename=WORKFLOW_SESSION_FILENAME,
        override_value=raw_value,
    )


SCHEMA_VERSION = 4
ANCHOR_IDS = ("open", "verify", "close")
DEFAULT_PRE_COMMIT_COMMAND = "pre-commit run --all-files"
_FRESHNESS_KINDS = {"ignore_paths", "any_change"}
_RUNNER_KINDS = {
    "command_group",
    "runtime_action",
    "policy_command",
    "manual_attestation",
}
_SUCCESS_CONTRACT_KINDS = {
    "all_commands_exit_zero",
    "runtime_action_success",
    "policy_command_success",
    "manual_attested",
    "external_artifact_check",
}
_DEFAULT_FRESHNESS_IGNORED_FILES = ("CHANGELOG.md",)


def _load_config_payload(repo_root: Path) -> dict[str, object]:
    """Load `devcovenant/config.yaml` into a mapping."""

    config_path = repo_root / "devcovenant" / "config.yaml"
    rendered = repository_paths.display_path(config_path, repo_root=repo_root)
    if not config_path.exists():
        raise ValueError(f"Missing config file: {rendered}")
    payload = repository_paths.load_yaml(config_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a YAML mapping: {rendered}")
    return payload


def resolve_pre_commit_command(repo_root: Path) -> str:
    """Return the configured workflow pre-commit command for one repo."""
    try:
        payload = _load_config_payload(repo_root)
    except ValueError:
        return DEFAULT_PRE_COMMIT_COMMAND
    workflow = payload.get("workflow")
    if isinstance(workflow, Mapping):
        command = str(workflow.get("pre_commit_command", "") or "").strip()
        if command:
            return command
    return DEFAULT_PRE_COMMIT_COMMAND


def _normalize_bool(raw_value: object, *, default: bool) -> bool:
    """Normalize a loose YAML-ish boolean token."""

    if isinstance(raw_value, bool):
        return raw_value
    token = str(raw_value or "").strip().lower()
    if not token:
        return default
    if token in {"true", "1", "yes", "on"}:
        return True
    if token in {"false", "0", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean token `{raw_value}`.")


def _normalize_int(raw_value: object, *, default: int) -> int:
    """Normalize one ordering integer."""

    if raw_value in {None, ""}:
        return default
    try:
        return int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid integer token `{raw_value}`.") from exc


def _normalize_position_reference(
    raw_value: object,
    *,
    default: str,
) -> str:
    """Normalize one workflow-position reference token."""

    return str(raw_value or default).strip().lower() or default


def _normalize_commands(raw_value: object, *, field_name: str) -> list[str]:
    """Normalize a command-group payload into an ordered string list."""

    if isinstance(raw_value, str):
        values = [raw_value]
    elif isinstance(raw_value, list):
        values = list(raw_value)
    else:
        raise ValueError(
            f"Invalid `{field_name}` payload: expected string or list."
        )
    commands: list[str] = []
    for entry in values:
        token = str(entry or "").strip()
        if token and token not in commands:
            commands.append(token)
    if not commands:
        raise ValueError(f"`{field_name}` must declare at least one command.")
    return commands


def _normalize_string_list(
    raw_value: object,
    *,
    field_name: str,
) -> list[str]:
    """Normalize one optional string-or-list payload into unique strings."""

    if raw_value is None or raw_value == "":
        return []
    if isinstance(raw_value, str):
        values = [raw_value]
    elif isinstance(raw_value, list):
        values = list(raw_value)
    else:
        raise ValueError(
            f"Invalid `{field_name}` payload: expected string or list."
        )
    normalized: list[str] = []
    for entry in values:
        token = str(entry or "").strip()
        if token and token not in normalized:
            normalized.append(token)
    return normalized


def _normalize_run_entry(
    profile_name: str,
    raw_entry: Mapping[str, object],
) -> dict[str, object]:
    """Normalize one workflow-run manifest entry."""

    run_id = str(raw_entry.get("id") or "").strip().lower()
    if not run_id:
        raise ValueError(
            f"Profile `{profile_name}` has a workflow run without id."
        )
    if run_id in ANCHOR_IDS:
        raise ValueError(
            f"Workflow run `{run_id}` in profile `{profile_name}` uses a "
            "reserved anchor id."
        )
    enabled = _normalize_bool(raw_entry.get("enabled"), default=True)
    after = _normalize_position_reference(
        raw_entry.get("after"),
        default="verify",
    )
    before = _normalize_position_reference(
        raw_entry.get("before"),
        default="close",
    )
    order = _normalize_int(raw_entry.get("order"), default=100)

    runner_raw = raw_entry.get("runner")
    if not isinstance(runner_raw, Mapping):
        raise ValueError(
            f"Workflow run `{run_id}` in profile `{profile_name}` "
            "must define runner as a mapping."
        )
    runner_kind = str(runner_raw.get("kind") or "").strip().lower()
    if runner_kind not in _RUNNER_KINDS:
        raise ValueError(
            f"Workflow run `{run_id}` in profile `{profile_name}` uses "
            f"unsupported runner kind `{runner_kind}`."
        )
    runner: dict[str, object] = {"kind": runner_kind}
    if runner_kind == "command_group":
        runner["commands"] = _normalize_commands(
            runner_raw.get("commands"),
            field_name=f"workflow_runs[{run_id}].runner.commands",
        )
    elif runner_kind in {"runtime_action", "policy_command"}:
        target = str(runner_raw.get("target") or "").strip()
        if not target:
            raise ValueError(
                f"Workflow run `{run_id}` in profile `{profile_name}` "
                f"must define runner.target for `{runner_kind}`."
            )
        runner["target"] = target
        payload_raw = runner_raw.get("payload")
        if payload_raw is None:
            payload = {}
        elif isinstance(payload_raw, Mapping):
            payload = dict(payload_raw)
        else:
            raise ValueError(
                f"Workflow run `{run_id}` in profile `{profile_name}` "
                "must define runner.payload as a mapping when present."
            )
        if payload:
            runner["payload"] = payload
        if runner_kind == "policy_command":
            runner["args"] = _normalize_string_list(
                runner_raw.get("args"),
                field_name=f"workflow_runs[{run_id}].runner.args",
            )
    elif runner_kind == "manual_attestation":
        attestation_key = str(runner_raw.get("attestation_key") or "").strip()
        if not attestation_key:
            raise ValueError(
                f"Workflow run `{run_id}` in profile `{profile_name}` "
                "must define runner.attestation_key for manual attestation."
            )
        runner["attestation_key"] = attestation_key

    success_raw = raw_entry.get("success_contract")
    if not isinstance(success_raw, Mapping):
        raise ValueError(
            f"Workflow run `{run_id}` in profile `{profile_name}` "
            "must define success_contract as a mapping."
        )
    success_kind = str(success_raw.get("kind") or "").strip().lower()
    if success_kind not in _SUCCESS_CONTRACT_KINDS:
        raise ValueError(
            f"Workflow run `{run_id}` in profile `{profile_name}` uses "
            f"unsupported success contract `{success_kind}`."
        )
    success_contract: dict[str, object] = {"kind": success_kind}
    if success_kind == "external_artifact_check":
        base_dir = str(success_raw.get("base_dir") or ".").strip() or "."
        required_files = _normalize_string_list(
            success_raw.get("required_files"),
            field_name=(
                f"workflow_runs[{run_id}].success_contract.required_files"
            ),
        )
        required_globs = _normalize_string_list(
            success_raw.get("required_globs"),
            field_name=(
                f"workflow_runs[{run_id}].success_contract.required_globs"
            ),
        )
        forbidden_globs = _normalize_string_list(
            success_raw.get("forbidden_globs"),
            field_name=(
                f"workflow_runs[{run_id}].success_contract." "forbidden_globs"
            ),
        )
        minimum_matches = _normalize_int(
            success_raw.get("minimum_matches"),
            default=1,
        )
        if minimum_matches < 0:
            raise ValueError(
                f"Workflow run `{run_id}` in profile `{profile_name}` "
                "must define a non-negative minimum_matches value."
            )
        if not (required_files or required_globs or forbidden_globs):
            raise ValueError(
                f"Workflow run `{run_id}` in profile `{profile_name}` "
                "must define required_files, required_globs, or "
                "forbidden_globs for external_artifact_check."
            )
        success_contract.update(
            {
                "base_dir": base_dir,
                "required_files": required_files,
                "required_globs": required_globs,
                "forbidden_globs": forbidden_globs,
                "minimum_matches": minimum_matches,
            }
        )

    recording_raw = raw_entry.get("recording")
    if recording_raw is None:
        recording_raw = {}
    if not isinstance(recording_raw, Mapping):
        raise ValueError(
            f"Workflow run `{run_id}` in profile `{profile_name}` "
            "must define recording as a mapping when present."
        )
    event_adapter_group = str(
        recording_raw.get("event_adapter_group") or ""
    ).strip()
    if event_adapter_group == "test_events":
        raise ValueError(
            f"Workflow run `{run_id}` in profile `{profile_name}` must "
            "use `run_events`, not legacy `test_events`."
        )
    recording = {
        "record_in_session": _normalize_bool(
            recording_raw.get("record_in_session"),
            default=True,
        ),
        "summary_label": (
            str(recording_raw.get("summary_label") or run_id).strip().title()
            or run_id.title()
        ),
        "output_mode_config_field": str(
            recording_raw.get("output_mode_config_field") or ""
        ).strip(),
        "event_adapter_group": event_adapter_group,
        "write_runtime_profile": _normalize_bool(
            recording_raw.get("write_runtime_profile"),
            default=False,
        ),
    }
    freshness_raw = raw_entry.get("freshness")
    if freshness_raw is None:
        freshness_raw = {}
    if not isinstance(freshness_raw, Mapping):
        raise ValueError(
            f"Workflow run `{run_id}` in profile `{profile_name}` "
            "must define freshness as a mapping when present."
        )
    freshness_kind = (
        str(freshness_raw.get("kind") or "ignore_paths").strip().lower()
        or "ignore_paths"
    )
    if freshness_kind not in _FRESHNESS_KINDS:
        raise ValueError(
            f"Workflow run `{run_id}` in profile `{profile_name}` uses "
            f"unsupported freshness kind `{freshness_kind}`."
        )
    freshness: dict[str, object] = {"kind": freshness_kind}
    if freshness_kind == "ignore_paths":
        ignored_files = _normalize_string_list(
            freshness_raw.get("ignored_files"),
            field_name=f"workflow_runs[{run_id}].freshness.ignored_files",
        )
        ignored_globs = _normalize_string_list(
            freshness_raw.get("ignored_globs"),
            field_name=f"workflow_runs[{run_id}].freshness.ignored_globs",
        )
        if not ignored_files and not ignored_globs:
            ignored_files = list(_DEFAULT_FRESHNESS_IGNORED_FILES)
        freshness.update(
            {
                "ignored_files": ignored_files,
                "ignored_globs": ignored_globs,
            }
        )

    run = {
        "id": run_id,
        "owner": "profile",
        "owner_id": profile_name,
        "enabled": enabled,
        "position": {
            "after": after,
            "before": before,
            "order": order,
        },
        "runner": runner,
        "success_contract": success_contract,
        "recording": recording,
        "freshness": freshness,
        "source_field": "workflow_runs",
    }
    return run


def _normalize_manifest_runs(
    profile_name: str,
    raw_value: object,
) -> list[dict[str, object]]:
    """Normalize workflow runs declared in one profile manifest."""

    if raw_value is None:
        return []
    if not isinstance(raw_value, list):
        raise ValueError(
            f"Profile `{profile_name}` must define workflow_runs as a list."
        )
    normalized: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for raw_entry in raw_value:
        if not isinstance(raw_entry, Mapping):
            raise ValueError(
                f"Profile `{profile_name}` has non-mapping workflow run "
                "entries."
            )
        entry = _normalize_run_entry(profile_name, raw_entry)
        run_id = str(entry["id"])
        if run_id in seen_ids:
            raise ValueError(
                f"Profile `{profile_name}` defines duplicate workflow run "
                f"`{run_id}`."
            )
        seen_ids.add(run_id)
        normalized.append(entry)
    return normalized


def _run_sort_key(run: Mapping[str, object]) -> tuple[int, str, str]:
    """Return deterministic sort key for resolved run entries."""

    position = run.get("position")
    if isinstance(position, Mapping):
        order = _normalize_int(position.get("order"), default=100)
    else:
        order = 100
    owner = str(run.get("owner_id") or "").strip().lower()
    run_id = str(run.get("id") or "").strip().lower()
    return (order, owner, run_id)


def _graph_node_sort_key(
    node_id: str,
    run_map: Mapping[str, Mapping[str, object]],
) -> tuple[int, int, int, str, str]:
    """Return deterministic sort key for one anchor-or-run graph node."""

    if node_id in ANCHOR_IDS:
        return (0, ANCHOR_IDS.index(node_id), 0, "", "")
    run = run_map.get(node_id, {})
    order, owner, run_id = _run_sort_key(run)
    return (1, 0, order, owner, run_id)


def _position_token(
    run: Mapping[str, object],
    key: str,
    *,
    default: str,
) -> str:
    """Return one normalized position token from a run mapping."""

    position = run.get("position")
    if not isinstance(position, Mapping):
        return default
    return _normalize_position_reference(position.get(key), default=default)


def _position_reference_error(
    run_id: str,
    field_name: str,
    ref_id: str,
    valid_ids: Sequence[str],
) -> ValueError:
    """Build one stable invalid-position-reference error."""

    allowed = ", ".join(valid_ids)
    return ValueError(
        f"Workflow run `{run_id}` references unknown `{field_name}` target "
        f"`{ref_id}`. Allowed targets: {allowed}."
    )


def _add_position_edge(
    graph: dict[str, set[str]],
    indegree: dict[str, int],
    source: str,
    target: str,
) -> None:
    """Add one directed ordering edge when it is not already present."""

    if target in graph[source]:
        return
    graph[source].add(target)
    indegree[target] += 1


def _resolve_positioned_runs(
    runs: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Return runs in validated deterministic workflow order."""

    run_map = {
        str(run.get("id") or "").strip().lower(): dict(run)
        for run in runs
        if str(run.get("id") or "").strip()
    }
    valid_ids = [*ANCHOR_IDS, *sorted(run_map)]
    graph = {node_id: set() for node_id in [*ANCHOR_IDS, *sorted(run_map)]}
    indegree = {node_id: 0 for node_id in graph}
    _add_position_edge(graph, indegree, "open", "verify")
    _add_position_edge(graph, indegree, "verify", "close")

    for run_id, run in run_map.items():
        after_id = _position_token(run, "after", default="verify")
        before_id = _position_token(run, "before", default="close")
        if after_id not in graph:
            raise _position_reference_error(
                run_id,
                "after",
                after_id,
                valid_ids,
            )
        if before_id not in graph:
            raise _position_reference_error(
                run_id,
                "before",
                before_id,
                valid_ids,
            )
        _add_position_edge(graph, indegree, after_id, run_id)
        _add_position_edge(graph, indegree, run_id, before_id)

    available = sorted(
        [node_id for node_id, degree in indegree.items() if degree == 0],
        key=lambda node_id: _graph_node_sort_key(node_id, run_map),
    )
    ordered_nodes: list[str] = []
    while available:
        node_id = available.pop(0)
        ordered_nodes.append(node_id)
        for neighbor in sorted(graph[node_id]):
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                available.append(neighbor)
                available.sort(
                    key=lambda candidate: _graph_node_sort_key(
                        candidate,
                        run_map,
                    )
                )
    if len(ordered_nodes) != len(graph):
        unresolved = sorted(
            node_id for node_id, degree in indegree.items() if degree > 0
        )
        raise ValueError(
            "Workflow runs declare cyclic ordering constraints involving: "
            f"{', '.join(unresolved)}."
        )
    return [
        run_map[node_id] for node_id in ordered_nodes if node_id in run_map
    ]


def _default_anchor_rows() -> list[dict[str, object]]:
    """Return the reserved anchor definitions for every contract."""

    return [
        {
            "id": anchor_id,
            "owner": "core",
            "anchor_kind": "gate_anchor",
        }
        for anchor_id in ANCHOR_IDS
    ]


def build_workflow_contract(
    repo_root: Path,
    profiles_registry: Mapping[str, Mapping[str, object]],
    active_profiles: Sequence[str],
) -> dict[str, object]:
    """Build the resolved workflow contract for the active profile set."""

    run_map: dict[str, dict[str, object]] = {}
    active_names = profile_registry_service._active_profile_names(
        active_profiles
    )
    for profile_name in active_names:
        profile_meta = profiles_registry.get(profile_name, {})
        if not isinstance(profile_meta, Mapping):
            continue
        for run in _normalize_manifest_runs(
            profile_name,
            profile_meta.get("workflow_runs"),
        ):
            run_map[str(run["id"])] = run
    runs = _resolve_positioned_runs(list(run_map.values()))
    run_ids = [
        str(run.get("id") or "") for run in runs if bool(run.get("enabled"))
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "anchors": _default_anchor_rows(),
        "runs": runs,
        "run_ids": run_ids,
        "active_profiles": list(active_names),
    }


def load_workflow_contract(repo_root: Path) -> dict[str, object]:
    """Resolve the active workflow contract from config and profiles."""

    config = _load_config_payload(repo_root)
    active_profiles = profile_registry_service.parse_active_profiles(
        config,
        include_global=True,
    )
    registry = profile_registry_service.load_profile_registry(repo_root)
    return build_workflow_contract(repo_root, registry, active_profiles)


def resolve_run(
    contract: Mapping[str, object],
    run_id: str,
) -> dict[str, object] | None:
    """Return one run definition by id from a workflow contract."""

    token = str(run_id or "").strip().lower()
    raw_runs = contract.get("runs")
    if not isinstance(raw_runs, list):
        return None
    for raw_run in raw_runs:
        if not isinstance(raw_run, Mapping):
            continue
        if str(raw_run.get("id") or "").strip().lower() != token:
            continue
        return dict(raw_run)
    return None


def run_ids(contract: Mapping[str, object]) -> list[str]:
    """Return enabled run ids from a workflow contract."""

    raw_ids = contract.get("run_ids")
    if isinstance(raw_ids, list):
        return [
            str(entry or "").strip().lower()
            for entry in raw_ids
            if str(entry or "").strip()
        ]
    raw_runs = contract.get("runs")
    if not isinstance(raw_runs, list):
        return []
    resolved: list[str] = []
    for raw_run in raw_runs:
        if not isinstance(raw_run, Mapping):
            continue
        run_id = str(raw_run.get("id") or "").strip().lower()
        if not run_id:
            continue
        if not _normalize_bool(raw_run.get("enabled"), default=True):
            continue
        resolved.append(run_id)
    return resolved


def run_relevant_paths_changed(
    run: Mapping[str, object],
    changed_paths: Sequence[str],
) -> bool:
    """Return whether changed paths invalidate one run result."""

    if not changed_paths:
        return False
    freshness = run.get("freshness")
    freshness_map = dict(freshness) if isinstance(freshness, Mapping) else {}
    freshness_kind = (
        str(freshness_map.get("kind") or "ignore_paths").strip().lower()
        or "ignore_paths"
    )
    if freshness_kind == "any_change":
        return True
    ignored_files = {
        str(entry).replace("\\", "/").strip().lower()
        for entry in freshness_map.get("ignored_files") or ()
        if str(entry).strip()
    }
    if not ignored_files:
        ignored_files = {
            token.lower() for token in _DEFAULT_FRESHNESS_IGNORED_FILES
        }
    ignored_globs = [
        str(entry).replace("\\", "/").strip().lower()
        for entry in freshness_map.get("ignored_globs") or ()
        if str(entry).strip()
    ]
    for raw_path in changed_paths:
        normalized_path = str(raw_path).replace("\\", "/").strip().lower()
        if not normalized_path:
            continue
        leaf = normalized_path.rsplit("/", 1)[-1]
        if normalized_path in ignored_files or leaf in ignored_files:
            continue
        if any(
            fnmatch.fnmatch(normalized_path, pattern)
            or fnmatch.fnmatch(leaf, pattern)
            for pattern in ignored_globs
        ):
            continue
        return True
    return False


CHECK_ID = "workflow-contract"
_PRE_COMMIT_EXECUTABLE_TOKENS = frozenset(
    {"pre-commit", "pre-commit.exe", "pre_commit", "pre_commit.exe"}
)


def _merged_section(
    repo_root: Path,
    context_config: dict[str, object],
    section_name: str,
) -> dict[str, object]:
    """Return one config section merged with in-context overrides."""
    merged: dict[str, object] = {}
    repo_payload = _load_config_payload_or_empty(repo_root)
    repo_section = repo_payload.get(section_name)
    if isinstance(repo_section, dict):
        merged.update(repo_section)
    context_section = context_config.get(section_name)
    if isinstance(context_section, dict):
        merged.update(context_section)
    return merged


def _resolve_status_path(context: CheckContext) -> Path:
    """Return the configured gate-status path for one repository."""
    paths = _merged_section(context.repo_root, context.config, "paths")
    return gate_status_path_from_option(
        context.repo_root,
        paths.get("gate_status_file"),
    )


def _resolve_workflow_session_path(context: CheckContext) -> Path:
    """Return the configured workflow-session path for one repository."""
    paths = _merged_section(context.repo_root, context.config, "paths")
    return workflow_session_path_from_option(
        context.repo_root,
        paths.get("workflow_session_file"),
    )


def _format_run_rerun_instructions(run_ids: list[str]) -> str:
    """Render the canonical rerun instruction."""
    del run_ids
    return "`devcovenant run`"


def _normalize_pre_commit_command(raw_value: object) -> str:
    """Normalize canonical pre-commit launchers for exact validation."""
    raw = str(raw_value or "").strip()
    if not raw:
        return ""
    try:
        tokens = shlex.split(raw)
    except ValueError:
        return raw.lower()
    if not tokens:
        return ""
    first = Path(tokens[0]).name.lower()
    normalized = [str(token).strip().lower() for token in tokens]
    if first in _PRE_COMMIT_EXECUTABLE_TOKENS:
        normalized[0] = "pre-commit"
        return shlex.join(normalized)
    return shlex.join(normalized)


def _required_pre_commit_command(context: CheckContext) -> str:
    """Return the workflow-owned canonical pre-commit command."""
    workflow = _merged_section(context.repo_root, context.config, "workflow")
    raw = str(workflow.get("pre_commit_command", "") or "").strip()
    if not raw:
        raw = DEFAULT_PRE_COMMIT_COMMAND
    return _normalize_pre_commit_command(raw)


def _load_gate_status(status_file: Path) -> dict | None:
    """Return parsed gate status, or None when file is missing."""
    if not status_file.is_file():
        return None
    try:
        payload = json.loads(status_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid gate status JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Gate status payload must be a JSON object.")
    return payload


def _load_workflow_session(session_file: Path) -> dict | None:
    """Return parsed workflow-session payload, or None when file is missing."""
    if not session_file.is_file():
        return None
    try:
        payload = json.loads(session_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid workflow session JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Workflow session payload must be a JSON object.")
    return payload


def _as_epoch(raw: object) -> float:
    """Parse one epoch field, returning 0 when missing or invalid."""
    try:
        return float(raw or 0.0)
    except (TypeError, ValueError):
        return 0.0


def check_workflow_contract(
    context: CheckContext,
) -> list[Violation]:
    """Enforce the recorded open, verify, run, and close workflow contract."""
    violations: list[Violation] = []
    status_path = _resolve_status_path(context)
    status_rel = status_path.relative_to(context.repo_root)
    workflow_session_path = _resolve_workflow_session_path(context)
    workflow_session_rel = workflow_session_path.relative_to(context.repo_root)
    stage = os.environ.get("DEVCOV_DEVFLOW_STAGE", "").strip().lower()
    in_pre_commit = bool(str(os.environ.get("PRE_COMMIT", "")).strip())

    if in_pre_commit and not stage:
        return violations
    if stage == "open":
        return violations

    try:
        status = _load_gate_status(status_path)
    except ValueError as error:
        return [
            Violation(
                policy_id=CHECK_ID,
                severity="error",
                file_path=status_rel,
                message=str(error),
            )
        ]
    if not status:
        top_command = (
            str(os.environ.get("DEVCOV_TOP_COMMAND", "")).strip().lower()
        )
        reason = str(context.change_state.session_reason_code or "").strip()
        if (
            not stage
            and top_command == "check"
            and reason == "missing_gate_status"
        ):
            return violations
        return [
            Violation(
                policy_id=CHECK_ID,
                severity="error",
                file_path=status_rel,
                message=(
                    "Gate status is missing. Run `devcovenant gate --open`, "
                    "then `devcovenant gate --verify`, then "
                    "`devcovenant run`, "
                    "then `devcovenant gate --close`."
                ),
            )
        ]

    session_id = str(status.get("session_id", "")).strip()
    session_state = str(status.get("session_state", "")).strip().lower()
    session_reason_code = str(
        context.change_state.session_reason_code or ""
    ).strip()
    has_unsessioned_edits = (
        not context.change_state.session_valid
        and session_reason_code == "unsessioned_edits_after_close"
    )
    if not session_id:
        if has_unsessioned_edits:
            return [
                Violation(
                    policy_id=CHECK_ID,
                    severity="error",
                    file_path=status_rel,
                    message=(
                        "Changes exist without a recorded session. Run "
                        "`devcovenant gate --open` before edits."
                    ),
                )
            ]
        return violations

    if stage == "close":
        if session_state != "open":
            violations.append(
                Violation(
                    policy_id=CHECK_ID,
                    severity="error",
                    file_path=status_rel,
                    message=(
                        "Close gate requires an active open session. Run "
                        "`devcovenant gate --open` first."
                    ),
                )
            )
            return violations
    else:
        if session_state == "open":
            violations.append(
                Violation(
                    policy_id=CHECK_ID,
                    severity="error",
                    file_path=status_rel,
                    message=(
                        "Session is still open. Complete the workflow with "
                        "`devcovenant gate --close`."
                    ),
                )
            )
        elif has_unsessioned_edits:
            violations.append(
                Violation(
                    policy_id=CHECK_ID,
                    severity="error",
                    file_path=status_rel,
                    message=(
                        "Detected edits outside an active session. Run "
                        "`devcovenant gate --open` before edits."
                    ),
                )
            )

    try:
        workflow_contract = load_workflow_contract(context.repo_root)
    except ValueError as error:
        return [
            Violation(
                policy_id=CHECK_ID,
                severity="error",
                file_path=workflow_session_rel,
                message=str(error),
            )
        ]
    configured_run_ids = run_ids(workflow_contract)
    if not configured_run_ids:
        return [
            Violation(
                policy_id=CHECK_ID,
                severity="error",
                file_path=workflow_session_rel,
                message=(
                    "No workflow runs are configured for the active "
                    "profiles. Declare `workflow_runs` before using "
                    "the workflow contract."
                ),
            )
        ]

    try:
        workflow_session = _load_workflow_session(workflow_session_path)
    except ValueError as error:
        return [
            Violation(
                policy_id=CHECK_ID,
                severity="error",
                file_path=workflow_session_rel,
                message=str(error),
            )
        ]
    if not workflow_session:
        return [
            Violation(
                policy_id=CHECK_ID,
                severity="error",
                file_path=workflow_session_rel,
                message=(
                    "Workflow session is missing. Run `devcovenant "
                    "gate --open`, then `devcovenant gate --verify`, then "
                    "execute workflow runs with `devcovenant run`, then "
                    "`devcovenant gate --close`."
                ),
            )
        ]

    pre_commit_command = _required_pre_commit_command(context)
    open_ts = _as_epoch(status.get("pre_commit_open_epoch"))
    close_ts = _as_epoch(status.get("pre_commit_close_epoch"))
    if open_ts <= 0.0:
        violations.append(
            Violation(
                policy_id=CHECK_ID,
                severity="error",
                file_path=status_rel,
                message=(
                    "Session open pre-commit run is missing. Run "
                    "`devcovenant gate --open` before edits."
                ),
            )
        )
    open_command = _normalize_pre_commit_command(
        status.get("pre_commit_open_command") or ""
    )
    if pre_commit_command and open_command != pre_commit_command:
        violations.append(
            Violation(
                policy_id=CHECK_ID,
                severity="error",
                file_path=status_rel,
                message=(
                    "Session open pre-commit command is missing or does "
                    f"not include `{pre_commit_command}`. Re-run "
                    "`devcovenant gate --open`."
                ),
            )
        )

    workflow_session_id = str(workflow_session.get("session_id", "")).strip()
    workflow_session_state = (
        str(workflow_session.get("session_state", "")).strip().lower()
    )
    if workflow_session_id and workflow_session_id != session_id:
        violations.append(
            Violation(
                policy_id=CHECK_ID,
                severity="error",
                file_path=workflow_session_rel,
                message=(
                    "Workflow session id does not match gate status. "
                    "Re-run `devcovenant gate --open`."
                ),
            )
        )
        return violations
    if stage == "close":
        if workflow_session_state != "open":
            violations.append(
                Violation(
                    policy_id=CHECK_ID,
                    severity="error",
                    file_path=workflow_session_rel,
                    message=(
                        "Workflow session must be open during "
                        "`devcovenant gate --close`."
                    ),
                )
            )
            return violations
    elif workflow_session_state == "open":
        violations.append(
            Violation(
                policy_id=CHECK_ID,
                severity="error",
                file_path=workflow_session_rel,
                message=(
                    "Workflow session is still open. Complete the "
                    "workflow with `devcovenant gate --close`."
                ),
            )
        )

    runs_raw = workflow_session.get("runs")
    run_map = dict(runs_raw) if isinstance(runs_raw, dict) else {}
    missing_runs: list[str] = []
    for run_id in configured_run_ids:
        run_entry = run_map.get(run_id)
        if not isinstance(run_entry, dict):
            missing_runs.append(run_id)
            continue
        if str(run_entry.get("status", "")).strip().lower() != "passed":
            missing_runs.append(run_id)
            continue
        last_run_session_id = str(
            run_entry.get("last_run_session_id", "")
        ).strip()
        if session_id and last_run_session_id != session_id:
            missing_runs.append(run_id)
    if missing_runs:
        rerun_instructions = _format_run_rerun_instructions(missing_runs)
        violations.append(
            Violation(
                policy_id=CHECK_ID,
                severity="error",
                file_path=workflow_session_rel,
                message=(
                    "Latest recorded workflow session is missing runs: "
                    f"{', '.join(missing_runs)}. Run {rerun_instructions}."
                ),
            )
        )

    if stage != "close":
        if close_ts <= 0.0:
            violations.append(
                Violation(
                    policy_id=CHECK_ID,
                    severity="error",
                    file_path=status_rel,
                    message=(
                        "Session close pre-commit run is missing. Run "
                        "`devcovenant gate --close`."
                    ),
                )
            )
        elif open_ts > 0.0 and close_ts < open_ts:
            violations.append(
                Violation(
                    policy_id=CHECK_ID,
                    severity="error",
                    file_path=status_rel,
                    message=(
                        "Session close timestamp predates session open. "
                        "Re-run `devcovenant gate --close`."
                    ),
                )
            )
        close_command = _normalize_pre_commit_command(
            status.get("pre_commit_close_command") or ""
        )
        if pre_commit_command and close_command != pre_commit_command:
            violations.append(
                Violation(
                    policy_id=CHECK_ID,
                    severity="error",
                    file_path=status_rel,
                    message=(
                        "Session close pre-commit command is missing or does "
                        f"not include `{pre_commit_command}`. Re-run "
                        "`devcovenant gate --close`."
                    ),
                )
            )

    return violations


def load_gate_status_payload(path: Path) -> dict[str, object]:
    """Load one gate-status payload, returning empty mapping when missing."""
    rendered = repository_paths.display_path(path)
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
            "Gate status payload is missing: "
            f"{repository_paths.display_path(path)}"
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
