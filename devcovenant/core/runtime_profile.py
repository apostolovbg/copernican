"""Workflow runtime profiling and rendering helpers."""

from __future__ import annotations

import datetime as _dt
import shlex
from pathlib import Path
from typing import Any, Mapping, Sequence

PROFILE_SCHEMA_VERSION = "1.0"
_SLOW_COMMAND_LIMIT = 10


def _normalize_command_tokens(raw: object) -> list[str]:
    """Normalize command payloads into one token list."""
    if isinstance(raw, list):
        return [str(token).strip() for token in raw if str(token).strip()]
    if isinstance(raw, str):
        token = raw.strip()
        if not token:
            return []
        return shlex.split(token)
    return []


def _coerce_duration_seconds(raw: object) -> float:
    """Parse one duration value into a non-negative float."""
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0
    if value < 0:
        return 0.0
    return round(value, 6)


def infer_workflow_run_command_module(
    command_tokens: Sequence[str],
) -> str:
    """Infer one module token from workflow-run command tokens."""
    if not command_tokens:
        return "unknown"
    head = Path(str(command_tokens[0])).name.strip().lower()
    if not head:
        return "unknown"
    if head in {"python", "python3", "python3.exe", "python.exe", "py"}:
        for index, token in enumerate(command_tokens):
            if str(token).strip() == "-m" and index + 1 < len(command_tokens):
                module_token = str(command_tokens[index + 1]).strip().lower()
                if module_token:
                    return module_token
        return "python"
    if head.endswith(".py"):
        stem = Path(head).stem.strip().lower()
        return stem or "python-script"
    return head


def infer_workflow_run_command_group(
    *,
    command_tokens: Sequence[str],
    module_name: str,
) -> str:
    """Infer one coarse command group for workflow-run profiling."""
    module_token = str(module_name).strip().lower()
    if module_token in {"pytest", "unittest"}:
        return module_token
    if module_token.startswith("pytest"):
        return "pytest"
    if module_token.startswith("unittest"):
        return "unittest"
    if module_token.startswith("tests"):
        return "tests-module"
    if command_tokens:
        head = Path(str(command_tokens[0])).name.strip().lower()
        if head in {"python", "python3", "python3.exe", "python.exe", "py"}:
            return "python-command"
    return "external-command"


def _aggregate_duration_rows(
    command_rows: Sequence[Mapping[str, Any]],
    *,
    key_name: str,
) -> list[dict[str, Any]]:
    """Aggregate duration totals/counts for one profile dimension."""
    buckets: dict[str, dict[str, Any]] = {}
    for row in command_rows:
        raw_key = str(row.get(key_name, "")).strip()
        bucket_key = raw_key or "unknown"
        duration = _coerce_duration_seconds(row.get("duration_seconds", 0.0))
        bucket = buckets.setdefault(
            bucket_key,
            {
                key_name: bucket_key,
                "duration_seconds": 0.0,
                "commands": 0,
            },
        )
        bucket["duration_seconds"] = round(
            float(bucket["duration_seconds"]) + duration,
            6,
        )
        bucket["commands"] = int(bucket["commands"]) + 1
    return sorted(
        buckets.values(),
        key=lambda item: (
            -float(item.get("duration_seconds", 0.0)),
            str(item.get(key_name, "")),
        ),
    )


def build_workflow_runtime_profile_payload(
    *,
    run_id: str,
    commands: Sequence[tuple[str, Sequence[str]]],
    events: Sequence[Mapping[str, Any]],
    workflow_run_output_mode: str,
    source_field: str,
    started: _dt.datetime,
    finished: _dt.datetime,
) -> dict[str, Any]:
    """Build one informational profiling payload for one workflow run."""
    normalized_commands = [
        (str(raw).strip(), [str(token) for token in command_tokens])
        for raw, command_tokens in commands
    ]
    command_rows: list[dict[str, Any]] = []
    max_rows = max(len(normalized_commands), len(events))
    for index in range(max_rows):
        raw_command = (
            normalized_commands[index][0]
            if index < len(normalized_commands)
            else ""
        )
        configured_tokens = (
            normalized_commands[index][1]
            if index < len(normalized_commands)
            else []
        )
        event_row = events[index] if index < len(events) else {}
        event_tokens = _normalize_command_tokens(event_row.get("command"))
        executed_tokens = event_tokens or configured_tokens
        module_name = infer_workflow_run_command_module(executed_tokens)
        group_name = infer_workflow_run_command_group(
            command_tokens=executed_tokens,
            module_name=module_name,
        )
        rendered_executed = (
            shlex.join(executed_tokens) if executed_tokens else ""
        )
        command_rows.append(
            {
                "index": index + 1,
                "raw_command": raw_command,
                "executed_command": rendered_executed,
                "module": module_name,
                "group": group_name,
                "status": str(event_row.get("status", "")).strip(),
                "duration_seconds": _coerce_duration_seconds(
                    event_row.get("duration_seconds", 0.0)
                ),
                "exit_code": (
                    event_row.get("metadata", {}).get("exit_code")
                    if isinstance(event_row.get("metadata"), Mapping)
                    else None
                ),
            }
        )

    duration_seconds = round(
        max(0.0, (finished - started).total_seconds()),
        3,
    )
    slowest_commands = sorted(
        command_rows,
        key=lambda item: (
            -float(item.get("duration_seconds", 0.0)),
            str(item.get("raw_command", "")),
        ),
    )[:_SLOW_COMMAND_LIMIT]
    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "informational_only": True,
        "run_id": str(run_id).strip().lower(),
        "workflow_run_output_mode": str(workflow_run_output_mode).strip(),
        "workflow_run_source_field": str(source_field).strip(),
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "duration_seconds": duration_seconds,
        "total_configured_commands": len(normalized_commands),
        "recorded_events": len(events),
        "commands": command_rows,
        "group_breakdown": _aggregate_duration_rows(
            command_rows,
            key_name="group",
        ),
        "module_breakdown": _aggregate_duration_rows(
            command_rows,
            key_name="module",
        ),
        "slowest_commands": slowest_commands,
    }


def render_workflow_runtime_profile_text(payload: Mapping[str, Any]) -> str:
    """Render one short human-readable workflow-run profiling report."""
    lines = [
        "Workflow Run Profile (informational)",
        f"Schema Version: {payload.get('schema_version', '')}",
        f"Run Id: {payload.get('run_id', '')}",
        (
            "Workflow Run Output Mode: "
            f"{payload.get('workflow_run_output_mode', '')}"
        ),
        (
            "Workflow Run Source Field: "
            f"{payload.get('workflow_run_source_field', '')}"
        ),
        f"Started At: {payload.get('started_at', '')}",
        f"Finished At: {payload.get('finished_at', '')}",
        f"Duration Seconds: {payload.get('duration_seconds', '')}",
        (
            "Configured Commands: "
            f"{payload.get('total_configured_commands', '')}"
        ),
        f"Recorded Events: {payload.get('recorded_events', '')}",
        "",
        "Group Breakdown:",
    ]
    for row in payload.get("group_breakdown", []):
        if not isinstance(row, Mapping):
            continue
        lines.append(
            "- "
            f"{row.get('group', 'unknown')}: "
            f"duration={row.get('duration_seconds', '')}, "
            f"commands={row.get('commands', '')}"
        )
    lines.append("")
    lines.append("Module Breakdown:")
    for row in payload.get("module_breakdown", []):
        if not isinstance(row, Mapping):
            continue
        lines.append(
            "- "
            f"{row.get('module', 'unknown')}: "
            f"duration={row.get('duration_seconds', '')}, "
            f"commands={row.get('commands', '')}"
        )
    lines.append("")
    lines.append("Slowest Commands:")
    for row in payload.get("slowest_commands", []):
        if not isinstance(row, Mapping):
            continue
        label = (
            str(row.get("raw_command", "")).strip()
            or str(row.get("executed_command", "")).strip()
        )
        lines.append(
            "- "
            f"{label}: "
            f"duration={row.get('duration_seconds', '')}, "
            f"group={row.get('group', '')}, "
            f"module={row.get('module', '')}"
        )
    return "\n".join(lines).rstrip() + "\n"
