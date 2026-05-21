"""Per-run logging substrate for DevCovenant command execution."""

from __future__ import annotations

import datetime as _dt
import json
import re
import shlex
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from devcovenant.core.repository_paths import display_path

RUN_LOG_SCHEMA_VERSION = "1.0"
_RUN_LOG_ROOT_RELATIVE = Path("devcovenant") / "logs"
_LATEST_POINTER_RELATIVE = (
    Path("devcovenant") / "registry" / "runtime" / "latest.json"
)
_COMMAND_SLUG_PATTERN = re.compile(r"[^a-z0-9]+")
_MAX_COMMAND_SLUG_LENGTH = 32
_REDACTED_VALUE = "[REDACTED]"
_SENSITIVE_KEY_TOKENS = frozenset(
    {
        "accesstoken",
        "apikey",
        "apitoken",
        "authorization",
        "authtoken",
        "bearertoken",
        "cookie",
        "credential",
        "credentials",
        "passwd",
        "passphrase",
        "password",
        "privatekey",
        "secret",
        "secrets",
        "token",
    }
)


@dataclass(frozen=True)
class RunLogPaths:
    """Resolved artifact paths for one per-run log context."""

    run_dir: Path
    run_json: Path
    summary_txt: Path
    summary_json: Path
    stdout_log: Path
    stderr_log: Path
    tail_txt: Path


@dataclass
class RunLogContext:
    """Runtime context describing one command-run logging allocation."""

    repo_root: Path
    logs_root: Path
    run_id: str
    command_name: str
    argv: tuple[str, ...]
    cwd: Path
    started_at: str
    gate_session_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    paths: RunLogPaths | None = None

    def require_paths(self) -> RunLogPaths:
        """Return resolved artifact paths or raise for invalid contexts."""
        if self.paths is None:
            raise ValueError("Run log context is missing resolved paths.")
        return self.paths


def resolve_run_logs_root(repo_root: Path) -> Path:
    """Return the canonical per-run log root for one repository."""
    return Path(repo_root) / _RUN_LOG_ROOT_RELATIVE


def latest_run_pointer_path(repo_root: Path) -> Path:
    """Return the path to the lightweight latest-run pointer file."""
    return Path(repo_root) / _LATEST_POINTER_RELATIVE


def create_run_log_context(
    repo_root: Path,
    command_name: str,
    argv: Sequence[str] | None = None,
    *,
    cwd: Path | None = None,
    gate_session_id: str | None = None,
    started_at: _dt.datetime | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> RunLogContext:
    """Allocate one per-run log directory and initialize core artifacts."""
    repo_root = Path(repo_root)
    logs_root = resolve_run_logs_root(repo_root)
    logs_root.mkdir(parents=True, exist_ok=True)
    latest_run_pointer_path(repo_root).parent.mkdir(
        parents=True, exist_ok=True
    )

    started = _coerce_utc_datetime(started_at)
    run_id = _allocate_run_id(
        logs_root=logs_root,
        command_name=command_name,
        argv=argv or (),
        started_at=started,
    )
    run_dir = logs_root / run_id
    run_dir.mkdir(parents=False, exist_ok=False)

    paths = _build_run_log_paths(run_dir)
    for path in (
        paths.stdout_log,
        paths.stderr_log,
        paths.tail_txt,
    ):
        path.touch(exist_ok=True)

    context = RunLogContext(
        repo_root=repo_root,
        logs_root=logs_root,
        run_id=run_id,
        command_name=str(command_name or "").strip() or "command",
        argv=tuple(str(token) for token in (argv or ())),
        cwd=Path(cwd) if cwd is not None else repo_root,
        started_at=_isoformat_utc(started),
        gate_session_id=(
            str(gate_session_id).strip() if gate_session_id else None
        ),
        metadata=dict(metadata or {}),
        paths=paths,
    )
    write_run_summary_text(
        context,
        _default_summary_text(context, status="running", exit_code=None),
    )
    write_run_summary_json(
        context,
        _default_summary_json(context, status="running", exit_code=None),
    )
    _write_run_metadata(
        context,
        status="running",
        exit_code=None,
        finished_at=None,
        metadata_updates=None,
    )
    record_latest_run_pointer(
        context,
        status="running",
        exit_code=None,
        finished_at=None,
    )
    return context


def load_run_log_context(
    repo_root: Path,
    *,
    run_id: str | None = None,
    run_dir: Path | None = None,
) -> RunLogContext:
    """Load an existing run-log context from a previously created folder."""
    repo_root = Path(repo_root)
    logs_root = resolve_run_logs_root(repo_root)
    selected_run_id = str(run_id or "").strip()
    if run_dir is None:
        if not selected_run_id:
            raise ValueError("Either `run_id` or `run_dir` is required.")
        run_dir = logs_root / selected_run_id
    else:
        run_dir = Path(run_dir)
        if not selected_run_id:
            selected_run_id = run_dir.name
    if not run_dir.is_dir():
        raise ValueError(f"Run log directory does not exist: {run_dir}")

    paths = _build_run_log_paths(run_dir)
    payload = _read_run_metadata(paths.run_json)
    command_name = str(payload.get("command_name", "")).strip() or "command"
    argv_raw = payload.get("argv", [])
    argv = tuple(
        str(token)
        for token in (argv_raw if isinstance(argv_raw, list) else [])
    )
    cwd_value = str(payload.get("cwd", "")).strip()
    started_at = str(payload.get("started_at", "")).strip() or _isoformat_utc(
        _utc_now()
    )
    gate_session_id_raw = payload.get("gate_session_id")
    gate_session_id = (
        str(gate_session_id_raw).strip()
        if gate_session_id_raw is not None and str(gate_session_id_raw).strip()
        else None
    )
    metadata_raw = payload.get("metadata", {})
    metadata = dict(metadata_raw) if isinstance(metadata_raw, Mapping) else {}
    return RunLogContext(
        repo_root=repo_root,
        logs_root=logs_root,
        run_id=selected_run_id,
        command_name=command_name,
        argv=argv,
        cwd=Path(cwd_value) if cwd_value else repo_root,
        started_at=started_at,
        gate_session_id=gate_session_id,
        metadata=metadata,
        paths=paths,
    )


def finalize_run_log_context(
    context: RunLogContext,
    *,
    exit_code: int | None,
    status: str | None = None,
    finished_at: _dt.datetime | None = None,
    summary_text: str | None = None,
    summary_data: Mapping[str, Any] | None = None,
    metadata_updates: Mapping[str, Any] | None = None,
) -> None:
    """Finalize run metadata and optional summaries for one context."""
    resolved_status = _resolve_run_status(exit_code, status)
    finished = _coerce_utc_datetime(finished_at)

    if summary_text is not None:
        write_run_summary_text(context, summary_text)
    else:
        write_run_summary_text(
            context,
            _default_summary_text(
                context,
                status=resolved_status,
                exit_code=exit_code,
            ),
        )

    if summary_data is not None:
        write_run_summary_json(context, summary_data)
    else:
        write_run_summary_json(
            context,
            _default_summary_json(
                context,
                status=resolved_status,
                exit_code=exit_code,
                finished_at=finished,
            ),
        )

    _write_run_metadata(
        context,
        status=resolved_status,
        exit_code=exit_code,
        finished_at=finished,
        metadata_updates=metadata_updates,
    )
    record_latest_run_pointer(
        context,
        status=resolved_status,
        exit_code=exit_code,
        finished_at=finished,
    )


def prune_run_log_directories(
    repo_root: Path,
    *,
    keep_last: int,
    preserve_run_id: str | None = None,
) -> list[str]:
    """Prune older run-log directories while preserving the newest runs."""
    try:
        normalized_keep = int(keep_last)
    except (TypeError, ValueError):
        normalized_keep = 0
    if normalized_keep < 0:
        normalized_keep = 0
    if normalized_keep == 0:
        return []

    logs_root = resolve_run_logs_root(Path(repo_root))
    if not logs_root.is_dir():
        return []

    run_dirs = _discover_run_log_directories(logs_root)
    if not run_dirs:
        return []
    run_dirs.sort(key=lambda path: path.name, reverse=True)

    keep_names: set[str] = set()
    for path in run_dirs[:normalized_keep]:
        keep_names.add(path.name)
    preserved = str(preserve_run_id or "").strip()
    if preserved:
        keep_names.add(preserved)

    removed: list[str] = []
    for path in run_dirs:
        if path.name in keep_names:
            continue
        try:
            shutil.rmtree(path)
        except OSError:
            continue
        removed.append(path.name)
    return removed


def append_run_stream_output(
    context: RunLogContext,
    stream_name: str,
    text: str,
) -> None:
    """Append text to one stream log (`stdout` or `stderr`)."""
    paths = context.require_paths()
    token = str(stream_name or "").strip().lower()
    if token == "stdout":
        path = paths.stdout_log
    elif token == "stderr":
        path = paths.stderr_log
    else:
        raise ValueError("Invalid stream_name. Expected `stdout` or `stderr`.")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(str(text))


def write_run_summary_text(context: RunLogContext, text: str) -> None:
    """Write `summary.txt` for one run context."""
    path = context.require_paths().summary_txt
    path.write_text(str(text), encoding="utf-8")


def write_run_summary_json(
    context: RunLogContext, payload: Mapping[str, Any]
) -> None:
    """Write `summary.json` using deterministic JSON serialization."""
    path = context.require_paths().summary_json
    _write_json(path, dict(payload))


def write_run_tail(context: RunLogContext, text: str) -> None:
    """Write or replace the bounded tail helper artifact."""
    path = context.require_paths().tail_txt
    path.write_text(str(text), encoding="utf-8")


def record_latest_run_pointer(
    context: RunLogContext,
    *,
    status: str,
    exit_code: int | None,
    finished_at: _dt.datetime | None,
) -> None:
    """Update the lightweight latest-run pointer for quick lookup."""
    context.logs_root.mkdir(parents=True, exist_ok=True)
    pointer_path = latest_run_pointer_path(context.repo_root)
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    paths = context.require_paths()
    payload = {
        "schema_version": RUN_LOG_SCHEMA_VERSION,
        "run_id": context.run_id,
        "command_name": context.command_name,
        "status": str(status),
        "exit_code": exit_code,
        "started_at": context.started_at,
        "finished_at": (
            _isoformat_utc(_coerce_utc_datetime(finished_at))
            if finished_at is not None
            else None
        ),
        "gate_session_id": context.gate_session_id,
        "run_dir": _repo_relative(context.repo_root, paths.run_dir),
        "summary_txt": _repo_relative(context.repo_root, paths.summary_txt),
        "summary_json": _repo_relative(context.repo_root, paths.summary_json),
        "updated_at": _isoformat_utc(_utc_now()),
    }
    _write_json(pointer_path, payload)


def _build_run_log_paths(run_dir: Path) -> RunLogPaths:
    """Return the standard artifact paths for one run directory."""
    return RunLogPaths(
        run_dir=run_dir,
        run_json=run_dir / "run.json",
        summary_txt=run_dir / "summary.txt",
        summary_json=run_dir / "summary.json",
        stdout_log=run_dir / "stdout.log",
        stderr_log=run_dir / "stderr.log",
        tail_txt=run_dir / "tail.txt",
    )


def _discover_run_log_directories(logs_root: Path) -> list[Path]:
    """Return candidate per-run directories under the logs root."""
    discovered: list[Path] = []
    for entry in logs_root.iterdir():
        if not entry.is_dir():
            continue
        if (entry / "run.json").is_file():
            discovered.append(entry)
    return discovered


def _allocate_run_id(
    *,
    logs_root: Path,
    command_name: str,
    argv: Sequence[str],
    started_at: _dt.datetime,
) -> str:
    """Allocate a unique run identifier using timestamp and command slug."""
    stamp = _run_timestamp_token(started_at)
    slug = _command_slug(command_name, argv)
    base = f"{stamp}-{slug}"
    candidate = base
    suffix = 1
    while (logs_root / candidate).exists():
        suffix += 1
        candidate = f"{base}-{suffix:03d}"
    return candidate


def _command_slug(command_name: str, argv: Sequence[str]) -> str:
    """Normalize a command label into a filesystem-safe slug."""
    label = str(command_name or "").strip()
    if not label and argv:
        label = str(argv[0]).strip()
    lowered = label.lower()
    slug = _COMMAND_SLUG_PATTERN.sub("-", lowered).strip("-")
    if not slug:
        slug = "command"
    return slug[:_MAX_COMMAND_SLUG_LENGTH]


def _run_timestamp_token(value: _dt.datetime) -> str:
    """Return one deterministic UTC timestamp token for run identifiers."""
    utc_value = _coerce_utc_datetime(value)
    return utc_value.strftime("%Y%m%dT%H%M%S%fZ")


def _write_run_metadata(
    context: RunLogContext,
    *,
    status: str,
    exit_code: int | None,
    finished_at: _dt.datetime | None,
    metadata_updates: Mapping[str, Any] | None,
) -> None:
    """Write `run.json` with deterministic metadata for one run."""
    effective_metadata = dict(context.metadata)
    if metadata_updates:
        effective_metadata.update(_json_safe(dict(metadata_updates)))
    persisted_argv = _redact_argv(context.argv)
    persisted_metadata = _redact_metadata(effective_metadata)
    invoked_python = str(effective_metadata.get("invoked_python", "")).strip()
    effective_python = str(
        effective_metadata.get("effective_python", "")
    ).strip()
    managed_active_raw = effective_metadata.get("managed_environment_active")
    managed_reexec_raw = effective_metadata.get("managed_reexec_applied")
    payload: dict[str, Any] = {
        "schema_version": RUN_LOG_SCHEMA_VERSION,
        "run_id": context.run_id,
        "command_name": context.command_name,
        "argv": persisted_argv,
        "cwd": str(context.cwd),
        "repo_root": str(context.repo_root),
        "started_at": context.started_at,
        "finished_at": (
            _isoformat_utc(_coerce_utc_datetime(finished_at))
            if finished_at is not None
            else None
        ),
        "status": str(status),
        "exit_code": exit_code,
        "gate_session_id": context.gate_session_id,
        "invoked_python": invoked_python or None,
        "effective_python": effective_python or None,
        "managed_environment_active": (
            bool(managed_active_raw)
            if isinstance(managed_active_raw, bool)
            else None
        ),
        "managed_reexec_applied": (
            bool(managed_reexec_raw)
            if isinstance(managed_reexec_raw, bool)
            else None
        ),
        "run_dir": _repo_relative(
            context.repo_root,
            context.require_paths().run_dir,
        ),
        "artifacts": _artifact_map(context),
        "metadata": persisted_metadata,
    }
    _write_json(context.require_paths().run_json, payload)


def _artifact_map(context: RunLogContext) -> dict[str, str]:
    """Return repo-relative artifact paths for serialized run metadata."""
    paths = context.require_paths()
    return {
        "run_json": _repo_relative(context.repo_root, paths.run_json),
        "summary_txt": _repo_relative(context.repo_root, paths.summary_txt),
        "summary_json": _repo_relative(context.repo_root, paths.summary_json),
        "stdout_log": _repo_relative(context.repo_root, paths.stdout_log),
        "stderr_log": _repo_relative(context.repo_root, paths.stderr_log),
        "tail_txt": _repo_relative(context.repo_root, paths.tail_txt),
    }


def _repo_relative(repo_root: Path, path: Path) -> str:
    """Return a repo-safe display path for run-log artifacts."""
    return display_path(path, repo_root=repo_root)


def _default_summary_text(
    context: RunLogContext,
    *,
    status: str,
    exit_code: int | None,
) -> str:
    """Build a deterministic summary text placeholder or final summary."""
    lines = [
        f"Run ID: {context.run_id}",
        f"Command: {context.command_name}",
        f"Status: {status}",
        f"Exit Code: {'' if exit_code is None else exit_code}",
        "Run Dir: "
        + _repo_relative(
            context.repo_root,
            context.require_paths().run_dir,
        ),
    ]
    return "\n".join(lines) + "\n"


def _default_summary_json(
    context: RunLogContext,
    *,
    status: str,
    exit_code: int | None,
    finished_at: _dt.datetime | None = None,
) -> dict[str, Any]:
    """Build a deterministic summary JSON payload for one run."""
    return {
        "schema_version": RUN_LOG_SCHEMA_VERSION,
        "run_id": context.run_id,
        "command_name": context.command_name,
        "status": status,
        "exit_code": exit_code,
        "started_at": context.started_at,
        "finished_at": (
            _isoformat_utc(_coerce_utc_datetime(finished_at))
            if finished_at is not None
            else None
        ),
        "artifacts": {
            "stdout_log": _repo_relative(
                context.repo_root, context.require_paths().stdout_log
            ),
            "stderr_log": _repo_relative(
                context.repo_root, context.require_paths().stderr_log
            ),
            "tail_txt": _repo_relative(
                context.repo_root, context.require_paths().tail_txt
            ),
        },
    }


def _resolve_run_status(exit_code: int | None, status: str | None) -> str:
    """Return finalized run status from explicit status or exit code."""
    token = str(status or "").strip().lower()
    if token:
        return token
    if exit_code is None:
        return "unknown"
    return "success" if exit_code == 0 else "failure"


def _utc_now() -> _dt.datetime:
    """Return the current UTC timestamp as an aware datetime."""
    return _dt.datetime.now(_dt.timezone.utc)


def _coerce_utc_datetime(value: _dt.datetime | None) -> _dt.datetime:
    """Return a timezone-aware UTC datetime."""
    if value is None:
        return _utc_now()
    if value.tzinfo is None:
        return value.replace(tzinfo=_dt.timezone.utc)
    return value.astimezone(_dt.timezone.utc)


def _isoformat_utc(value: _dt.datetime) -> str:
    """Return UTC ISO-8601 text for one datetime value."""
    return _coerce_utc_datetime(value).isoformat()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON deterministically with UTF-8 encoding."""
    normalized = _json_safe(dict(payload))
    path.write_text(
        json.dumps(normalized, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_run_metadata(path: Path) -> dict[str, Any]:
    """Read `run.json` when present and return a mapping payload."""
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _redact_argv(argv: Sequence[str]) -> list[str]:
    """Redact obvious secret-bearing CLI argument values for persisted logs."""
    redacted: list[str] = []
    redact_next = False
    for token in argv:
        text = str(token)
        if redact_next:
            redacted.append(_REDACTED_VALUE)
            redact_next = False
            continue
        if "=" in text:
            name, value = text.split("=", 1)
            if value and _looks_sensitive_name(name):
                redacted.append(f"{name}={_REDACTED_VALUE}")
                continue
        if _looks_sensitive_name(text):
            redacted.append(text)
            redact_next = True
            continue
        redacted.append(text)
    return redacted


def _redact_metadata(value: Any, *, key_name: str | None = None) -> Any:
    """Redact obvious secret-bearing metadata fields for persisted logs."""
    if key_name is not None and _looks_sensitive_name(key_name):
        return _REDACTED_VALUE
    if isinstance(value, Mapping):
        return {
            str(key): _redact_metadata(item, key_name=str(key))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_metadata(item) for item in value]
    return value


def _looks_sensitive_name(token: str) -> bool:
    """Return whether one CLI or metadata token looks secret-bearing."""
    normalized = _normalize_sensitive_name(token)
    if not normalized:
        return False
    if normalized in _SENSITIVE_KEY_TOKENS:
        return True
    return any(
        normalized.endswith(suffix) or normalized.startswith(suffix)
        for suffix in _SENSITIVE_KEY_TOKENS
    )


def _normalize_sensitive_name(token: str) -> str:
    """Normalize one metadata or CLI key name for secret matching."""
    cleaned = str(token).strip().lower().lstrip("-")
    return re.sub(r"[^a-z0-9]+", "", cleaned)


def _json_safe(value: Any) -> Any:
    """Convert values into JSON-serializable primitives deterministically."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, _dt.datetime):
        return _isoformat_utc(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    return str(value)


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
