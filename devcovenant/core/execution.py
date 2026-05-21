"""Execution helpers for command entrypoints and test orchestration."""

from __future__ import annotations

import argparse
import datetime as _dt
import errno
import hashlib
import importlib
import json
import os
import re
import select
import shlex
import shutil
import subprocess  # nosec B404
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence, TextIO

import yaml

from devcovenant.core.repository_paths import display_path

try:
    import pty
except ImportError:  # pragma: no cover - non-POSIX runtimes
    pty = None  # type: ignore[assignment]

import devcovenant.core.cli_support as cli_args_module
import devcovenant.core.cli_support as output_runtime_module
from devcovenant import __version__ as package_version


class _LazyModuleProxy:
    """Lazily import one runtime module on first attribute access."""

    def __init__(self, module_path: str) -> None:
        """Store the fully qualified import path for deferred loading."""
        self._module_path = module_path
        self._module: Any | None = None

    def _load(self) -> Any:
        """Return the imported module, loading it once on demand."""
        if self._module is None:
            self._module = importlib.import_module(self._module_path)
        return self._module

    def __getattr__(self, name: str) -> Any:
        """Resolve attribute access through the lazily imported module."""
        return getattr(self._load(), name)


workflow_contract_module = _LazyModuleProxy(
    "devcovenant.core.workflow_support"
)
event_runtime_module = _LazyModuleProxy("devcovenant.core.run_events")
registry_runtime_module = _LazyModuleProxy("devcovenant.core.workflow_support")
run_logging_runtime_module = _LazyModuleProxy("devcovenant.core.run_logs")
session_snapshot_runtime_module = _LazyModuleProxy(
    "devcovenant.core.gate_runtime"
)
workflow_session_runtime_module = _LazyModuleProxy(
    "devcovenant.core.gate_runtime"
)
workflow_profile_runtime_module = _LazyModuleProxy("devcovenant.core.run_logs")
yaml_cache_service = _LazyModuleProxy("devcovenant.core.repository_paths")

OutputMode = output_runtime_module.OutputMode
DevCovenantArgumentParser = cli_args_module.DevCovenantArgumentParser
add_output_mode_override_arguments = (
    cli_args_module.add_output_mode_override_arguments
)
output_mode_override_from_namespace = (
    cli_args_module.output_mode_override_from_namespace
)
resolve_cli_output_mode_override = (
    cli_args_module.resolve_cli_output_mode_override
)
strip_leading_cli_output_mode_overrides = (
    cli_args_module.strip_leading_cli_output_mode_overrides
)


def apply_output_mode_override_from_namespace(
    namespace: argparse.Namespace | object | None,
) -> OutputMode | None:
    """Apply one parsed per-invocation output-mode override, if present."""
    override = output_mode_override_from_namespace(namespace)
    if override is not None:
        configure_output_mode(override)
    return override


def build_command_parser(
    command_name: str,
    description: str,
) -> argparse.ArgumentParser:
    """Build a command-scoped parser with stable root-command usage text."""
    return cli_args_module.build_command_parser(command_name, description)


ChildOutputChannel = output_runtime_module.ChildOutputChannel
_OUTPUT_MODE_DEFAULT: OutputMode = output_runtime_module.OUTPUT_MODE_DEFAULT
_MANAGED_ENV_POLICY_ID = "managed-environment"
_MANAGED_ENV_ACTION_RESOLVE_STAGE = "resolve-stage"
_WORKFLOW_RUN_COMMAND_OUTPUT_MODE: OutputMode | None = None
_WORKFLOW_RUN_COMMAND_LABEL = ""
_PYCACHE_PREFIX_ENABLED = False
_PYCACHE_PREFIX_VALUE: str | None = None
_LOGS_KEEP_LAST_DEFAULT = 0
_LOGS_KEEP_LAST = _LOGS_KEEP_LAST_DEFAULT
_RUN_LOG_TAIL_MAX_LINES = 160
_RUN_LOG_TAIL_MAX_CHARS = 12000
_ACTIVE_RUN_LOG_CONTEXT: run_logging_runtime_module.RunLogContext | None = None
_ACTIVE_RUN_TAIL_LINES: list[str] = []
_ACTIVE_RUN_LOG_POINTER_EMITTED = False
_TOP_LEVEL_COMMAND_ENV = "DEVCOV_TOP_COMMAND"
_PACKAGE_VERSION_CACHE: str | None = None
_WAIT_PROGRESS_MESSAGE = output_runtime_module.WAIT_PROGRESS_MESSAGE
_WAIT_PROGRESS_INITIAL_SECONDS = 15.0
_WAIT_PROGRESS_REPEAT_SECONDS = 60.0
_PTY_EOF_EXIT_WAIT_SECONDS = 0.1


def _normalize_output_mode(raw_value: str | None) -> OutputMode:
    """Normalize an output mode token to one of the allowed runtime modes."""
    return output_runtime_module.normalize_output_mode(
        raw_value,
        default=_OUTPUT_MODE_DEFAULT,
    )


class Reporter(Protocol):
    """Output boundary contract for user-visible runtime messages."""

    mode: OutputMode

    def emit(
        self,
        message: str,
        *,
        stream: TextIO | None = None,
        end: str = "\n",
        flush: bool = False,
        verbose_only: bool = False,
    ) -> None:
        """Emit one message through the configured output boundary."""

    def banner(self, title: str, emoji: str) -> None:
        """Emit a stage banner message."""

    def step(
        self, message: str, emoji: str = "•", *, verbose_only: bool = False
    ) -> None:
        """Emit a short status step."""


class ConsoleReporter:
    """Console output adapter implementing the runtime Reporter contract."""

    def __init__(self, mode: OutputMode) -> None:
        """Initialize reporter with one deterministic output mode."""
        self.mode = mode

    def emit(
        self,
        message: str,
        *,
        stream: TextIO | None = None,
        end: str = "\n",
        flush: bool = False,
        verbose_only: bool = False,
    ) -> None:
        """Write one message to stdout/stderr with mode-aware filtering."""
        target = stream if stream is not None else sys.stdout
        if self.mode == "quiet" and target is sys.stdout:
            return
        if verbose_only and self.mode != "verbose":
            return
        target.write(f"{message}{end}")
        # Line-flush console output by default so normal/verbose status lines
        # remain visible during long-running commands without waiting for
        # process exit or large buffer fills.
        if flush or target in {sys.stdout, sys.stderr}:
            target.flush()

    def banner(self, title: str, emoji: str) -> None:
        """Emit a decorative stage banner in verbose mode only."""
        self.emit("\n" + "=" * 70, verbose_only=True)
        self.emit(f"{emoji} {title}", verbose_only=True)
        self.emit("=" * 70, verbose_only=True)

    def step(
        self, message: str, emoji: str = "•", *, verbose_only: bool = False
    ) -> None:
        """Emit a one-line status message."""
        self.emit(f"{emoji} {message}", verbose_only=verbose_only)


_OUTPUT_MODE: OutputMode = _OUTPUT_MODE_DEFAULT
_REPORTER: Reporter = ConsoleReporter(_OUTPUT_MODE)


def set_active_run_log_context(
    context: run_logging_runtime_module.RunLogContext | None,
) -> None:
    """Activate one per-run log context for runtime output capture."""
    global _ACTIVE_RUN_LOG_CONTEXT, _ACTIVE_RUN_TAIL_LINES
    global _ACTIVE_RUN_LOG_POINTER_EMITTED
    _ACTIVE_RUN_LOG_CONTEXT = context
    _ACTIVE_RUN_TAIL_LINES = []
    _ACTIVE_RUN_LOG_POINTER_EMITTED = False


def get_active_run_log_context() -> (
    run_logging_runtime_module.RunLogContext | None
):
    """Return the active per-run log context, if any."""
    return _ACTIVE_RUN_LOG_CONTEXT


def clear_active_run_log_context() -> None:
    """Clear active per-run log capture state."""
    set_active_run_log_context(None)


def merge_active_run_log_metadata(updates: Mapping[str, Any]) -> None:
    """Merge metadata into the active run context when one is present."""
    context = _ACTIVE_RUN_LOG_CONTEXT
    if context is None:
        return
    context.metadata.update(dict(updates))


def merge_active_run_phase_timings(
    command_name: str,
    phase_timings: Sequence[Mapping[str, Any]],
) -> None:
    """Merge per-command phase timing rows into the active run metadata."""
    context = _ACTIVE_RUN_LOG_CONTEXT
    if context is None:
        return
    normalized_name = str(command_name or "").strip()
    if not normalized_name:
        return
    normalized_rows = [
        dict(row)
        for row in phase_timings
        if isinstance(row, Mapping) and dict(row)
    ]
    if not normalized_rows:
        return
    existing = context.metadata.get("phase_timings")
    merged = dict(existing) if isinstance(existing, Mapping) else {}
    merged[normalized_name] = normalized_rows
    context.metadata["phase_timings"] = merged


def append_active_run_log_output(stream_name: str, text: str) -> None:
    """Append captured output text into the active run-log artifacts."""
    context = _ACTIVE_RUN_LOG_CONTEXT
    payload = str(text)
    if context is None or not payload:
        return
    try:
        run_logging_runtime_module.append_run_stream_output(
            context,
            stream_name,
            payload,
        )
        _record_active_run_tail_text(payload)
    except (OSError, ValueError):
        return


def emit_active_run_log_pointer(
    *,
    file: TextIO | None = None,
    verbose_only: bool = False,
    once: bool = False,
) -> str | None:
    """Emit a deterministic pointer to the active run folder and summaries."""
    global _ACTIVE_RUN_LOG_POINTER_EMITTED
    context = _ACTIVE_RUN_LOG_CONTEXT
    if context is None:
        return None
    paths = context.require_paths()
    run_dir = _run_log_repo_relative(context.repo_root, paths.run_dir)
    summary_txt = _run_log_repo_relative(context.repo_root, paths.summary_txt)
    summary_json = _run_log_repo_relative(
        context.repo_root,
        paths.summary_json,
    )
    message = (
        "Run logs: "
        f"{run_dir} (summary: {summary_txt}, summary.json: {summary_json})"
    )
    if once and _ACTIVE_RUN_LOG_POINTER_EMITTED:
        return message
    runtime_print(message, file=file, verbose_only=verbose_only)
    _ACTIVE_RUN_LOG_POINTER_EMITTED = True
    return message


def finalize_active_run_log_context(
    *,
    exit_code: int | None,
    status: str | None = None,
    metadata_updates: Mapping[str, Any] | None = None,
) -> run_logging_runtime_module.RunLogContext | None:
    """Finalize the active run-log context and write summary artifacts."""
    context = _ACTIVE_RUN_LOG_CONTEXT
    if context is None:
        return None
    tail_text = _active_run_tail_text()
    try:
        run_logging_runtime_module.write_run_tail(context, tail_text)
        run_logging_runtime_module.finalize_run_log_context(
            context,
            exit_code=exit_code,
            status=status,
            summary_text=_build_active_run_summary_text(
                context,
                status=status,
                exit_code=exit_code,
            ),
            summary_data=_build_active_run_summary_json(
                context,
                status=status,
                exit_code=exit_code,
            ),
            metadata_updates=metadata_updates,
        )
        run_logging_runtime_module.prune_run_log_directories(
            context.repo_root,
            keep_last=get_logs_keep_last(),
            preserve_run_id=context.run_id,
        )
    except OSError:
        return context
    return context


def _build_active_run_summary_text(
    context: run_logging_runtime_module.RunLogContext,
    *,
    status: str | None,
    exit_code: int | None,
) -> str:
    """Build a generic command-run summary text for CLI-dispatched runs."""
    final_status = _resolve_run_log_status(exit_code, status)
    paths = context.require_paths()
    lines = [
        f"Run ID: {context.run_id}",
        f"Command: {context.command_name}",
        "Argv: " + (" ".join(context.argv) if context.argv else ""),
        f"Status: {final_status}",
        f"Exit Code: {'' if exit_code is None else exit_code}",
        "Run Dir: " + _run_log_repo_relative(context.repo_root, paths.run_dir),
        "stdout.log: "
        + _run_log_repo_relative(context.repo_root, paths.stdout_log),
        "stderr.log: "
        + _run_log_repo_relative(context.repo_root, paths.stderr_log),
        "tail.txt: "
        + _run_log_repo_relative(context.repo_root, paths.tail_txt),
    ]
    workflow_run_summary = context.metadata.get("workflow_run_summary")
    if isinstance(workflow_run_summary, Mapping):
        run_id = str(workflow_run_summary.get("run_id", "")).strip()
        if run_id:
            lines.append(f"Workflow Run: {run_id}")
        mode = str(
            workflow_run_summary.get("workflow_run_output_mode", "")
        ).strip()
        if mode:
            lines.append(f"Workflow Run Output Mode: {mode}")
        total = workflow_run_summary.get("total_commands")
        passed = workflow_run_summary.get("passed_commands")
        failed = workflow_run_summary.get("failed_commands")
        if any(value is not None for value in (total, passed, failed)):
            lines.append(
                "Run Commands: "
                f"total={'' if total is None else total}, "
                f"passed={'' if passed is None else passed}, "
                f"failed={'' if failed is None else failed}"
            )
        duration_seconds = workflow_run_summary.get("duration_seconds")
        if duration_seconds is not None:
            lines.append(f"Duration Seconds: {duration_seconds}")
        min_command = workflow_run_summary.get("duration_seconds_min_command")
        avg_command = workflow_run_summary.get("duration_seconds_avg_command")
        max_command = workflow_run_summary.get("duration_seconds_max_command")
        duration_events = workflow_run_summary.get("duration_events_count")
        if any(
            value is not None
            for value in (min_command, avg_command, max_command)
        ):
            lines.append(
                "Run Command Duration Seconds: "
                f"min={'' if min_command is None else min_command}, "
                f"avg={'' if avg_command is None else avg_command}, "
                f"max={'' if max_command is None else max_command}, "
                f"events={'' if duration_events is None else duration_events}"
            )
        first_failed = str(
            workflow_run_summary.get("first_failed_command", "")
        ).strip()
        if first_failed:
            lines.append(f"First Failed Run Command: {first_failed}")
        failure_hint = str(
            workflow_run_summary.get("failure_hint", "")
        ).strip()
        if failure_hint:
            lines.append(f"Failure Hint: {failure_hint}")
    workflow_profile_artifacts = context.metadata.get(
        "workflow_profile_artifacts"
    )
    if isinstance(workflow_profile_artifacts, Mapping):
        profile_txt = str(
            workflow_profile_artifacts.get("workflow_profile_txt", "")
        ).strip()
        profile_json = str(
            workflow_profile_artifacts.get("workflow_profile_json", "")
        ).strip()
        if profile_txt:
            lines.append(f"Workflow Profile txt: {profile_txt}")
        if profile_json:
            lines.append(f"Workflow Profile json: {profile_json}")
    clean_summary = context.metadata.get("clean_summary")
    if isinstance(clean_summary, Mapping):
        scopes = clean_summary.get("selected_scopes")
        if isinstance(scopes, Sequence) and not isinstance(scopes, str):
            scope_text = ", ".join(
                str(item).strip() for item in scopes if str(item).strip()
            )
            if scope_text:
                lines.append(f"Cleanup Scope: {scope_text}")
        removed_count = clean_summary.get("removed_count")
        if removed_count is not None:
            lines.append(f"Removed Targets: {removed_count}")
        skipped_count = clean_summary.get("skipped_protected_count")
        if skipped_count is not None:
            lines.append(f"Skipped Protected Targets: {skipped_count}")
        skipped_matches = clean_summary.get("skipped_protected_match_count")
        if skipped_matches is not None:
            lines.append(f"Skipped Protected Matches: {skipped_matches}")
        skipped_paths = clean_summary.get("skipped_protected_paths")
        if isinstance(skipped_paths, Sequence) and not isinstance(
            skipped_paths, str
        ):
            skipped_text = ", ".join(
                str(item).strip()
                for item in skipped_paths
                if str(item).strip()
            )
            if skipped_text:
                lines.append(f"Skipped Protected Paths: {skipped_text}")
    phase_timings = context.metadata.get("phase_timings")
    if isinstance(phase_timings, Mapping):
        for command_name, rows in phase_timings.items():
            if not isinstance(rows, Sequence) or isinstance(rows, str):
                continue
            rendered_rows: list[str] = []
            for row in rows:
                if not isinstance(row, Mapping):
                    continue
                phase_name = str(row.get("phase", "")).strip()
                duration = row.get("duration_seconds")
                if not phase_name or duration is None:
                    continue
                extras: list[str] = [
                    f"duration={duration}",
                ]
                if "changed" in row:
                    extras.append(f"changed={bool(row.get('changed'))}")
                if "skipped" in row:
                    extras.append(f"skipped={bool(row.get('skipped'))}")
                rendered_rows.append(f"{phase_name} ({', '.join(extras)})")
            if rendered_rows:
                lines.append(
                    f"Phase Timings [{str(command_name).strip()}]: "
                    + "; ".join(rendered_rows)
                )
    return "\n".join(lines) + "\n"


def _build_active_run_summary_json(
    context: run_logging_runtime_module.RunLogContext,
    *,
    status: str | None,
    exit_code: int | None,
) -> dict[str, Any]:
    """Build a generic command-run summary JSON payload for CLI dispatch."""
    final_status = _resolve_run_log_status(exit_code, status)
    paths = context.require_paths()
    payload = {
        "schema_version": run_logging_runtime_module.RUN_LOG_SCHEMA_VERSION,
        "run_id": context.run_id,
        "command_name": context.command_name,
        "command_family": context.command_name,
        "argv": list(context.argv),
        "status": final_status,
        "exit_code": exit_code,
        "started_at": context.started_at,
        "gate_session_id": context.gate_session_id,
        "artifacts": {
            "run_json": _run_log_repo_relative(
                context.repo_root,
                paths.run_json,
            ),
            "summary_txt": _run_log_repo_relative(
                context.repo_root, paths.summary_txt
            ),
            "summary_json": _run_log_repo_relative(
                context.repo_root, paths.summary_json
            ),
            "stdout_log": _run_log_repo_relative(
                context.repo_root, paths.stdout_log
            ),
            "stderr_log": _run_log_repo_relative(
                context.repo_root, paths.stderr_log
            ),
            "tail_txt": _run_log_repo_relative(
                context.repo_root,
                paths.tail_txt,
            ),
        },
        "metadata": dict(context.metadata),
    }
    clean_summary = context.metadata.get("clean_summary")
    if isinstance(clean_summary, Mapping):
        payload["clean_summary"] = dict(clean_summary)
    phase_timings = context.metadata.get("phase_timings")
    if isinstance(phase_timings, Mapping):
        payload["phase_timings"] = {
            str(command_name): [dict(row) for row in rows]
            for command_name, rows in phase_timings.items()
            if isinstance(rows, Sequence) and not isinstance(rows, str)
        }
    return payload


def _resolve_run_log_status(
    exit_code: int | None,
    status: str | None,
) -> str:
    """Resolve final run-log status token from explicit or exit-code values."""
    token = str(status or "").strip().lower()
    if token:
        return token
    if exit_code is None:
        return "unknown"
    return "success" if int(exit_code) == 0 else "failure"


def _record_active_run_tail_text(text: str) -> None:
    """Keep a bounded in-memory tail for the active run summary artifact."""
    global _ACTIVE_RUN_TAIL_LINES
    if not text:
        return
    _ACTIVE_RUN_TAIL_LINES.extend(text.splitlines(keepends=True))
    if len(_ACTIVE_RUN_TAIL_LINES) > (_RUN_LOG_TAIL_MAX_LINES * 2):
        _ACTIVE_RUN_TAIL_LINES = _ACTIVE_RUN_TAIL_LINES[
            -_RUN_LOG_TAIL_MAX_LINES:
        ]
    while len(_ACTIVE_RUN_TAIL_LINES) > _RUN_LOG_TAIL_MAX_LINES:
        _ACTIVE_RUN_TAIL_LINES.pop(0)
    while (
        sum(len(line) for line in _ACTIVE_RUN_TAIL_LINES)
        > _RUN_LOG_TAIL_MAX_CHARS
    ):
        if not _ACTIVE_RUN_TAIL_LINES:
            break
        _ACTIVE_RUN_TAIL_LINES.pop(0)


def _active_run_tail_text() -> str:
    """Return bounded tail text for the active run context."""
    return "".join(_ACTIVE_RUN_TAIL_LINES)


def _run_log_repo_relative(repo_root: Path, path: Path) -> str:
    """Return repo-relative path when available for runtime log pointers."""
    return display_path(path, repo_root=repo_root)


def configure_output_mode(mode: str | None) -> OutputMode:
    """Configure global output mode for this process runtime."""
    global _OUTPUT_MODE, _REPORTER
    normalized = (
        _normalize_output_mode(mode)
        if mode is not None
        else _OUTPUT_MODE_DEFAULT
    )
    _OUTPUT_MODE = normalized
    _REPORTER = ConsoleReporter(normalized)
    return normalized


def get_output_mode() -> OutputMode:
    """Return active runtime output mode."""
    return _OUTPUT_MODE


def configure_logs_keep_last(value: object | None) -> int:
    """Configure run-log retention (`0` keeps all run folders)."""
    global _LOGS_KEEP_LAST
    _LOGS_KEEP_LAST = _normalize_logs_keep_last(value)
    return _LOGS_KEEP_LAST


def get_logs_keep_last() -> int:
    """Return run-log retention count (`0` means unlimited retention)."""
    return _LOGS_KEEP_LAST


def _read_repo_config(repo_root: Path) -> dict[str, Any]:
    """Read full repo config mapping from `devcovenant/config.yaml`."""
    config_path = repo_root / "devcovenant" / "config.yaml"
    if not config_path.exists():
        return {}
    try:
        payload = yaml_cache_service.load_yaml(config_path)
    except (OSError, yaml.YAMLError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _read_engine_config(repo_root: Path) -> dict[str, Any]:
    """Read `engine` config mapping from repo config when available."""

    payload = _read_repo_config(repo_root)
    engine_cfg = payload.get("engine")
    if not isinstance(engine_cfg, dict):
        return {}
    return engine_cfg


def _read_config_value_by_path(
    repo_root: Path,
    path_token: str,
) -> object | None:
    """Resolve one dotted config path against `devcovenant/config.yaml`."""

    normalized_path = str(path_token or "").strip()
    if not normalized_path:
        return None
    current: object = _read_repo_config(repo_root)
    for segment in normalized_path.split("."):
        token = str(segment or "").strip()
        if not token or not isinstance(current, Mapping):
            return None
        current = current.get(token)
    return current


def resolve_engine_auto_fix_enabled(repo_root: Path) -> bool:
    """Resolve gate-managed autofix enablement from repo config."""
    engine_cfg = _read_engine_config(repo_root)
    raw_value = engine_cfg.get("auto_fix_enabled")
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, str):
        token = raw_value.strip().lower()
        if token in {"true", "1", "yes", "on", "enabled"}:
            return True
        if token in {"false", "0", "no", "off", "disabled"}:
            return False
    return False


def _normalize_engine_bool(raw_value: object, *, default: bool) -> bool:
    """Normalize common boolean config tokens to a bool value."""
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, str):
        token = raw_value.strip().lower()
        if token in {"true", "1", "yes", "on", "enabled"}:
            return True
        if token in {"false", "0", "no", "off", "disabled"}:
            return False
    return default


def _read_pycache_prefix_enabled_from_config(repo_root: Path) -> bool:
    """Read explicit `engine.pycache_prefix_enabled` from repo config."""
    engine_cfg = _read_engine_config(repo_root)
    return _normalize_engine_bool(
        engine_cfg.get("pycache_prefix_enabled"),
        default=False,
    )


def _default_repo_pycache_prefix(repo_root: Path) -> str:
    """Return a stable temp pycache root for one repository checkout."""
    try:
        repo_token = str(repo_root.resolve())
    except OSError:
        repo_token = str(repo_root)
    suffix = hashlib.sha256(repo_token.encode("utf-8")).hexdigest()[:12]
    return str(Path(tempfile.gettempdir()) / "devcovenant-pycache" / suffix)


def _read_pycache_prefix_from_config(repo_root: Path) -> str:
    """Read and resolve `engine.pycache_prefix` (empty => auto temp path)."""
    engine_cfg = _read_engine_config(repo_root)
    raw_value = engine_cfg.get("pycache_prefix")
    token = str(raw_value or "").strip()
    if not token:
        return _default_repo_pycache_prefix(repo_root)
    path = Path(token).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return str(path)


def configure_repo_pycache_prefix(repo_root: Path) -> bool:
    """Configure repo-scoped Python bytecode cache routing when enabled."""
    global _PYCACHE_PREFIX_ENABLED, _PYCACHE_PREFIX_VALUE
    if not _read_pycache_prefix_enabled_from_config(repo_root):
        _PYCACHE_PREFIX_ENABLED = False
        _PYCACHE_PREFIX_VALUE = None
        return False
    prefix = _read_pycache_prefix_from_config(repo_root)
    try:
        Path(prefix).mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    _PYCACHE_PREFIX_ENABLED = True
    _PYCACHE_PREFIX_VALUE = prefix
    try:
        setattr(sys, "pycache_prefix", prefix)
    except (AttributeError, TypeError):
        pass
    os.environ["PYTHONPYCACHEPREFIX"] = prefix
    return True


def cleanup_source_checkout_import_cache(
    repo_root: Path,
    *,
    runtime_file: str | Path | None = None,
) -> bool:
    """Remove owned source-tree import caches when earlier runs left drift."""
    module_path = Path(runtime_file or __file__).resolve()
    try:
        package_root = module_path.parents[3]
    except IndexError:
        return False
    if package_root != repo_root.resolve():
        return False
    removed = False
    for cache_root in (
        repo_root / "devcovenant",
        repo_root / "tests" / "devcovenant",
    ):
        if not cache_root.exists():
            continue
        for cache_dir in cache_root.rglob("__pycache__"):
            shutil.rmtree(cache_dir, ignore_errors=True)
            removed = True
        for compiled_file in cache_root.rglob("*.py[co]"):
            try:
                compiled_file.unlink()
            except OSError:
                continue
            removed = True
    return removed


def cleanup_repo_bytecode_artifacts(repo_root: Path) -> bool:
    """Remove repository-local bytecode artifacts when routing is enabled."""
    if not _read_pycache_prefix_enabled_from_config(repo_root):
        return False
    protected_dirs = {
        ".git",
        ".venv",
        ".python",
        "node_modules",
    }
    removed = False

    def _repo_local_pycache_root(raw_value: str | None) -> Path | None:
        """Resolve one repository-local pycache root candidate when it is
        safe."""
        token = str(raw_value or "").strip()
        if not token:
            return None
        path = Path(token).expanduser()
        if not path.is_absolute():
            path = repo_root / path
        try:
            resolved = path.resolve(strict=False)
            repo_resolved = repo_root.resolve(strict=False)
        except OSError:
            return None
        if resolved == repo_resolved:
            return None
        if not resolved.is_relative_to(repo_resolved):
            return None
        if any(
            part in protected_dirs
            for part in resolved.relative_to(repo_resolved).parts
        ):
            return None
        return resolved

    repo_local_prefixes = {
        path
        for path in (
            _repo_local_pycache_root(os.environ.get("PYTHONPYCACHEPREFIX")),
            _repo_local_pycache_root(_PYCACHE_PREFIX_VALUE),
            _repo_local_pycache_root(
                _read_pycache_prefix_from_config(repo_root)
            ),
        )
        if path is not None
    }
    for prefix in sorted(repo_local_prefixes):
        if prefix.is_dir():
            shutil.rmtree(prefix, ignore_errors=True)
            removed = True
        elif prefix.exists():
            try:
                prefix.unlink()
            except OSError:
                continue
            removed = True

    for root, dirs, names in os.walk(repo_root):
        root_path = Path(root)
        dirs[:] = [name for name in dirs if name not in protected_dirs]
        if ".gha-pycache" in dirs:
            gha_cache_dir = root_path / ".gha-pycache"
            shutil.rmtree(gha_cache_dir, ignore_errors=True)
            dirs.remove(".gha-pycache")
            removed = True
        if "__pycache__" in dirs:
            cache_dir = root_path / "__pycache__"
            shutil.rmtree(cache_dir, ignore_errors=True)
            dirs.remove("__pycache__")
            removed = True
        for name in names:
            if not name.endswith((".pyc", ".pyo", ".pyd")):
                continue
            file_path = root_path / name
            try:
                rel_parts = file_path.relative_to(repo_root).parts
            except ValueError:
                continue
            if any(part in protected_dirs for part in rel_parts):
                continue
            try:
                file_path.unlink()
            except OSError:
                continue
            removed = True
    return removed


def _apply_repo_bytecode_env(env: dict[str, str]) -> dict[str, str]:
    """Attach repo runtime env flags for bytecode hygiene and live output."""
    if _PYCACHE_PREFIX_ENABLED and _PYCACHE_PREFIX_VALUE:
        env["PYTHONPYCACHEPREFIX"] = _PYCACHE_PREFIX_VALUE
    # Keep child Python commands unbuffered so line-by-line streaming remains
    # live in both normal and verbose modes.
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _read_output_mode_from_config(repo_root: Path) -> str | None:
    """Read optional `engine.output_mode` from repo config."""
    engine_cfg = _read_engine_config(repo_root)
    token = str(engine_cfg.get("output_mode", "")).strip()
    return token or None


def _normalize_logs_keep_last(raw_value: object) -> int:
    """Normalize `engine.logs_keep_last` to a non-negative integer."""
    if isinstance(raw_value, bool):
        return _LOGS_KEEP_LAST_DEFAULT
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return _LOGS_KEEP_LAST_DEFAULT
    return max(0, value)


def _read_logs_keep_last_from_config(repo_root: Path) -> int:
    """Read `engine.logs_keep_last` from repo config (`0` keeps all)."""
    engine_cfg = _read_engine_config(repo_root)
    return _normalize_logs_keep_last(engine_cfg.get("logs_keep_last"))


def configure_output_mode_from_config(repo_root: Path) -> OutputMode:
    """Configure output mode from `devcovenant/config.yaml`."""
    return configure_output_mode(_read_output_mode_from_config(repo_root))


def configure_logs_keep_last_from_config(repo_root: Path) -> int:
    """Configure run-log retention from `devcovenant/config.yaml`."""
    return configure_logs_keep_last(
        _read_logs_keep_last_from_config(repo_root)
    )


def resolve_workflow_run_output_mode(
    repo_root: Path,
    run: Mapping[str, object] | str,
) -> OutputMode:
    """Resolve console output mode for one declared workflow run."""
    run_payload = (
        dict(run)
        if isinstance(run, Mapping)
        else resolve_declared_workflow_run(repo_root, str(run))
    )
    recording = run_payload.get("recording")
    recording_map = dict(recording) if isinstance(recording, Mapping) else {}
    config_field = str(
        recording_map.get("output_mode_config_field") or ""
    ).strip()
    if config_field:
        return _normalize_output_mode(
            str(_read_config_value_by_path(repo_root, config_field) or "")
        )
    return get_output_mode()


def runtime_print(
    *args: object,
    sep: str = " ",
    end: str = "\n",
    file: TextIO | None = None,
    flush: bool = False,
    verbose_only: bool = False,
) -> None:
    """
    Print via the output boundary with built-in-print-compatible semantics.

    Existing runtime call sites can migrate from direct `print()` usage
    without changing caller-side argument shapes.
    """
    message = sep.join(str(arg) for arg in args)
    stream = file if file is not None else sys.stdout
    if stream is sys.stdout:
        append_active_run_log_output("stdout", f"{message}{end}")
    elif stream is sys.stderr:
        append_active_run_log_output("stderr", f"{message}{end}")
    if stream in {sys.stdout, sys.stderr}:
        _REPORTER.emit(
            message,
            stream=stream,
            end=end,
            flush=flush,
            verbose_only=verbose_only,
        )
        return
    if verbose_only and _OUTPUT_MODE != "verbose":
        return
    stream.write(f"{message}{end}")
    if flush:
        stream.flush()


def print_banner(title: str, emoji: str) -> None:
    """Print a readable stage banner via the output boundary."""
    _REPORTER.banner(title, emoji)


def read_package_version() -> str:
    """Read the packaged DevCovenant version from VERSION."""
    global _PACKAGE_VERSION_CACHE
    if _PACKAGE_VERSION_CACHE is not None:
        return _PACKAGE_VERSION_CACHE
    version_path = Path(__file__).resolve().parents[1] / "VERSION"
    try:
        version_text = version_path.read_text(encoding="utf-8").strip()
    except OSError:
        version_text = ""
    _PACKAGE_VERSION_CACHE = version_text or package_version
    return _PACKAGE_VERSION_CACHE


def devcovenant_banner_title() -> str:
    """Return the canonical top banner title with the active package."""
    return f"DevCovenant {read_package_version()}"


def print_step(message: str, emoji: str = "•") -> None:
    """Print a short, single-line status step via output boundary."""
    _REPORTER.step(message, emoji)


def top_level_command_name() -> str:
    """Return the normalized top-level CLI command name from environment."""
    return str(os.environ.get(_TOP_LEVEL_COMMAND_ENV, "")).strip().lower()


def normal_mode_prefers_live_streaming_for_command(
    command_name: str | None = None,
) -> bool:
    """Return whether normal mode should keep console streaming live."""
    del command_name
    plan = output_runtime_module.resolve_child_output_plan(
        get_output_mode(),
        "generic_child",
    )
    return plan.emit_console


def resolve_child_output_plan_for_channel(
    channel: ChildOutputChannel,
    *,
    output_mode: str | None = None,
) -> output_runtime_module.ChildOutputPlan:
    """Resolve child-command output behavior for one output channel."""
    if output_mode is None:
        effective_mode = get_output_mode()
    else:
        effective_mode = _normalize_output_mode(output_mode)
    return output_runtime_module.resolve_child_output_plan(
        effective_mode,
        channel,
    )


def run_child_command_with_output_policy(
    command: Sequence[str],
    *,
    channel: ChildOutputChannel,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    capture_combined_output: bool = False,
    output_mode: str | None = None,
    heartbeat_initial_seconds: float = _WAIT_PROGRESS_INITIAL_SECONDS,
    heartbeat_repeat_seconds: float = _WAIT_PROGRESS_REPEAT_SECONDS,
    verbose_only_console: bool = False,
) -> tuple[subprocess.CompletedProcess, str]:
    """
    Run one child command through the shared mode-aware output pipeline.

    This is the single policy gateway for child-process streaming behavior.
    Channel + mode resolve console suppression and heartbeat behavior in one
    place before command execution.
    """
    output_plan = resolve_child_output_plan_for_channel(
        channel,
        output_mode=output_mode,
    )
    return run_subprocess_with_runtime_output(
        command,
        env=env,
        cwd=cwd,
        emit_console=output_plan.emit_console,
        capture_combined_output=capture_combined_output,
        heartbeat_message=output_plan.heartbeat_message,
        heartbeat_initial_seconds=heartbeat_initial_seconds,
        heartbeat_repeat_seconds=heartbeat_repeat_seconds,
        verbose_only_console=verbose_only_console,
    )


def run_subprocess_with_runtime_output(
    command: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    emit_console: bool = True,
    capture_combined_output: bool = False,
    heartbeat_message: str | None = None,
    heartbeat_initial_seconds: float = _WAIT_PROGRESS_INITIAL_SECONDS,
    heartbeat_repeat_seconds: float = _WAIT_PROGRESS_REPEAT_SECONDS,
    verbose_only_console: bool = False,
) -> tuple[subprocess.CompletedProcess, str]:
    """
    Run one subprocess with live line handling, log capture, and heartbeat.

    Output is always appended to the active run log context when one exists.
    Console emission is caller-controlled so normal-mode flood suppression can
    hide child output while still providing heartbeat liveness lines.
    """
    if emit_console and pty is not None:
        return _run_subprocess_with_runtime_output_pty(
            command,
            env=env,
            cwd=cwd,
            capture_combined_output=capture_combined_output,
            heartbeat_message=heartbeat_message,
            heartbeat_initial_seconds=heartbeat_initial_seconds,
            heartbeat_repeat_seconds=heartbeat_repeat_seconds,
            verbose_only_console=verbose_only_console,
        )

    return _run_subprocess_with_runtime_output_pipe(
        command,
        env=env,
        cwd=cwd,
        emit_console=emit_console,
        capture_combined_output=capture_combined_output,
        heartbeat_message=heartbeat_message,
        heartbeat_initial_seconds=heartbeat_initial_seconds,
        heartbeat_repeat_seconds=heartbeat_repeat_seconds,
        verbose_only_console=verbose_only_console,
    )


def _emit_subprocess_chunk(
    chunk: str,
    *,
    emit_console: bool,
    verbose_only_console: bool,
) -> None:
    """Route one subprocess output chunk through runtime output/log sinks."""
    if not chunk:
        return
    if emit_console:
        runtime_print(
            chunk,
            end="",
            verbose_only=verbose_only_console,
        )
    else:
        append_active_run_log_output("stdout", chunk)


def _run_subprocess_with_runtime_output_pty(
    command: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    capture_combined_output: bool = False,
    heartbeat_message: str | None = None,
    heartbeat_initial_seconds: float = _WAIT_PROGRESS_INITIAL_SECONDS,
    heartbeat_repeat_seconds: float = _WAIT_PROGRESS_REPEAT_SECONDS,
    verbose_only_console: bool = False,
) -> tuple[subprocess.CompletedProcess, str]:
    """Run one subprocess through a PTY to avoid child-output buffering."""
    if pty is None:
        raise RuntimeError("PTY runtime path is unavailable.")
    command_env = _apply_repo_bytecode_env(dict(env or os.environ))
    command_tokens = [str(token) for token in command]
    master_fd, slave_fd = pty.openpty()
    # Reviewed tokenized child command execution; shell use stays forbidden.
    process = subprocess.Popen(  # nosec B603
        command_tokens,
        stdin=subprocess.DEVNULL,
        stdout=slave_fd,
        stderr=slave_fd,
        env=command_env,
        cwd=cwd,
        bufsize=0,
        close_fds=True,
    )
    combined_chunks: list[str] = []
    os.close(slave_fd)

    heartbeat_token = str(heartbeat_message or "").strip()
    next_heartbeat: float | None = None
    if heartbeat_token:
        next_heartbeat = time.monotonic() + max(
            0.0, float(heartbeat_initial_seconds)
        )

    try:
        while True:
            timeout = 1.0
            now = time.monotonic()
            if next_heartbeat is not None:
                timeout = max(0.01, min(1.0, next_heartbeat - now))
            ready, _, _ = select.select([master_fd], [], [], timeout)
            if not ready:
                if (
                    next_heartbeat is not None
                    and process.poll() is None
                    and time.monotonic() >= next_heartbeat
                ):
                    runtime_print(heartbeat_token)
                    next_heartbeat = time.monotonic() + max(
                        1.0, float(heartbeat_repeat_seconds)
                    )
                if process.poll() is not None:
                    break
                continue

            try:
                chunk_bytes = os.read(master_fd, 4096)
            except OSError as exc:
                if exc.errno == errno.EIO:
                    # Linux PTYs can raise EIO at EOF before poll() reflects
                    # the child exit. Give the process a brief reap window
                    # before treating this as a real runtime failure.
                    try:
                        process.wait(timeout=_PTY_EOF_EXIT_WAIT_SECONDS)
                    except subprocess.TimeoutExpired:
                        if process.poll() is None:
                            continue
                    break
                raise
            if not chunk_bytes:
                if process.poll() is not None:
                    break
                continue
            chunk = chunk_bytes.decode("utf-8", errors="replace")
            if capture_combined_output:
                combined_chunks.append(chunk)
            _emit_subprocess_chunk(
                chunk,
                emit_console=True,
                verbose_only_console=verbose_only_console,
            )
            if next_heartbeat is not None:
                next_heartbeat = time.monotonic() + max(
                    1.0, float(heartbeat_repeat_seconds)
                )
    finally:
        process.wait()
        try:
            os.close(master_fd)
        except OSError:
            pass

    completed = subprocess.CompletedProcess(
        command_tokens,
        int(process.returncode or 0),
    )
    return completed, "".join(combined_chunks)


def _run_subprocess_with_runtime_output_pipe(
    command: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    emit_console: bool = True,
    capture_combined_output: bool = False,
    heartbeat_message: str | None = None,
    heartbeat_initial_seconds: float = _WAIT_PROGRESS_INITIAL_SECONDS,
    heartbeat_repeat_seconds: float = _WAIT_PROGRESS_REPEAT_SECONDS,
    verbose_only_console: bool = False,
) -> tuple[subprocess.CompletedProcess, str]:
    """Run one subprocess through pipes with live queue-based streaming."""
    command_env = _apply_repo_bytecode_env(dict(env or os.environ))
    command_tokens = [str(token) for token in command]
    # Reviewed tokenized child command execution; shell use stays forbidden.
    process = subprocess.Popen(  # nosec B603
        command_tokens,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=False,
        env=command_env,
        cwd=cwd,
        bufsize=0,
    )
    output_stream = process.stdout
    if output_stream is None:
        process.wait()
        completed = subprocess.CompletedProcess(
            command_tokens,
            int(process.returncode or 0),
        )
        return completed, ""
    output_fd = output_stream.fileno()
    combined_chunks: list[str] = []
    heartbeat_token = str(heartbeat_message or "").strip()
    next_heartbeat: float | None = None
    if heartbeat_token:
        next_heartbeat = time.monotonic() + max(
            0.0, float(heartbeat_initial_seconds)
        )

    try:
        while True:
            timeout = 1.0
            now = time.monotonic()
            if next_heartbeat is not None:
                timeout = max(0.01, min(1.0, next_heartbeat - now))
            ready, _, _ = select.select([output_fd], [], [], timeout)
            if not ready:
                if (
                    next_heartbeat is not None
                    and process.poll() is None
                    and time.monotonic() >= next_heartbeat
                ):
                    runtime_print(heartbeat_token)
                    next_heartbeat = time.monotonic() + max(
                        1.0, float(heartbeat_repeat_seconds)
                    )
                if process.poll() is not None:
                    break
                continue

            try:
                chunk_bytes = os.read(output_fd, 4096)
            except OSError as exc:
                if exc.errno in {errno.EIO, errno.EBADF} and (
                    process.poll() is not None
                ):
                    break
                raise
            if not chunk_bytes:
                if process.poll() is not None:
                    break
                continue
            chunk = chunk_bytes.decode("utf-8", errors="replace")
            if capture_combined_output:
                combined_chunks.append(chunk)
            _emit_subprocess_chunk(
                chunk,
                emit_console=emit_console,
                verbose_only_console=verbose_only_console,
            )
            if next_heartbeat is not None and emit_console:
                next_heartbeat = time.monotonic() + max(
                    1.0, float(heartbeat_repeat_seconds)
                )
    finally:
        process.wait()
        try:
            output_stream.close()
        except OSError:
            pass

    completed = subprocess.CompletedProcess(
        command_tokens,
        int(process.returncode or 0),
    )
    return completed, "".join(combined_chunks)


def find_git_root(path: Path) -> Path | None:
    """Return the nearest git root for a path."""
    current = path.resolve()
    for candidate in [current] + list(current.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def resolve_repo_root(*, require_install: bool = False) -> Path:
    """Resolve and validate the current git repository root."""
    repo_root = find_git_root(Path.cwd())
    if repo_root is None:
        raise SystemExit(
            "DevCovenant commands must run inside a git repository."
        )
    if require_install and not (repo_root / "devcovenant").exists():
        raise SystemExit(
            "DevCovenant is not installed in this repository. "
            "Run `devcovenant install` first."
        )
    configure_repo_pycache_prefix(repo_root)
    configure_output_mode_from_config(repo_root)
    return repo_root


def _snapshot_ignored_dirs(repo_root: Path) -> set[str]:
    """Return snapshot ignored directories from defaults plus config."""
    return session_snapshot_runtime_module._snapshot_ignored_dirs(repo_root)


def _snapshot_files(repo_root: Path, ignored_dirs: set[str]) -> list[Path]:
    """Collect snapshot files under the repository root using ignore-dir
    filtering."""
    return session_snapshot_runtime_module._snapshot_files(
        repo_root,
        ignored_dirs,
    )


def _sha256_file(path: Path) -> str:
    """Return SHA-256 digest for one file path."""
    return session_snapshot_runtime_module._sha256_file(path)


def _hash_lines(lines: list[str]) -> str:
    """Return deterministic SHA-256 digest for normalized text lines."""
    return session_snapshot_runtime_module._hash_lines(lines)


def capture_current_numstat_snapshot(repo_root: Path) -> dict[str, str]:
    """Return deterministic filesystem-hash snapshot rows."""
    rows: dict[str, str] = {}
    ignored_dirs = _snapshot_ignored_dirs(repo_root)
    files = _snapshot_files(repo_root, ignored_dirs)
    for file_path in files:
        rel = file_path.relative_to(repo_root).as_posix()
        if rel in session_snapshot_runtime_module._SNAPSHOT_IGNORED_FILES:
            continue
        if any(
            rel == prefix.rstrip("/") or rel.startswith(prefix)
            for prefix in (
                session_snapshot_runtime_module._SNAPSHOT_IGNORED_PREFIXES
            )
        ):
            continue
        digest = _sha256_file(file_path)
        rows[rel] = f"{digest}\t{rel}"
    return rows


def capture_current_snapshot_paths(repo_root: Path) -> list[str]:
    """Return deterministic repo-relative path list from filesystem scan."""
    ignored_dirs = _snapshot_ignored_dirs(repo_root)
    files = _snapshot_files(repo_root, ignored_dirs)
    return [path.relative_to(repo_root).as_posix() for path in files]


def changed_numstat_paths(*args: Any, **kwargs: Any) -> Any:
    """Delegate changed-snapshot path calculation lazily."""
    return session_snapshot_runtime_module.changed_numstat_paths(
        *args,
        **kwargs,
    )


def diff_snapshot_paths(*args: Any, **kwargs: Any) -> Any:
    """Delegate snapshot-path diff calculation lazily."""
    return session_snapshot_runtime_module.diff_snapshot_paths(
        *args,
        **kwargs,
    )


def snapshot_signature(*args: Any, **kwargs: Any) -> Any:
    """Delegate snapshot-signature calculation lazily."""
    return session_snapshot_runtime_module.snapshot_signature(
        *args,
        **kwargs,
    )


def normalize_snapshot_rows(*args: Any, **kwargs: Any) -> Any:
    """Delegate snapshot-row normalization lazily."""
    return session_snapshot_runtime_module.normalize_snapshot_rows(
        *args,
        **kwargs,
    )


def snapshot_row_style(*args: Any, **kwargs: Any) -> Any:
    """Delegate snapshot-row-style detection lazily."""
    return session_snapshot_runtime_module.snapshot_row_style(
        *args,
        **kwargs,
    )


def snapshot_paths_changed_since(repo_root: Path, epoch: float) -> set[str]:
    """Return snapshot paths whose mtime is at or after the given epoch."""
    return session_snapshot_runtime_module.snapshot_paths_changed_since(
        repo_root,
        epoch,
    )


def session_delta_paths(
    repo_root: Path,
    open_snapshot: dict[str, str],
    current_snapshot: dict[str, str],
    *,
    session_open_epoch: float | None = None,
) -> set[str]:
    """Return session delta paths using shared snapshot comparison logic."""
    return session_snapshot_runtime_module.session_delta_paths(
        repo_root,
        open_snapshot,
        current_snapshot,
        session_open_epoch=session_open_epoch,
    )


def capture_agents_section_hashes(*args: Any, **kwargs: Any) -> Any:
    """Delegate AGENTS section hashing lazily."""
    return session_snapshot_runtime_module.capture_agents_section_hashes(
        *args,
        **kwargs,
    )


def document_exemption_fingerprint_for_path(*args: Any, **kwargs: Any) -> Any:
    """Delegate document exemption fingerprint calculation lazily."""
    document_fingerprint = (
        session_snapshot_runtime_module.document_exemption_fingerprint_for_path
    )
    return document_fingerprint(*args, **kwargs)


def capture_document_exemption_baseline(*args: Any, **kwargs: Any) -> Any:
    """Delegate document exemption baseline capture lazily."""
    return session_snapshot_runtime_module.capture_document_exemption_baseline(
        *args,
        **kwargs,
    )


def load_session_snapshot_payload(*args: Any, **kwargs: Any) -> Any:
    """Delegate session-snapshot payload loading lazily."""
    return session_snapshot_runtime_module.load_session_snapshot_payload(
        *args,
        **kwargs,
    )


def merge_session_snapshot_payload(*args: Any, **kwargs: Any) -> Any:
    """Delegate session-snapshot payload merge lazily."""
    return session_snapshot_runtime_module.merge_session_snapshot_payload(
        *args,
        **kwargs,
    )


def prune_inline_session_snapshot_fields(*args: Any, **kwargs: Any) -> Any:
    """Delegate inline session-snapshot pruning lazily."""
    return (
        session_snapshot_runtime_module.prune_inline_session_snapshot_fields(
            *args,
            **kwargs,
        )
    )


def read_local_version(repo_root: Path) -> str | None:
    """Read the local devcovenant version from repo_root."""
    version_path = repo_root / "devcovenant" / "VERSION"
    if not version_path.exists():
        return None
    version_text = version_path.read_text(encoding="utf-8").strip()
    return version_text or None


def warn_version_mismatch(repo_root: Path) -> None:
    """Warn when the local devcovenant version differs from the CLI."""
    local_version = read_local_version(repo_root)
    if not local_version:
        return
    cli_version = read_package_version()
    if local_version != cli_version:
        message = (
            "⚠️  Local DevCovenant version differs from CLI.\n"
            f"   Local: {local_version}\n"
            f"   CLI:   {cli_version}\n"
            "Use the local version via `python3 -m devcovenant` or update."
        )
        runtime_print(message)


def run_bootstrap_registry_refresh(repo_root: Path) -> None:
    """Run lightweight registry refresh for command startup."""
    print_step("Refreshing tracked registry", "🔄")
    from devcovenant.core.refresh_runtime import refresh_policy_registry

    refresh_exit = refresh_policy_registry(repo_root)
    if refresh_exit != 0:
        raise SystemExit("Registry refresh failed.")
    print_step("Registry refresh complete", "✅")


def _run_policy_runtime_action(
    repo_root: Path,
    *,
    policy_id: str,
    action: str,
    payload: dict[str, Any] | None = None,
) -> Any:
    """Run one policy runtime action through engine dispatch."""
    from devcovenant.core.policy_runtime import run_policy_runtime_action

    return run_policy_runtime_action(
        repo_root,
        policy_id=policy_id,
        action=action,
        payload=payload or {},
    )


def resolve_managed_environment_for_stage(
    repo_root: Path,
    stage: str,
    *,
    base_env: Mapping[str, str] | None = None,
) -> tuple[dict[str, str] | None, str | None]:
    """Resolve managed-environment state via policy-owned runtime action."""
    stage_token = str(stage or "").strip().lower()
    payload = {
        "stage": stage_token,
        "base_env": dict(base_env or os.environ),
    }
    result = _run_policy_runtime_action(
        repo_root,
        policy_id=_MANAGED_ENV_POLICY_ID,
        action=_MANAGED_ENV_ACTION_RESOLVE_STAGE,
        payload=payload,
    )
    if not isinstance(result, tuple) or len(result) != 2:
        raise ValueError(
            "managed-environment runtime action returned invalid payload."
        )
    env_raw, managed_python_raw = result
    if env_raw is None and managed_python_raw is None:
        return None, None
    if not isinstance(env_raw, dict):
        raise ValueError(
            "managed-environment runtime returned invalid environment payload."
        )
    if not isinstance(managed_python_raw, str):
        raise ValueError(
            "managed-environment runtime returned invalid interpreter payload."
        )
    return dict(env_raw), managed_python_raw


def _looks_like_python_launcher(token: str) -> bool:
    """Return True when token points to a Python launcher."""
    name = Path(str(token).strip()).name.lower()
    if name in {"py", "py.exe"}:
        return True
    return name.startswith("python")


def rewrite_command_for_managed_python(
    command: Sequence[str],
    managed_python: str | None,
) -> list[str]:
    """Replace command python launcher with managed interpreter path."""
    rewritten = [str(token) for token in command]
    if not rewritten or not managed_python:
        return rewritten
    if not _looks_like_python_launcher(rewritten[0]):
        return rewritten
    rewritten[0] = managed_python
    return rewritten


def rewrite_command_string_for_managed_python(
    command: str,
    managed_python: str | None,
) -> str:
    """Rewrite shell command string with managed Python launcher."""
    if not managed_python:
        return command
    tokens = shlex.split(command)
    rewritten = rewrite_command_for_managed_python(tokens, managed_python)
    return shlex.join(rewritten)


def registry_run_commands(
    repo_root: Path,
    run_id: str = "tests",
) -> list[tuple[str, list[str]]]:
    """Read command-group commands from one declared workflow run."""
    _, commands, _ = resolve_workflow_run_commands(repo_root, run_id)
    return commands


def resolve_workflow_runs(
    repo_root: Path,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Return the resolved workflow contract and enabled runs in order."""
    contract = workflow_contract_module.load_workflow_contract(repo_root)
    runs: list[dict[str, object]] = []
    for run_id in workflow_contract_module.run_ids(contract):
        run = workflow_contract_module.resolve_run(contract, run_id)
        if run is None:
            raise ValueError(
                "Workflow run "
                f"`{run_id}` is missing from the active contract."
            )
        runs.append(run)
    return contract, runs


def resolve_declared_workflow_run(
    repo_root: Path,
    run_id: str,
) -> dict[str, object]:
    """Resolve one declared workflow run from the active contract."""
    contract = workflow_contract_module.load_workflow_contract(repo_root)
    run = workflow_contract_module.resolve_run(contract, run_id)
    if run is None:
        raise ValueError(
            "No "
            f"`{run_id}` workflow run is configured for the active "
            "profiles."
        )
    return run


def _normalize_required_commands(
    raw_commands: object,
    *,
    field_name: str,
) -> list[tuple[str, list[str]]]:
    """Normalize one command metadata field into raw/tokens tuples."""
    if isinstance(raw_commands, str):
        raw_commands = [
            item.strip()
            for item in raw_commands.replace("\n", ",").split(",")
            if item.strip()
        ]
    elif isinstance(raw_commands, list):
        normalized: list[object] = []
        for command_entry in raw_commands:
            if isinstance(command_entry, str):
                normalized.extend(
                    entry.strip()
                    for entry in command_entry.replace("\n", ",").split(",")
                    if entry.strip()
                )
            else:
                normalized.append(command_entry)
        raw_commands = normalized
    else:
        raise ValueError(
            f"Invalid `{field_name}` payload: expected string or list."
        )

    commands: list[tuple[str, list[str]]] = []
    for entry in raw_commands:
        if isinstance(entry, list):
            raw = " ".join(
                str(part).strip() for part in entry if str(part).strip()
            )
        else:
            raw = str(entry).strip()
        if not raw:
            raise ValueError(f"Invalid `{field_name}` command: empty token.")
        tokens = shlex.split(raw)
        if not tokens:
            raise ValueError(f"Invalid `{field_name}` command: `{raw}`.")
        commands.append((raw, tokens))
    return commands


def resolve_workflow_run_commands(
    repo_root: Path,
    run_id: str,
) -> tuple[dict[str, object], list[tuple[str, list[str]]], str]:
    """Resolve one command-group workflow run into runnable commands."""
    run = resolve_declared_workflow_run(repo_root, run_id)
    runner = run.get("runner")
    if not isinstance(runner, Mapping):
        raise ValueError(f"Workflow run `{run_id}` runner is invalid.")
    if str(runner.get("kind") or "").strip().lower() != "command_group":
        raise ValueError(
            f"Workflow run `{run_id}` does not use runner.kind: "
            "command_group."
        )
    source_field = (
        str(run.get("source_field") or "workflow_runs").strip()
        or "workflow_runs"
    )
    commands = _normalize_required_commands(
        runner.get("commands"),
        field_name=f"{source_field}[{run_id}]",
    )
    return run, commands, source_field


def _workflow_run_recording_map(
    run: Mapping[str, object],
) -> dict[str, object]:
    """Return normalized recording metadata for one workflow run."""

    recording = run.get("recording")
    return dict(recording) if isinstance(recording, Mapping) else {}


def _workflow_run_output_mode(
    repo_root: Path,
    run: Mapping[str, object],
) -> OutputMode:
    """Resolve console output mode for one workflow run."""

    return resolve_workflow_run_output_mode(repo_root, run)


def _workflow_run_event_adapter_group(
    run: Mapping[str, object],
) -> str:
    """Return the configured event-adapter group for one run."""

    return str(
        _workflow_run_recording_map(run).get("event_adapter_group") or ""
    ).strip()


def _workflow_run_writes_runtime_profile(
    run: Mapping[str, object],
) -> bool:
    """Return whether one run should emit run-profile artifacts."""

    return bool(_workflow_run_recording_map(run).get("write_runtime_profile"))


def _workflow_run_uses_reporting_hooks(
    run: Mapping[str, object],
) -> bool:
    """Return whether a run declared any richer reporting hooks."""

    recording = _workflow_run_recording_map(run)
    return bool(
        str(recording.get("output_mode_config_field") or "").strip()
        or str(recording.get("event_adapter_group") or "").strip()
        or bool(recording.get("write_runtime_profile"))
    )


def _load_workflow_run_event_manager(
    repo_root: Path,
    run: Mapping[str, object],
) -> event_runtime_module.RunEventManager:
    """Return the configured event manager for one workflow run."""

    adapter_group = _workflow_run_event_adapter_group(run)
    if not adapter_group:
        return event_runtime_module.RunEventManager(())
    # Clear stale warnings from prior calls in this process.
    event_runtime_module.consume_run_event_adapter_warnings()
    adapters = event_runtime_module.load_profile_event_adapters(
        repo_root,
        adapter_group,
    )
    adapter_warnings = (
        event_runtime_module.consume_run_event_adapter_warnings()
    )
    for warning in adapter_warnings:
        runtime_print(
            f"WARNING: workflow run event-adapter load issue: {warning}",
            file=sys.stderr,
        )
    return event_runtime_module.RunEventManager(adapters)


def _parse_runner_target(
    target: str,
    *,
    kind: str,
) -> tuple[str, str]:
    """Split one `policy:action` or `policy:command` target token."""

    owner_id, _, action_id = str(target or "").partition(":")
    policy_id = owner_id.strip()
    resolved = action_id.strip()
    if not policy_id or not resolved:
        raise ValueError(
            f"Workflow runner target `{target}` for `{kind}` must use the "
            "`policy-id:target-id` format."
        )
    return policy_id, resolved


def _manual_attestation_env_key(attestation_key: str) -> str:
    """Return the environment variable used for one manual attestation."""

    normalized = re.sub(
        r"[^A-Za-z0-9]+",
        "_",
        str(attestation_key or "").strip().upper(),
    ).strip("_")
    if not normalized:
        raise ValueError("Manual attestation key is empty.")
    return f"DEVCOV_WORKFLOW_ATTEST_{normalized}"


def _render_runtime_action_result(result: object) -> None:
    """Emit a short human-readable summary for workflow runtime results."""

    if isinstance(result, dict):
        message = str(result.get("message", "")).strip()
        if message:
            runtime_print(message)
        lines = result.get("lines")
        if isinstance(lines, list):
            for entry in lines:
                token = str(entry).rstrip()
                if token:
                    runtime_print(token)
        return
    if result is not None:
        runtime_print(str(result))


def _verify_external_artifact_check(
    repo_root: Path,
    run_id: str,
    success_contract: Mapping[str, object],
) -> None:
    """Validate artifact expectations declared by one workflow run."""

    base_dir_raw = str(success_contract.get("base_dir") or ".").strip() or "."
    base_dir = Path(base_dir_raw)
    if not base_dir.is_absolute():
        base_dir = repo_root / base_dir
    base_dir = base_dir.resolve()

    required_files = success_contract.get("required_files")
    required_files = (
        list(required_files) if isinstance(required_files, list) else []
    )
    required_globs = success_contract.get("required_globs")
    required_globs = (
        list(required_globs) if isinstance(required_globs, list) else []
    )
    forbidden_globs = success_contract.get("forbidden_globs")
    forbidden_globs = (
        list(forbidden_globs) if isinstance(forbidden_globs, list) else []
    )
    minimum_matches = success_contract.get("minimum_matches", 1)
    try:
        minimum_matches_value = int(minimum_matches)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Workflow run `{run_id}` declares an invalid "
            "external_artifact_check minimum_matches value."
        ) from exc

    matched_required = 0
    missing_files: list[str] = []
    for raw_path in required_files:
        token = str(raw_path or "").strip()
        if not token:
            continue
        path = Path(token)
        if not path.is_absolute():
            path = base_dir / path
        if not path.exists():
            missing_files.append(display_path(path, repo_root=repo_root))
            continue
        matched_required += 1

    matched_globs: list[str] = []
    for raw_pattern in required_globs:
        pattern = str(raw_pattern or "").strip()
        if not pattern:
            continue
        matches = sorted(base_dir.glob(pattern))
        matched_globs.extend(
            display_path(match, repo_root=repo_root)
            for match in matches
            if match.exists()
        )

    forbidden_matches: list[str] = []
    for raw_pattern in forbidden_globs:
        pattern = str(raw_pattern or "").strip()
        if not pattern:
            continue
        matches = sorted(base_dir.glob(pattern))
        forbidden_matches.extend(
            display_path(match, repo_root=repo_root)
            for match in matches
            if match.exists()
        )

    if missing_files:
        raise SystemExit(
            "Workflow run "
            f"`{run_id}` failed external artifact verification. Missing "
            f"required files: {', '.join(missing_files)}."
        )
    if forbidden_matches:
        raise SystemExit(
            "Workflow run "
            f"`{run_id}` failed external artifact verification. Forbidden "
            f"artifacts exist: {', '.join(forbidden_matches)}."
        )
    if matched_required + len(matched_globs) < minimum_matches_value:
        raise SystemExit(
            "Workflow run "
            f"`{run_id}` failed external artifact verification. Expected "
            f"at least {minimum_matches_value} artifact matches under "
            f"`{base_dir}`."
        )


def _run_command(
    command: Sequence[str],
    allow_codes: set[int] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess:
    """Execute command and raise when it fails."""
    effective_mode = _WORKFLOW_RUN_COMMAND_OUTPUT_MODE or get_output_mode()
    command_env = _apply_repo_bytecode_env(dict(env or os.environ))
    result, _ = run_child_command_with_output_policy(
        command,
        channel="workflow_child",
        env=command_env,
        cwd=cwd,
        capture_combined_output=False,
        output_mode=effective_mode,
    )
    allowed = allow_codes or {0}
    if result.returncode not in allowed:
        output_plan = resolve_child_output_plan_for_channel(
            "workflow_child",
            output_mode=effective_mode,
        )
        if output_plan.child_output_suppressed:
            rendered = shlex.join([str(token) for token in command])
            runtime_print(
                "Workflow run child command failed while child output is "
                f"suppressed by mode `{effective_mode}` "
                f"(exit {result.returncode}): {rendered}",
                file=sys.stderr,
            )
        raise subprocess.CalledProcessError(result.returncode, command)
    return result


def _parse_commands(command: str) -> list[str]:
    """Return an ordered command list parsed from a shell chain."""
    return [part.strip() for part in command.split("&&") if part.strip()]


def record_gate_status(
    repo_root: Path,
    command: str,
    notes: str = "",
    run_events: Iterable[Mapping[str, Any]] | None = None,
    workflow_run_output_mode: str | None = None,
    workflow_run_source_field: str | None = None,
) -> None:
    """Record gate status payload at the configured runtime evidence path."""
    status_path = registry_runtime_module.gate_status_path(repo_root)
    status_path.parent.mkdir(parents=True, exist_ok=True)

    existing: dict[str, object] = {}
    if status_path.exists():
        try:
            existing = json.loads(status_path.read_text(encoding="utf-8"))
            if not isinstance(existing, dict):
                existing = {}
        except json.JSONDecodeError:
            existing = {}

    now = _dt.datetime.now(tz=_dt.timezone.utc)
    run_snapshot = capture_current_numstat_snapshot(repo_root)
    active_session_id = str(existing.get("session_id", "")).strip()
    normalized_events = [dict(entry) for entry in run_events or ()]
    snapshot_rel_path, _ = merge_session_snapshot_payload(
        repo_root,
        existing,
        updates={"last_run_snapshot": run_snapshot}
        | ({"run_events": normalized_events} if normalized_events else {}),
        remove_keys=() if normalized_events else ("run_events",),
    )
    payload = {
        **existing,
        "last_run_utc": now.isoformat(),
        "last_run_epoch": now.timestamp(),
        "commands": _parse_commands(command),
        "notes": notes.strip(),
        "session_snapshot_file": snapshot_rel_path,
        "session_snapshot_updated_utc": now.isoformat(),
        "session_snapshot_updated_epoch": now.timestamp(),
    }
    if active_session_id:
        payload["last_run_session_id"] = active_session_id
    else:
        payload.pop("last_run_session_id", None)
    if normalized_events:
        payload["run_events_count"] = len(normalized_events)
    else:
        payload.pop("run_events_count", None)
    if workflow_run_output_mode:
        payload["workflow_run_output_mode"] = _normalize_output_mode(
            workflow_run_output_mode
        )
    else:
        payload.pop("workflow_run_output_mode", None)
    token = str(workflow_run_source_field or "").strip()
    if token:
        payload["workflow_run_source_field"] = token
    else:
        payload.pop("workflow_run_source_field", None)
    # Purge legacy gate-status keys instead of carrying them forward.
    payload.pop("sha", None)
    payload.pop("tests_coverage_evidence", None)
    payload.pop("changelog_open_diff_numstat", None)
    payload.pop("changelog_open_exemption_fingerprints", None)
    payload.pop("cache_enabled", None)
    payload.pop("cache_control_env", None)
    payload.pop("last_run", None)
    payload.pop("command", None)
    prune_inline_session_snapshot_fields(payload)
    status_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    commands_text = " && ".join(payload.get("commands") or [])
    runtime_print(
        f"Recorded gate status at {payload['last_run_utc']} "
        f"for commands `{commands_text}`.",
        verbose_only=True,
    )


def record_workflow_run_result(
    repo_root: Path,
    *,
    run_id: str,
    command: str,
    notes: str = "",
    command_name: str = "",
    workflow_run_output_mode: str | None = None,
    workflow_run_source_field: str | None = None,
    run_events: Iterable[Mapping[str, Any]] | None = None,
) -> None:
    """Record one workflow-run result in the runtime workflow session."""

    payload = workflow_session_runtime_module.load_workflow_session(repo_root)
    contract = workflow_contract_module.load_workflow_contract(repo_root)
    run = workflow_contract_module.resolve_run(contract, run_id)
    if run is None:
        raise ValueError(
            "No "
            f"`{run_id}` workflow run is configured for the active "
            "profiles."
        )
    now = _dt.datetime.now(tz=_dt.timezone.utc)
    run_snapshot = capture_current_numstat_snapshot(repo_root)
    snapshot_rel_path, _ = workflow_session_runtime_module.merge_run_snapshot(
        repo_root,
        payload,
        run_id,
        run_snapshot,
    )
    runs = payload.get("runs")
    run_map = dict(runs) if isinstance(runs, dict) else {}
    current_entry = run_map.get(run_id)
    entry = dict(current_entry) if isinstance(current_entry, dict) else {}
    recording = run.get("recording")
    recording_map = dict(recording) if isinstance(recording, Mapping) else {}
    summary_label = (
        str(recording_map.get("summary_label") or run_id).strip() or run_id
    )
    active_session_id = str(payload.get("session_id", "")).strip()
    if str(payload.get("session_state", "")).strip().lower() != "open":
        active_session_id = ""
    event_count = len([dict(event) for event in run_events or ()])
    entry.update(
        {
            "id": run_id,
            "enabled": bool(run.get("enabled")),
            "status": "passed",
            "summary_label": summary_label,
            "runner_kind": str(
                (run.get("runner") or {}).get("kind", "")
            ).strip(),
            "success_contract_kind": str(
                (run.get("success_contract") or {}).get("kind", "")
            ).strip(),
            "last_run_utc": now.isoformat(),
            "last_run_epoch": now.timestamp(),
            "last_run_session_id": active_session_id,
            "commands": _parse_commands(command),
            "command_name": command_name.strip(),
            "notes": notes.strip(),
            "workflow_run_output_mode": (
                _normalize_output_mode(workflow_run_output_mode)
                if workflow_run_output_mode
                else ""
            ),
            "workflow_run_source_field": str(
                workflow_run_source_field or ""
            ).strip(),
            "events_count": event_count,
        }
    )
    entry.pop("last_run", None)
    entry.pop("command", None)
    run_map[run_id] = entry
    payload["schema_version"] = workflow_session_runtime_module.SCHEMA_VERSION
    payload["workflow_contract_schema_version"] = contract.get(
        "schema_version", workflow_contract_module.SCHEMA_VERSION
    )
    payload["run_ids"] = workflow_contract_module.run_ids(contract)
    payload["runs"] = run_map
    payload["session_snapshot_file"] = snapshot_rel_path
    payload["session_snapshot_updated_utc"] = now.isoformat()
    payload["session_snapshot_updated_epoch"] = now.timestamp()
    workflow_session_runtime_module.write_workflow_session(repo_root, payload)
    runtime_print(
        f"Recorded workflow run `{run_id}` at " f"{entry['last_run_utc']}.",
        verbose_only=True,
    )


class _WorkflowCommandProgress:
    """Track workflow commands with sparse deterministic console lines."""

    def __init__(self, total: int, output_mode: OutputMode):
        """Initialize counter state for sparse normal-mode progress lines."""
        self.total = total
        self._count = 0
        self._normal_mode = output_mode == "normal"
        self._current_description = ""
        self._completed_descriptions: list[str] = []

    def __enter__(self):
        """Return self for context-manager parity."""
        return self

    def describe(self, description: str) -> None:
        """Store the current command description for deterministic updates."""
        self._current_description = str(description)

    def start_step(self, description: str) -> None:
        """Emit a deterministic start marker so long runs show liveness."""
        if not self._normal_mode:
            return
        runtime_print(f"▶ [{self._count + 1}/{self.total}] {description}")

    def complete_step(self, description: str) -> None:
        """Advance state and keep normal-mode progress output non-duplicate."""
        self._count += 1
        self._completed_descriptions.append(str(description))
        if self._normal_mode:
            # Normal mode already emitted the start marker and heartbeats.
            # Keep completion silent to avoid duplicate progress lines.
            return

    def fail_step(
        self,
        description: str,
        exit_code: int | None = None,
    ) -> None:
        """Emit a deterministic failure marker in normal mode."""
        if not self._normal_mode:
            return
        code_text = "" if exit_code is None else f" (exit {int(exit_code)})"
        runtime_print(
            f"[{self._count + 1}/{self.total}] "
            f"FAILED: {description}{code_text}"
        )

    def close(self) -> None:
        """Preserve context-manager API; no bar resources are allocated."""
        return None

    def __exit__(self, exc_type, exc, exc_tb):
        """Preserve context-manager semantics."""
        self.close()


def _emit_workflow_runtime_message(
    message: str,
    workflow_run_output_mode: OutputMode,
    *,
    verbose_only: bool = False,
) -> None:
    """Emit one workflow-runtime line according to the run output mode."""
    if verbose_only and workflow_run_output_mode != "verbose":
        return
    runtime_print(message)


def _execute_command_group_workflow_run(
    repo_root: Path,
    *,
    run: Mapping[str, object],
    notes: str,
    command_name: str,
) -> dict[str, object]:
    """Run one command-group workflow run and return result details."""

    global _WORKFLOW_RUN_COMMAND_LABEL
    global _WORKFLOW_RUN_COMMAND_OUTPUT_MODE
    run_id = str(run.get("id") or "").strip().lower()
    summary_label = (
        str(
            ((run.get("recording") or {}).get("summary_label") or run_id)
        ).strip()
        or run_id
    )
    run, commands, source_field = resolve_workflow_run_commands(
        repo_root,
        run_id,
    )
    run_output_mode = _workflow_run_output_mode(repo_root, run)
    try:
        managed_env, managed_python = resolve_managed_environment_for_stage(
            repo_root,
            "run",
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    event_manager = _load_workflow_run_event_manager(repo_root, run)
    reporting_enabled = _workflow_run_uses_reporting_hooks(run)
    writes_runtime_profile = _workflow_run_writes_runtime_profile(run)
    run_started = _dt.datetime.now(tz=_dt.timezone.utc)
    first_failed_command = ""
    first_failed_exit_code: int | None = None
    passed_commands = 0
    failed_commands = 0

    merge_active_run_log_metadata(
        {
            "workflow_run_id": run_id,
            "workflow_run_summary_label": summary_label,
            "workflow_run_source_field": source_field,
            "workflow_run_output_mode": run_output_mode,
            "normal_console_mode": run_output_mode == "normal",
            "quiet_console_mode": run_output_mode == "quiet",
            "full_output_in_logs": True,
            "console_output_policy": (
                "workflow run output mode controls console detail; "
                "run logs retain full child output"
            ),
        }
    )
    if run_output_mode == "normal":
        _emit_workflow_runtime_message(
            "Please wait for workflow run commands to execute. Full output "
            "is available in run logs.",
            run_output_mode,
        )
        emit_active_run_log_pointer(once=True)

    with _WorkflowCommandProgress(
        len(commands),
        output_mode=run_output_mode,
    ) as progress:
        for raw, command in commands:
            command_tokens = rewrite_command_for_managed_python(
                command,
                managed_python,
            )
            command_str = " ".join(command_tokens)
            progress.describe(raw)
            progress.start_step(raw)
            _emit_workflow_runtime_message(
                f"Running: {command_str}",
                run_output_mode,
                verbose_only=True,
            )
            started = _dt.datetime.now(tz=_dt.timezone.utc)
            try:
                run_kwargs: dict[str, Any] = {"allow_codes": {0}}
                if managed_env is not None:
                    run_kwargs["env"] = managed_env
                    run_kwargs["cwd"] = repo_root
                previous_mode = _WORKFLOW_RUN_COMMAND_OUTPUT_MODE
                previous_label = _WORKFLOW_RUN_COMMAND_LABEL
                _WORKFLOW_RUN_COMMAND_OUTPUT_MODE = run_output_mode
                _WORKFLOW_RUN_COMMAND_LABEL = raw
                try:
                    result = _run_command(command_tokens, **run_kwargs)
                finally:
                    _WORKFLOW_RUN_COMMAND_OUTPUT_MODE = previous_mode
                    _WORKFLOW_RUN_COMMAND_LABEL = previous_label
            except subprocess.CalledProcessError as exc:
                finished = _dt.datetime.now(tz=_dt.timezone.utc)
                failed_commands += 1
                if not first_failed_command:
                    first_failed_command = command_str
                    first_failed_exit_code = int(exc.returncode or 1)
                event_manager.record_command(
                    command=command_tokens,
                    command_str=command_str,
                    started=started,
                    finished=finished,
                    exit_code=int(exc.returncode or 1),
                )
                progress.fail_step(raw, int(exc.returncode or 1))
                if reporting_enabled:
                    merge_active_run_log_metadata(
                        _build_workflow_run_metadata_bundle(
                            commands=commands,
                            events=event_manager.events,
                            workflow_run_output_mode=run_output_mode,
                            source_field=source_field,
                            run_id=run_id,
                            write_runtime_profile=writes_runtime_profile,
                            started=run_started,
                            finished=finished,
                            first_failed_command=first_failed_command,
                            first_failed_exit_code=first_failed_exit_code,
                            passed_commands=passed_commands,
                            failed_commands=failed_commands,
                        )
                    )
                raise
            finished = _dt.datetime.now(tz=_dt.timezone.utc)
            passed_commands += 1
            event_manager.record_command(
                command=command_tokens,
                command_str=command_str,
                started=started,
                finished=finished,
                exit_code=result.returncode,
            )
            progress.complete_step(raw)

    success_contract = run.get("success_contract")
    if (
        isinstance(success_contract, Mapping)
        and str(success_contract.get("kind") or "").strip().lower()
        == "external_artifact_check"
    ):
        _verify_external_artifact_check(repo_root, run_id, success_contract)

    if reporting_enabled:
        merge_active_run_log_metadata(
            _build_workflow_run_metadata_bundle(
                commands=commands,
                events=event_manager.events,
                workflow_run_output_mode=run_output_mode,
                source_field=source_field,
                run_id=run_id,
                write_runtime_profile=writes_runtime_profile,
                started=run_started,
                finished=_dt.datetime.now(tz=_dt.timezone.utc),
                first_failed_command=first_failed_command,
                first_failed_exit_code=first_failed_exit_code,
                passed_commands=passed_commands,
                failed_commands=failed_commands,
            )
        )

    return {
        "run_id": run_id,
        "command": " && ".join(raw for raw, _ in commands),
        "notes": notes,
        "command_name": command_name,
        "workflow_run_output_mode": (
            run_output_mode if reporting_enabled else None
        ),
        "workflow_run_source_field": source_field,
        "run_events": [event.to_dict() for event in event_manager.events],
    }


def _execute_runtime_action_workflow_run(
    repo_root: Path,
    *,
    run: Mapping[str, object],
    notes: str,
    command_name: str,
) -> dict[str, object]:
    """Run one runtime-action workflow run and return result details."""

    runner = run.get("runner")
    if not isinstance(runner, Mapping):
        raise ValueError("Workflow run runner is invalid.")
    target = str(runner.get("target") or "").strip()
    payload = runner.get("payload")
    payload_map = dict(payload) if isinstance(payload, Mapping) else {}
    policy_id, action_id = _parse_runner_target(target, kind="runtime_action")
    result = _run_policy_runtime_action(
        repo_root,
        policy_id=policy_id,
        action=action_id,
        payload=payload_map,
    )
    _render_runtime_action_result(result)
    success_contract = run.get("success_contract")
    if (
        isinstance(success_contract, Mapping)
        and str(success_contract.get("kind") or "").strip().lower()
        == "external_artifact_check"
    ):
        _verify_external_artifact_check(
            repo_root,
            str(run.get("id") or "").strip(),
            success_contract,
        )
    return {
        "run_id": str(run.get("id") or "").strip().lower(),
        "command": f"runtime_action:{policy_id}:{action_id}",
        "notes": notes,
        "command_name": command_name,
        "workflow_run_output_mode": None,
        "workflow_run_source_field": "workflow_runs",
        "run_events": [],
    }


def _execute_policy_command_workflow_run(
    repo_root: Path,
    *,
    run: Mapping[str, object],
    notes: str,
    command_name: str,
) -> dict[str, object]:
    """Run one policy-command workflow run and return result details."""

    import devcovenant.core.policy_commands as policy_commands_service

    runner = run.get("runner")
    if not isinstance(runner, Mapping):
        raise ValueError("Workflow run runner is invalid.")
    target = str(runner.get("target") or "").strip()
    argv_raw = runner.get("args")
    argv = list(argv_raw) if isinstance(argv_raw, list) else []
    policy_id, command_token = _parse_runner_target(
        target, kind="policy_command"
    )
    command = policy_commands_service.find_policy_command(
        repo_root,
        policy_id=policy_id,
        command_name=command_token,
    )
    if command is None:
        raise SystemExit(
            "Workflow run "
            f"`{run.get('id', '')}` references missing policy command "
            f"`{policy_id}:{command_token}`. Run `devcovenant refresh` if "
            "policy metadata changed."
        )
    payload = policy_commands_service.parse_policy_command_payload(
        policy_id,
        command,
        [str(entry) for entry in argv],
    )
    result = _run_policy_runtime_action(
        repo_root,
        policy_id=policy_id,
        action=command.runtime_action,
        payload=payload,
    )
    _render_runtime_action_result(result)
    success_contract = run.get("success_contract")
    if (
        isinstance(success_contract, Mapping)
        and str(success_contract.get("kind") or "").strip().lower()
        == "external_artifact_check"
    ):
        _verify_external_artifact_check(
            repo_root,
            str(run.get("id") or "").strip(),
            success_contract,
        )
    invocation = policy_commands_service.canonical_policy_command_invocation(
        policy_id,
        command.name,
    )
    if argv:
        invocation = (
            f"{invocation} {shlex.join([str(entry) for entry in argv])}"
        )
    return {
        "run_id": str(run.get("id") or "").strip().lower(),
        "command": invocation,
        "notes": notes,
        "command_name": command_name,
        "workflow_run_output_mode": None,
        "workflow_run_source_field": "workflow_runs",
        "run_events": [],
    }


def _execute_manual_attestation_workflow_run(
    repo_root: Path,
    *,
    run: Mapping[str, object],
    notes: str,
    command_name: str,
) -> dict[str, object]:
    """Run one manual-attestation workflow run and return result details."""

    del repo_root
    runner = run.get("runner")
    if not isinstance(runner, Mapping):
        raise ValueError("Workflow run runner is invalid.")
    attestation_key = str(runner.get("attestation_key") or "").strip()
    env_key = _manual_attestation_env_key(attestation_key)
    value = str(os.environ.get(env_key, "")).strip().lower()
    if value not in {"1", "true", "yes", "on", "attested"}:
        raise SystemExit(
            "Workflow run "
            f"`{run.get('id', '')}` requires manual attestation. Set "
            f"`{env_key}=true` and rerun `devcovenant run`."
        )
    return {
        "run_id": str(run.get("id") or "").strip().lower(),
        "command": f"manual_attestation:{env_key}",
        "notes": notes,
        "command_name": command_name,
        "workflow_run_output_mode": None,
        "workflow_run_source_field": "workflow_runs",
        "run_events": [],
    }


def _execute_workflow_run(
    repo_root: Path,
    run_id: str,
    *,
    notes: str = "",
    command_name: str,
) -> dict[str, object]:
    """Execute one declared workflow run and return its recorded details."""

    run_token = str(run_id or "").strip().lower()
    run = resolve_declared_workflow_run(repo_root, run_token)
    runner = run.get("runner")
    success_contract = run.get("success_contract")
    if not isinstance(runner, Mapping):
        raise ValueError(f"Workflow run `{run_token}` runner is invalid.")
    if not isinstance(success_contract, Mapping):
        raise ValueError(
            f"Workflow run `{run_token}` success_contract is invalid."
        )
    runner_kind = str(runner.get("kind") or "").strip().lower()
    success_kind = str(success_contract.get("kind") or "").strip().lower()

    if runner_kind == "command_group":
        if success_kind not in {
            "all_commands_exit_zero",
            "external_artifact_check",
        }:
            raise SystemExit(
                "Workflow run "
                f"`{run_token}` uses incompatible success contract "
                f"`{success_kind}` for runner `{runner_kind}`."
            )
        return _execute_command_group_workflow_run(
            repo_root,
            run=run,
            notes=notes,
            command_name=command_name,
        )
    if runner_kind == "runtime_action":
        if success_kind not in {
            "runtime_action_success",
            "external_artifact_check",
        }:
            raise SystemExit(
                "Workflow run "
                f"`{run_token}` uses incompatible success contract "
                f"`{success_kind}` for runner `{runner_kind}`."
            )
        return _execute_runtime_action_workflow_run(
            repo_root,
            run=run,
            notes=notes,
            command_name=command_name,
        )
    if runner_kind == "policy_command":
        if success_kind not in {
            "policy_command_success",
            "external_artifact_check",
        }:
            raise SystemExit(
                "Workflow run "
                f"`{run_token}` uses incompatible success contract "
                f"`{success_kind}` for runner `{runner_kind}`."
            )
        return _execute_policy_command_workflow_run(
            repo_root,
            run=run,
            notes=notes,
            command_name=command_name,
        )
    if runner_kind == "manual_attestation":
        if success_kind != "manual_attested":
            raise SystemExit(
                "Workflow run "
                f"`{run_token}` uses incompatible success contract "
                f"`{success_kind}` for runner `{runner_kind}`."
            )
        return _execute_manual_attestation_workflow_run(
            repo_root,
            run=run,
            notes=notes,
            command_name=command_name,
        )
    raise SystemExit(
        f"Workflow run `{run_token}` uses unsupported runner "
        f"kind `{runner_kind}`."
    )


def run_and_record_workflow_run(
    repo_root: Path,
    run_id: str,
    *,
    notes: str = "",
    command_name: str | None = None,
    record_gate_status_entry: bool = True,
) -> int:
    """Run one declared workflow run and record its result."""

    configure_repo_pycache_prefix(repo_root)
    run_token = str(run_id or "").strip().lower()
    invocation_name = str(command_name or "run").strip()
    details = _execute_workflow_run(
        repo_root,
        run_token,
        notes=notes,
        command_name=invocation_name,
    )
    record_workflow_run_result(
        repo_root,
        run_id=run_token,
        command=str(details.get("command", "")).strip(),
        notes=notes,
        command_name=invocation_name,
        run_events=details.get("run_events") or [],
        workflow_run_output_mode=(
            str(details.get("workflow_run_output_mode", "")).strip() or None
        ),
        workflow_run_source_field=(
            str(details.get("workflow_run_source_field", "")).strip() or None
        ),
    )
    if record_gate_status_entry:
        record_gate_status(
            repo_root,
            f"devcovenant {invocation_name}",
            notes=notes,
            run_events=details.get("run_events") or [],
            workflow_run_output_mode=(
                str(details.get("workflow_run_output_mode", "")).strip()
                or None
            ),
            workflow_run_source_field=(
                str(details.get("workflow_run_source_field", "")).strip()
                or None
            ),
        )
    return 0


def run_workflow_runs(repo_root: Path, notes: str = "") -> int:
    """Run all configured workflow runs in declared order."""

    _, runs = resolve_workflow_runs(repo_root)
    if not runs:
        raise SystemExit(
            "No workflow runs are configured for the active profiles."
        )
    executed_run_ids: list[str] = []
    for run in runs:
        run_id = str(run.get("id") or "").strip().lower()
        if not run_id:
            continue
        run_and_record_workflow_run(
            repo_root,
            run_id,
            notes=notes,
            command_name="run",
            record_gate_status_entry=False,
        )
        executed_run_ids.append(run_id)
    record_gate_status(
        repo_root,
        "devcovenant run",
        notes=notes,
    )
    merge_active_run_log_metadata(
        {
            "workflow_run": {
                "run_ids": executed_run_ids,
                "run_count": len(executed_run_ids),
            }
        }
    )
    return 0


def _build_workflow_run_metadata_bundle(
    *,
    run_id: str,
    commands: Sequence[tuple[str, Sequence[str]]],
    events: Sequence[Any],
    workflow_run_output_mode: OutputMode,
    source_field: str,
    write_runtime_profile: bool,
    started: _dt.datetime,
    finished: _dt.datetime,
    first_failed_command: str,
    first_failed_exit_code: int | None,
    passed_commands: int,
    failed_commands: int,
) -> dict[str, Any]:
    """Build run metadata bundle for one workflow run."""
    summary_payload = _build_workflow_run_summary_metadata(
        run_id=run_id,
        commands=commands,
        events=events,
        workflow_run_output_mode=workflow_run_output_mode,
        source_field=source_field,
        started=started,
        finished=finished,
        first_failed_command=first_failed_command,
        first_failed_exit_code=first_failed_exit_code,
        passed_commands=passed_commands,
        failed_commands=failed_commands,
    )
    payload = {"workflow_run_summary": summary_payload}
    if not write_runtime_profile:
        return payload
    profile_payload, profile_artifacts = (
        _build_and_write_workflow_profile_artifacts(
            run_id=run_id,
            commands=commands,
            events=events,
            workflow_run_output_mode=workflow_run_output_mode,
            source_field=source_field,
            started=started,
            finished=finished,
        )
    )
    payload["workflow_profile"] = profile_payload
    payload["workflow_profile_artifacts"] = profile_artifacts
    return payload


def _build_and_write_workflow_profile_artifacts(
    *,
    run_id: str,
    commands: Sequence[tuple[str, Sequence[str]]],
    events: Sequence[Any],
    workflow_run_output_mode: OutputMode,
    source_field: str,
    started: _dt.datetime,
    finished: _dt.datetime,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Build and persist per-run workflow profile artifacts when logs exist."""
    event_rows: list[dict[str, Any]] = []
    for event in events:
        to_dict = getattr(event, "to_dict", None)
        if not callable(to_dict):
            continue
        payload = to_dict()
        if isinstance(payload, dict):
            event_rows.append(dict(payload))
    profile_payload = (
        workflow_profile_runtime_module.build_workflow_runtime_profile_payload(
            run_id=run_id,
            commands=commands,
            events=event_rows,
            workflow_run_output_mode=workflow_run_output_mode,
            source_field=source_field,
            started=started,
            finished=finished,
        )
    )
    profile_text = (
        workflow_profile_runtime_module.render_workflow_runtime_profile_text(
            profile_payload
        )
    )
    context = get_active_run_log_context()
    if context is None:
        return profile_payload, {}
    run_dir = context.require_paths().run_dir
    profile_json_path = run_dir / "workflow_profile.json"
    profile_txt_path = run_dir / "workflow_profile.txt"
    profile_json_path.write_text(
        json.dumps(profile_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    profile_txt_path.write_text(profile_text, encoding="utf-8")
    artifacts = {
        "workflow_profile_json": _run_log_repo_relative(
            context.repo_root,
            profile_json_path,
        ),
        "workflow_profile_txt": _run_log_repo_relative(
            context.repo_root,
            profile_txt_path,
        ),
    }
    return profile_payload, artifacts


def _build_workflow_run_summary_metadata(
    *,
    run_id: str,
    commands: Sequence[tuple[str, Sequence[str]]],
    events: Sequence[Any],
    workflow_run_output_mode: OutputMode,
    source_field: str,
    started: _dt.datetime,
    finished: _dt.datetime,
    first_failed_command: str,
    first_failed_exit_code: int | None,
    passed_commands: int,
    failed_commands: int,
) -> dict[str, Any]:
    """Build structured summary metadata for one workflow run."""
    total_commands = len(commands)
    duration_seconds = round(
        max(
            0.0,
            (finished - started).total_seconds(),
        ),
        3,
    )
    event_rows: list[dict[str, Any]] = []
    for event in events:
        to_dict = getattr(event, "to_dict", None)
        if callable(to_dict):
            payload = to_dict()
            if isinstance(payload, dict):
                event_rows.append(dict(payload))
    duration_values: list[float] = []
    command_durations: list[dict[str, Any]] = []
    for index, (raw_command, command_tokens) in enumerate(commands, start=1):
        event_payload = (
            event_rows[index - 1] if index <= len(event_rows) else {}
        )
        metadata = (
            event_payload.get("metadata")
            if isinstance(event_payload.get("metadata"), Mapping)
            else {}
        )
        event_command = event_payload.get("command")
        if isinstance(event_command, list):
            command_text = " ".join(str(token) for token in event_command)
        else:
            command_text = " ".join(str(token) for token in command_tokens)
        duration_raw = event_payload.get("duration_seconds")
        duration_value: float | None = None
        try:
            duration_value = round(max(0.0, float(duration_raw)), 6)
        except (TypeError, ValueError):
            duration_value = None
        if duration_value is not None:
            duration_values.append(duration_value)
        command_durations.append(
            {
                "index": index,
                "raw_command": str(raw_command),
                "command": command_text.strip(),
                "status": str(event_payload.get("status", "")).strip(),
                "duration_seconds": duration_value,
                "started_at": str(event_payload.get("started_at", "")).strip(),
                "finished_at": str(
                    event_payload.get("finished_at", "")
                ).strip(),
                "exit_code": metadata.get("exit_code"),
            }
        )
    min_duration = min(duration_values) if duration_values else None
    max_duration = max(duration_values) if duration_values else None
    avg_duration = (
        round(sum(duration_values) / len(duration_values), 6)
        if duration_values
        else None
    )
    return {
        "run_id": run_id,
        "workflow_run_output_mode": workflow_run_output_mode,
        "workflow_run_source_field": source_field,
        "normal_console_flood_suppressed": workflow_run_output_mode
        in {"normal", "quiet"},
        "normal_console_streaming": workflow_run_output_mode == "verbose",
        "quiet_console_mode": workflow_run_output_mode == "quiet",
        "full_output_in_logs": True,
        "total_commands": total_commands,
        "passed_commands": passed_commands,
        "failed_commands": failed_commands,
        "duration_seconds": duration_seconds,
        "duration_breakdown_version": "1.0",
        "duration_events_count": len(duration_values),
        "duration_seconds_min_command": min_duration,
        "duration_seconds_max_command": max_duration,
        "duration_seconds_avg_command": avg_duration,
        "first_failed_command": first_failed_command or "",
        "first_failed_exit_code": first_failed_exit_code,
        "failure_hint": (
            "See tail.txt and stdout.log/stderr.log in the run folder."
            if failed_commands
            else ""
        ),
        "commands": [raw for raw, _ in commands],
        "command_durations": command_durations,
        "events": event_rows,
    }
