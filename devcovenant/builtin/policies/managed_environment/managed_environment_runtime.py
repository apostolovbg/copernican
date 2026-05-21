"""Runtime helpers owned by managed-environment policy."""

from __future__ import annotations

import os
import re
import shlex
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

import devcovenant.core.policy_metadata as metadata_runtime_module
import devcovenant.core.policy_registry as registry_service_module
import devcovenant.core.repository_paths as yaml_cache_service
import devcovenant.core.tracked_registry as tracked_registry_module
from devcovenant.core.execution import (
    run_child_command_with_output_policy,
    runtime_print,
)
from devcovenant.core.repository_paths import display_path

POLICY_ID = "managed-environment"
RUNTIME_ACTION_RESOLVE_STAGE = "resolve-stage"
_MANAGED_ENV_STAGES = frozenset(
    {"open", "run", "close", "bootstrap", "managed", "all"}
)
_MANAGED_STAGE_RUNS_ENV = "DEVCOV_MANAGED_STAGE_RUNS"
_GUIDANCE_TOKEN_PATTERN = re.compile(r"{([a-zA-Z0-9_]+)}")


def _load_policy_entry(repo_root: Path) -> dict[str, Any] | None:
    """Load managed-environment policy entry from the tracked registry."""
    registry_path = tracked_registry_module.policy_registry_path(repo_root)
    rendered = display_path(registry_path, repo_root=repo_root)
    if not registry_path.exists():
        config_path = repo_root / "devcovenant" / "config.yaml"
        if not config_path.exists():
            raise ValueError(
                "managed-environment runtime requires tracked registry "
                f"at {rendered}. Run `devcovenant refresh`."
            )
        descriptor = registry_service_module.load_policy_descriptor(
            repo_root,
            POLICY_ID,
        )
        if descriptor is None:
            return None
        current_order, current_values = (
            metadata_runtime_module.descriptor_metadata_order_values(
                descriptor
            )
        )
        context = metadata_runtime_module.build_metadata_context(repo_root)
        bundle = metadata_runtime_module.resolve_policy_metadata_bundle(
            POLICY_ID,
            current_order,
            current_values,
            descriptor,
            context,
        )
        enabled_token = bundle.string_map.get("enabled", "")
        return {
            "enabled": enabled_token,
            "metadata": dict(bundle.string_map),
        }

    try:
        registry_data = yaml_cache_service.load_yaml(registry_path)
    except yaml.YAMLError as exc:
        raise ValueError(
            f"Invalid YAML in policy registry {rendered}: {exc}"
        ) from exc
    except OSError as exc:
        raise ValueError(
            f"Unable to read policy registry {rendered}: {exc}"
        ) from exc

    if not isinstance(registry_data, dict):
        raise ValueError(
            f"Invalid policy registry payload in {rendered}: "
            "expected a mapping."
        )
    policies = registry_data.get("policies")
    if not isinstance(policies, dict):
        raise ValueError(
            "Invalid policy registry payload: `policies` must be a mapping."
        )
    entry = policies.get(POLICY_ID)
    if not isinstance(entry, dict):
        return None
    return entry


def _normalize_metadata_tokens(raw_value: object) -> list[str]:
    """Normalize metadata values into non-empty string tokens."""
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        return [item.strip() for item in raw_value.split(",") if item.strip()]
    if isinstance(raw_value, list):
        return [str(item).strip() for item in raw_value if str(item).strip()]
    token = str(raw_value).strip()
    return [token] if token else []


def _is_enabled_token(raw_value: object) -> bool:
    """Normalize enabled-like metadata tokens to bool."""
    if isinstance(raw_value, bool):
        return raw_value
    token = str(raw_value or "").strip().lower()
    return token in {"1", "true", "yes", "on"}


def _resolve_metadata_paths(
    repo_root: Path,
    entries: list[str],
    *,
    resolve_symlinks: bool = True,
) -> list[Path]:
    """Resolve metadata path entries relative to the repository root."""
    resolved: list[Path] = []
    for entry in entries:
        token = entry.strip()
        if not token:
            continue
        path = Path(token)
        if not path.is_absolute():
            path = repo_root / path
        absolute_path = Path(os.path.abspath(str(path)))
        if resolve_symlinks:
            try:
                resolved.append(absolute_path.resolve())
            except OSError:
                resolved.append(absolute_path)
            continue
        resolved.append(absolute_path)
    return resolved


def _path_aliases(path: Path) -> tuple[Path, ...]:
    """Return stable absolute and resolved aliases for one path."""
    absolute_path = Path(os.path.abspath(str(path)))
    aliases = [absolute_path]
    try:
        resolved_path = absolute_path.resolve()
    except OSError:
        resolved_path = absolute_path
    if resolved_path != absolute_path:
        aliases.append(resolved_path)
    return tuple(aliases)


def _paths_equivalent(left: Path, right: Path) -> bool:
    """Return True when two path texts point to the same location."""
    left_aliases = set(_path_aliases(left))
    right_aliases = set(_path_aliases(right))
    return bool(left_aliases.intersection(right_aliases))


def _path_within_root(candidate: Path, root: Path) -> bool:
    """Return True when one candidate path lives under one root path."""
    for candidate_alias in _path_aliases(candidate):
        for root_alias in _path_aliases(root):
            if (
                candidate_alias == root_alias
                or root_alias in candidate_alias.parents
            ):
                return True
    return False


def _dedupe_paths(paths: Sequence[Path]) -> list[Path]:
    """Return paths in order while dropping equivalent duplicates."""
    deduped: list[Path] = []
    for path in paths:
        if any(_paths_equivalent(path, existing) for existing in deduped):
            continue
        deduped.append(path)
    return deduped


def _resolve_command_search_paths(
    repo_root: Path,
    command_search_path_tokens: list[str],
    managed_python: Path | None,
    managed_root: Path | None,
) -> list[Path]:
    """Return PATH entries used to resolve managed-environment commands."""
    resolved: list[Path] = []
    if managed_python is not None:
        resolved.append(managed_python.parent)
    elif managed_root is not None:
        resolved.extend(
            [
                managed_root / "bin",
                managed_root / "Scripts",
            ]
        )
    resolved.extend(
        _resolve_metadata_paths(
            repo_root,
            command_search_path_tokens,
            resolve_symlinks=False,
        )
    )
    return _dedupe_paths(resolved)


def _parse_managed_commands(entries: list[str]) -> list[tuple[str, str]]:
    """Parse metadata-managed command entries into stage/command pairs."""
    parsed: list[tuple[str, str]] = []
    for entry in entries:
        stage = "open"
        command_text = entry.strip()
        if "=>" in command_text:
            raw_stage, raw_command = command_text.split("=>", 1)
            stage = raw_stage.strip().lower()
            command_text = raw_command.strip()
        if stage not in _MANAGED_ENV_STAGES:
            allowed = ", ".join(sorted(_MANAGED_ENV_STAGES))
            raise ValueError(
                "Invalid managed-environment stage "
                f"`{stage}`. Allowed values: {allowed}."
            )
        if not command_text:
            raise ValueError("Managed command entry is empty.")
        parsed.append((stage, command_text))
    return parsed


def _allows_current_interpreter_fallback(
    metadata_map: Mapping[str, Any],
) -> bool:
    """Return True when current-interpreter fallback is explicitly enabled."""
    return _is_enabled_token(
        metadata_map.get("allow_current_interpreter_fallback")
    )


def _select_managed_command_for_stage(
    managed_commands: list[tuple[str, str]],
    *,
    target_stage: str,
) -> str | None:
    """Select one stage command, preferring exact stage over `all`."""
    for command_stage, command_text in managed_commands:
        if command_stage == target_stage:
            return command_text
    for command_stage, command_text in managed_commands:
        if command_stage == "all":
            return command_text
    return None


def _detect_managed_python(
    expected_interpreters: list[Path],
    expected_paths: list[Path],
) -> tuple[Path | None, Path | None]:
    """Detect managed interpreter and root from expected metadata paths."""
    for interpreter in expected_interpreters:
        if interpreter.exists():
            for root in expected_paths:
                if _path_within_root(interpreter, root):
                    return interpreter, root
            parent_name = interpreter.parent.name.lower()
            if parent_name in {"bin", "scripts"}:
                return interpreter, interpreter.parent.parent
            return interpreter, None

    for root in expected_paths:
        if not root.exists():
            continue
        posix_candidate = root / "bin" / "python"
        if posix_candidate.exists():
            return posix_candidate, root
        windows_candidate = root / "Scripts" / "python.exe"
        if windows_candidate.exists():
            return windows_candidate, root
    return None, None


def _derive_managed_root(
    interpreter: Path,
    expected_paths: list[Path],
) -> Path | None:
    """Derive one managed root for an interpreter path when possible."""
    for root in expected_paths:
        if _path_within_root(interpreter, root):
            return root
    parent_name = interpreter.parent.name.lower()
    if parent_name in {"bin", "scripts"}:
        return interpreter.parent.parent
    return None


def _matches_expected_interpreter(
    interpreter: Path,
    expected_interpreters: list[Path],
    expected_paths: list[Path],
) -> tuple[Path | None, Path | None]:
    """Return interpreter/root when a candidate already matches metadata."""
    absolute_interpreter = Path(os.path.abspath(str(interpreter)))
    for expected in expected_interpreters:
        if _paths_equivalent(absolute_interpreter, expected):
            managed_root = _derive_managed_root(expected, expected_paths)
            if managed_root is None:
                parent_name = expected.parent.name.lower()
                if parent_name in {"bin", "scripts"}:
                    managed_root = expected.parent.parent
            return expected, managed_root
    for root in expected_paths:
        if _path_within_root(absolute_interpreter, root):
            return absolute_interpreter, root
    return None, None


def _select_managed_environment(
    expected_interpreters: list[Path],
    expected_paths: list[Path],
) -> tuple[Path | None, Path | None]:
    """Prefer the current matching interpreter, then inspect targets."""
    current_python, current_root = _matches_expected_interpreter(
        Path(sys.executable),
        expected_interpreters,
        expected_paths,
    )
    if current_python is not None:
        return current_python, current_root
    return _detect_managed_python(expected_interpreters, expected_paths)


def _command_candidates(command: str) -> list[str]:
    """Return one command token plus common dash/underscore variants."""
    token = str(command or "").strip()
    if not token:
        return []
    candidates = [token]
    if "_" in token:
        candidates.append(token.replace("_", "-"))
    if "-" in token:
        candidates.append(token.replace("-", "_"))
    return list(dict.fromkeys(candidates))


def _command_available_in_env(
    command: str,
    env: Mapping[str, str],
) -> bool:
    """Return whether a command resolves inside one execution environment."""
    path_value = str(env.get("PATH", "")).strip() or None
    return any(
        shutil.which(candidate, path=path_value) is not None
        for candidate in _command_candidates(command)
    )


def _missing_required_commands_in_env(
    required_commands: list[str],
    env: Mapping[str, str],
) -> list[str]:
    """Return required commands that are unavailable in one environment."""
    return [
        command
        for command in required_commands
        if not _command_available_in_env(command, env)
    ]


def _environment_satisfies_contract(
    env: Mapping[str, str],
    interpreter: Path | None,
    managed_root: Path | None,
    *,
    required_commands: list[str],
    command_search_paths: Sequence[Path] | None = None,
) -> tuple[bool, dict[str, str]]:
    """Return whether one candidate environment already satisfies policy."""
    if interpreter is None:
        return False, dict(env)
    if not interpreter.exists():
        return False, dict(env)
    if not os.access(interpreter, os.X_OK):
        return False, dict(env)
    prepared_env = _apply_managed_env(
        env,
        interpreter,
        managed_root,
        command_search_paths=command_search_paths,
    )
    missing_commands = _missing_required_commands_in_env(
        required_commands,
        prepared_env,
    )
    return not missing_commands, prepared_env


def _guidance_token_value(
    token: str,
    repo_root: Path,
    managed_python: Path | None,
    managed_root: Path | None,
) -> str:
    """Return rendered guidance token values with safe display paths."""
    normalized = str(token or "").strip()
    if normalized == "repo_root":
        return display_path(repo_root, repo_root=repo_root)
    if normalized == "current_python":
        return display_path(Path(sys.executable), repo_root=repo_root)
    if normalized == "current_bin":
        return display_path(
            Path(sys.executable).parent,
            repo_root=repo_root,
        )
    if normalized == "managed_root":
        if managed_root is None:
            return "<managed_root>"
        return display_path(managed_root, repo_root=repo_root)
    if normalized == "managed_python":
        if managed_python is None:
            return "<managed_python>"
        return display_path(managed_python, repo_root=repo_root)
    if normalized == "managed_bin":
        if managed_python is None:
            return "<managed_bin>"
        return display_path(
            managed_python.parent,
            repo_root=repo_root,
        )
    if normalized:
        return f"<{normalized}>"
    return "<token>"


def _expand_guidance_command_tokens(
    command_text: str,
    repo_root: Path,
    managed_python: Path | None,
    managed_root: Path | None,
) -> str:
    """Expand guidance tokens with safe placeholders for missing context."""
    if "{" not in command_text:
        return command_text
    return _GUIDANCE_TOKEN_PATTERN.sub(
        lambda match: _guidance_token_value(
            match.group(1),
            repo_root,
            managed_python,
            managed_root,
        ),
        command_text,
    )


def _managed_guidance_suffix(
    manual_commands: list[str],
    *,
    repo_root: Path,
    managed_python: Path | None,
    managed_root: Path | None,
) -> str:
    """Build manual-commands suffix for managed-environment errors."""
    if not manual_commands:
        return ""
    expanded = [
        _expand_guidance_command_tokens(
            command,
            repo_root,
            managed_python,
            managed_root,
        )
        for command in manual_commands
    ]
    return " Manual commands: " + " | ".join(expanded)


def _apply_managed_env(
    env: Mapping[str, str],
    interpreter: Path,
    root: Path | None,
    *,
    command_search_paths: Sequence[Path] | None = None,
) -> dict[str, str]:
    """Return env with managed interpreter PATH and identity markers."""
    updated = dict(env)
    path_entries: list[str] = []
    for path_entry in [interpreter.parent, *(command_search_paths or [])]:
        token = str(path_entry).strip()
        if token and token not in path_entries:
            path_entries.append(token)
    existing_path = str(updated.get("PATH", "")).strip()
    if existing_path:
        for token in existing_path.split(os.pathsep):
            normalized = token.strip()
            if normalized and normalized not in path_entries:
                path_entries.append(normalized)
    updated["PATH"] = os.pathsep.join(path_entries)
    updated["DEVCOV_MANAGED_PYTHON"] = str(interpreter)
    if root is not None:
        updated["VIRTUAL_ENV"] = str(root)
    return updated


def _current_interpreter_root(env: Mapping[str, str]) -> Path | None:
    """Return one trusted root for the current interpreter when available."""
    current_python = Path(os.path.abspath(str(sys.executable)))
    raw_virtual_env = str(env.get("VIRTUAL_ENV", "")).strip()
    if raw_virtual_env:
        virtual_env_root = Path(raw_virtual_env)
        if _path_within_root(current_python, virtual_env_root):
            return virtual_env_root
    parent_name = current_python.parent.name.lower()
    if parent_name in {"bin", "scripts"}:
        candidate_root = current_python.parent.parent
        if (candidate_root / "pyvenv.cfg").exists():
            return candidate_root
    return None


def _resolve_current_interpreter_environment(
    env: Mapping[str, str],
    *,
    command_search_paths: Sequence[Path] | None = None,
) -> tuple[dict[str, str] | None, str | None]:
    """Return the current interpreter env when it can host bootstrap setup."""
    current_python = Path(os.path.abspath(str(sys.executable)))
    if not current_python.exists():
        return None, None
    if not os.access(current_python, os.X_OK):
        return None, None
    current_root = _current_interpreter_root(env)
    prepared_env = _apply_managed_env(
        env,
        current_python,
        current_root,
        command_search_paths=command_search_paths,
    )
    return prepared_env, str(current_python)


def _read_managed_stage_runs(env: Mapping[str, str]) -> set[str]:
    """Return normalized set of stages already prepared in this process env."""
    raw = str(env.get(_MANAGED_STAGE_RUNS_ENV, "")).strip()
    if not raw:
        return set()
    stages: set[str] = set()
    for token in raw.split(","):
        stage = token.strip().lower()
        if stage in _MANAGED_ENV_STAGES:
            stages.add(stage)
    return stages


def _write_managed_stage_runs(env: dict[str, str], stages: set[str]) -> None:
    """Persist prepared-stage set into process environment."""
    ordered = [
        stage
        for stage in ("open", "run", "close", "bootstrap", "managed", "all")
        if stage in stages
    ]
    env[_MANAGED_STAGE_RUNS_ENV] = ",".join(ordered)


def _expand_managed_command_tokens(
    command_text: str,
    repo_root: Path,
    managed_python: Path | None,
    managed_root: Path | None,
) -> list[str]:
    """Expand managed-command placeholders and return argv tokens."""
    tokens = shlex.split(command_text)
    expanded: list[str] = []
    for token in tokens:
        resolved = token.replace("{repo_root}", str(repo_root))
        resolved = resolved.replace("{current_python}", sys.executable)
        resolved = resolved.replace(
            "{current_bin}", str(Path(sys.executable).parent)
        )
        if "{managed_root}" in resolved:
            if managed_root is None:
                raise ValueError(
                    "Managed command uses `{managed_root}` before a "
                    "managed root exists. Run bootstrap commands first."
                )
            resolved = resolved.replace("{managed_root}", str(managed_root))
        if "{managed_python}" in resolved:
            if managed_python is None:
                raise ValueError(
                    "Managed command uses `{managed_python}` before an "
                    "expected interpreter exists. Run bootstrap commands "
                    "first."
                )
            resolved = resolved.replace(
                "{managed_python}", str(managed_python)
            )
        if "{managed_bin}" in resolved:
            if managed_python is None:
                raise ValueError(
                    "Managed command uses `{managed_bin}` before an "
                    "expected interpreter exists. Run bootstrap commands "
                    "first."
                )
            resolved = resolved.replace(
                "{managed_bin}", str(managed_python.parent)
            )
        expanded.append(resolved)
    return expanded


def _run_command(
    command: Sequence[str],
    *,
    env: Mapping[str, str],
    cwd: Path,
) -> None:
    """Run one command and raise ValueError when it fails."""
    command_list = list(command)
    result, _ = run_child_command_with_output_policy(
        command_list,
        channel="managed_child",
        env=env,
        cwd=cwd,
        capture_combined_output=False,
        verbose_only_console=False,
    )
    return_code = int(result.returncode or 0)
    if return_code != 0:
        rendered = " ".join(command_list)
        raise ValueError(
            "Managed-environment command failed "
            f"({return_code}): {rendered}"
        )


def _run_managed_commands_for_stage(
    repo_root: Path,
    env: dict[str, str],
    managed_commands: list[tuple[str, str]],
    *,
    target_stage: str,
    expected_interpreters: list[Path],
    expected_paths: list[Path],
    include_all_stage: bool,
) -> tuple[dict[str, str], bool]:
    """Run managed commands for a stage and return updated environment."""
    updated_env = dict(env)
    ran_commands = False
    for command_stage, command_text in managed_commands:
        if include_all_stage:
            if command_stage not in {"all", target_stage}:
                continue
        elif command_stage != target_stage:
            continue
        ran_commands = True
        managed_python, managed_root = _detect_managed_python(
            expected_interpreters,
            expected_paths,
        )
        if managed_python is not None:
            updated_env = _apply_managed_env(
                updated_env,
                managed_python,
                managed_root,
            )
        command_tokens = _expand_managed_command_tokens(
            command_text,
            repo_root,
            managed_python,
            managed_root,
        )
        runtime_print(
            "Running managed-environment command "
            f"({target_stage}): {' '.join(command_tokens)}",
            verbose_only=True,
        )
        _run_command(command_tokens, env=updated_env, cwd=repo_root)
    return updated_env, ran_commands


def resolve_managed_environment_for_stage(
    repo_root: Path,
    stage: str,
    *,
    base_env: Mapping[str, str] | None = None,
) -> tuple[dict[str, str] | None, str | None]:
    """Resolve and optionally prepare managed-environment execution state."""
    stage_token = str(stage or "").strip().lower()
    if stage_token not in {"open", "run", "close", "bootstrap", "managed"}:
        raise ValueError(
            "Invalid managed-environment stage "
            f"`{stage}`. Allowed: open, run, close, bootstrap, managed."
        )
    entry = _load_policy_entry(repo_root)
    if entry is None:
        return None, None
    if not _is_enabled_token(entry.get("enabled")):
        return None, None

    metadata_map = entry.get("metadata")
    if not isinstance(metadata_map, dict):
        raise ValueError(
            "Invalid policy registry payload: "
            "`managed-environment.metadata` must be a mapping."
        )

    expected_path_tokens = _normalize_metadata_tokens(
        metadata_map.get("expected_paths")
    )
    expected_interpreter_tokens = _normalize_metadata_tokens(
        metadata_map.get("expected_interpreters")
    )
    manual_commands = _normalize_metadata_tokens(
        metadata_map.get("manual_commands")
    )
    required_commands = _normalize_metadata_tokens(
        metadata_map.get("required_commands")
    )
    command_search_path_tokens = _normalize_metadata_tokens(
        metadata_map.get("command_search_paths")
    )
    managed_commands_raw = _normalize_metadata_tokens(
        metadata_map.get("managed_commands")
    )
    managed_commands = _parse_managed_commands(managed_commands_raw)
    allow_current_interpreter_fallback = _allows_current_interpreter_fallback(
        metadata_map
    )

    expected_paths = _resolve_metadata_paths(repo_root, expected_path_tokens)
    expected_interpreters = _resolve_metadata_paths(
        repo_root,
        expected_interpreter_tokens,
        resolve_symlinks=False,
    )
    if not expected_paths and not expected_interpreters:
        guidance = _managed_guidance_suffix(
            manual_commands,
            repo_root=repo_root,
            managed_python=None,
            managed_root=None,
        )
        raise ValueError(
            "managed-environment is enabled, but no expected_paths or "
            f"expected_interpreters are configured.{guidance}"
        )

    env: dict[str, str] = (
        dict(base_env) if base_env is not None else dict(os.environ)
    )
    prepared_stages = _read_managed_stage_runs(env)
    managed_python, managed_root = _select_managed_environment(
        expected_interpreters,
        expected_paths,
    )
    command_search_paths = _resolve_command_search_paths(
        repo_root,
        command_search_path_tokens,
        managed_python,
        managed_root,
    )
    environment_ready, env = _environment_satisfies_contract(
        env,
        managed_python,
        managed_root,
        required_commands=required_commands,
        command_search_paths=command_search_paths,
    )

    if stage_token == "open" and environment_ready:
        prepared_stages.add("open")
        _write_managed_stage_runs(env, prepared_stages)
    elif stage_token not in prepared_stages:
        env, ran_stage_commands = _run_managed_commands_for_stage(
            repo_root,
            env,
            managed_commands,
            target_stage=stage_token,
            expected_interpreters=expected_interpreters,
            expected_paths=expected_paths,
            include_all_stage=True,
        )
        if ran_stage_commands:
            prepared_stages.add(stage_token)
            _write_managed_stage_runs(env, prepared_stages)
        managed_python, managed_root = _select_managed_environment(
            expected_interpreters,
            expected_paths,
        )
        command_search_paths = _resolve_command_search_paths(
            repo_root,
            command_search_path_tokens,
            managed_python,
            managed_root,
        )
        environment_ready, env = _environment_satisfies_contract(
            env,
            managed_python,
            managed_root,
            required_commands=required_commands,
            command_search_paths=command_search_paths,
        )

    if (
        not environment_ready
        and stage_token in {"bootstrap", "run"}
        and "open" not in prepared_stages
    ):
        env, ran_open_commands = _run_managed_commands_for_stage(
            repo_root,
            env,
            managed_commands,
            target_stage="open",
            expected_interpreters=expected_interpreters,
            expected_paths=expected_paths,
            include_all_stage=False,
        )
        if ran_open_commands:
            prepared_stages.add("open")
            _write_managed_stage_runs(env, prepared_stages)
        managed_python, managed_root = _select_managed_environment(
            expected_interpreters,
            expected_paths,
        )
        command_search_paths = _resolve_command_search_paths(
            repo_root,
            command_search_path_tokens,
            managed_python,
            managed_root,
        )
        environment_ready, env = _environment_satisfies_contract(
            env,
            managed_python,
            managed_root,
            required_commands=required_commands,
            command_search_paths=command_search_paths,
        )
    guidance = _managed_guidance_suffix(
        manual_commands,
        repo_root=repo_root,
        managed_python=managed_python,
        managed_root=managed_root,
    )
    if not environment_ready or managed_python is None:
        if (
            managed_python is None
            and stage_token == "bootstrap"
            and not managed_commands
        ):
            if allow_current_interpreter_fallback:
                bootstrap_env, bootstrap_python = (
                    _resolve_current_interpreter_environment(
                        env,
                        command_search_paths=command_search_paths,
                    )
                )
                if bootstrap_env is not None and bootstrap_python is not None:
                    return bootstrap_env, bootstrap_python
            if not allow_current_interpreter_fallback:
                raise ValueError(
                    "managed-environment is enabled, but bootstrap-stage "
                    "fallback to the current interpreter is disabled. "
                    "Declare `managed_commands` or set "
                    "`allow_current_interpreter_fallback: true`." + guidance
                )
        if managed_python is None:
            raise ValueError(
                "managed-environment is enabled, but no expected "
                f"interpreter was found.{guidance}"
            )
        missing_commands = _missing_required_commands_in_env(
            required_commands,
            env,
        )
        if missing_commands:
            missing_text = ", ".join(missing_commands)
            raise ValueError(
                "managed-environment resolved an interpreter, but the "
                "execution environment is still missing required "
                f"commands: {missing_text}.{guidance}"
            )
        raise ValueError(
            "managed-environment resolved an interpreter, but the "
            "execution environment is not ready." + guidance
        )
    return env, str(managed_python)


def resolve_cleanup_protected_paths(repo_root: Path) -> tuple[Path, ...]:
    """Return cleanup-protected roots from managed-environment metadata."""
    entry = _load_policy_entry(repo_root)
    if entry is None:
        return ()
    if not _is_enabled_token(entry.get("enabled")):
        return ()

    metadata_map = entry.get("metadata")
    if not isinstance(metadata_map, dict):
        raise ValueError(
            "Invalid policy registry payload: "
            "`managed-environment.metadata` must be a mapping."
        )

    cleanup_tokens = _normalize_metadata_tokens(
        metadata_map.get("cleanup_protected_paths")
    )
    if cleanup_tokens:
        return tuple(_resolve_metadata_paths(repo_root, cleanup_tokens))

    expected_path_tokens = _normalize_metadata_tokens(
        metadata_map.get("expected_paths")
    )
    expected_paths = _resolve_metadata_paths(repo_root, expected_path_tokens)
    if expected_paths:
        return tuple(expected_paths)

    expected_interpreter_tokens = _normalize_metadata_tokens(
        metadata_map.get("expected_interpreters")
    )
    expected_interpreters = _resolve_metadata_paths(
        repo_root,
        expected_interpreter_tokens,
        resolve_symlinks=False,
    )
    managed_python, managed_root = _detect_managed_python(
        expected_interpreters,
        expected_paths,
    )
    if managed_root is not None:
        return (managed_root,)
    if managed_python is not None:
        return (managed_python,)
    return tuple(expected_interpreters)
