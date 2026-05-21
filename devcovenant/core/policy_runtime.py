"""Policy check-context, scope, reporting, and engine orchestration."""

from __future__ import annotations

import fnmatch
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Collection, Dict, List, Optional, Set

import yaml

import devcovenant.core.policy_autofix as policy_autofix
import devcovenant.core.policy_metadata as metadata_runtime
import devcovenant.core.repository_paths as yaml_cache_service
import devcovenant.core.repository_validation as repository_validation
import devcovenant.core.workflow_support as workflow_support_module
from devcovenant.core import policy_runtime_actions as runtime_actions
from devcovenant.core.execution import (
    capture_current_numstat_snapshot,
    capture_current_snapshot_paths,
    changed_numstat_paths,
    diff_snapshot_paths,
    get_output_mode,
    load_session_snapshot_payload,
    normalize_snapshot_rows,
    runtime_print,
    snapshot_row_style,
)
from devcovenant.core.policy_contract import (
    ChangeState,
    CheckContext,
    PolicyCheck,
    PolicyFixer,
    Violation,
)
from devcovenant.core.policy_metadata import PolicyDefinition, PolicyParser
from devcovenant.core.policy_registry import (
    PolicyRegistry,
    PolicySyncIssue,
    load_policy_descriptor,
)
from devcovenant.core.profile_registry import (
    load_profile_registry,
    parse_active_profiles,
    resolve_profile_ignore_dirs,
    resolve_profile_suffixes,
)
from devcovenant.core.repository_paths import display_path
from devcovenant.core.repository_validation import ensure_manifest
from devcovenant.core.tracked_registry import policy_registry_path
from devcovenant.core.translator import TranslatorRuntime


def build_check_context(
    repo_root: Path,
    *,
    config: dict[str, Any] | None,
    translator_runtime: Any,
    gate_status_path: Path,
    autofix_enabled: bool,
    autofix_requested: bool,
    is_ignored_path: Callable[[Path], bool],
    resolve_file_suffixes: Callable[[], list[str]],
    collect_all_files: Callable[[set[str]], list[Path]],
) -> CheckContext:
    """Build the `CheckContext` used by policy checks."""
    change_state = build_change_state(
        repo_root,
        gate_status_path=gate_status_path,
        is_ignored_path=is_ignored_path,
    )
    suffixes = set(resolve_file_suffixes())
    snapshot_files = [
        path
        for path in change_state.current_snapshot_paths
        if path.suffix.lower() in suffixes
    ]
    all_files = snapshot_files or collect_all_files(suffixes)
    changed_files = (
        list(change_state.session_paths) if change_state.session_valid else []
    )
    return CheckContext(
        repo_root=repo_root,
        changed_files=changed_files,
        all_files=all_files,
        config=config or {},
        translator_runtime=translator_runtime,
        change_state=change_state,
        autofix_enabled=autofix_enabled,
        autofix_requested=autofix_requested,
    )


def build_change_state(
    repo_root: Path,
    *,
    gate_status_path: Path,
    is_ignored_path: Callable[[Path], bool],
) -> ChangeState:
    """Build current-snapshot and session scopes for policy checks."""
    stage = os.environ.get("DEVCOV_DEVFLOW_STAGE", "").strip().lower()
    state = ChangeState(
        stage=stage,
        gate_status_path=gate_status_path.as_posix(),
    )

    def _set_invalid(reason_code: str, message: str) -> ChangeState:
        """Populate one explicit invalid-session reason and message."""
        state.session_valid = False
        state.session_reason_code = reason_code
        state.session_error = message
        return state

    if stage == "open":
        try:
            current_paths = capture_current_snapshot_paths(repo_root)
        except ValueError as error:
            _set_invalid("snapshot_error", str(error))
            return state
        filtered_paths = sorted(
            path
            for path in current_paths
            if not is_ignored_path(repo_root / path)
        )
        state.current_snapshot_paths = [
            repo_root / path for path in filtered_paths
        ]
        state.session_valid = True
        state.session_reason_code = "open_stage"
        state.session_paths = []
        state.session_error = ""
        return state

    try:
        current_snapshot = capture_current_numstat_snapshot(repo_root)
    except ValueError as error:
        _set_invalid("snapshot_error", str(error))
        return state
    current_snapshot = {
        path: row
        for path, row in current_snapshot.items()
        if not is_ignored_path(repo_root / path)
    }
    state.current_snapshot_numstat = dict(current_snapshot)
    state.current_snapshot_paths = [
        repo_root / path for path in sorted(current_snapshot)
    ]

    def _validate_snapshot_style(
        snapshot: dict[str, str],
        *,
        field_name: str,
    ) -> str | None:
        """Reject unsupported historical snapshot row styles explicitly."""
        style = snapshot_row_style(snapshot)
        if style == "unsupported_legacy":
            _set_invalid(
                "unsupported_snapshot_style",
                "Invalid gate status payload: "
                f"`{field_name}` uses unsupported legacy snapshot rows. "
                "Run `devcovenant gate --open` to record a fresh session.",
            )
            return None
        if style == "mixed":
            _set_invalid(
                "unsupported_snapshot_style",
                "Invalid gate status payload: "
                f"`{field_name}` mixes snapshot row formats. "
                "Run `devcovenant gate --open` to record a fresh session.",
            )
            return None
        return style

    status_path = repo_root / gate_status_path
    if not status_path.exists():
        return _set_invalid(
            "missing_gate_status",
            f"Gate status file is missing: {gate_status_path.as_posix()}. "
            "Run `devcovenant gate --open` first.",
        )

    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return _set_invalid(
            "invalid_gate_status_json",
            f"Invalid gate status JSON in {gate_status_path.as_posix()}: "
            f"{error}",
        )
    if not isinstance(payload, dict):
        return _set_invalid(
            "invalid_gate_status_payload",
            "Invalid gate status payload: expected a mapping.",
        )
    state.gate_status_payload = payload

    session_id = str(payload.get("session_id", "")).strip()
    if not session_id:
        return _set_invalid(
            "missing_session_id",
            "Gate status payload is missing `session_id`. "
            "Run `devcovenant gate --open` first.",
        )

    session_state = str(payload.get("session_state", "")).strip().lower()
    if session_state not in {"open", "closed"}:
        return _set_invalid(
            "invalid_session_state",
            "Invalid gate status payload: `session_state` must be "
            "`open` or `closed`.",
        )
    try:
        snapshot_payload = load_session_snapshot_payload(
            repo_root,
            payload,
            require=True,
        )
    except ValueError as error:
        return _set_invalid("invalid_session_snapshot", str(error))
    state.session_snapshot_path = str(
        payload.get("session_snapshot_file", "")
    ).strip()
    state.session_snapshot_payload = snapshot_payload

    def _load_snapshot_field(
        field_name: str,
        *,
        missing_reason_code: str,
    ) -> dict[str, str] | None:
        """Load one snapshot mapping field from gate status."""
        if field_name not in snapshot_payload:
            _set_invalid(
                missing_reason_code,
                "Invalid session snapshot payload: "
                f"`{field_name}` is required for session checks.",
            )
            return None
        try:
            snapshot = normalize_snapshot_rows(
                snapshot_payload.get(field_name),
                field_name=field_name,
            )
        except ValueError as error:
            _set_invalid("invalid_snapshot_payload", str(error))
            return None
        return {
            path: row
            for path, row in snapshot.items()
            if not is_ignored_path(repo_root / path)
        }

    if session_state == "closed":
        close_snapshot = _load_snapshot_field(
            "session_close_snapshot",
            missing_reason_code="missing_session_close_snapshot",
        )
        if close_snapshot is None:
            return state
        close_style = _validate_snapshot_style(
            close_snapshot,
            field_name="session_close_snapshot",
        )
        if close_style is None:
            return state
        post_close_paths = diff_snapshot_paths(
            close_snapshot,
            current_snapshot,
        )
        if post_close_paths:
            _set_invalid(
                "unsessioned_edits_after_close",
                "Detected edits after the previous `devcovenant gate "
                "--close`. Run `devcovenant gate --open` to open a new "
                "session.",
            )
            return state
        state.session_valid = True
        state.session_reason_code = "closed_clean"
        state.session_paths = []
        state.session_error = ""
        return state

    open_snapshot = _load_snapshot_field(
        "session_open_snapshot",
        missing_reason_code="missing_session_open_snapshot",
    )
    if open_snapshot is None:
        return state

    baseline_snapshot = open_snapshot
    baseline_field_name = "session_open_snapshot"
    if "session_baseline_snapshot" in payload:
        loaded_baseline = _load_snapshot_field(
            "session_baseline_snapshot",
            missing_reason_code="missing_session_baseline_snapshot",
        )
        if loaded_baseline is None:
            return state
        baseline_snapshot = loaded_baseline
        baseline_field_name = "session_baseline_snapshot"
    baseline_style = _validate_snapshot_style(
        baseline_snapshot,
        field_name=baseline_field_name,
    )
    if baseline_style is None:
        return state
    session_rel_paths = changed_numstat_paths(
        baseline_snapshot,
        current_snapshot,
    )
    state.session_paths = sorted(
        [repo_root / path for path in session_rel_paths]
    )
    state.session_valid = True
    if not state.session_reason_code:
        state.session_reason_code = "open_session"
    state.session_error = ""
    return state


@dataclass
class PolicyCheckRunResult:
    """Counted results from one `run_policy_checks` execution pass."""

    violations: list[Violation]
    passed_count: int
    failed_count: int


def critical_disable_attempted(
    policy: PolicyDefinition,
    *,
    normalized_policy_state: dict[str, bool] | None,
    config: dict[str, Any] | None,
) -> bool:
    """Return True when config attempts to disable a critical policy."""
    severity_token = str(policy.severity or "").strip().lower()
    if severity_token != "critical":
        return False
    policy_state = normalized_policy_state
    if not isinstance(policy_state, dict):
        policy_state = metadata_runtime.normalize_policy_state(
            (config or {}).get("policy_state")
        )
    if policy.policy_id not in policy_state:
        return False
    return policy_state[policy.policy_id] is False


def critical_disable_attempt_violation(
    policy: PolicyDefinition,
    *,
    config_path: Path | None,
) -> Violation:
    """Build a deterministic violation for critical disable attempts."""
    if policy.custom:
        remediation = (
            "Update the custom policy metadata in tracked sources to "
            "change severity/enforcement, then refresh."
        )
    else:
        remediation = (
            "Change tracked policy metadata (or copy the builtin policy "
            "to a custom policy and change metadata there), then refresh."
        )
    return Violation(
        policy_id=policy.policy_id,
        severity="critical",
        file_path=config_path,
        message=(
            "Config `policy_state` attempted to disable a critical "
            f"policy (`{policy.policy_id}`), but critical policies "
            "remain enforced."
        ),
        suggestion=(
            f"Remove or set `policy_state.{policy.policy_id}: true` in "
            f"`devcovenant/config.yaml`. {remediation}"
        ),
        can_auto_fix=False,
    )


def extract_policy_options(
    policy: PolicyDefinition,
    *,
    reserved_metadata_keys: set[str],
) -> dict[str, Any]:
    """Pull custom metadata options from a policy definition."""
    options: dict[str, Any] = {"severity": policy.severity}
    options.update(
        metadata_runtime.decode_metadata_options_map(
            policy.raw_metadata,
            reserved_keys=reserved_metadata_keys,
        )
    )
    return options


def run_policy_checks(
    policies: list[PolicyDefinition],
    *,
    context: CheckContext,
    load_policy_script: Callable[[str], PolicyCheck | None],
    extract_policy_options_fn: Callable[[PolicyDefinition], dict[str, Any]],
    critical_disable_attempted_fn: Callable[[PolicyDefinition], bool],
    critical_disable_attempt_violation_fn: Callable[
        [PolicyDefinition], Violation
    ],
) -> PolicyCheckRunResult:
    """Load and run policy checks while tracking pass/fail counts."""
    violations: list[Violation] = []
    passed_count = 0
    failed_count = 0

    for policy in policies:
        policy_violations: list[Violation] = []
        forced_enabled = False
        if critical_disable_attempted_fn(policy):
            forced_enabled = True
            policy_violations.append(
                critical_disable_attempt_violation_fn(policy)
            )

        if not policy.enabled and not forced_enabled:
            continue

        try:
            checker = load_policy_script(policy.policy_id)
            if checker:
                options = extract_policy_options_fn(policy)
                config_overrides = context.get_policy_config(policy.policy_id)
                checker.set_options(options, config_overrides)
                checker_violations = checker.check(context)
                policy_violations.extend(checker_violations)
                violations.extend(policy_violations)
                if not policy_violations:
                    passed_count += 1
                else:
                    failed_count += 1
        # DEVCOV_ALLOW_BROAD_ONCE policy execution isolation boundary.
        except Exception as error:
            failed_count += 1
            violations.append(
                Violation(
                    policy_id=policy.policy_id,
                    severity="error",
                    message=f"Policy execution failed: {error}",
                    suggestion=(
                        "Fix the policy script/runtime error before "
                        "continuing."
                    ),
                )
            )

    return PolicyCheckRunResult(
        violations=violations,
        passed_count=passed_count,
        failed_count=failed_count,
    )


def _normalized_name_entries(raw_entries: object) -> list[str]:
    """Return stripped, non-empty names from scalar or list metadata."""
    if isinstance(raw_entries, str):
        candidates = [raw_entries]
    elif isinstance(raw_entries, list):
        candidates = raw_entries
    else:
        candidates = [raw_entries] if raw_entries else []
    names: list[str] = []
    for entry in candidates:
        name = str(entry).strip()
        if name:
            names.append(name)
    return names


def configured_ignore_dir_names(
    config: dict[str, Any] | None,
) -> list[str]:
    """Return extra ignored directory names from engine config."""
    engine_cfg = config.get("engine", {}) if isinstance(config, dict) else {}
    extra_dirs = engine_cfg.get("ignore_dirs", [])
    return _normalized_name_entries(extra_dirs)


def config_ignore_patterns(config: dict[str, Any] | None) -> list[str]:
    """Return normalized ignore glob patterns from config metadata."""
    ignore_cfg = config.get("ignore", {}) if isinstance(config, dict) else {}
    if isinstance(ignore_cfg, dict):
        raw_patterns = ignore_cfg.get("patterns", [])
    else:
        raw_patterns = []
    if isinstance(raw_patterns, str):
        candidates = [entry.strip() for entry in raw_patterns.split(",")]
    elif isinstance(raw_patterns, list):
        candidates = [str(entry).strip() for entry in raw_patterns]
    else:
        candidates = [str(raw_patterns).strip()] if raw_patterns else []
    patterns: list[str] = []
    for entry in candidates:
        pattern = entry.replace("\\", "/").lstrip("/")
        if not pattern or pattern.startswith("#"):
            continue
        if pattern.endswith("/"):
            pattern = pattern.rstrip("/") + "/**"
        patterns.append(pattern)
    return patterns


def matches_config_ignore_pattern(
    repo_root: Path,
    candidate: Path,
    patterns: list[str],
) -> bool:
    """Return True when candidate matches configured ignore patterns."""
    if not patterns:
        return False
    try:
        rel_path = candidate.relative_to(repo_root)
    except ValueError:
        rel_path = candidate
    rel_text = PurePosixPath(rel_path.as_posix()).as_posix()
    for pattern in patterns:
        if pattern.endswith("/**"):
            dir_token = pattern[: -len("/**")].rstrip("/")
            if rel_text == dir_token or rel_text.startswith(f"{dir_token}/"):
                return True
        if "*" not in pattern and "?" not in pattern and "[" not in pattern:
            if rel_text == pattern:
                return True
            continue
        if fnmatch.fnmatch(rel_text, pattern):
            return True
    return False


def core_exclusion_paths(
    repo_root: Path,
    config: dict[str, Any] | None,
) -> list[Path]:
    """Return repository-rooted core exclusion paths based on config."""
    developer_mode = bool((config or {}).get("developer_mode", False))
    if developer_mode:
        return []
    profiles_cfg = (config or {}).get("profiles", {})
    if isinstance(profiles_cfg, dict):
        generated_cfg = profiles_cfg.get("generated", {})
    else:
        generated_cfg = {}
    if isinstance(generated_cfg, dict):
        core_paths = generated_cfg.get(
            "devcov_core_paths",
            repository_validation.default_scan_excluded_core_paths(),
        )
    else:
        core_paths = repository_validation.default_scan_excluded_core_paths()
    entries = [core_paths] if isinstance(core_paths, str) else list(core_paths)
    results: list[Path] = []
    for entry in entries:
        rel = str(entry).strip()
        if rel:
            results.append(repo_root / rel)
    return results


def discover_custom_policy_overrides(repo_root: Path) -> set[str]:
    """Return policy ids overridden by custom policy scripts."""
    overrides: set[str] = set()
    custom_dir = repo_root / "devcovenant" / "custom" / "policies"
    if not custom_dir.exists():
        return overrides
    for policy_dir in custom_dir.iterdir():
        if not policy_dir.is_dir():
            continue
        script_path = policy_dir / f"{policy_dir.name}.py"
        if not script_path.exists():
            continue
        overrides.add(policy_dir.name.replace("_", "-"))
    return overrides


def is_ignored_path(
    candidate: Path,
    *,
    repo_root: Path,
    ignored_dirs: Collection[str],
    ignored_paths: list[Path],
    config_ignore_patterns: list[str],
) -> bool:
    """Return True when candidate hits ignore names, prefixes, or patterns."""
    if matches_config_ignore_pattern(
        repo_root, candidate, config_ignore_patterns
    ):
        return True
    for part in candidate.parts:
        if part in ignored_dirs:
            return True
    for root in ignored_paths:
        try:
            candidate.relative_to(root)
        except ValueError:
            continue
        return True
    return False


def profile_ignored_dir_names(
    profile_registry: object,
    active_profiles: list[str],
) -> list[str]:
    """Return normalized ignored directory names from active profiles."""
    ignored = resolve_profile_ignore_dirs(profile_registry, active_profiles)
    return _normalized_name_entries(list(ignored))


def resolve_engine_file_suffixes(
    config: dict[str, Any] | None,
    profile_registry: object,
    active_profiles: list[str],
) -> list[str]:
    """Return configured + profile-provided file suffixes for scanning."""
    engine_cfg = config.get("engine", {}) if isinstance(config, dict) else {}
    suffixes = list(
        engine_cfg.get(
            "file_suffixes",
            [".py", ".md", ".yml", ".yaml"],
        )
    )
    suffixes.extend(
        resolve_profile_suffixes(profile_registry, active_profiles)
    )
    cleaned: list[str] = []
    for entry in suffixes:
        text = str(entry).strip()
        if text:
            cleaned.append(text)
    return cleaned


def should_descend_dir(
    candidate: Path,
    *,
    repo_root: Path,
    ignored_dirs: Collection[str],
    ignored_paths: list[Path],
    config_ignore_patterns: list[str],
) -> bool:
    """Return True when repository walk should recurse into candidate."""
    name = candidate.name
    if name in ignored_dirs:
        return False
    if is_ignored_path(
        candidate,
        repo_root=repo_root,
        ignored_dirs=ignored_dirs,
        ignored_paths=ignored_paths,
        config_ignore_patterns=config_ignore_patterns,
    ):
        return False
    if name.startswith("__pycache__"):
        return False
    return True


def collect_all_files(
    repo_root: Path,
    suffixes: set[str],
    *,
    ignored_dirs: Collection[str],
    ignored_paths: list[Path],
    config_ignore_patterns: list[str],
) -> list[Path]:
    """Collect files matching suffixes while honoring ignore rules."""
    matched: list[Path] = []
    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [
            name
            for name in dirs
            if should_descend_dir(
                Path(root) / name,
                repo_root=repo_root,
                ignored_dirs=ignored_dirs,
                ignored_paths=ignored_paths,
                config_ignore_patterns=config_ignore_patterns,
            )
        ]
        for name in files:
            file_path = Path(root) / name
            if is_ignored_path(
                file_path,
                repo_root=repo_root,
                ignored_dirs=ignored_dirs,
                ignored_paths=ignored_paths,
                config_ignore_patterns=config_ignore_patterns,
            ):
                continue
            if file_path.suffix.lower() in suffixes:
                matched.append(file_path)
    return matched


_SEVERITY_ORDER = ("critical", "error", "warning", "info")
_SEVERITY_LEVELS = {
    "critical": 4,
    "error": 3,
    "warning": 2,
    "info": 1,
}


def _sync_issue_test_file(issue: PolicySyncIssue) -> str:
    """Return the suggested test file path for a sync issue."""
    policy_slug = issue.policy_id.replace("-", "_")
    script_path = str(issue.script_path).replace("\\", "/")
    if "/builtin/policies/" in script_path:
        return (
            "tests/devcovenant/builtin/policies/"
            f"{policy_slug}/test_{policy_slug}.py"
        )
    if "/custom/policies/" in script_path:
        return (
            "tests/devcovenant/custom/policies/"
            f"{policy_slug}/test_{policy_slug}.py"
        )
    return (
        "tests/devcovenant/builtin/policies/"
        f"{policy_slug}/test_{policy_slug}.py"
    )


def report_sync_issues(
    issues: list[PolicySyncIssue],
    *,
    print_fn: Callable[..., None],
    repo_root: Path | None = None,
) -> None:
    """Report policy sync issues in AI-friendly format."""
    print_fn("\n" + "=" * 70)
    print_fn("🔄 POLICY SYNC REQUIRED")
    print_fn("=" * 70)
    print_fn()

    for issue in issues:
        print_fn(f"Policy '{issue.policy_id}' requires attention.")
        print_fn(f"Issue: {issue.issue_type.replace('_', ' ').title()}")
        print_fn()

        test_file = _sync_issue_test_file(issue)

        print_fn("📋 Current Policy (from AGENTS.md):")
        print_fn("━" * 70)
        policy_preview = issue.policy_text[:500]
        if len(issue.policy_text) > 500:
            policy_preview += "..."
        print_fn(policy_preview)
        print_fn("━" * 70)
        print_fn()

        print_fn("🎯 Action Required:")
        is_new = issue.issue_type in {"script_missing", "new_policy"}
        script_path = display_path(issue.script_path, repo_root=repo_root)
        if is_new:
            print_fn(f"1. Create: {script_path}")
            print_fn("2. Implement the policy described above")
            print_fn(
                "3. Use the PolicyCheck contract from "
                "devcovenant.core.policy_contract"
            )
            print_fn(f"4. Add tests in {test_file}")
            print_fn(f"5. Run tests: pytest {test_file} -v")
        else:
            print_fn(f"1. Update: {script_path}")
            print_fn("2. Modify the script to implement the updated policy")
            print_fn(f"3. Update tests in {test_file}")
            print_fn(f"4. Run tests: pytest {test_file} -v")

        print_fn("6. Re-run `devcovenant refresh` to sync policy hashes")
        print_fn()
        print_fn("⚠️  Complete this BEFORE working on user's request.")
        print_fn()
        print_fn("=" * 70)
        print_fn()


def report_single_violation(
    violation: Violation,
    *,
    print_fn: Callable[..., None],
    repo_root: Path | None = None,
) -> None:
    """Report one violation with full context."""
    icons = {
        "critical": "❌",
        "error": "🚫",
        "warning": "⚠️",
        "info": "💡",
    }
    icon = icons.get(violation.severity, "•")

    print_fn(f"{icon} {violation.severity.upper()}: {violation.policy_id}")

    if violation.file_path:
        location = display_path(violation.file_path, repo_root=repo_root)
        if violation.line_number:
            location += f":{violation.line_number}"
        print_fn(f"📍 {location}")

    print_fn()
    print_fn(f"Issue: {violation.message}")

    if violation.suggestion:
        print_fn()
        print_fn("Fix:")
        print_fn(violation.suggestion)

    if violation.can_auto_fix:
        print_fn()
        print_fn("Auto-fix: Available in gate workflow (check is audit-only)")

    print_fn()
    print_fn(f"Policy: AGENTS.md#{violation.policy_id}")
    print_fn("━" * 70)
    print_fn()


def violations_by_severity(
    violations: list[Violation],
) -> dict[str, list[Violation]]:
    """Group violations by severity."""
    grouped: dict[str, list[Violation]] = {}
    for violation in violations:
        grouped.setdefault(violation.severity, []).append(violation)
    return grouped


def should_block(
    violations: list[Violation],
    *,
    fail_threshold: str = "error",
) -> bool:
    """Return True when any violation meets the configured threshold."""
    if not violations:
        return False
    threshold_token = str(fail_threshold or "error").strip().lower()
    threshold_level = _SEVERITY_LEVELS.get(threshold_token, 3)
    for violation in violations:
        if _SEVERITY_LEVELS.get(violation.severity, 1) >= threshold_level:
            return True
    return False


def report_summary(
    by_severity: dict[str, list[Violation]],
    *,
    print_fn: Callable[..., None],
    fail_threshold: str = "error",
    auto_fix_enabled: bool = False,
) -> None:
    """Report violation counts and blocking status summary."""
    critical = len(by_severity.get("critical", []))
    errors = len(by_severity.get("error", []))
    warnings = len(by_severity.get("warning", []))
    info = len(by_severity.get("info", []))

    print_fn(
        f"Summary: {critical} critical, {errors} errors, "
        f"{warnings} warnings, {info} info"
    )
    print_fn()

    threshold_token = str(fail_threshold or "error").strip().lower()
    threshold_level = _SEVERITY_LEVELS.get(threshold_token, 3)
    counts_by_severity = {
        "critical": critical,
        "error": errors,
        "warning": warnings,
        "info": info,
    }
    blocks_at_threshold = any(
        counts_by_severity[severity] > 0
        and _SEVERITY_LEVELS[severity] >= threshold_level
        for severity in _SEVERITY_ORDER
    )
    if blocks_at_threshold:
        print_fn(
            "Status: 🚫 BLOCKED "
            f"(violations >= {threshold_token} threshold)"
        )
    elif any(counts_by_severity.values()):
        print_fn("Status: ✅ PASSED (violations below fail threshold)")
    else:
        print_fn("Status: ✅ PASSED")

    print_fn()
    if auto_fix_enabled:
        print_fn(
            "💡 `devcovenant check` is read-only; use the gate workflow "
            "to run refresh + autofix with lifecycle recording"
        )
    print_fn("=" * 70)


def report_violations(
    violations: list[Violation],
    *,
    passed_count: int,
    failed_count: int,
    print_fn: Callable[..., None],
    fail_threshold: str = "error",
    auto_fix_enabled: bool = False,
    repo_root: Path | None = None,
) -> None:
    """Report policy violations with grouped detail and summary."""
    if not violations:
        print_fn("\n✅ All policy checks passed!")
        return

    print_fn("\n" + "=" * 70)
    print_fn("📊 DEVCOVENANT CHECK RESULTS")
    print_fn("=" * 70)
    print_fn()
    print_fn(f"✅ Passed: {passed_count} policies")
    print_fn(f"⚠️  Violations: {len(violations)} issues found")
    print_fn()

    by_severity = violations_by_severity(violations)
    for severity in _SEVERITY_ORDER:
        for violation in by_severity.get(severity, []):
            report_single_violation(
                violation,
                print_fn=print_fn,
                repo_root=repo_root,
            )

    print_fn("=" * 70)
    report_summary(
        by_severity,
        print_fn=print_fn,
        fail_threshold=fail_threshold,
        auto_fix_enabled=auto_fix_enabled,
    )


def config_fail_threshold(config: dict[str, object] | None) -> str:
    """Return normalized fail-threshold token from config."""
    if not isinstance(config, dict):
        return "error"
    engine_cfg = config.get("engine", {})
    if not isinstance(engine_cfg, dict):
        return "error"
    return str(engine_cfg.get("fail_threshold", "error")).strip().lower()


def config_auto_fix_enabled(config: dict[str, object] | None) -> bool:
    """Return auto-fix enablement from config."""
    if not isinstance(config, dict):
        return False
    engine_cfg = config.get("engine", {})
    if not isinstance(engine_cfg, dict):
        return False
    return bool(engine_cfg.get("auto_fix_enabled", False))


def load_policy_check_instance(
    repo_root: Path, policy_id: str
) -> Optional[PolicyCheck]:
    """Load one policy script and return its PolicyCheck instance."""
    return runtime_actions.load_policy_check_instance(repo_root, policy_id)


def run_policy_runtime_action(
    repo_root: Path,
    *,
    policy_id: str,
    action: str,
    payload: Dict[str, Any] | None = None,
) -> Any:
    """Run one policy-owned runtime action through the policy contract."""
    return runtime_actions.run_policy_runtime_action(
        repo_root,
        policy_id=policy_id,
        action=action,
        payload=payload,
        checker_loader=load_policy_check_instance,
        metadata_loader=_runtime_policy_metadata_options,
        config_loader=_runtime_policy_config_overrides,
    )


def _runtime_policy_config_overrides(
    repo_root: Path, policy_id: str
) -> dict[str, Any]:
    """Return merged config overrides for one policy runtime action."""
    return runtime_actions.runtime_policy_config_overrides(
        repo_root, policy_id
    )


def _runtime_policy_metadata_options(
    repo_root: Path, policy_id: str
) -> dict[str, Any]:
    """Return runtime metadata options for a policy action."""
    return runtime_actions.runtime_policy_metadata_options(
        repo_root,
        policy_id,
        descriptor_loader=load_policy_descriptor,
        registry_path_resolver=policy_registry_path,
    )


class DevCovenantEngine:
    """
    Main engine for devcovenant policy enforcement.
    """

    _RESERVED_METADATA_KEYS = {
        "id",
        "severity",
        "auto_fix",
        "updated",
        "enabled",
        "custom",
        "hash",
        "enforcement",
    }

    # Directories we never traverse for policy checks
    _BASE_IGNORED_DIRS = frozenset(
        {
            ".git",
            ".venv",
            ".python",
            "output",
            "logs",
            "build",
            "dist",
            "node_modules",
            "__pycache__",
            ".cache",
            ".venv.lock",
        }
    )
    _DEFAULT_GATE_STATUS_PATH = (
        Path("devcovenant") / "registry" / "runtime" / "gate_status.json"
    )

    def __init__(self, repo_root: Optional[Path] = None):
        """
        Initialize the engine.

        Args:
            repo_root: Root directory of the repository (default: current dir)
        """
        if repo_root is None:
            repo_root = Path.cwd()

        self.repo_root = Path(repo_root).resolve()
        self.devcovenant_dir = self.repo_root / "devcovenant"
        self.agents_md_path = self.repo_root / "AGENTS.md"
        self.config_path = self.devcovenant_dir / "config.yaml"
        self.registry_path = policy_registry_path(self.repo_root)

        # Load configuration and apply overrides
        self.config = self._load_config()
        self._normalized_policy_state = (
            metadata_runtime.normalize_policy_state(
                self.config.get("policy_state")
            )
        )
        self._apply_config_paths()
        self._ignored_dirs = set(self._BASE_IGNORED_DIRS)
        self._ignored_paths: list[Path] = []
        self._config_ignore_patterns = self._load_config_ignore_patterns()
        self._merge_configured_ignored_dirs()
        self._apply_core_exclusions()

        try:
            self._profile_registry = load_profile_registry(self.repo_root)
        except ValueError as error:
            raise SystemExit(f"Invalid profile metadata: {error}") from error
        self._active_profiles = parse_active_profiles(
            self.config, include_global=True
        )
        self.translator_runtime = TranslatorRuntime(
            self.repo_root,
            self._profile_registry,
            self._active_profiles,
        )
        self._merge_profile_ignored_dirs()

        ensure_manifest(self.repo_root)

        # Initialize parser and registry
        self.parser = PolicyParser(self.agents_md_path)
        self.registry = PolicyRegistry(self.registry_path, self.repo_root)

        # Statistics
        self.passed_count = 0
        self.failed_count = 0
        self._custom_policy_overrides = (
            self._discover_custom_policy_overrides()
        )
        self.fixers: List[PolicyFixer] = self._load_fixers()

    def _load_config(self) -> Dict:
        """Load configuration from config.yaml."""
        rendered = display_path(self.config_path, repo_root=self.repo_root)
        if not self.config_path.exists():
            raise SystemExit(
                f"Missing config file: {rendered}. "
                "Run `devcovenant install` or restore config."
            )
        try:
            payload = yaml_cache_service.load_yaml(self.config_path)
        except yaml.YAMLError as exc:
            raise SystemExit(
                f"Invalid YAML in config file {rendered}: {exc}"
            ) from exc
        except OSError as exc:
            raise SystemExit(
                f"Unable to read config file {rendered}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise SystemExit(f"Config file must be a YAML mapping: {rendered}")
        return payload

    def _load_policies_from_agents(self) -> List[PolicyDefinition]:
        """Load policy definitions directly from AGENTS policy blocks."""
        if not self.agents_md_path.exists():
            raise ValueError(
                f"Missing policy definitions file: {self.agents_md_path}."
            )
        try:
            parsed = self.parser.parse_agents_md()
        # DEVCOV_ALLOW_BROAD_ONCE AGENTS parser boundary.
        except Exception as exc:
            raise ValueError(
                f"Failed to parse AGENTS policies: {exc}"
            ) from exc
        policies: List[PolicyDefinition] = []
        for policy in parsed:
            if policy.policy_id:
                policies.append(policy)
        return sorted(policies, key=lambda policy: policy.policy_id)

    def _apply_config_paths(self) -> None:
        """Apply configurable path overrides after the config loads."""
        paths_cfg = self.config.get("paths", {})
        policy_doc = paths_cfg.get("policy_definitions")
        if policy_doc:
            self.agents_md_path = self.repo_root / Path(policy_doc)
        registry_file = paths_cfg.get("registry_file")
        if registry_file:
            self.registry_path = self.repo_root / Path(registry_file)

    def _merge_configured_ignored_dirs(self) -> None:
        """Extend the default ignored directory set via configuration."""
        for name in configured_ignore_dir_names(self.config):
            self._ignored_dirs.add(name)

    def _load_config_ignore_patterns(self) -> list[str]:
        """Return normalized ignore patterns from config.ignore.patterns."""
        return config_ignore_patterns(self.config)

    def _matches_config_ignore_pattern(self, candidate: Path) -> bool:
        """Return True when candidate matches config ignore glob patterns."""
        return matches_config_ignore_pattern(
            self.repo_root,
            candidate,
            self._config_ignore_patterns,
        )

    def _apply_core_exclusions(self) -> None:
        """Apply devcovenant core exclusion rules from configuration."""
        self._ignored_paths.extend(
            core_exclusion_paths(self.repo_root, self.config)
        )

    def _discover_custom_policy_overrides(self) -> set[str]:
        """Return policy ids overridden by custom policy scripts."""
        return discover_custom_policy_overrides(self.repo_root)

    def _is_ignored_path(self, candidate: Path) -> bool:
        """Return True when candidate is within an ignored path prefix."""
        return is_ignored_path(
            candidate,
            repo_root=self.repo_root,
            ignored_dirs=self._ignored_dirs,
            ignored_paths=self._ignored_paths,
            config_ignore_patterns=self._config_ignore_patterns,
        )

    def _merge_profile_ignored_dirs(self) -> None:
        """Extend ignored directories with active profile declarations."""
        for name in profile_ignored_dir_names(
            self._profile_registry,
            self._active_profiles,
        ):
            self._ignored_dirs.add(name)

    def _load_fixers(self) -> List[PolicyFixer]:
        """Dynamically import policy fixers bundled with DevCovenant."""
        return policy_autofix.load_fixers(
            self.repo_root,
            custom_policy_overrides=self._custom_policy_overrides,
        )

    def check(self, apply_fixes: bool = False) -> "CheckResult":
        """
        Main entry point for policy checking.

        Returns:
            CheckResult object
        """
        # Runtime policy input is the compiled AGENTS policy block.
        try:
            policies = self._load_policies_from_agents()
        except ValueError as exc:
            return self._agents_parse_failure_result(str(exc))
        if not policies:
            return self._agents_parse_failure_result(
                "AGENTS policy blocks are empty or invalid. "
                "Checks cannot run without resolved policy metadata."
            )

        # Registry remains hash/diagnostic state only.
        self.registry.load()
        sync_issues = self.registry.check_policy_sync(policies)

        if sync_issues:
            self.report_sync_issues(sync_issues)

        auto_fix_enabled = self.config.get("engine", {}).get(
            "auto_fix_enabled", False
        )
        violations = self._run_check_cycle(
            policies,
            apply_fixes=bool(apply_fixes),
            auto_fix_enabled=bool(auto_fix_enabled),
        )

        # Report violations
        self.report_violations(violations)

        # Determine if should block
        should_block = self.should_block(violations)

        return CheckResult(
            violations,
            should_block,
            sync_issues=sync_issues,
        )

    def _agents_parse_failure_result(self, message: str) -> "CheckResult":
        """Build and report one deterministic AGENTS parse failure result."""
        violation = Violation(
            policy_id="agents-parse",
            severity="error",
            file_path=self.agents_md_path,
            message=message,
            suggestion=(
                "Run `python3 -m devcovenant refresh` to regenerate "
                "AGENTS.md policy blocks from descriptors."
            ),
        )
        self.report_violations([violation])
        return CheckResult([violation], should_block=True, sync_issues=[])

    def _reset_check_counts(self) -> None:
        """Reset aggregate pass/fail counters before one full check pass."""
        self.passed_count = 0
        self.failed_count = 0

    def _run_checks_for_context(
        self,
        policies: List[PolicyDefinition],
        *,
        context: CheckContext,
    ) -> List[Violation]:
        """Run built-in checks and policy checks for one resolved context."""
        self._reset_check_counts()
        built_in_violations: List[Violation] = []
        for check_fn in (
            repository_validation.check_integrity,
            repository_validation.check_structure,
            workflow_support_module.check_workflow_contract,
        ):
            current = list(check_fn(context))
            built_in_violations.extend(current)
            if current:
                self.failed_count += 1
            else:
                self.passed_count += 1
        violations = list(built_in_violations)
        violations.extend(self.run_policy_checks(policies, context))
        return violations

    def _run_check_cycle(
        self,
        policies: List[PolicyDefinition],
        *,
        apply_fixes: bool,
        auto_fix_enabled: bool,
    ) -> List[Violation]:
        """Run one full check cycle with one optional autofix rerun."""
        context = self._build_check_context(
            apply_fixes=apply_fixes,
            auto_fix_enabled=auto_fix_enabled,
        )
        violations = self._run_checks_for_context(policies, context=context)
        if not (apply_fixes and auto_fix_enabled):
            return violations
        if not self.apply_auto_fixes(violations):
            return violations
        context = self._build_check_context(
            apply_fixes=apply_fixes,
            auto_fix_enabled=auto_fix_enabled,
        )
        return self._run_checks_for_context(policies, context=context)

    def report_sync_issues(self, issues: List[PolicySyncIssue]):
        """
        Report policy sync issues in AI-friendly format.

        Args:
            issues: List of PolicySyncIssue objects
        """
        report_sync_issues(
            issues,
            print_fn=self._report_print_fn(error_channel=bool(issues)),
            repo_root=self.repo_root,
        )

    def run_policy_checks(
        self,
        policies: List[PolicyDefinition],
        context: Optional[CheckContext] = None,
    ) -> List[Violation]:
        """
        Load and run all policy check scripts.

        Args:
            policies: List of policy definitions

        Returns:
            List of all violations found
        """
        violations = []

        # Build check context when not provided
        if context is None:
            context = self._build_check_context()
        result = run_policy_checks(
            policies,
            context=context,
            load_policy_script=self._load_policy_script,
            extract_policy_options_fn=self._extract_policy_options,
            critical_disable_attempted_fn=self._critical_disable_attempted,
            critical_disable_attempt_violation_fn=(
                self._critical_disable_attempt_violation
            ),
        )
        self.passed_count += result.passed_count
        self.failed_count += result.failed_count
        violations.extend(result.violations)
        return violations

    def _critical_disable_attempted(self, policy: PolicyDefinition) -> bool:
        """Return True when config attempts to disable a critical policy."""
        return critical_disable_attempted(
            policy,
            normalized_policy_state=getattr(
                self, "_normalized_policy_state", None
            ),
            config=self.config,
        )

    def _critical_disable_attempt_violation(
        self, policy: PolicyDefinition
    ) -> Violation:
        """Build a deterministic violation for critical disable attempts."""
        return critical_disable_attempt_violation(
            policy,
            config_path=getattr(self, "config_path", None),
        )

    def _build_check_context(
        self,
        *,
        apply_fixes: bool = False,
        auto_fix_enabled: bool = False,
    ) -> CheckContext:
        """
        Build the CheckContext for policy checks.

        Returns:
            CheckContext object
        """
        return build_check_context(
            self.repo_root,
            config=self.config,
            translator_runtime=self.translator_runtime,
            gate_status_path=self._DEFAULT_GATE_STATUS_PATH,
            autofix_enabled=bool(auto_fix_enabled),
            autofix_requested=bool(apply_fixes and auto_fix_enabled),
            is_ignored_path=self._is_ignored_path,
            resolve_file_suffixes=self._resolve_file_suffixes,
            collect_all_files=self._collect_all_files,
        )

    def _build_change_state(self) -> ChangeState:
        """Build current-snapshot and session scopes for policy checks."""
        return build_change_state(
            self.repo_root,
            gate_status_path=self._DEFAULT_GATE_STATUS_PATH,
            is_ignored_path=self._is_ignored_path,
        )

    def _collect_all_files(self, suffixes: Set[str]) -> List[Path]:
        """
        Walk the repository tree and collect files matching the given suffixes,
        skipping large or third-party directories.
        """
        return collect_all_files(
            self.repo_root,
            set(suffixes),
            ignored_dirs=self._ignored_dirs,
            ignored_paths=self._ignored_paths,
            config_ignore_patterns=self._config_ignore_patterns,
        )

    def apply_auto_fixes(self, violations: List[Violation]) -> bool:
        """
        Attempt to auto-fix any violations that advertise a fixer.

        Returns:
            True when at least one file was modified.
        """
        return policy_autofix.apply_auto_fixes(
            violations,
            self.fixers,
            print_fn=runtime_print,
        )

    def _should_descend_dir(self, candidate: Path) -> bool:
        """
        Decide whether to continue walking into a directory.
        """
        return should_descend_dir(
            candidate,
            repo_root=self.repo_root,
            ignored_dirs=self._ignored_dirs,
            ignored_paths=self._ignored_paths,
            config_ignore_patterns=self._config_ignore_patterns,
        )

    def _resolve_file_suffixes(self) -> list[str]:
        """Resolve file suffixes using profiles and overrides."""
        return resolve_engine_file_suffixes(
            self.config,
            self._profile_registry,
            self._active_profiles,
        )

    def _load_policy_script(self, policy_id: str) -> Optional[PolicyCheck]:
        """
        Dynamically load a policy script.

        Args:
            policy_id: ID of the policy

        Returns:
            PolicyCheck instance or None if not found
        """
        return load_policy_check_instance(self.repo_root, policy_id)

    def _extract_policy_options(
        self, policy: PolicyDefinition
    ) -> Dict[str, Any]:
        """Pull custom metadata options from a policy definition."""
        return extract_policy_options(
            policy,
            reserved_metadata_keys=set(self._RESERVED_METADATA_KEYS),
        )

    @staticmethod
    def _parse_metadata_value(raw_value: str) -> Any:
        """Decode scalar/list metadata from the policy-def block."""
        return metadata_runtime.decode_metadata_option_value(raw_value)

    def report_violations(self, violations: List[Violation]):
        """
        Report violations in AI-friendly, actionable format.

        Args:
            violations: List of violations
        """
        if not violations and get_output_mode() == "quiet":
            return
        report_violations(
            violations,
            passed_count=self.passed_count,
            failed_count=self.failed_count,
            print_fn=self._report_print_fn(error_channel=bool(violations)),
            fail_threshold=config_fail_threshold(self.config),
            auto_fix_enabled=config_auto_fix_enabled(self.config),
            repo_root=self.repo_root,
        )

    def _report_single_violation(self, violation: Violation):
        """Report a single violation with full context."""
        report_single_violation(
            violation,
            print_fn=runtime_print,
            repo_root=self.repo_root,
        )

    def _report_summary(self, by_severity: Dict[str, List[Violation]]):
        """Report summary of violations."""
        report_summary(
            by_severity,
            print_fn=self._report_print_fn(
                error_channel=any(by_severity.values())
            ),
            fail_threshold=config_fail_threshold(self.config),
            auto_fix_enabled=config_auto_fix_enabled(self.config),
        )

    @staticmethod
    def _report_print_fn(*, error_channel: bool):
        """Return report printer routed for the active output mode."""
        if not (error_channel and get_output_mode() == "quiet"):
            return runtime_print

        def _stderr_print(message: str = "", **kwargs: Any) -> None:
            """Route quiet-mode violation output to stderr."""
            runtime_print(str(message), file=sys.stderr, **kwargs)

        return _stderr_print

    def should_block(self, violations: List[Violation]) -> bool:
        """
        Determine if violations should block the commit/operation.

        Args:
            violations: List of violations

        Returns:
            True if should block
        """
        return should_block(
            violations,
            fail_threshold=config_fail_threshold(self.config),
        )


class CheckResult:
    """Result of a devcovenant check operation."""

    def __init__(
        self,
        violations: List[Violation],
        should_block: bool,
        sync_issues: List[PolicySyncIssue],
    ):
        """Store the check result metadata for later inspection."""
        self.violations = violations
        self.should_block = should_block
        self.sync_issues = sync_issues

    def has_sync_issues(self) -> bool:
        """Check if there are policy sync issues."""
        return len(self.sync_issues) > 0

    def has_violations(self) -> bool:
        """Check if there are any violations."""
        return len(self.violations) > 0
