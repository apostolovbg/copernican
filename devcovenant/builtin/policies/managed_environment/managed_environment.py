"""Ensure DevCovenant runs inside a declared managed environment."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Any, List

from devcovenant.builtin.policies.managed_environment import (
    managed_environment_runtime,
)
from devcovenant.core.policy_contract import (
    CheckContext,
    PolicyCheck,
    Violation,
)


class ManagedEnvironmentCheck(PolicyCheck):
    """Verify the active interpreter matches the managed environment."""

    policy_id = "managed-environment"
    version = "1.0.0"

    def run_runtime_action(
        self,
        action: str,
        *,
        repo_root: Path,
        payload: dict[str, object] | None = None,
    ) -> Any:
        """Run managed-environment runtime actions for CLI orchestration."""
        payload_map = dict(payload or {})
        runtime_helpers = managed_environment_runtime
        if action == runtime_helpers.RUNTIME_ACTION_RESOLVE_STAGE:
            stage = str(payload_map.get("stage", "")).strip().lower()
            base_env_raw = payload_map.get("base_env")
            base_env = (
                dict(base_env_raw) if isinstance(base_env_raw, dict) else None
            )
            return runtime_helpers.resolve_managed_environment_for_stage(
                repo_root,
                stage,
                base_env=base_env,
            )
        raise ValueError(
            "Unsupported managed-environment runtime action: " f"`{action}`."
        )

    def check(self, context: CheckContext) -> List[Violation]:
        """Error when DevCovenant runs outside configured environments."""
        repo_root = context.repo_root.resolve()
        expected_paths = self._normalize_entries(
            self.get_option("expected_paths", [])
        )
        expected_interpreters = self._normalize_entries(
            self.get_option("expected_interpreters", [])
        )
        manual_commands = self._normalize_entries(
            self.get_option("manual_commands", [])
        )
        required_commands = self._normalize_entries(
            self.get_option("required_commands", [])
        )
        command_search_path_tokens = self._normalize_entries(
            self.get_option("command_search_paths", [])
        )
        managed_commands = self._normalize_entries(
            self.get_option("managed_commands", [])
        )
        allow_current_interpreter_fallback = (
            managed_environment_runtime._allows_current_interpreter_fallback(
                {
                    "allow_current_interpreter_fallback": self.get_option(
                        "allow_current_interpreter_fallback", False
                    )
                }
            )
        )

        warnings: list[Violation] = []
        if not expected_paths and not expected_interpreters:
            warnings.append(
                self._warning(
                    repo_root,
                    "managed-environment is enabled, but no expected_paths or "
                    "expected_interpreters are configured.",
                )
            )

        resolved_paths = self._resolve_paths(repo_root, expected_paths)
        resolved_interpreters = self._resolve_paths(
            repo_root, expected_interpreters
        )

        if not resolved_paths and not resolved_interpreters:
            return warnings

        if resolved_paths and not any(
            path.exists() for path in resolved_paths
        ):
            warnings.append(
                self._warning(
                    repo_root,
                    "managed-environment expected_paths do not exist on disk.",
                )
            )

        if self._matches_environment(resolved_paths, resolved_interpreters):
            return warnings

        managed_python, managed_root = (
            managed_environment_runtime._detect_managed_python(
                resolved_interpreters,
                resolved_paths,
            )
        )
        command_search_paths = (
            managed_environment_runtime._resolve_command_search_paths(
                repo_root,
                command_search_path_tokens,
                managed_python,
                managed_root,
            )
        )
        if required_commands and command_search_paths:
            missing = [
                cmd
                for cmd in required_commands
                if cmd
                and not self._command_available(cmd, command_search_paths)
            ]
            if missing:
                warnings.append(
                    self._warning(
                        repo_root,
                        "Required commands are missing from the resolved "
                        "managed-environment command paths: "
                        f"{', '.join(missing)}.",
                    )
                )
        if not managed_commands:
            if allow_current_interpreter_fallback:
                return warnings
            primary_path = resolved_paths[0] if resolved_paths else repo_root
            guidance = managed_environment_runtime._managed_guidance_suffix(
                manual_commands,
                repo_root=repo_root,
                managed_python=managed_python,
                managed_root=managed_root,
            )
            return warnings + [
                Violation(
                    policy_id=self.policy_id,
                    severity="error",
                    file_path=primary_path,
                    line_number=1,
                    message=(
                        "managed-environment is enabled, but no "
                        "`managed_commands` are declared and bootstrap-stage "
                        "fallback to the current interpreter is not "
                        "explicitly enabled. Declare `managed_commands` or "
                        "set `allow_current_interpreter_fallback: true`."
                        + guidance
                    ),
                    suggestion=(
                        "Add managed bootstrap commands for the target "
                        "environment or set `allow_current_interpreter_"
                        "fallback: true` when bootstrap-stage reuse is "
                        "intentional."
                    ),
                )
            ]
        guidance = managed_environment_runtime._managed_guidance_suffix(
            manual_commands,
            repo_root=repo_root,
            managed_python=managed_python,
            managed_root=managed_root,
        )

        message = (
            "DevCovenant must run from the managed environment declared by "
            "managed-environment metadata." + guidance
        )
        primary_path = resolved_paths[0] if resolved_paths else repo_root
        return warnings + [
            Violation(
                policy_id=self.policy_id,
                severity="error",
                file_path=primary_path,
                line_number=1,
                message=message,
            )
        ]

    def _command_available(
        self,
        command: str,
        search_paths: list[Path],
    ) -> bool:
        """Return True when a command resolves in managed search paths."""
        token = str(command or "").strip()
        if not token:
            return False
        path_value = os.pathsep.join(
            str(path) for path in search_paths if str(path).strip()
        ).strip()
        if not path_value:
            return False
        candidates = [token]
        if "_" in token:
            candidates.append(token.replace("_", "-"))
        if "-" in token:
            candidates.append(token.replace("-", "_"))
        deduped = list(dict.fromkeys(candidates))
        return any(
            shutil.which(candidate, path=path_value) is not None
            for candidate in deduped
        )

    def _normalize_entries(self, entries: object) -> list[str]:
        """Normalize metadata entries into a list of strings."""
        if isinstance(entries, str):
            return [
                entry.strip() for entry in entries.split(",") if entry.strip()
            ]
        if isinstance(entries, (list, tuple, set)):
            return [
                str(entry).strip() for entry in entries if str(entry).strip()
            ]
        return []

    def _resolve_paths(
        self, repo_root: Path, entries: list[str]
    ) -> List[Path]:
        """Resolve metadata paths relative to the repository root when
        needed."""
        resolved: list[Path] = []
        for entry in entries:
            entry_path = Path(entry)
            if not entry_path.is_absolute():
                entry_path = repo_root / entry_path
            try:
                resolved.append(entry_path.resolve())
            except OSError:
                resolved.append(entry_path)
        return resolved

    def _matches_environment(
        self, expected_paths: List[Path], expected_interpreters: List[Path]
    ) -> bool:
        """Return True when the active interpreter matches expectations."""
        candidates = []
        env_path = os.environ.get("VIRTUAL_ENV")
        if env_path:
            candidates.append(Path(env_path))
        candidates.append(Path(sys.executable))
        candidates.append(Path(sys.executable).parent)

        for candidate in candidates:
            try:
                resolved = candidate.resolve()
            except OSError:
                resolved = candidate
            for interpreter in expected_interpreters:
                if resolved == interpreter:
                    return True
            for directory in expected_paths:
                if directory in resolved.parents or resolved == directory:
                    return True
        return False

    def _warning(self, repo_root: Path, message: str) -> Violation:
        """Return a warning-level violation for config guidance."""
        return Violation(
            policy_id=self.policy_id,
            severity="warning",
            file_path=repo_root,
            line_number=1,
            message=message,
        )
