"""Detect risky constructs via translators and optional scanner backends."""

from __future__ import annotations

import json
import subprocess  # nosec B404
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Sequence

from devcovenant.core.policy_contract import (
    CheckContext,
    PolicyCheck,
    Violation,
)
from devcovenant.core.selectors import SelectorSet

_SCANNER_SUCCESS_CODES = {0, 1}
_SCANNER_KIND_BANDIT = "bandit"


def _normalize_string_list(raw_value: object | None) -> List[str]:
    """Return one normalized string list from metadata or config input."""
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        candidates: Sequence[object] = raw_value.split(",")
    elif isinstance(raw_value, Sequence):
        candidates = raw_value
    else:
        candidates = [raw_value]
    normalized: List[str] = []
    for entry in candidates:
        text = str(entry or "").strip()
        if text:
            normalized.append(text)
    return normalized


def _normalize_suffixes(raw_value: object | None) -> tuple[str, ...]:
    """Return dotted lowercase suffixes from raw metadata values."""
    normalized: list[str] = []
    for entry in _normalize_string_list(raw_value):
        suffix = entry.lower()
        if not suffix.startswith("."):
            suffix = f".{suffix}"
        normalized.append(suffix)
    return tuple(normalized)


def _normalize_enabled(raw_value: object, default: bool = True) -> bool:
    """Return one bool from permissive metadata values."""
    if raw_value is None:
        return default
    if isinstance(raw_value, bool):
        return raw_value
    token = str(raw_value).strip().lower()
    if token in {"true", "1", "yes", "y", "on"}:
        return True
    if token in {"false", "0", "no", "n", "off"}:
        return False
    return default


def _bandit_severity(raw_value: object) -> str:
    """Map Bandit severities onto DevCovenant severities."""
    token = str(raw_value or "").strip().lower()
    if token in {"medium", "high"}:
        return "error"
    if token == "low":
        return "warning"
    return "error"


def _bandit_output_message(
    scanner_id: str,
    issue: Mapping[str, object],
) -> str:
    """Render one stable violation message from Bandit JSON output."""
    test_id = str(issue.get("test_id") or "").strip()
    issue_text = str(issue.get("issue_text") or "").strip()
    confidence = str(issue.get("issue_confidence") or "").strip()
    fragments = [f"External scanner `{scanner_id}` reported"]
    if test_id:
        fragments.append(f"{test_id}:")
    if issue_text:
        fragments.append(issue_text)
    if confidence:
        fragments.append(f"(confidence: {confidence.lower()})")
    return " ".join(fragment for fragment in fragments if fragment)


@dataclass(frozen=True)
class ExternalScannerConfig:
    """Normalized external-scanner metadata owned by one policy overlay."""

    scanner_id: str
    kind: str
    config_file: str
    include_suffixes: tuple[str, ...]
    target_dirs: tuple[str, ...]


class SecurityScannerCheck(PolicyCheck):
    """Flag insecure constructs and run configured scanner backends."""

    policy_id = "security-scanner"
    version = "1.3.0"
    DEFAULT_SUFFIXES = [
        ".py",
        ".pyi",
        ".pyw",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".go",
        ".rs",
        ".java",
        ".cs",
    ]

    def check(self, context: CheckContext) -> List[Violation]:
        """Search repository modules and scanner-backed targets."""
        files = context.all_files or context.changed_files or []
        selector = SelectorSet.from_policy(
            self, defaults={"include_suffixes": self.DEFAULT_SUFFIXES}
        )
        selected_files = [
            path
            for path in files
            if path.is_file() and selector.matches(path, context.repo_root)
        ]
        violations = self._scan_translators(
            context=context,
            selected_files=selected_files,
        )
        scanners, config_violations = self._load_scanners()
        violations.extend(config_violations)
        for scanner in scanners:
            violations.extend(
                self._run_scanner(
                    scanner=scanner,
                    context=context,
                    selected_files=selected_files,
                )
            )
        return violations

    def _scan_translators(
        self,
        *,
        context: CheckContext,
        selected_files: Sequence[Path],
    ) -> List[Violation]:
        """Run the shared translator-backed risk scan."""
        violations: List[Violation] = []
        runtime = context.translator_runtime
        if runtime is None:
            return violations
        for path in selected_files:
            resolution = runtime.resolve(
                path=path,
                policy_id=self.policy_id,
                context=context,
            )
            if not resolution.is_resolved:
                if any(
                    "ambiguous" in violation.message.lower()
                    for violation in resolution.violations
                ):
                    violations.extend(resolution.violations)
                continue
            try:
                source = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                violations.append(
                    Violation(
                        policy_id=self.policy_id,
                        severity="error",
                        file_path=path,
                        message=(
                            "Unable to read file as UTF-8 while scanning "
                            f"security constructs: {exc}"
                        ),
                    )
                )
                continue
            unit = runtime.translate(
                resolution,
                path=path,
                source=source,
                context=context,
            )
            if unit is None:
                continue
            for fact in unit.risk_facts:
                violations.append(
                    Violation(
                        policy_id=self.policy_id,
                        severity=fact.severity,
                        file_path=path,
                        line_number=fact.line_number,
                        message=(
                            "Insecure construct detected: " f"{fact.message}"
                        ),
                    )
                )
        return violations

    def _load_scanners(
        self,
    ) -> tuple[List[ExternalScannerConfig], List[Violation]]:
        """Validate and normalize configured external-scanner metadata."""
        raw_entries = self.get_option("scanners", [])
        if raw_entries in (None, "", []):
            return [], []
        if not isinstance(raw_entries, Sequence) or isinstance(
            raw_entries, (str, bytes, bytearray)
        ):
            return [], [
                Violation(
                    policy_id=self.policy_id,
                    severity="error",
                    message=(
                        "`security-scanner.scanners` must be a list of "
                        "mappings with stable `id` keys."
                    ),
                )
            ]
        scanners: List[ExternalScannerConfig] = []
        violations: List[Violation] = []
        for index, raw_entry in enumerate(raw_entries):
            if not isinstance(raw_entry, Mapping):
                violations.append(
                    Violation(
                        policy_id=self.policy_id,
                        severity="error",
                        message=(
                            "`security-scanner.scanners["
                            f"{index}]` must be a mapping."
                        ),
                    )
                )
                continue
            if not _normalize_enabled(raw_entry.get("enabled"), True):
                continue
            scanner_id = str(raw_entry.get("id") or "").strip()
            if not scanner_id:
                violations.append(
                    Violation(
                        policy_id=self.policy_id,
                        severity="error",
                        message=(
                            "`security-scanner.scanners["
                            f"{index}]` must declare `id`."
                        ),
                    )
                )
                continue
            kind = str(raw_entry.get("kind") or "").strip().lower()
            if kind != _SCANNER_KIND_BANDIT:
                violations.append(
                    Violation(
                        policy_id=self.policy_id,
                        severity="error",
                        message=(
                            "Unsupported `security-scanner` backend "
                            f"`{kind or '<missing>'}` for scanner "
                            f"`{scanner_id}`."
                        ),
                    )
                )
                continue
            scanners.append(
                ExternalScannerConfig(
                    scanner_id=scanner_id,
                    kind=kind,
                    config_file=str(
                        raw_entry.get("config_file") or ""
                    ).strip(),
                    include_suffixes=_normalize_suffixes(
                        raw_entry.get("include_suffixes")
                    )
                    or (".py",),
                    target_dirs=tuple(
                        _normalize_string_list(raw_entry.get("target_dirs"))
                    ),
                )
            )
        return scanners, violations

    def _run_scanner(
        self,
        *,
        scanner: ExternalScannerConfig,
        context: CheckContext,
        selected_files: Sequence[Path],
    ) -> List[Violation]:
        """Dispatch one configured external scanner backend."""
        if scanner.kind == _SCANNER_KIND_BANDIT:
            return self._run_bandit(
                scanner=scanner,
                context=context,
                selected_files=selected_files,
            )
        return [
            Violation(
                policy_id=self.policy_id,
                severity="error",
                message=(
                    "Unsupported `security-scanner` backend "
                    f"`{scanner.kind}` for scanner `{scanner.scanner_id}`."
                ),
            )
        ]

    def _run_bandit(
        self,
        *,
        scanner: ExternalScannerConfig,
        context: CheckContext,
        selected_files: Sequence[Path],
    ) -> List[Violation]:
        """Run Bandit through the launched interpreter and map results."""
        target_files = [
            path
            for path in selected_files
            if path.suffix.lower() in scanner.include_suffixes
        ]
        target_dirs = self._resolve_target_dirs(
            repo_root=context.repo_root,
            scanner=scanner,
        )
        if any(isinstance(entry, Violation) for entry in target_dirs):
            return [
                entry for entry in target_dirs if isinstance(entry, Violation)
            ]
        resolved_dirs = [
            entry for entry in target_dirs if isinstance(entry, Path)
        ]
        if not target_files and not resolved_dirs:
            return []

        command = [sys.executable, "-m", "bandit", "-q", "-f", "json"]
        config_path = self._resolve_config_path(
            repo_root=context.repo_root,
            scanner=scanner,
        )
        if isinstance(config_path, Violation):
            return [config_path]
        if isinstance(config_path, Path):
            command.extend(["-c", str(config_path)])
        if resolved_dirs:
            command.append("-r")
            command.extend(str(path) for path in resolved_dirs)
        else:
            command.extend(str(path) for path in target_files)

        result = subprocess.run(  # nosec B603
            command,
            cwd=context.repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode not in _SCANNER_SUCCESS_CODES:
            output = (result.stderr or result.stdout).strip()
            return [
                Violation(
                    policy_id=self.policy_id,
                    severity="error",
                    message=(
                        f"External scanner `{scanner.scanner_id}` failed "
                        f"with exit code {result.returncode}: {output}"
                    ),
                )
            ]
        try:
            payload = json.loads(result.stdout or "{}")
        except json.JSONDecodeError as exc:
            return [
                Violation(
                    policy_id=self.policy_id,
                    severity="error",
                    message=(
                        f"External scanner `{scanner.scanner_id}` returned "
                        f"invalid JSON output: {exc}"
                    ),
                )
            ]

        violations: List[Violation] = []
        for raw_error in payload.get("errors", []):
            message = str(raw_error or "").strip()
            if not message:
                continue
            violations.append(
                Violation(
                    policy_id=self.policy_id,
                    severity="error",
                    message=(
                        f"External scanner `{scanner.scanner_id}` reported "
                        f"an error: {message}"
                    ),
                )
            )
        for issue in payload.get("results", []):
            if not isinstance(issue, Mapping):
                continue
            file_path = self._resolve_bandit_file_path(
                repo_root=context.repo_root,
                raw_value=issue.get("filename"),
            )
            violations.append(
                Violation(
                    policy_id=self.policy_id,
                    severity=_bandit_severity(issue.get("issue_severity")),
                    file_path=file_path,
                    line_number=self._line_number(issue.get("line_number")),
                    message=_bandit_output_message(
                        scanner.scanner_id,
                        issue,
                    ),
                )
            )
        return violations

    def _resolve_bandit_file_path(
        self,
        *,
        repo_root: Path,
        raw_value: object,
    ) -> Path | None:
        """Return one repo-root-relative file path from Bandit output."""
        filename = str(raw_value or "").strip()
        if not filename:
            return None
        path = Path(filename)
        if not path.is_absolute():
            path = repo_root / path
        return path.resolve()

    def _resolve_config_path(
        self,
        *,
        repo_root: Path,
        scanner: ExternalScannerConfig,
    ) -> Path | Violation | None:
        """Resolve one optional scanner config path from metadata."""
        if not scanner.config_file:
            return None
        candidate = Path(scanner.config_file)
        if not candidate.is_absolute():
            candidate = repo_root / candidate
        if candidate.exists():
            return candidate.resolve()
        return Violation(
            policy_id=self.policy_id,
            severity="error",
            message=(
                f"External scanner `{scanner.scanner_id}` expects config "
                f"`{scanner.config_file}`, but that file does not exist."
            ),
        )

    def _resolve_target_dirs(
        self,
        *,
        repo_root: Path,
        scanner: ExternalScannerConfig,
    ) -> List[Path | Violation]:
        """Resolve configured recursive target directories for one scanner."""
        resolved: List[Path | Violation] = []
        for raw_dir in scanner.target_dirs:
            candidate = Path(raw_dir)
            if not candidate.is_absolute():
                candidate = repo_root / candidate
            if candidate.is_dir():
                resolved.append(candidate.resolve())
                continue
            resolved.append(
                Violation(
                    policy_id=self.policy_id,
                    severity="error",
                    message=(
                        f"External scanner `{scanner.scanner_id}` declares "
                        f"target dir `{raw_dir}`, but that directory does "
                        "not exist."
                    ),
                )
            )
        return resolved

    @staticmethod
    def _line_number(raw_value: object) -> int | None:
        """Parse one optional line number from scanner output."""
        if isinstance(raw_value, int):
            return raw_value
        text = str(raw_value or "").strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            return None
