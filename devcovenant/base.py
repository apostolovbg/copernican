"""
Base classes and interfaces for devcovenant policies and fixers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional


@dataclass
class CheckContext:
    """
    Context provided to policy checks.

    Attributes:
        repo_root: Root directory of the repository
        changed_files: List of files that have changed
        all_files: List of all files in the repo (optional, for full checks)
        git_diff: Git diff output (optional)
        mode: Check mode (startup, lint, pre-commit, normal)
    """

    repo_root: Path
    changed_files: List[Path] = field(default_factory=list)
    all_files: List[Path] = field(default_factory=list)
    git_diff: Optional[str] = None
    mode: str = "normal"
    config: Dict[str, Any] = field(default_factory=dict)
    _ignore_patterns: List[str] = field(
        default_factory=list, init=False, repr=False
    )

    def __post_init__(self) -> None:
        """Load ignore patterns and sanitize file lists."""
        self._ignore_patterns = self._load_ignore_patterns()
        self.changed_files = [
            path for path in self.changed_files if not self.is_ignored(path)
        ]
        self.all_files = [
            path for path in self.all_files if not self.is_ignored(path)
        ]

    def _load_ignore_patterns(self) -> List[str]:
        """Return ignore patterns defined in the configuration."""
        config_section = (self.config or {}).get("ignore", {})
        raw_patterns = config_section.get("patterns", [])
        if isinstance(raw_patterns, str):
            candidates = [entry.strip() for entry in raw_patterns.split(",")]
        elif isinstance(raw_patterns, List):
            candidates = [str(entry).strip() for entry in raw_patterns]
        else:
            candidates = [str(raw_patterns).strip()] if raw_patterns else []
        patterns: List[str] = []
        for entry in candidates:
            pattern = entry.replace("\\", "/").lstrip("/")
            if not pattern or pattern.startswith("#"):
                continue
            if pattern.endswith("/"):
                pattern = pattern.rstrip("/") + "/**"
            patterns.append(pattern)
        return patterns

    def is_ignored(self, path: Path) -> bool:
        """Return True when *path* matches an ignore rule."""
        if not self._ignore_patterns:
            return False
        try:
            rel_path = path.relative_to(self.repo_root)
        except ValueError:
            rel_path = path
        rel_posix = PurePosixPath(rel_path.as_posix())
        for pattern in self._ignore_patterns:
            if rel_posix.match(pattern):
                return True
        return False

    def get_policy_config(self, policy_id: str) -> Dict[str, Any]:
        """Return the configuration dictionary for a specific policy."""
        policies = self.config.get("policies", {}) if self.config else {}
        entry = policies.get(policy_id, {})
        # Always return a dictionary to avoid cascading None checks.
        return entry if isinstance(entry, dict) else {}


@dataclass
class Violation:
    """
    A single policy violation.

    Attributes:
        policy_id: ID of the violated policy
        severity: Severity level (critical, error, warning, info)
        file_path: Path to the file with violation (optional)
        line_number: Line number of violation (optional)
        column: Column number (optional)
        message: Human-readable description of the violation
        suggestion: Suggested fix (optional)
        can_auto_fix: Whether this violation can be auto-fixed
    """

    policy_id: str
    severity: str
    message: str
    file_path: Optional[Path] = None
    line_number: Optional[int] = None
    column: Optional[int] = None
    suggestion: Optional[str] = None
    can_auto_fix: bool = False
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FixResult:
    """
    Result of attempting to fix a violation.

    Attributes:
        success: Whether the fix was successful
        message: Description of what was done
        files_modified: List of files that were modified
    """

    success: bool
    message: str
    files_modified: List[Path] = field(default_factory=list)


class PolicyCheck(ABC):
    """
    Base class for all policy checks.

    Subclasses must implement the check() method and set policy_id.
    """

    policy_id: str = ""
    version: str = "1.0.0"

    def __init__(self) -> None:
        """Initialise storage for metadata/config-driven options."""
        self.metadata_options: Dict[str, Any] = {}
        self.policy_config: Dict[str, Any] = {}

    @abstractmethod
    def check(self, context: CheckContext) -> List[Violation]:
        """
        Check for policy violations.

        Args:
            context: Context containing files to check and other metadata

        Returns:
            List of violations found (empty list if none)
        """
        pass

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about this policy check.

        Returns:
            Dictionary with policy_id, version, and other metadata
        """
        return {
            "policy_id": self.policy_id,
            "version": self.version,
            "class": self.__class__.__name__,
        }

    def set_options(
        self,
        metadata_options: Dict[str, Any] | None,
        config_overrides: Dict[str, Any] | None,
    ) -> None:
        """
        Store policy options coming from AGENTS.md and config.yaml.

        Metadata options originate from the policy-def block, while
        config overrides map to devcovenant/config.yaml entries.
        """

        self.metadata_options = metadata_options or {}
        self.policy_config = config_overrides or {}

    def get_option(self, key: str, default: Any = None) -> Any:
        """
        Return a merged option value.

        Config overrides in devcovenant/config.yaml win over
        policy-def metadata, which in turn falls back to the default.
        """

        if key in self.policy_config:
            candidate = self.policy_config[key]
            if candidate is not None:
                return candidate
        if key in self.metadata_options:
            candidate = self.metadata_options[key]
            if candidate is not None:
                return candidate
        return default


class PolicyFixer(ABC):
    """
    Base class for automated policy fixers.

    Subclasses must implement can_fix() and fix() methods.
    """

    policy_id: str = ""

    @abstractmethod
    def can_fix(self, violation: Violation) -> bool:
        """
        Determine if this specific violation can be fixed automatically.

        Args:
            violation: The violation to check

        Returns:
            True if this fixer can handle this violation
        """
        pass

    @abstractmethod
    def fix(self, violation: Violation) -> FixResult:
        """
        Attempt to fix the violation.

        Args:
            violation: The violation to fix

        Returns:
            FixResult indicating success/failure and what was changed
        """
        pass
