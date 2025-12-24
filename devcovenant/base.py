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
        """Read ignore patterns from ``devcovignore.md``."""
        ignore_file = self.repo_root / "devcovenant" / "devcovignore.md"
        patterns: List[str] = []
        if not ignore_file.exists():
            return patterns
        try:
            for raw_line in ignore_file.read_text(
                encoding="utf-8"
            ).splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                normalized = line.lstrip("/")
                if normalized.endswith("/"):
                    normalized = normalized.rstrip("/") + "/**"
                patterns.append(normalized)
        except OSError:
            return patterns
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
