"""Shared helpers for policy include/exclude selectors."""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Iterable, List, Mapping, Sequence

from devcovenant.base import PolicyCheck


def _normalize_list(raw_value: object | None) -> List[str]:
    """
    Return a list of non-empty strings parsed from metadata/config values.
    """

    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        candidates: Sequence[str] = raw_value.split(",")
    elif isinstance(raw_value, Sequence):
        candidates = raw_value  # type: ignore[assignment]
    else:
        candidates = [raw_value]  # type: ignore[list-item]
    normalized: List[str] = []
    for entry in candidates:
        text = str(entry).strip()
        if text:
            normalized.append(text)
    return normalized


def _normalize_suffixes(values: Iterable[str]) -> List[str]:
    """Return suffixes prefixed with '.' and folded to lowercase."""
    suffixes: List[str] = []
    for entry in values:
        suffix = entry.strip()
        if not suffix:
            continue
        if not suffix.startswith("."):
            suffix = f".{suffix}"
        suffixes.append(suffix.lower())
    return suffixes


def _normalize_prefixes(values: Iterable[str]) -> List[str]:
    """Return prefixes in forward-slash form without leading slashes."""
    prefixes: List[str] = []
    for entry in values:
        prefix = entry.strip().replace("\\", "/").lstrip("/")
        if prefix:
            prefixes.append(prefix)
    return prefixes


def _normalize_globs(values: Iterable[str]) -> List[str]:
    """Return glob patterns with forward slashes."""
    globs: List[str] = []
    for entry in values:
        pattern = entry.strip().replace("\\", "/")
        if pattern:
            globs.append(pattern)
    return globs


def _normalize_paths(values: Iterable[str]) -> List[str]:
    """Return repository-relative paths in forward-slash form."""
    return [
        entry.strip().replace("\\", "/").lstrip("/")
        for entry in values
        if entry.strip()
    ]


def _match_suffix(name: str, suffixes: Iterable[str]) -> bool:
    """Return True when the filename ends with one of the suffixes."""
    lowered = name.lower()
    return any(lowered.endswith(suffix) for suffix in suffixes)


def _match_prefix(rel_path: str, prefixes: Iterable[str]) -> bool:
    """Return True when the relative path starts with one of the prefixes."""
    for prefix in prefixes:
        if rel_path == prefix or rel_path.startswith(f"{prefix}/"):
            return True
    return False


def _match_globs(rel_path: PurePosixPath, globs: Iterable[str]) -> bool:
    """Return True when the relative path matches any glob pattern."""
    rel_str = rel_path.as_posix()
    return any(fnmatch.fnmatch(rel_str, pattern) for pattern in globs)


def _relative(path: Path, repo_root: Path | None) -> PurePosixPath:
    """Return a PurePosixPath relative to the repo root when possible."""
    if repo_root is not None:
        try:
            rel = path.relative_to(repo_root)
        except ValueError:
            rel = path
    else:
        rel = path
    return PurePosixPath(rel.as_posix())


@dataclass
class SelectorSet:
    """
    Unified include/exclude metadata for policy path selection.

    Policies can populate this structure via ``from_policy`` to ensure
    consistent behaviour when combining suffix, prefix and glob filters.
    """

    include_suffixes: List[str] = field(default_factory=list)
    include_prefixes: List[str] = field(default_factory=list)
    include_globs: List[str] = field(default_factory=list)
    exclude_suffixes: List[str] = field(default_factory=list)
    exclude_prefixes: List[str] = field(default_factory=list)
    exclude_globs: List[str] = field(default_factory=list)
    force_include_globs: List[str] = field(default_factory=list)

    @classmethod
    def from_policy(
        cls,
        policy: PolicyCheck,
        prefix: str = "",
        defaults: Mapping[str, object] | None = None,
    ) -> "SelectorSet":
        """
        Build a selector set from metadata/config options.

        Args:
            policy: Policy providing ``get_option`` overrides.
            prefix: Optional option prefix (e.g., ``dataset_``) when
                policies expose multiple selector groups.
        """

        def option(name: str, default: object | None = None) -> object | None:
            """Return the merged metadata/config value for ``name``."""
            key = f"{prefix}{name}" if prefix else name
            fallback = default
            if fallback is None and defaults:
                fallback = defaults.get(name)
            return policy.get_option(key, fallback)

        return cls(
            include_suffixes=_normalize_suffixes(
                _normalize_list(option("include_suffixes"))
            ),
            include_prefixes=_normalize_prefixes(
                _normalize_list(option("include_prefixes"))
            ),
            include_globs=_normalize_globs(
                _normalize_list(option("include_globs"))
            ),
            exclude_suffixes=_normalize_suffixes(
                _normalize_list(option("exclude_suffixes"))
            ),
            exclude_prefixes=_normalize_prefixes(
                _normalize_list(option("exclude_prefixes"))
            ),
            exclude_globs=_normalize_globs(
                _normalize_list(option("exclude_globs"))
            ),
            force_include_globs=_normalize_globs(
                _normalize_list(option("force_include_globs"))
            ),
        )

    def matches(self, path: Path, repo_root: Path | None = None) -> bool:
        """
        Return True when *path* falls inside the selector scope.

        Force-include globs win over all excludes, excludes take
        precedence over includes, and when any include list is
        populated the path must match at least one include rule.
        """

        rel_path = _relative(path, repo_root)
        rel_str = rel_path.as_posix()
        name = rel_path.name or rel_str

        if self.force_include_globs and _match_globs(
            rel_path, self.force_include_globs
        ):
            return True

        if self.exclude_suffixes and _match_suffix(
            name, self.exclude_suffixes
        ):
            return False
        if self.exclude_prefixes and _match_prefix(
            rel_str, self.exclude_prefixes
        ):
            return False
        if self.exclude_globs and _match_globs(rel_path, self.exclude_globs):
            return False

        include_checks: List[bool] = []
        if self.include_suffixes:
            include_checks.append(_match_suffix(name, self.include_suffixes))
        if self.include_prefixes:
            include_checks.append(
                _match_prefix(rel_str, self.include_prefixes)
            )
        if self.include_globs:
            include_checks.append(_match_globs(rel_path, self.include_globs))

        if include_checks and not any(include_checks):
            return False

        return True


def build_watchlists(
    policy: PolicyCheck,
    prefix: str = "",
    defaults: Mapping[str, Iterable[str]] | None = None,
) -> tuple[List[str], List[str]]:
    """
    Parse ``watch_files``/``watch_dirs`` metadata using the standard format.

    Returns:
        Tuple of (files, directories) as repository-relative strings.
    """

    def option(name: str) -> object | None:
        """Return the merged metadata/config value for ``name``."""
        key = f"{prefix}{name}" if prefix else name
        fallback: object | None = None
        if defaults:
            fallback = defaults.get(name)
        return policy.get_option(key, fallback)

    return (
        _normalize_paths(_normalize_list(option("watch_files"))),
        _normalize_paths(_normalize_list(option("watch_dirs"))),
    )


__all__ = ["SelectorSet", "build_watchlists"]
