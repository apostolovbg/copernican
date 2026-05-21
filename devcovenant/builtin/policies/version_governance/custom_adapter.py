"""Custom adapter loader for the version-governance policy."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, List

from devcovenant.core.policy_contract import Violation

if TYPE_CHECKING:
    from .version_governance import (
        VersionGovernanceCheck,
        VersionReleaseContext,
        VersionScheme,
    )


class CustomAdapterScheme:
    """Delegate version-governance to one repository-local scheme module."""

    name = "custom_adapter"

    def __init__(self) -> None:
        """Initialize a per-run cache for loaded custom schemes."""
        self._cache: dict[Path, VersionScheme] = {}

    def preflight(
        self,
        check: "VersionGovernanceCheck",
        repo_root: Path,
        version_path: Path,
    ) -> List[Violation]:
        """Validate the configured adapter path and required interface."""
        try:
            scheme = self._load_scheme(check, repo_root)
        except ValueError as exc:
            return [
                Violation(
                    policy_id=check.policy_id,
                    severity="error",
                    file_path=version_path,
                    message=str(exc),
                )
            ]
        preflight = getattr(scheme, "preflight", None)
        if callable(preflight):
            return list(preflight(check, repo_root, version_path))
        return []

    def version_pattern(
        self,
        check: "VersionGovernanceCheck",
        repo_root: Path,
    ) -> str:
        """Return the custom adapter's changelog-header regex fragment."""
        scheme = self._load_scheme(check, repo_root)
        return str(scheme.version_pattern(check, repo_root))

    def parse_version(
        self,
        value: str,
        check: "VersionGovernanceCheck",
        repo_root: Path,
    ) -> Any:
        """Parse one version string through the repository-local adapter."""
        scheme = self._load_scheme(check, repo_root)
        return scheme.parse_version(value, check, repo_root)

    def compare_versions(self, left: Any, right: Any) -> int:
        """Compare parsed versions through the repository-local adapter."""
        compare = getattr(self._loaded_scheme, "compare_versions", None)
        if not callable(compare):
            raise ValueError("Custom adapter is missing `compare_versions`.")
        return int(compare(left, right))

    def canonicalize_version(
        self,
        parsed: Any,
        check: "VersionGovernanceCheck",
        repo_root: Path,
    ) -> str | None:
        """Return one custom canonical spelling when supported."""
        scheme = self._load_scheme(check, repo_root)
        canonicalize = getattr(scheme, "canonicalize_version", None)
        if not callable(canonicalize):
            return None
        value = canonicalize(parsed, check, repo_root)
        if value is None:
            return None
        return str(value)

    def validate_progression(
        self,
        check: "VersionGovernanceCheck",
        release: "VersionReleaseContext",
    ) -> List:
        """Apply optional custom progression validation."""
        scheme = self._load_scheme(check, release.repo_root)
        validate = getattr(scheme, "validate_progression", None)
        if not callable(validate):
            return []
        return list(validate(check, release))

    def validate_release(
        self,
        check: "VersionGovernanceCheck",
        release: "VersionReleaseContext",
    ) -> List:
        """Apply optional scheme-specific release validation."""
        scheme = self._load_scheme(check, release.repo_root)
        return list(scheme.validate_release(check, release))

    @property
    def _loaded_scheme(self) -> "VersionScheme":
        """Return the only cached scheme after preflight loading."""
        if not self._cache:
            raise ValueError("Custom adapter scheme not loaded.")
        return next(iter(self._cache.values()))

    def _load_scheme(
        self,
        check: "VersionGovernanceCheck",
        repo_root: Path,
    ) -> "VersionScheme":
        """Load and cache the repository-local scheme object for this run."""
        path = self._resolve_adapter_path(check, repo_root)
        cached = self._cache.get(path)
        if cached is not None:
            return cached

        module_name = (
            "devcovenant_builtin_version_governance_custom_"
            f"{abs(hash(path.as_posix()))}"
        )
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ValueError(
                f"Cannot load custom adapter module from `{path}`."
            )
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        scheme = getattr(module, "SCHEME", None)
        if scheme is None:
            raise ValueError(
                "Custom adapter module must export `SCHEME` with the "
                "version-governance scheme interface."
            )

        missing = [
            name
            for name in (
                "version_pattern",
                "parse_version",
                "compare_versions",
                "validate_release",
            )
            if not callable(getattr(scheme, name, None))
        ]
        if missing:
            names = ", ".join(missing)
            raise ValueError(
                "Custom adapter `SCHEME` is missing required callables: "
                f"{names}."
            )
        self._cache[path] = scheme
        return scheme

    def _resolve_adapter_path(
        self,
        check: "VersionGovernanceCheck",
        repo_root: Path,
    ) -> Path:
        """Resolve and validate the repository-local adapter path."""
        raw = str(check.get_option("custom_adapter_path", "")).strip()
        if not raw:
            raise ValueError(
                "Set `custom_adapter_path` when using the "
                "`custom_adapter` version-governance scheme."
            )
        candidate = Path(raw)
        if candidate.is_absolute():
            raise ValueError("`custom_adapter_path` must stay repo-relative.")
        resolved = (repo_root / candidate).resolve()
        try:
            resolved.relative_to(repo_root.resolve())
        except ValueError as exc:
            raise ValueError(
                "`custom_adapter_path` must resolve inside the repository."
            ) from exc
        if resolved.suffix != ".py":
            raise ValueError(
                "`custom_adapter_path` must point to a Python module."
            )
        if not resolved.is_file():
            raise ValueError(
                "Custom adapter module "
                f"`{candidate.as_posix()}` does not exist."
            )
        return resolved
