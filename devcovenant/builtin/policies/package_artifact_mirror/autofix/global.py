"""Fixer: Package Artifact Mirror.

Sync package-shipped mirror artifacts from their canonical
repository-root sources.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from devcovenant.core.policy_contract import FixResult, PolicyFixer, Violation


class PackageArtifactMirrorFixer(PolicyFixer):
    """Rewrite configured package-artifact mirrors from their source paths."""

    policy_id = "package-artifact-mirror"

    def can_fix(self, violation: Violation) -> bool:
        """Return True when the violation belongs to this mirror policy."""
        return violation.policy_id == self.policy_id

    def fix(self, violation: Violation) -> FixResult:
        """Sync one file or directory mirror from source to target."""
        kind = str(violation.context.get("kind") or "").strip()
        source_value = str(violation.context.get("source_path") or "").strip()
        target_value = str(violation.context.get("target_path") or "").strip()
        if not kind or not source_value or not target_value:
            return FixResult(
                success=False,
                message="Missing kind, source_path, or target_path context.",
            )

        source = Path(source_value)
        target = Path(target_value)
        if kind == "file":
            if not source.is_file():
                return FixResult(
                    success=False,
                    message=f"Mirror source file is missing: {source}.",
                )
            if target.exists() and target.is_dir():
                shutil.rmtree(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            return FixResult(
                success=True,
                message=f"Synced {target} from {source}.",
                files_modified=[target],
            )
        if kind == "dir":
            if not source.is_dir():
                return FixResult(
                    success=False,
                    message=f"Mirror source directory is missing: {source}.",
                )
            preserved_paths = self._preserved_file_payloads(violation)
            ignored_paths = self._ignored_rel_paths(violation)
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, target)
            for relative_path in ignored_paths:
                ignored_target = target / relative_path
                if ignored_target.is_dir():
                    shutil.rmtree(ignored_target)
                elif ignored_target.exists():
                    ignored_target.unlink()
            for relative_path, payload in preserved_paths.items():
                preserved_target = target / relative_path
                preserved_target.parent.mkdir(parents=True, exist_ok=True)
                preserved_target.write_bytes(payload)
            return FixResult(
                success=True,
                message=f"Synced {target} from {source}.",
                files_modified=[target],
            )
        return FixResult(
            success=False,
            message=f"Unsupported package-artifact mirror kind: {kind}.",
        )

    def _preserved_file_payloads(
        self, violation: Violation
    ) -> dict[Path, bytes]:
        """Return separately mirrored file payloads under one dir target."""
        target_value = str(violation.context.get("target_path") or "").strip()
        preserved_values = violation.context.get("preserved_paths", [])
        if not target_value or not isinstance(preserved_values, list):
            return {}
        target_root = Path(target_value)
        preserved_payloads: dict[Path, bytes] = {}
        for raw_path in preserved_values:
            rel_path = Path(str(raw_path).strip())
            if not str(rel_path):
                continue
            file_path = target_root / rel_path
            if file_path.is_file():
                preserved_payloads[rel_path] = file_path.read_bytes()
        return preserved_payloads

    def _ignored_rel_paths(self, violation: Violation) -> list[Path]:
        """Return relative target paths that should not be mirrored."""
        ignored_values = violation.context.get("ignored_paths", [])
        if not isinstance(ignored_values, list):
            return []
        ignored_paths: list[Path] = []
        for raw_path in ignored_values:
            rel_path = Path(str(raw_path).strip())
            if str(rel_path):
                ignored_paths.append(rel_path)
        return ignored_paths
