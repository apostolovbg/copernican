"""
Policy: Package Artifact Mirror

Ensure package-shipped artifacts mirror the canonical repository-root artifacts
they are derived from.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from devcovenant.core.policy_contract import (
    CheckContext,
    PolicyCheck,
    Violation,
)


class PackageArtifactMirrorCheck(PolicyCheck):
    """Verify configured repository-root files and directories mirror into
    the package."""

    policy_id = "package-artifact-mirror"
    version = "0.1.0"

    def check(self, context: CheckContext) -> List[Violation]:
        """Report missing, divergent, or stale package mirror targets."""
        repo_root = context.repo_root
        violations: List[Violation] = []

        for source_rel, target_rel in self._mirror_pairs("file_mirrors"):
            source_path = repo_root / source_rel
            target_path = repo_root / target_rel
            if not source_path.is_file():
                violations.append(
                    Violation(
                        policy_id=self.policy_id,
                        severity="error",
                        file_path=source_path,
                        message=(
                            "Configured package-artifact file mirror source "
                            f"`{source_rel.as_posix()}` is missing."
                        ),
                    )
                )
                continue
            if not target_path.is_file():
                violations.append(
                    self._sync_violation(
                        source_path=source_path,
                        target_path=target_path,
                        message=(
                            f"`{target_rel.as_posix()}` is missing and must "
                            f"mirror `{source_rel.as_posix()}`."
                        ),
                        kind="file",
                    )
                )
                continue
            if source_path.read_bytes() != target_path.read_bytes():
                violations.append(
                    self._sync_violation(
                        source_path=source_path,
                        target_path=target_path,
                        message=(
                            f"`{target_rel.as_posix()}` diverges from "
                            f"canonical source `{source_rel.as_posix()}`."
                        ),
                        kind="file",
                    )
                )

        for source_rel, target_rel in self._mirror_pairs("dir_mirrors"):
            source_dir = repo_root / source_rel
            target_dir = repo_root / target_rel
            if not source_dir.is_dir():
                violations.append(
                    Violation(
                        policy_id=self.policy_id,
                        severity="error",
                        file_path=source_dir,
                        message=(
                            "Configured package-artifact dir mirror source "
                            f"`{source_rel.as_posix()}` is missing."
                        ),
                    )
                )
                continue
            exempt_rel_paths = self._dir_exempt_target_rel_paths(
                repo_root,
                target_dir,
            )
            skipped_rel_paths = self._dir_skip_rel_paths(source_rel)
            mismatch = self._dir_mismatch(
                repo_root,
                source_dir,
                target_dir,
                exempt_rel_paths=exempt_rel_paths,
                skipped_rel_paths=skipped_rel_paths,
            )
            if mismatch is None:
                continue
            violations.append(
                self._sync_violation(
                    source_path=source_dir,
                    target_path=target_dir,
                    message=mismatch,
                    kind="dir",
                    extra_context={
                        "preserved_paths": [
                            path.as_posix()
                            for path in sorted(
                                exempt_rel_paths | skipped_rel_paths
                            )
                        ],
                        "ignored_paths": [
                            path.as_posix()
                            for path in sorted(skipped_rel_paths)
                        ],
                    },
                )
            )

        return violations

    def _sync_violation(
        self,
        *,
        source_path: Path,
        target_path: Path,
        message: str,
        kind: str,
        extra_context: dict[str, object] | None = None,
    ) -> Violation:
        """Build one auto-fixable mirror-sync violation."""
        context: dict[str, object] = {
            "kind": kind,
            "source_path": str(source_path),
            "target_path": str(target_path),
        }
        if extra_context:
            context.update(extra_context)
        return Violation(
            policy_id=self.policy_id,
            severity="error",
            file_path=target_path,
            message=message,
            suggestion=(
                "Sync package mirror artifacts from the canonical "
                "repository-root sources."
            ),
            can_auto_fix=True,
            context=context,
        )

    def _mirror_pairs(self, option_key: str) -> list[tuple[Path, Path]]:
        """Return normalized `source=>target` metadata pairs."""
        raw_value = self.get_option(option_key, [])
        tokens: list[str] = []
        if isinstance(raw_value, str):
            tokens = [
                token.strip()
                for token in raw_value.replace("\n", ",").split(",")
                if token.strip()
            ]
        elif isinstance(raw_value, (list, tuple, set)):
            tokens = [str(token).strip() for token in raw_value if str(token)]
        pairs: list[tuple[Path, Path]] = []
        for token in tokens:
            if "=>" not in token:
                continue
            left, right = token.split("=>", 1)
            left_token = left.strip()
            right_token = right.strip()
            if left_token and right_token:
                pairs.append((Path(left_token), Path(right_token)))
        return pairs

    def _dir_mismatch(
        self,
        repo_root: Path,
        source_dir: Path,
        target_dir: Path,
        *,
        exempt_rel_paths: set[Path],
        skipped_rel_paths: set[Path],
    ) -> str | None:
        """Return one summary mismatch message for one mirrored directory."""
        source_rel = source_dir.relative_to(repo_root).as_posix()
        target_rel = target_dir.relative_to(repo_root).as_posix()
        if not target_dir.exists():
            return (
                f"`{target_rel}` is missing and must mirror "
                f"`{source_rel}`."
            )
        if not target_dir.is_dir():
            return f"`{target_rel}` exists but is not a directory mirror."

        source_files = {
            rel_path: path
            for rel_path, path in self._relative_file_map(source_dir).items()
            if rel_path not in skipped_rel_paths
        }
        target_files = {
            rel_path: path
            for rel_path, path in self._relative_file_map(target_dir).items()
            if rel_path not in skipped_rel_paths
        }
        missing = sorted(set(source_files) - set(target_files))
        extra = sorted(
            rel_path
            for rel_path in set(target_files) - set(source_files)
            if rel_path not in exempt_rel_paths
        )
        changed = sorted(
            rel_path
            for rel_path in set(source_files) & set(target_files)
            if source_files[rel_path].read_bytes()
            != target_files[rel_path].read_bytes()
        )
        if not missing and not extra and not changed:
            return None

        details: list[str] = []
        if missing:
            details.append(
                "missing " + ", ".join(path.as_posix() for path in missing[:4])
            )
        if extra:
            details.append(
                "extra " + ", ".join(path.as_posix() for path in extra[:4])
            )
        if changed:
            details.append(
                "changed " + ", ".join(path.as_posix() for path in changed[:4])
            )
        return (
            f"`{target_rel}` must mirror `{source_rel}` exactly "
            f"({'; '.join(details)})."
        )

    def _dir_exempt_target_rel_paths(
        self, repo_root: Path, target_dir: Path
    ) -> set[Path]:
        """Return file-mirror targets that live under one dir mirror root."""
        exempt_paths: set[Path] = set()
        for _, target_rel in self._mirror_pairs("file_mirrors"):
            absolute_target = repo_root / target_rel
            if absolute_target == target_dir:
                continue
            try:
                exempt_paths.add(absolute_target.relative_to(target_dir))
            except ValueError:
                continue
        return exempt_paths

    def _dir_skip_rel_paths(self, source_dir_rel: Path) -> set[Path]:
        """Return configured relative paths skipped for one dir mirror."""
        raw_value = self.get_option("dir_skip_paths", [])
        if isinstance(raw_value, str):
            tokens = [
                token.strip()
                for token in raw_value.replace("\n", ",").split(",")
                if token.strip()
            ]
        elif isinstance(raw_value, (list, tuple, set)):
            tokens = [str(token).strip() for token in raw_value if str(token)]
        else:
            tokens = []

        skipped_paths: set[Path] = set()
        for token in tokens:
            if "=>" not in token:
                continue
            left, right = token.split("=>", 1)
            if Path(left.strip()) != source_dir_rel:
                continue
            right_token = right.strip()
            if right_token:
                skipped_paths.add(Path(right_token))
        return skipped_paths

    def _relative_file_map(self, root: Path) -> dict[Path, Path]:
        """Return regular files under one root keyed by relative path."""
        return {
            path.relative_to(root): path
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }
