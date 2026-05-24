"""Policy: keep package-facing docs synchronized with repo sources."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import List

from devcovenant.core.policy_contract import (
    CheckContext,
    PolicyCheck,
    Violation,
)


class PackageDocSyncCheck(PolicyCheck):
    """Verify configured package docs match their repo-owned source docs."""

    policy_id = "package-doc-sync"
    version = "1.0.0"
    MARKDOWN_LINK_PATTERN = re.compile(r"(?<!!)\[([^\]]+)\]\(([^)\s]+)\)")
    MARKDOWN_IMAGE_PATTERN = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)\)")
    ABSOLUTE_TARGET_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")

    def check(self, context: CheckContext) -> List[Violation]:
        """Check configured source=>target doc sync pairs."""
        repo_root = context.repo_root
        violations: List[Violation] = []

        for source_rel, target_rel in self._sync_pairs():
            source_path = repo_root / source_rel
            target_path = repo_root / target_rel
            if not source_path.exists():
                violations.append(
                    Violation(
                        policy_id=self.policy_id,
                        severity="error",
                        file_path=source_path,
                        message=(
                            "Configured package-doc source "
                            f"`{source_rel.as_posix()}` is missing."
                        ),
                    )
                )
                continue

            source_text = source_path.read_text(encoding="utf-8")
            stripped, strip_error = self._strip_omitted_blocks(
                source_text,
                source_label=source_rel.as_posix(),
            )
            if strip_error:
                violations.append(
                    Violation(
                        policy_id=self.policy_id,
                        severity="error",
                        file_path=source_path,
                        message=strip_error,
                    )
                )
                continue

            rewritten, link_error = self._rewrite_packaged_links(
                repo_root,
                stripped,
            )
            if link_error:
                violations.append(
                    Violation(
                        policy_id=self.policy_id,
                        severity="error",
                        file_path=source_path,
                        message=link_error,
                    )
                )
                continue

            expected = self._normalize_text(rewritten)
            if not target_path.exists():
                violations.append(
                    self._sync_violation(
                        source_path=source_path,
                        target_path=target_path,
                        expected_text=expected,
                        message=(
                            f"`{target_rel.as_posix()}` is missing and must "
                            f"sync from `{source_rel.as_posix()}`."
                        ),
                    )
                )
                continue

            target_text = target_path.read_text(encoding="utf-8")
            if self._normalize_text(target_text) != expected:
                violations.append(
                    self._sync_violation(
                        source_path=source_path,
                        target_path=target_path,
                        expected_text=expected,
                        message=(
                            f"`{target_rel.as_posix()}` diverges from "
                            f"`{source_rel.as_posix()}` after configured "
                            "package-doc transforms."
                        ),
                    )
                )

        return violations

    def _sync_violation(
        self,
        *,
        source_path: Path,
        target_path: Path,
        expected_text: str,
        message: str,
    ) -> Violation:
        """Build one auto-fixable doc-sync violation."""
        return Violation(
            policy_id=self.policy_id,
            severity="error",
            file_path=target_path,
            message=message,
            suggestion=(
                "Sync package-facing docs from the configured repository "
                "source docs."
            ),
            can_auto_fix=True,
            context={
                "expected_text": expected_text,
                "source_path": str(source_path),
                "target_path": str(target_path),
            },
        )

    def _sync_pairs(self) -> list[tuple[Path, Path]]:
        """Return normalized `source=>target` sync pairs."""
        return self._pair_tokens("sync_pairs")

    def _omit_block_pairs(self) -> list[tuple[str, str]]:
        """Return configured begin=>end omit marker pairs."""
        return [
            (left.as_posix(), right.as_posix())
            for left, right in self._pair_tokens("omit_block_pairs")
        ]

    def _pair_tokens(self, option_key: str) -> list[tuple[Path, Path]]:
        """Return normalized `source=>target` option pairs."""
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

    def _strip_omitted_blocks(
        self,
        text: str,
        *,
        source_label: str,
    ) -> tuple[str, str | None]:
        """Remove configured omit blocks from one source doc."""
        stripped = text
        for begin, end in self._omit_block_pairs():
            if not begin or not end:
                continue
            has_begin = begin in stripped
            has_end = end in stripped
            if has_begin and not has_end:
                return (
                    text,
                    f"`{source_label}` has an unclosed package-doc omit "
                    "block.",
                )
            if has_end and not has_begin:
                return (
                    text,
                    f"`{source_label}` has a package-doc omit end marker "
                    "without a begin marker.",
                )
            while True:
                start = stripped.find(begin)
                if start == -1:
                    break
                finish = stripped.find(end, start)
                if finish == -1:
                    return (
                        text,
                        f"`{source_label}` has an unclosed package-doc omit "
                        "block.",
                    )
                finish += len(end)
                before = stripped[:start].rstrip()
                after = stripped[finish:].lstrip()
                if before and after:
                    stripped = before + "\n\n" + after
                else:
                    stripped = (before + "\n" + after).strip("\n")
                stripped = stripped.rstrip() + "\n"
        return stripped, None

    def _rewrite_packaged_links(
        self,
        repo_root: Path,
        text: str,
    ) -> tuple[str, str | None]:
        """Rewrite repo-relative Markdown links for packaged docs."""
        if not bool(self.get_option("rewrite_repo_relative_links", True)):
            return text, None

        has_repo_relative_links = any(
            self._is_repo_relative_target(match.group(2).strip())
            for match in self.MARKDOWN_LINK_PATTERN.finditer(text)
        )
        has_repo_relative_images = any(
            self._is_repo_relative_target(match.group(2).strip())
            for match in self.MARKDOWN_IMAGE_PATTERN.finditer(text)
        )
        (
            repository_url,
            blob_base,
            raw_base,
            raw_image_base,
            error,
        ) = self._resolve_repository_link_bases(repo_root)
        if error and (has_repo_relative_links or has_repo_relative_images):
            return None, error
        if error:
            return text, None

        normalized_repo_url = str(repository_url or "").rstrip("/")
        has_same_repo_absolute_links = any(
            self._normalize_packaged_target(
                match.group(2).strip(),
                repository_url=normalized_repo_url,
                blob_base=str(blob_base or ""),
                raw_base=str(raw_base or ""),
            )
            is not None
            for match in self.MARKDOWN_LINK_PATTERN.finditer(text)
        )
        has_same_repo_absolute_images = any(
            self._normalize_packaged_target(
                match.group(2).strip(),
                repository_url=normalized_repo_url,
                blob_base=str(blob_base or ""),
                raw_base=str(raw_base or ""),
            )
            is not None
            for match in self.MARKDOWN_IMAGE_PATTERN.finditer(text)
        )
        if not any(
            (
                has_repo_relative_links,
                has_repo_relative_images,
                has_same_repo_absolute_links,
                has_same_repo_absolute_images,
            )
        ):
            return text, None

        # Keep image targets on the raw-content base so PyPI renders them.
        def _replace_image(match: re.Match[str]) -> str:
            label = match.group(1)
            target = match.group(2).strip()
            normalized = self._normalize_packaged_target(
                target,
                repository_url=normalized_repo_url,
                blob_base=str(blob_base or ""),
                raw_base=str(raw_base or ""),
            )
            if normalized is None:
                return match.group(0)
            return f"![{label}]({raw_image_base}{normalized})"

        # Keep document links on the blob base so package docs stay browsable.
        def _replace(match: re.Match[str]) -> str:
            label = match.group(1)
            target = match.group(2).strip()
            normalized = self._normalize_packaged_target(
                target,
                repository_url=normalized_repo_url,
                blob_base=str(blob_base or ""),
                raw_base=str(raw_base or ""),
            )
            if normalized is None:
                return match.group(0)
            return f"[{label}]({blob_base}{normalized})"

        rewritten = self.MARKDOWN_IMAGE_PATTERN.sub(_replace_image, text)
        rewritten = self.MARKDOWN_LINK_PATTERN.sub(_replace, rewritten)
        return rewritten, None

    def _resolve_repository_link_bases(
        self,
        repo_root: Path,
    ) -> tuple[str | None, str | None, str | None, str | None, str | None]:
        """Resolve repo link bases from `pyproject.toml`."""
        pyproject_path = repo_root / "pyproject.toml"
        if not pyproject_path.exists():
            return (
                None,
                None,
                None,
                None,
                "Package-doc sync found repo-relative public links, but "
                "`pyproject.toml` is missing.",
            )
        try:
            with pyproject_path.open("rb") as handle:
                payload = tomllib.load(handle)
        except OSError as exc:
            return (
                None,
                None,
                None,
                None,
                f"Failed to read `pyproject.toml`: {exc}.",
            )

        project = payload.get("project")
        if not isinstance(project, dict):
            return (
                None,
                None,
                None,
                None,
                "Package-doc sync found repo-relative public links, but "
                "`pyproject.toml` is missing `[project]` metadata.",
            )
        urls = project.get("urls")
        if not isinstance(urls, dict):
            urls = {}
        repository_url = str(
            urls.get("Repository") or urls.get("Homepage") or ""
        ).strip()
        if not repository_url:
            return (
                None,
                None,
                None,
                None,
                "Package-doc sync found repo-relative public links, but "
                "`pyproject.toml` is missing `project.urls.Repository` "
                "or `project.urls.Homepage`.",
            )
        normalized = repository_url.removesuffix(".git").rstrip("/")
        blob_base = f"{normalized}/blob/main/"
        if normalized.startswith("https://github.com/"):
            owner_repo = normalized.removeprefix("https://github.com/")
            raw_base = f"https://raw.githubusercontent.com/{owner_repo}/main/"
            raw_image_base = (
                f"https://raw.githubusercontent.com/{owner_repo}/main/"
            )
        else:
            raw_base = f"{normalized}/raw/main/"
            raw_image_base = f"{normalized}/raw/main/"
        return normalized, blob_base, raw_base, raw_image_base, None

    def _normalize_packaged_target(
        self,
        target: str,
        *,
        repository_url: str,
        blob_base: str,
        raw_base: str,
    ) -> str | None:
        """Normalize one target path for
        main-branch package-doc rewriting."""
        stripped = target.strip()
        if self._is_repo_relative_target(stripped):
            return stripped[2:] if stripped.startswith("./") else stripped

        if not repository_url:
            return None

        raw_prefixes = (
            f"{repository_url}/raw/main/",
            f"{repository_url}/raw/master/",
        )
        if repository_url.startswith("https://github.com/"):
            owner_repo = repository_url.removeprefix("https://github.com/")
            raw_prefixes += (
                f"https://raw.githubusercontent.com/{owner_repo}/main/",
                f"https://raw.githubusercontent.com/{owner_repo}/master/",
            )

        for prefix in raw_prefixes:
            if stripped.startswith(prefix):
                return stripped[len(prefix) :]

        for prefix in (
            f"{repository_url}/blob/main/",
            f"{repository_url}/blob/master/",
            f"{repository_url}/tree/main/",
            f"{repository_url}/tree/master/",
            blob_base,
            raw_base,
        ):
            if prefix and stripped.startswith(prefix):
                return stripped[len(prefix) :]

        return None

    def _is_repo_relative_target(self, target: str) -> bool:
        """Return True when one Markdown target is repo-relative."""
        if target.startswith("#"):
            return False
        return not self.ABSOLUTE_TARGET_PATTERN.match(target)

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text for comparison."""
        lines = [line.rstrip() for line in text.splitlines()]
        return "\n".join(lines).rstrip() + "\n"
