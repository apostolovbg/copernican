"""Remind contributors to grow and maintain documentation quality."""

import fnmatch
import re
from pathlib import Path, PurePosixPath
from typing import Iterable, List, Sequence

from devcovenant.core.policy_contract import (
    CheckContext,
    PolicyCheck,
    Violation,
)


def _normalize_list(raw: object | None) -> List[str]:
    """Return a flattened list of non-empty strings."""
    if raw is None:
        return []
    if isinstance(raw, str):
        candidates: Iterable[str] = raw.split(",")
    elif isinstance(raw, Iterable):
        candidates = raw  # type: ignore[assignment]
    else:
        candidates = [str(raw)]
    normalized: List[str] = []
    for entry in candidates:
        text = str(entry).strip()
        if text:
            normalized.append(text)
    return normalized


def _matches_doc_target(rel_path: PurePosixPath, targets: List[str]) -> bool:
    """Return True when rel_path matches a configured documentation target."""
    for raw_target in targets:
        target = raw_target.strip().replace("\\", "/")
        if not target:
            continue
        target_path = PurePosixPath(target)
        if "/" in target and rel_path.as_posix() == target_path.as_posix():
            return True
        if rel_path.name == target_path.name:
            return True
    return False


def _is_doc_target_touched(target: str, touched: set[PurePosixPath]) -> bool:
    """Return True when a target doc is present in touched doc paths."""
    normalized = target.strip().replace("\\", "/")
    if not normalized:
        return False
    target_path = PurePosixPath(normalized)
    if "/" in normalized:
        return target_path in touched
    return any(path.name == target_path.name for path in touched)


def _parse_doc_routes(
    raw: object | None,
) -> tuple[List[tuple[str, List[str]]], List[str]]:
    """Return parsed route rules and configuration errors."""
    routes: List[tuple[str, List[str]]] = []
    errors: List[str] = []
    for entry in _normalize_list(raw):
        if "=>" not in entry:
            errors.append(
                "Each `doc_routes` entry must use "
                "`trigger => doc1, doc2` format."
            )
            continue
        trigger_raw, docs_raw = entry.split("=>", 1)
        trigger = trigger_raw.strip()
        docs = [
            doc.strip().replace("\\", "/")
            for doc in docs_raw.split(",")
            if doc.strip()
        ]
        if not trigger:
            errors.append("`doc_routes` entries require a non-empty trigger.")
            continue
        if not docs:
            errors.append(
                "`doc_routes` entries require at least one documentation path."
            )
            continue
        routes.append((trigger, docs))
    return routes, errors


def _parse_bool_option(value: object, key: str) -> bool:
    """Parse a strict boolean option value."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"`{key}` must be boolean.")


def _parse_int_option(value: object, key: str) -> int:
    """Parse an integer option value."""
    if value is None:
        raise ValueError(f"`{key}` is required.")
    try:
        return int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"`{key}` must be integer.") from exc


def _parse_severity_option(value: object) -> str:
    """Parse and validate policy severity."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("`severity` is required.")
    normalized = value.strip().lower()
    allowed = {"critical", "error", "warning", "info"}
    if normalized not in allowed:
        raise ValueError(
            "`severity` must be one of: critical, error, warning, info."
        )
    return normalized


def _required_bool_option(
    policy: PolicyCheck,
    key: str,
    errors: List[str],
) -> bool | None:
    """Parse a required boolean option or append a config error."""
    value = policy.get_option(key)
    try:
        return _parse_bool_option(value, key)
    except ValueError as exc:
        errors.append(str(exc))
        return None


def _required_int_option(
    policy: PolicyCheck,
    key: str,
    errors: List[str],
) -> int | None:
    """Parse a required integer option or append a config error."""
    value = policy.get_option(key)
    try:
        return _parse_int_option(value, key)
    except ValueError as exc:
        errors.append(str(exc))
        return None


def _config_violation(policy_id: str, message: str) -> Violation:
    """Build a strict configuration violation for policy metadata issues."""
    return Violation(
        policy_id=policy_id,
        severity="error",
        message=f"Invalid documentation policy configuration: {message}",
    )


def _route_matches(rel_path: PurePosixPath, trigger: str) -> bool:
    """Return True when rel_path matches a route trigger."""
    rel = rel_path.as_posix()
    normalized = trigger.strip().replace("\\", "/")
    if not normalized:
        return False
    if normalized.endswith("/"):
        prefix = normalized.rstrip("/")
        return rel == prefix or rel.startswith(f"{prefix}/")
    if any(char in normalized for char in "*?[]"):
        pattern = normalized
        if pattern.startswith("./"):
            pattern = pattern[2:]
        return rel_path.match(pattern)
    return rel == normalized or rel_path.name == PurePosixPath(normalized).name


def _extract_headings(text: str) -> List[str]:
    """Return lower-cased Markdown heading text."""
    headings: List[str] = []
    for line in text.splitlines():
        if not line.lstrip().startswith("#"):
            continue
        title = line.lstrip("#").strip()
        if title:
            headings.append(title.lower())
    return headings


def _count_sections(text: str) -> int:
    """Count Markdown section headings (level 2 or deeper)."""
    return sum(
        1 for line in text.splitlines() if line.lstrip().startswith("##")
    )


def _word_count(text: str) -> int:
    """Return a simple word count."""
    return len([word for word in text.split() if word.strip()])


def _normalize_headings(raw: object | None) -> List[str]:
    """Return lower-cased required headings."""
    return [heading.lower() for heading in _normalize_list(raw)]


def _matches_prefixes(rel_path: PurePosixPath, prefixes: List[str]) -> bool:
    """Return True when rel_path starts with any prefix."""
    if not prefixes:
        return False
    rel = rel_path.as_posix()
    for prefix in prefixes:
        cleaned = prefix.strip().rstrip("/")
        if cleaned and (rel == cleaned or rel.startswith(f"{cleaned}/")):
            return True
    return False


def _matches_globs(rel_path: PurePosixPath, globs: List[str]) -> bool:
    """Return True when rel_path matches any glob pattern."""
    if not globs:
        return False
    rel = rel_path.as_posix()
    return any(fnmatch.fnmatch(rel, glob) for glob in globs if glob)


def _matches_suffixes(rel_path: PurePosixPath, suffixes: List[str]) -> bool:
    """Return True when rel_path ends with any configured suffix."""
    if not suffixes:
        return False
    return rel_path.suffix in {suffix for suffix in suffixes if suffix}


def _matches_files(rel_path: PurePosixPath, files: List[str]) -> bool:
    """Return True when rel_path matches any filename or path."""
    if not files:
        return False
    rel = rel_path.as_posix()
    for entry in files:
        cleaned = entry.strip().replace("\\", "/")
        if not cleaned:
            continue
        if "/" in cleaned and rel == cleaned:
            return True
        if rel_path.name == cleaned:
            return True
    return False


def _matches_keywords(rel_path: PurePosixPath, keywords: List[str]) -> bool:
    """Return True when rel_path contains any user-facing keyword."""
    if not keywords:
        return False
    tokens = {
        token.lower()
        for token in re.split(r"[\\/._-]+", rel_path.as_posix())
        if token
    }
    keyword_set = {word.lower() for word in keywords if word}
    return bool(tokens & keyword_set)


def _tokenize_path(
    rel_path: PurePosixPath,
    min_length: int,
    stopwords: List[str],
) -> List[str]:
    """Return mention tokens derived from a path."""
    tokens: set[str] = set()
    stopset = {word.lower() for word in stopwords}
    text = "/".join(rel_path.parts)
    for raw in re.split(r"[\\/._-]+", text):
        token = raw.strip().lower()
        if not token or token in stopset:
            continue
        if len(token) < min_length:
            continue
        tokens.add(token)
    return sorted(tokens)


class DocumentationGrowthTrackingCheck(PolicyCheck):
    """Remind contributors to add and maintain documentation quality."""

    policy_id = "documentation-growth-tracking"
    version = "1.2.1"

    def check(self, context: CheckContext):
        """Remind contributors to expand and maintain documentation."""
        state = context.change_state
        has_runtime_scope = bool(
            state.current_snapshot_paths
            or state.session_paths
            or state.session_error
            or state.session_valid
            or state.stage
        )
        if has_runtime_scope:
            try:
                files = self.scoped_changed_files(context)
            except ValueError as error:
                return [
                    Violation(
                        policy_id=self.policy_id,
                        severity="error",
                        file_path=context.repo_root,
                        message=str(error),
                    )
                ]
        else:
            files = list(context.changed_files)
        config_errors: List[str] = []
        raw_severity = self.get_option("severity")
        try:
            policy_severity = _parse_severity_option(raw_severity)
        except ValueError as exc:
            return [_config_violation(self.policy_id, str(exc))]

        doc_targets = _normalize_list(self.get_option("user_visible_files"))
        quality_targets = _normalize_list(self.get_option("doc_quality_files"))
        user_prefixes = _normalize_list(
            self.get_option("user_facing_prefixes")
        )
        user_globs = _normalize_list(self.get_option("user_facing_globs"))
        user_suffixes = _normalize_list(
            self.get_option("user_facing_suffixes")
        )
        user_keywords = _normalize_list(
            self.get_option("user_facing_keywords")
        )
        user_files = _normalize_list(self.get_option("user_facing_files"))
        exclude_prefixes = _normalize_list(
            self.get_option("user_facing_exclude_prefixes")
        )
        exclude_globs = _normalize_list(
            self.get_option("user_facing_exclude_globs")
        )
        exclude_suffixes = _normalize_list(
            self.get_option("user_facing_exclude_suffixes")
        )
        required_headings = _normalize_headings(
            self.get_option("required_headings")
        )
        require_toc = _required_bool_option(self, "require_toc", config_errors)
        min_sections = _required_int_option(
            self, "min_section_count", config_errors
        )
        min_words = _required_int_option(self, "min_word_count", config_errors)
        mention_required = _required_bool_option(
            self, "require_mentions", config_errors
        )
        mention_min_length = _required_int_option(
            self, "mention_min_length", config_errors
        )
        mention_stopwords = _normalize_list(
            self.get_option("mention_stopwords")
        )
        doc_routes, route_errors = _parse_doc_routes(
            self.get_option("doc_routes")
        )
        for error_text in route_errors:
            config_errors.append(error_text)
        if not doc_targets:
            config_errors.append("`user_visible_files` must not be empty.")
        if not quality_targets:
            config_errors.append("`doc_quality_files` must not be empty.")
        if not any(
            [
                user_prefixes,
                user_globs,
                user_suffixes,
                user_keywords,
                user_files,
            ]
        ):
            config_errors.append(
                "At least one user-facing selector is required: "
                "`user_facing_prefixes`, `user_facing_globs`, "
                "`user_facing_suffixes`, `user_facing_keywords`, or "
                "`user_facing_files`."
            )
        if require_toc is None:
            config_errors.append("`require_toc` must be a boolean.")
        if min_sections is None:
            config_errors.append("`min_section_count` must be an integer.")
        if min_words is None:
            config_errors.append("`min_word_count` must be an integer.")
        if mention_required is None:
            config_errors.append("`require_mentions` must be a boolean.")
        if mention_min_length is None:
            config_errors.append("`mention_min_length` must be an integer.")
        if min_sections is not None and min_sections < 0:
            config_errors.append("`min_section_count` must be >= 0.")
        if min_words is not None and min_words < 0:
            config_errors.append("`min_word_count` must be >= 0.")
        if mention_min_length is not None and mention_min_length < 1:
            config_errors.append("`mention_min_length` must be >= 1.")

        if config_errors:
            return [
                _config_violation(self.policy_id, message)
                for message in dict.fromkeys(config_errors)
            ]

        doc_touched: List[PurePosixPath] = []
        user_facing_touched: List[PurePosixPath] = []
        doc_quality_violations: List[Violation] = []
        scope_selectors = any(
            [user_prefixes, user_globs, user_files, user_keywords]
        )

        for path in files:
            rel = self._relative_path(path, context.repo_root)
            if rel is None:
                continue
            if _matches_doc_target(rel, doc_targets):
                doc_touched.append(rel)
                continue
            if _matches_prefixes(rel, exclude_prefixes):
                continue
            if _matches_globs(rel, exclude_globs):
                continue
            if _matches_suffixes(rel, exclude_suffixes):
                continue
            file_match = _matches_files(rel, user_files)
            prefix_match = _matches_prefixes(rel, user_prefixes)
            glob_match = _matches_globs(rel, user_globs)
            suffix_match = _matches_suffixes(rel, user_suffixes)
            keyword_match = _matches_keywords(rel, user_keywords)
            if scope_selectors:
                if file_match:
                    user_facing_touched.append(rel)
                elif keyword_match and (not user_suffixes or suffix_match):
                    user_facing_touched.append(rel)
                elif (prefix_match or glob_match) and (
                    not user_suffixes or suffix_match
                ):
                    user_facing_touched.append(rel)
            elif suffix_match:
                user_facing_touched.append(rel)

        doc_candidates: Sequence[Path] = context.all_files or files
        doc_texts: dict[PurePosixPath, str] = {}
        for path in doc_candidates:
            rel = self._relative_path(path, context.repo_root)
            if rel is None:
                continue
            if not _matches_doc_target(rel, quality_targets):
                continue
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8")
            doc_texts[rel] = text.lower()
            headings = _extract_headings(text)
            section_count = _count_sections(text)
            word_count = _word_count(text)

            missing = [
                heading
                for heading in required_headings
                if heading not in headings
            ]
            if require_toc and "table of contents" not in headings:
                missing.append("table of contents")

            quality_messages: List[str] = []
            if missing:
                missing_list = ", ".join(sorted(set(missing)))
                quality_messages.append(f"missing headings: {missing_list}")
            if min_sections and section_count < min_sections:
                quality_messages.append(
                    "requires at least "
                    f"{min_sections} sections (found {section_count})"
                )
            if min_words and word_count < min_words:
                quality_messages.append(
                    "requires at least "
                    f"{min_words} words (found {word_count})"
                )

            if quality_messages:
                doc_quality_violations.append(
                    Violation(
                        policy_id=self.policy_id,
                        severity=policy_severity,
                        message=(
                            "Documentation quality issue: "
                            + "; ".join(quality_messages)
                        ),
                        file_path=path,
                    )
                )

        violations = doc_quality_violations
        if not user_facing_touched:
            return violations
        doc_touched_set = set(doc_touched)
        for rel in user_facing_touched:
            route_matched = False
            for trigger, required_docs in doc_routes:
                if not _route_matches(rel, trigger):
                    continue
                route_matched = True
                missing = [
                    target
                    for target in required_docs
                    if not _is_doc_target_touched(target, doc_touched_set)
                ]
                if not missing:
                    continue
                violations.append(
                    Violation(
                        policy_id=self.policy_id,
                        severity=policy_severity,
                        file_path=context.repo_root / rel,
                        message=(
                            "User-facing change requires documentation "
                            f"updates for route `{trigger}`. Missing: "
                            f"{', '.join(missing)}."
                        ),
                    )
                )
            if doc_routes and not route_matched:
                violations.append(
                    Violation(
                        policy_id=self.policy_id,
                        severity=policy_severity,
                        file_path=context.repo_root / rel,
                        message=(
                            "User-facing change has no doc_routes mapping "
                            f"for `{rel}`."
                        ),
                    )
                )
        if not doc_touched:
            targets = ", ".join(sorted(doc_targets))
            violations.append(
                Violation(
                    policy_id=self.policy_id,
                    severity=policy_severity,
                    message=(
                        "User-facing changes detected without doc updates. "
                        f"Expand {targets} before committing."
                    ),
                )
            )
            return violations
        if mention_required:
            doc_values = list(doc_texts.values())
            for rel in user_facing_touched:
                tokens = _tokenize_path(
                    rel,
                    mention_min_length,
                    mention_stopwords,
                )
                if not tokens:
                    continue
                if any(
                    any(token in doc for token in tokens) for doc in doc_values
                ):
                    continue
                violations.append(
                    Violation(
                        policy_id=self.policy_id,
                        severity=policy_severity,
                        file_path=context.repo_root / rel,
                        message=(
                            "Docs updated but missing references to "
                            f"user-facing change `{rel}`. Mention at least "
                            f"one of: {', '.join(tokens)}."
                        ),
                    )
                )

        return violations

    @staticmethod
    def _relative_path(path: Path, repo_root: Path) -> PurePosixPath | None:
        """Return a posix relative path when possible."""
        try:
            rel = path.relative_to(repo_root)
        except ValueError:
            return None
        return PurePosixPath(rel.as_posix())
