"""Ensure in-scope modules carry tests via shared translator LanguageUnit."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set, Tuple

import devcovenant.core.repository_paths as yaml_cache_service
from devcovenant.core.gate_runtime import capture_current_snapshot_paths
from devcovenant.core.policy_contract import (
    CheckContext,
    PolicyCheck,
    Violation,
)
from devcovenant.core.selectors import SelectorSet, build_watchlists

IGNORED_MODULE_STEMS = {"__init__", "__main__"}
IGNORED_TEST_FILE_NAMES = {"__init__.py", ".DS_Store"}
IGNORED_TEST_SUFFIXES = {".pyc", ".pyo"}
IGNORED_TEST_DIR_NAMES = {"__pycache__", ".pytest_cache"}
PLACEHOLDER_TEST_METHODS = {"test_placeholder"}
PLACEHOLDER_TEXT_MARKERS = (
    "placeholder-marker-alpha",
    "placeholder-marker-beta",
    "placeholder-marker-gamma",
)
DEFAULT_TEST_STYLE_REQUIREMENTS = ("python=>python_unittest",)


@dataclass(frozen=True)
class IndexedTestPath:
    """Pre-indexed test-path strings for repeated related-test lookups."""

    path: Path
    relative_lower: str
    test_name_lower: str
    compact_test_name: str


def _normalize_tokens(raw: object) -> list[str]:
    """Normalize metadata tokens to non-empty strings."""
    if raw is None:
        return []
    if isinstance(raw, str):
        values = [entry.strip() for entry in raw.split(",")]
    elif isinstance(raw, list):
        values = [str(entry).strip() for entry in raw]
    else:
        values = [str(raw).strip()]
    return [entry for entry in values if entry]


def _normalize_mirror_roots(raw_value: object) -> List[Tuple[str, str]]:
    """Parse mirror_roots metadata into (source, tests_root) pairs."""
    if raw_value is None:
        return []
    entries: list[str]
    if isinstance(raw_value, str):
        entries = [raw_value]
    elif isinstance(raw_value, list):
        entries = [str(entry).strip() for entry in raw_value]
    else:
        entries = [str(raw_value).strip()]

    rules: List[Tuple[str, str]] = []
    for raw_entry in entries:
        token = raw_entry.strip()
        if not token:
            continue
        if "=>" in token:
            source, target = token.split("=>", 1)
        elif ":" in token:
            source, target = token.split(":", 1)
        else:
            continue
        source_prefix = source.strip().replace("\\", "/").strip("/")
        target_prefix = target.strip().replace("\\", "/").strip("/")
        if not source_prefix or not target_prefix:
            continue
        rules.append((source_prefix, target_prefix))
    return rules


def _parse_lang_values(raw: object) -> dict[str, list[str]]:
    """Parse `language=>value` metadata into language-keyed lists."""
    mapping: dict[str, list[str]] = {}
    for token in _normalize_tokens(raw):
        if "=>" not in token:
            mapping.setdefault("*", []).append(token)
            continue
        language, value = token.split("=>", 1)
        language_token = language.strip().lower() or "*"
        value_token = value.strip()
        if not value_token:
            continue
        mapping.setdefault(language_token, []).append(value_token)
    return mapping


def _values_for_language(
    mapping: dict[str, list[str]], language: str
) -> tuple[str, ...]:
    """Resolve language values with wildcard defaults."""
    token = str(language or "").strip().lower()
    values: list[str] = []
    values.extend(mapping.get("*", []))
    values.extend(mapping.get(token, []))
    return tuple(values)


def _is_under_tests(
    path: Path, repo_root: Path, tests_dirs: List[str]
) -> bool:
    """Return True when path resolves under configured tests roots."""
    try:
        rel = path.relative_to(repo_root).as_posix()
    except ValueError:
        return False
    return any(
        rel == test_dir or rel.startswith(f"{test_dir}/")
        for test_dir in tests_dirs
    )


def _collect_repo_files(
    repo_root: Path,
    *,
    context: CheckContext | None = None,
) -> Set[Path]:
    """Return repository files using shared snapshot-scanner semantics."""
    del context
    return {
        repo_root / rel_path
        for rel_path in capture_current_snapshot_paths(repo_root)
    }


def _test_roots(policy: PolicyCheck) -> List[str]:
    """Resolve configured tests roots using watch and tests_watch metadata."""
    _, configured_watch_dirs = build_watchlists(
        policy,
        defaults={"watch_dirs": ["tests"]},
    )
    _, prefixed_tests_dirs = build_watchlists(
        policy,
        prefix="tests_",
        defaults={"watch_dirs": configured_watch_dirs or ["tests"]},
    )
    if prefixed_tests_dirs:
        return prefixed_tests_dirs
    if configured_watch_dirs:
        return configured_watch_dirs
    return ["tests"]


def _is_module_candidate(
    path: Path,
    *,
    selector: SelectorSet,
    repo_root: Path,
    tests_dirs: List[str],
) -> bool:
    """Return True when path represents a source module to track."""
    if not path.is_file():
        return False
    if path.name.startswith("test_"):
        return False
    if path.stem in IGNORED_MODULE_STEMS:
        return False
    if _is_under_tests(path, repo_root, tests_dirs):
        return False
    return selector.matches(path, repo_root)


def _list_existing_tests(
    repo_root: Path,
    tests_dirs: List[str],
    *,
    context: CheckContext | None = None,
) -> list[Path]:
    """Return existing files under configured tests roots."""
    del context
    discovered: list[Path] = []
    for tests_dir in tests_dirs:
        root = repo_root / tests_dir
        if not root.exists():
            continue
        for candidate in root.rglob("*"):
            if not candidate.is_file():
                continue
            if not _is_test_file_candidate(candidate, repo_root):
                continue
            discovered.append(candidate)
    return discovered


def _index_tests(
    tests: list[Path],
    *,
    repo_root: Path,
    context: CheckContext | None = None,
) -> tuple[IndexedTestPath, ...]:
    """Return reusable lowercase test-path metadata."""
    cache_bucket = (
        context.runtime_cache_bucket("modules_need_tests")
        if context is not None
        else None
    )
    cache_key = (
        "indexed_tests",
        tuple(path.relative_to(repo_root).as_posix() for path in tests),
    )
    if cache_bucket is not None and cache_key in cache_bucket:
        return tuple(cache_bucket[cache_key])
    indexed: list[IndexedTestPath] = []
    for test_path in tests:
        relative = test_path.relative_to(repo_root).as_posix().lower()
        test_name = Path(relative).name
        indexed.append(
            IndexedTestPath(
                path=test_path,
                relative_lower=relative,
                test_name_lower=test_name,
                compact_test_name=re.sub(r"[^a-zA-Z0-9]+", "", test_name),
            )
        )
    result = tuple(indexed)
    if cache_bucket is not None:
        cache_bucket[cache_key] = result
    return result


def _is_test_file_candidate(path: Path, repo_root: Path) -> bool:
    """Return True when one file path should count as a test file."""
    try:
        rel_parts = path.relative_to(repo_root).parts
    except ValueError:
        rel_parts = path.parts
    if any(part in IGNORED_TEST_DIR_NAMES for part in rel_parts):
        return False
    if path.name in IGNORED_TEST_FILE_NAMES:
        return False
    if path.suffix.lower() in IGNORED_TEST_SUFFIXES:
        return False
    return True


def _related_tests(
    module: Path,
    *,
    unit_templates: tuple[str, ...],
    indexed_tests: tuple[IndexedTestPath, ...],
) -> set[Path]:
    """Return related test paths for a source module."""
    stem = module.stem
    compact_stem = re.sub(r"[^a-zA-Z0-9]+", "", stem.lower())
    related: set[Path] = set()
    candidate_names = tuple(
        template.format(stem=stem).lower() for template in unit_templates
    )

    for indexed in indexed_tests:
        test = indexed.relative_lower
        for candidate_name in candidate_names:
            if test.endswith(f"/{candidate_name}") or test == candidate_name:
                related.add(indexed.path)
                break

    for indexed in indexed_tests:
        if stem.lower() in indexed.test_name_lower:
            related.add(indexed.path)
            continue
        if compact_stem and compact_stem in indexed.compact_test_name:
            related.add(indexed.path)

    return related


def _mirror_candidates(
    module: Path,
    *,
    repo_root: Path,
    mirror_roots: list[tuple[str, str]],
    language: str,
    unit_templates: tuple[str, ...],
    template_overrides: dict[str, list[str]],
) -> list[Path]:
    """Return mirror candidates for configured mirror roots."""
    try:
        rel = module.relative_to(repo_root).as_posix()
    except ValueError:
        return []
    templates = _values_for_language(template_overrides, language)
    if not templates:
        templates = unit_templates
    expected: list[Path] = []
    seen: set[Path] = set()
    for source_prefix, tests_prefix in mirror_roots:
        if rel != source_prefix and not rel.startswith(f"{source_prefix}/"):
            continue
        remainder = rel[len(source_prefix) :].lstrip("/")
        if not remainder:
            continue
        remainder_path = Path(remainder)
        source_dir = remainder_path.parent.as_posix()
        for template in templates:
            try:
                rendered = template.format(
                    stem=remainder_path.stem,
                    source_dir=source_dir,
                    source_file=remainder_path.name,
                )
            except (KeyError, ValueError):
                continue
            normalized = rendered.strip().replace("\\", "/").lstrip("/")
            if not normalized:
                continue
            candidate_rel = Path(normalized)
            if candidate_rel.parent == Path("."):
                candidate = (
                    repo_root
                    / tests_prefix
                    / remainder_path.parent
                    / candidate_rel.name
                )
            else:
                candidate = repo_root / tests_prefix / candidate_rel
            if candidate in seen:
                continue
            seen.add(candidate)
            expected.append(candidate)
    return expected


def _within_root(path: Path, root: Path) -> bool:
    """Return True when path is under the provided root directory."""
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _resolve_language_for_path(
    *,
    path: Path,
    runtime: object,
    context: CheckContext,
    policy_id: str,
) -> str:
    """Resolve translator language ID for one path, if available."""
    resolver = getattr(runtime, "resolve", None)
    if resolver is None:
        return ""
    try:
        resolution = resolver(
            path=path,
            policy_id=policy_id,
            context=context,
        )
    # DEVCOV_ALLOW_BROAD_ONCE translator resolution boundary.
    except Exception:
        return ""
    if not getattr(resolution, "is_resolved", False):
        return ""
    declaration = getattr(resolution, "declaration", None)
    if declaration is None:
        return ""
    return str(getattr(declaration, "translator_id", "")).strip().lower()


def _validate_python_unittest_style(
    path: Path,
    *,
    placeholder_methods: set[str],
    placeholder_markers: tuple[str, ...],
) -> str | None:
    """Return violation message when Python tests are not unittest-style."""
    if path.suffix.lower() != ".py" or not path.name.startswith("test_"):
        return None
    try:
        content = yaml_cache_service.read_text(path)
    except UnicodeDecodeError as error:
        return (
            "Python test modules must be UTF-8 decodable; "
            f"unable to read {path.name}: {error}"
        )
    except OSError as error:
        return (
            "Unable to read Python test module while validating unittest "
            f"style: {error}"
        )
    normalized = content.lower()
    if any(marker in normalized for marker in placeholder_markers):
        return (
            "Placeholder test scaffolds are not allowed; "
            "replace with behavioral or contract assertions."
        )

    if "export_unittest_cases(" in content:
        return (
            "Remove unittest bridge usage and define explicit "
            "unittest.TestCase tests."
        )

    tree = yaml_cache_service.parse_python_ast(path)
    if tree is None:
        return None

    has_top_level = any(
        isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
        for node in tree.body
    )
    if has_top_level:
        return (
            "Module-level test_* functions are not allowed; "
            "use unittest.TestCase methods."
        )

    has_unittest = False
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        is_testcase = any(
            (
                isinstance(base, ast.Name)
                and base.id == "TestCase"
                or isinstance(base, ast.Attribute)
                and base.attr == "TestCase"
            )
            for base in node.bases
        )
        if not is_testcase:
            continue
        has_method = any(
            isinstance(method, ast.FunctionDef)
            and method.name.startswith("test_")
            for method in node.body
        )
        has_placeholder_method = any(
            isinstance(method, ast.FunctionDef)
            and method.name in placeholder_methods
            for method in node.body
        )
        if has_placeholder_method:
            return (
                "Placeholder test methods are not allowed; replace "
                "`test_placeholder` with concrete checks."
            )
        if has_method:
            has_unittest = True
            break

    if has_unittest:
        return None
    return (
        "Python test modules must define unittest.TestCase "
        "classes with test_* methods."
    )


def _validate_test_style(
    path: Path,
    *,
    language: str,
    style: str,
    placeholder_methods: set[str],
    placeholder_markers: tuple[str, ...],
) -> str | None:
    """Validate one configured test-style rule for a test path."""
    style_token = str(style or "").strip().lower()
    if style_token != "python_unittest":
        return None
    if str(language or "").strip().lower() != "python":
        return None
    return _validate_python_unittest_style(
        path,
        placeholder_methods=placeholder_methods,
        placeholder_markers=placeholder_markers,
    )


class ModulesNeedTestsCheck(PolicyCheck):
    """Ensure in-scope modules ship with accompanying tests."""

    policy_id = "modules-need-tests"
    version = "2.0.0"

    def check(self, context: CheckContext) -> List[Violation]:
        """Check that in-scope modules have corresponding tests."""
        runtime = context.translator_runtime
        if runtime is None:
            return []

        violations: List[Violation] = []
        repo_files = _collect_repo_files(context.repo_root, context=context)
        selector = SelectorSet.from_policy(self)
        tests_dirs = _test_roots(self)
        tests_label = ", ".join(sorted(tests_dirs))
        mirror_roots = _normalize_mirror_roots(
            self.get_option("mirror_roots", [])
        )
        mirror_templates = _parse_lang_values(
            self.get_option("mirror_test_name_templates", [])
        )
        style_requirements = _parse_lang_values(
            self.get_option(
                "test_style_requirements",
                list(DEFAULT_TEST_STYLE_REQUIREMENTS),
            )
        )
        placeholder_methods = {
            item
            for item in _normalize_tokens(
                self.get_option("placeholder_test_methods")
            )
        }
        if not placeholder_methods:
            placeholder_methods = set(PLACEHOLDER_TEST_METHODS)
        placeholder_markers = tuple(
            item.lower()
            for item in _normalize_tokens(
                self.get_option("placeholder_text_markers")
            )
        )
        if not placeholder_markers:
            placeholder_markers = PLACEHOLDER_TEXT_MARKERS

        test_files = _list_existing_tests(
            context.repo_root,
            tests_dirs,
            context=context,
        )
        indexed_tests = _index_tests(
            test_files,
            repo_root=context.repo_root,
            context=context,
        )
        module_files = [
            path
            for path in sorted(repo_files)
            if _is_module_candidate(
                path,
                selector=selector,
                repo_root=context.repo_root,
                tests_dirs=tests_dirs,
            )
        ]

        if module_files and not test_files:
            modules = ", ".join(
                path.relative_to(context.repo_root).as_posix()
                for path in module_files
            )
            violations.append(
                Violation(
                    policy_id=self.policy_id,
                    severity="error",
                    file_path=module_files[0],
                    message=(
                        f"No tests found under {tests_label}; "
                        "add tests before "
                        f"changing modules:\n{modules}"
                    ),
                )
            )
            return violations

        expected_mirror_paths: set[Path] = set()
        for module in module_files:
            resolution = runtime.resolve(
                path=module,
                policy_id=self.policy_id,
                context=context,
            )
            if not resolution.is_resolved:
                violations.extend(resolution.violations)
                continue
            try:
                source = yaml_cache_service.read_text(module)
            except UnicodeDecodeError as error:
                violations.append(
                    Violation(
                        policy_id=self.policy_id,
                        severity="error",
                        file_path=module,
                        message=(
                            "Module sources must be UTF-8 decodable while "
                            "validating related tests: "
                            f"{error}"
                        ),
                    )
                )
                continue
            except OSError as error:
                violations.append(
                    Violation(
                        policy_id=self.policy_id,
                        severity="error",
                        file_path=module,
                        message=(
                            "Unable to read module source while validating "
                            f"related tests: {error}"
                        ),
                    )
                )
                continue
            unit = runtime.translate(
                resolution,
                path=module,
                source=source,
                context=context,
            )
            if unit is None:
                continue

            expected = _mirror_candidates(
                module,
                repo_root=context.repo_root,
                mirror_roots=mirror_roots,
                language=unit.language,
                unit_templates=unit.test_name_templates,
                template_overrides=mirror_templates,
            )
            if expected:
                expected_mirror_paths.update(expected)
                if any(path.exists() for path in expected):
                    continue
                expected_display = ", ".join(
                    sorted(
                        path.relative_to(context.repo_root).as_posix()
                        for path in expected
                    )
                )
                violations.append(
                    Violation(
                        policy_id=self.policy_id,
                        severity="error",
                        file_path=module,
                        message=(
                            "Mirror mode requires tests under mirrored paths. "
                            f"Add one of: {expected_display}"
                        ),
                    )
                )
                continue

            related_tests = _related_tests(
                module=module,
                unit_templates=unit.test_name_templates,
                indexed_tests=indexed_tests,
            )
            if related_tests:
                continue
            module_rel = module.relative_to(context.repo_root).as_posix()
            violations.append(
                Violation(
                    policy_id=self.policy_id,
                    severity="error",
                    file_path=module,
                    message=(
                        f"Add related tests under {tests_label}/ for module: "
                        f"{module_rel}"
                    ),
                )
            )

        if mirror_roots:
            mirror_roots_abs = [
                context.repo_root / tests_prefix
                for _, tests_prefix in mirror_roots
            ]
            for test_file in sorted(test_files):
                if not any(
                    _within_root(test_file, mirror_root)
                    for mirror_root in mirror_roots_abs
                ):
                    continue
                if test_file in expected_mirror_paths:
                    continue
                test_rel = test_file.relative_to(context.repo_root).as_posix()
                violations.append(
                    Violation(
                        policy_id=self.policy_id,
                        severity="error",
                        file_path=test_file,
                        message=(
                            "Remove stale mirrored file with no valid "
                            f"source module: {test_rel}"
                        ),
                    )
                )

        for test_path in sorted(test_files):
            language = _resolve_language_for_path(
                path=test_path,
                runtime=runtime,
                context=context,
                policy_id=self.policy_id,
            )
            styles = _values_for_language(style_requirements, language)
            if not styles:
                continue
            message = None
            for style in styles:
                message = _validate_test_style(
                    test_path,
                    language=language,
                    style=style,
                    placeholder_methods=placeholder_methods,
                    placeholder_markers=placeholder_markers,
                )
                if message:
                    break
            if message is None:
                continue
            violations.append(
                Violation(
                    policy_id=self.policy_id,
                    severity="error",
                    file_path=test_path,
                    message=message,
                )
            )

        return violations
