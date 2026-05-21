"""Centralized translator runtime and policy-agnostic LanguageUnit model."""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from devcovenant.core.policy_contract import Violation
from devcovenant.core.repository_paths import display_path

MODULE_FUNCTION = "module_function"
_CAN_HANDLE_CACHE_MAX_ENTRIES = 2048
_TRANSLATE_CACHE_MAX_ENTRIES = 512


@dataclass(frozen=True)
class IdentifierFact:
    """Identifier discovered by a translator."""

    name: str
    line_number: int
    kind: str


@dataclass(frozen=True)
class SymbolDocFact:
    """Symbol documentation status emitted by a translator."""

    kind: str
    name: str
    line_number: int
    documented: bool


@dataclass(frozen=True)
class RiskFact:
    """Potentially risky construct discovered by a translator."""

    severity: str
    line_number: int
    message: str


@dataclass(frozen=True)
class LanguageUnit:
    """Policy-agnostic translated representation for one source file."""

    translator_id: str
    profile_name: str
    language: str
    path: str
    suffix: str
    source: str
    module_documented: bool
    identifier_facts: tuple[IdentifierFact, ...]
    symbol_doc_facts: tuple[SymbolDocFact, ...]
    risk_facts: tuple[RiskFact, ...]
    test_name_templates: tuple[str, ...]


@dataclass(frozen=True)
class TranslatorDeclaration:
    """Normalized translator declaration owned by one language profile."""

    translator_id: str
    profile_name: str
    extensions: tuple[str, ...]
    can_handle_strategy: str
    can_handle_entrypoint: str
    translate_strategy: str
    translate_entrypoint: str


@dataclass(frozen=True)
class TranslatorResolution:
    """Result of resolving a translator for one file."""

    declaration: TranslatorDeclaration | None
    violations: tuple[Violation, ...]

    @property
    def is_resolved(self) -> bool:
        """Return True when exactly one translator matched."""
        return self.declaration is not None and not self.violations


def can_handle_declared_extensions(
    *, path: Path, declaration: TranslatorDeclaration, **_: Any
) -> bool:
    """Default can_handle strategy using declaration extensions."""
    return path.suffix.lower() in declaration.extensions


def translate_language_unit(
    *, path: Path, source: str, declaration: TranslatorDeclaration, **_: Any
) -> LanguageUnit:
    """Default translate strategy returning a minimal LanguageUnit."""
    language = str(declaration.translator_id or "").strip().lower()
    return LanguageUnit(
        translator_id=declaration.translator_id,
        profile_name=declaration.profile_name,
        language=language,
        path=str(path),
        suffix=path.suffix.lower(),
        source=source,
        module_documented=False,
        identifier_facts=tuple(),
        symbol_doc_facts=tuple(),
        risk_facts=tuple(),
        test_name_templates=tuple(),
    )


def can_handle(**kwargs: Any) -> bool:
    """Short alias for profile entrypoints."""
    return can_handle_declared_extensions(**kwargs)


def translate(**kwargs: Any) -> LanguageUnit:
    """Short alias for profile entrypoints."""
    return translate_language_unit(**kwargs)


def _bounded_cache_store(
    cache: dict[Any, Any],
    *,
    key: Any,
    value: Any,
    max_entries: int,
) -> None:
    """Store a cache value with deterministic bounded growth."""
    if key in cache:
        cache.pop(key, None)
    cache[key] = value
    while len(cache) > max_entries:
        oldest_key = next(iter(cache))
        cache.pop(oldest_key, None)


class TranslatorRuntime:
    """Resolve and invoke translators declared by active language profiles."""

    def __init__(
        self,
        repo_root: Path,
        profile_registry: dict[str, dict[str, Any]],
        active_profiles: list[str],
    ) -> None:
        """Store runtime state and precompute declarations by extension."""
        self.repo_root = Path(repo_root).resolve()
        self.profile_registry = profile_registry
        self.active_profiles = sorted(set(active_profiles))
        self._by_extension = self._build_extension_map()
        self._file_module_cache: dict[str, Any] = {}
        self._entrypoint_function_cache: dict[tuple[str, str], Any] = {}
        self._can_handle_result_cache: dict[tuple[Any, ...], bool] = {}
        self._translate_result_cache: dict[tuple[Any, ...], LanguageUnit] = {}

    def resolve(
        self,
        *,
        path: Path,
        policy_id: str,
        context: Any | None = None,
    ) -> TranslatorResolution:
        """Resolve one translator declaration for a file path."""
        extension = path.suffix.lower()
        candidates = list(self._by_extension.get(extension, []))
        if not candidates:
            violation = Violation(
                policy_id=policy_id,
                severity="error",
                file_path=path,
                message=(
                    f"No translator matched extension '{extension}' for "
                    f"policy '{policy_id}'."
                ),
                suggestion=(
                    "Declare a language-profile translator with matching "
                    "extensions in active profile metadata."
                ),
            )
            return TranslatorResolution(None, (violation,))

        matched: list[TranslatorDeclaration] = []
        for declaration in candidates:
            if self._can_handle(declaration, path=path, context=context):
                matched.append(declaration)

        if not matched:
            candidate_ids = ", ".join(
                sorted(d.translator_id for d in candidates)
            )
            violation = Violation(
                policy_id=policy_id,
                severity="error",
                file_path=path,
                message=(
                    "Translator arbitration found no accepted candidate for "
                    f"extension '{extension}'. Candidates: {candidate_ids}."
                ),
                suggestion=(
                    "Review can_handle strategy/entrypoint declarations in "
                    "active language profiles."
                ),
            )
            return TranslatorResolution(None, (violation,))

        if len(matched) > 1:
            ids = ", ".join(sorted(d.translator_id for d in matched))
            violation = Violation(
                policy_id=policy_id,
                severity="error",
                file_path=path,
                message=(
                    f"Translator arbitration is ambiguous for extension "
                    f"'{extension}'. Matched translators: {ids}."
                ),
                suggestion=(
                    "Adjust translator extensions or can_handle logic so "
                    "exactly one translator matches."
                ),
            )
            return TranslatorResolution(None, (violation,))

        return TranslatorResolution(matched[0], ())

    def translate(
        self,
        resolution: TranslatorResolution,
        *,
        path: Path,
        source: str,
        context: Any | None = None,
    ) -> LanguageUnit | None:
        """Invoke the translate strategy for a resolved declaration."""
        declaration = resolution.declaration
        if declaration is None:
            return None
        translate_cache_key = (
            id(context),  # name-clarity: allow
            declaration,
            str(path),
            source,
        )
        cached_unit = self._translate_result_cache.get(translate_cache_key)
        if cached_unit is not None:
            return cached_unit
        translated = self._invoke_strategy(
            declaration.translate_strategy,
            declaration.translate_entrypoint,
            path=path,
            source=source,
            context=context,
            declaration=declaration,
        )
        if isinstance(translated, LanguageUnit):
            safe_path = display_path(path, repo_root=self.repo_root)
            if translated.path != safe_path:
                translated = replace(translated, path=safe_path)
            _bounded_cache_store(
                self._translate_result_cache,
                key=translate_cache_key,
                value=translated,
                max_entries=_TRANSLATE_CACHE_MAX_ENTRIES,
            )
            return translated
        return None

    def _build_extension_map(self) -> dict[str, list[TranslatorDeclaration]]:
        """Build extension->declarations map from active language profiles."""
        mapping: dict[str, list[TranslatorDeclaration]] = {}
        for profile_name in self.active_profiles:
            metadata = self.profile_registry.get(profile_name, {})
            if metadata.get("category") != "language":
                continue
            translators = metadata.get("translators") or []
            for raw in translators:
                declaration = TranslatorDeclaration(
                    translator_id=str(raw["id"]),
                    profile_name=profile_name,
                    extensions=tuple(raw["extensions"]),
                    can_handle_strategy=str(raw["can_handle"]["strategy"]),
                    can_handle_entrypoint=str(raw["can_handle"]["entrypoint"]),
                    translate_strategy=str(raw["translate"]["strategy"]),
                    translate_entrypoint=str(raw["translate"]["entrypoint"]),
                )
                for extension in declaration.extensions:
                    mapping.setdefault(extension, []).append(declaration)
        return mapping

    def _can_handle(
        self,
        declaration: TranslatorDeclaration,
        *,
        path: Path,
        context: Any | None,
    ) -> bool:
        """Run the can_handle strategy and coerce to bool."""
        path_key = self._path_cache_key(path)
        can_handle_cache_key = (
            id(context),  # name-clarity: allow
            declaration,
            path_key,
        )
        if can_handle_cache_key in self._can_handle_result_cache:
            return self._can_handle_result_cache[can_handle_cache_key]
        raw_result = self._invoke_strategy(
            declaration.can_handle_strategy,
            declaration.can_handle_entrypoint,
            path=path,
            context=context,
            declaration=declaration,
        )
        result = bool(raw_result)
        _bounded_cache_store(
            self._can_handle_result_cache,
            key=can_handle_cache_key,
            value=result,
            max_entries=_CAN_HANDLE_CACHE_MAX_ENTRIES,
        )
        return result

    def _path_cache_key(self, path: Path) -> tuple[Any, ...]:
        """Return a stable path fingerprint for run-scoped caches."""
        normalized = str(path)
        try:
            stat = path.stat()
        except OSError:
            return (normalized, None, None)
        return (normalized, stat.st_mtime_ns, stat.st_size)

    def _invoke_strategy(
        self,
        strategy: str,
        entrypoint: str,
        **kwargs: Any,
    ) -> Any:
        """Invoke one strategy with deterministic supported modes."""
        if strategy != MODULE_FUNCTION:
            raise ValueError(
                f"Unsupported translator strategy '{strategy}'. "
                f"Expected '{MODULE_FUNCTION}'."
            )
        declaration = kwargs.get("declaration")
        if isinstance(entrypoint, str) and ":" in entrypoint:
            function = self._resolve_profile_file_entrypoint(
                entrypoint, declaration
            )
            return function(**kwargs)
        module_name, _, function_name = entrypoint.rpartition(".")
        if not module_name or not function_name:
            raise ValueError(
                f"Translator entrypoint '{entrypoint}' must be "
                "module.function."
            )
        module = importlib.import_module(module_name)
        function = getattr(module, function_name)
        return function(**kwargs)

    def _resolve_profile_file_entrypoint(
        self,
        entrypoint: str,
        declaration: TranslatorDeclaration | Any,
    ) -> Any:
        """Resolve file-style translator entrypoints."""
        if not isinstance(declaration, TranslatorDeclaration):
            raise ValueError(
                "File-style translator entrypoints need declaration context."
            )
        relative_path, _, function_name = entrypoint.partition(":")
        relative_path = str(relative_path or "").strip()
        function_name = str(function_name or "").strip()
        if not relative_path or not function_name:
            raise ValueError(
                f"Translator entrypoint '{entrypoint}' must be "
                "file.py:function."
            )
        relative_file_path = Path(relative_path)
        if relative_file_path.is_absolute():
            raise ValueError(
                f"Translator entrypoint '{entrypoint}' must use "
                "a relative file path."
            )
        if any(part == ".." for part in relative_file_path.parts):
            raise ValueError(
                f"Translator entrypoint '{entrypoint}' must stay "
                "within the profile root."
            )
        profile_meta = self.profile_registry.get(declaration.profile_name, {})
        profile_path = str(profile_meta.get("path", "")).strip()
        if not profile_path:
            raise ValueError(
                f"Profile '{declaration.profile_name}' has no path metadata."
            )
        profile_root = (self.repo_root / profile_path).resolve()
        try:
            profile_root.relative_to(self.repo_root)
        except ValueError as error:
            raise ValueError(
                f"Profile '{declaration.profile_name}' path escapes "
                "the repository root."
            ) from error
        module_path = (profile_root / relative_file_path).resolve()
        try:
            module_path.relative_to(profile_root)
        except ValueError as error:
            raise ValueError(
                f"Translator entrypoint '{entrypoint}' escapes profile "
                f"'{declaration.profile_name}' root."
            ) from error
        if not module_path.exists():
            raise ValueError(
                "Translator module path not found: "
                f"{display_path(module_path, repo_root=self.repo_root)}"
            )
        if not module_path.is_file():
            raise ValueError(
                "Translator module path is not a file: "
                f"{display_path(module_path, repo_root=self.repo_root)}"
            )
        function_cache_key = (str(module_path), function_name)
        cached_function = self._entrypoint_function_cache.get(
            function_cache_key
        )
        if cached_function is not None:
            return cached_function
        cache_key = str(module_path)
        module = self._file_module_cache.get(cache_key)
        if module is None:
            spec = importlib.util.spec_from_file_location(
                f"devcovenant_translator_{declaration.profile_name}",
                module_path,
            )
            if spec is None or spec.loader is None:
                raise ValueError(
                    f"Failed to load translator module: {module_path}"
                )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self._file_module_cache[cache_key] = module
        function = getattr(module, function_name, None)
        if function is None:
            raise ValueError(
                f"Translator function '{function_name}' "
                f"not found in {module_path}"
            )
        self._entrypoint_function_cache[function_cache_key] = function
        return function
