"""Autofixer loading and execution helpers for `policy_engine`."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Callable

from devcovenant.core.policy_contract import PolicyFixer, Violation


def load_fixers(
    repo_root: Path, custom_policy_overrides: set[str] | None = None
) -> list[PolicyFixer]:
    """Dynamically import bundled policy fixers for a repository."""
    fixers: list[PolicyFixer] = []
    custom_policy_overrides = set(custom_policy_overrides or set())
    roots: list[tuple[str, Path]] = [
        ("custom", repo_root / "devcovenant" / "custom" / "policies"),
        ("builtin", repo_root / "devcovenant" / "builtin" / "policies"),
    ]
    loaded_builtin_policy_dirs: set[str] = set()

    for origin, root in roots:
        if not root.exists():
            continue
        for policy_dir in root.iterdir():
            if not policy_dir.is_dir() or policy_dir.name.startswith("_"):
                continue
            policy_id = policy_dir.name.replace("_", "-")
            if origin == "builtin" and policy_id in custom_policy_overrides:
                continue
            if (
                origin == "builtin"
                and policy_dir.name in loaded_builtin_policy_dirs
            ):
                continue
            fixers_dir = policy_dir / "autofix"
            if not fixers_dir.exists():
                continue
            if origin == "builtin":
                loaded_builtin_policy_dirs.add(policy_dir.name)
            for module_file in fixers_dir.glob("*.py"):
                if (
                    module_file.name.startswith("_")
                    or module_file.name == "__init__.py"
                ):
                    continue
                module_name = (
                    f"devcovenant.{origin}.policies."
                    f"{policy_dir.name}.autofix.{module_file.stem}"
                )
                try:
                    module = importlib.import_module(module_name)
                # DEVCOV_ALLOW_BROAD_ONCE plugin import boundary.
                except Exception as exc:
                    raise RuntimeError(
                        "Failed to import fixer module "
                        f"`{module_name}`: {exc}"
                    ) from exc
                for member in module.__dict__.values():
                    if (
                        inspect.isclass(member)
                        and issubclass(member, PolicyFixer)
                        and member is not PolicyFixer
                    ):
                        try:
                            instance = member()
                            setattr(instance, "repo_root", repo_root)
                            setattr(instance, "_origin", origin)
                            fixers.append(instance)
                        # DEVCOV_ALLOW_BROAD_ONCE plugin init boundary.
                        except Exception as exc:
                            raise RuntimeError(
                                "Failed to initialize fixer "
                                f"`{member.__name__}` in {module_name}: "
                                f"{exc}"
                            ) from exc
    return fixers


def apply_auto_fixes(
    violations: list[Violation],
    fixers: list[PolicyFixer],
    *,
    print_fn: Callable[..., object],
) -> bool:
    """Run fixer instances against auto-fixable violations."""
    if not violations or not fixers:
        return False

    applied = False
    print_fn("\n🔧 Running auto-fixers...\n")
    for violation in violations:
        if not violation.can_auto_fix:
            continue
        for fixer in fixers:
            if not fixer.can_fix(violation):
                continue
            result = fixer.fix(violation)
            message = result.message or ""
            if result.success:
                if message:
                    print_fn(f"  • {message}")
                if result.files_modified:
                    applied = True
            else:
                print_fn(
                    f"  • Auto-fix failed for {violation.policy_id}: "
                    f"{message or 'unknown error'}"
                )
            break

    if applied:
        print_fn("\n🔁 Re-running policy checks after auto-fix.\n")
    else:
        print_fn("⚪ No auto-fixable violations were modified.\n")

    return applied
