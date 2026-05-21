"""Command dispatcher for DevCovenant."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import traceback
from pathlib import Path
from types import ModuleType

import devcovenant.core.cli_support as cli_args_module
import devcovenant.core.cli_support as output_runtime_module
from devcovenant import __version__
from devcovenant.core.repository_paths import display_path

_COMMAND_MODULES = {
    "asset": "devcovenant.asset",
    "check": "devcovenant.check",
    "clean": "devcovenant.clean",
    "gate": "devcovenant.gate",
    "custom": "devcovenant.custom",
    "demo": "devcovenant.demo",
    "run": "devcovenant.run",
    "install": "devcovenant.install",
    "deploy": "devcovenant.deploy",
    "upgrade": "devcovenant.upgrade",
    "refresh": "devcovenant.refresh",
    "uninstall": "devcovenant.uninstall",
    "undeploy": "devcovenant.undeploy",
    "policy": "devcovenant.policy",
    "quickstart": "devcovenant.quickstart",
}

_COMMAND_SUMMARIES = {
    "asset": (
        "Write one Desktop copy of a shipped profile asset or managed doc."
    ),
    "check": "Run read-only DevCovenant audit checks.",
    "clean": (
        "Remove disposable build, cache, runtime-registry, or log artifacts."
    ),
    "custom": (
        "Promote or retract builtin policy/profile custom copies and any "
        "mirrored tests."
    ),
    "deploy": "Deploy managed docs/assets in the current repository.",
    "demo": "Run a disposable evaluation demo.",
    "gate": "Run DevCovenant gate session lifecycle commands.",
    "install": "Install DevCovenant into the current repository.",
    "policy": (
        "Run one explicit policy-born command declared by an enabled policy."
    ),
    "quickstart": "Print a terse static reminder.",
    "refresh": "Run a full refresh.",
    "run": "Run all declared DevCovenant workflow runs.",
    "undeploy": "Remove deployed managed artifacts and keep core files.",
    "uninstall": "Remove DevCovenant from the current repository.",
    "upgrade": "Upgrade DevCovenant core in the current repository.",
}

_MANAGED_REEXEC_GUARD_ENV = "DEVCOV_MANAGED_REEXEC_ACTIVE"
_MANAGED_REEXEC_SOURCE_ENV = "DEVCOV_MANAGED_REEXEC_SOURCE"
_RUN_LOG_HANDOFF_REPO_ENV = "DEVCOV_RUN_LOG_REPO_ROOT"
_RUN_LOG_HANDOFF_RUN_ID_ENV = "DEVCOV_RUN_LOG_ID"
_TOP_LEVEL_COMMAND_ENV = "DEVCOV_TOP_COMMAND"
_MANAGED_REEXEC_BYPASS_COMMANDS = {
    "install",
    "deploy",
    "undeploy",
    "uninstall",
    "quickstart",
    "demo",
}
_RUN_LOG_BYPASS_COMMANDS = {"uninstall"}
_RUNTIME_SETUP_SKIP_COMMANDS = {"quickstart", "demo"}

_execution_runtime_module: ModuleType | None = None
_runtime_errors_module: ModuleType | None = None


def _execution_runtime() -> ModuleType:
    """Return the shared execution runtime module on demand."""
    global _execution_runtime_module
    if _execution_runtime_module is None:
        _execution_runtime_module = importlib.import_module(
            "devcovenant.core.execution"
        )
    return _execution_runtime_module


def _runtime_errors() -> ModuleType:
    """Return the runtime error helpers module on demand."""
    global _runtime_errors_module
    if _runtime_errors_module is None:
        _runtime_errors_module = importlib.import_module(
            "devcovenant.core.runtime_errors"
        )
    return _runtime_errors_module


def _build_parser() -> argparse.ArgumentParser:
    """Build the root dispatcher parser."""
    parser = cli_args_module.DevCovenantArgumentParser(
        description="DevCovenant - repository governance framework",
        epilog=_render_command_help_epilog(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    cli_args_module.add_output_mode_override_arguments(parser)
    parser.add_argument(
        "command",
        choices=sorted(_COMMAND_MODULES.keys()),
        help="Command to run",
    )
    return parser


def _render_command_help_epilog() -> str:
    """Render one root-help command summary block."""
    lines = ["Command summary:"]
    for command in sorted(_COMMAND_MODULES):
        summary = _COMMAND_SUMMARIES.get(command, "").strip()
        lines.append(f"  {command:<10} {summary}")
    lines.append("")
    lines.append(
        "Run `devcovenant <command> --help` for command-specific options."
    )
    return "\n".join(lines)


def _load_command_module(command: str) -> ModuleType:
    """Import and return command module for a command id."""
    module_path = _COMMAND_MODULES[command]
    return importlib.import_module(module_path)


def _managed_stage_for_command(
    command: str,
    command_args: list[str],
) -> str:
    """Resolve managed-environment stage for one CLI command invocation."""
    if command == "gate":
        if "--close" in command_args:
            return "close"
        if "--open" in command_args:
            return "open"
        if "--verify" in command_args:
            return "managed"
        if "--status" in command_args:
            return "bootstrap"
    if command == "run":
        return "run"
    return "bootstrap"


def _same_interpreter_path(current: str, expected: str) -> bool:
    """Return True when two interpreter paths resolve to the same file."""
    current_path = Path(current)
    expected_path = Path(expected)
    try:
        current_resolved = current_path.resolve()
    except OSError:
        current_resolved = current_path
    try:
        expected_resolved = expected_path.resolve()
    except OSError:
        expected_resolved = expected_path
    return current_resolved == expected_resolved


def _managed_python_is_executable(path_text: str) -> bool:
    """Return True when a managed interpreter path is executable."""
    path = Path(path_text)
    try:
        if not path.exists() or not path.is_file():
            return False
    except OSError:
        return False
    return os.access(path, os.X_OK)


def _should_skip_managed_reexec(command_args: list[str]) -> bool:
    """Skip managed re-exec for help/version invocations."""
    return any(
        token in {"-h", "--help", "-V", "--version"} for token in command_args
    )


def _apply_run_log_handoff_env(env: dict[str, str]) -> dict[str, str]:
    """Copy active run-log handoff variables into a re-exec environment."""
    for key in (_RUN_LOG_HANDOFF_REPO_ENV, _RUN_LOG_HANDOFF_RUN_ID_ENV):
        value = str(os.environ.get(key, "")).strip()
        if value:
            env[key] = value
    return env


def _same_path_text(left: str, right: str) -> bool:
    """Return True when two path strings resolve to the same location."""
    try:
        return Path(left).resolve() == Path(right).resolve()
    except OSError:
        return Path(left) == Path(right)


def _has_local_managed_environment_policy(repo_root: Path) -> bool:
    """Return True when local managed-environment policy runtime exists."""
    policy_script = (
        repo_root
        / "devcovenant"
        / "builtin"
        / "policies"
        / "managed_environment"
        / "managed_environment.py"
    )
    return policy_script.exists()


def _initialize_cli_run_logging(
    repo_root: Path | None,
    command: str,
    command_args: list[str],
):
    """Create or adopt the per-run log context for one CLI command."""
    if command in _RUN_LOG_BYPASS_COMMANDS:
        _execution_runtime().clear_active_run_log_context()
        os.environ.pop(_RUN_LOG_HANDOFF_REPO_ENV, None)
        os.environ.pop(_RUN_LOG_HANDOFF_RUN_ID_ENV, None)
        return None
    if _should_skip_managed_reexec(command_args):
        _execution_runtime().clear_active_run_log_context()
        return None
    if repo_root is None:
        _execution_runtime().clear_active_run_log_context()
        return None
    execution_runtime_module = _execution_runtime()
    run_logging_module = execution_runtime_module.run_logging_runtime_module
    handoff_repo = str(os.environ.get(_RUN_LOG_HANDOFF_REPO_ENV, "")).strip()
    handoff_run_id = str(
        os.environ.get(_RUN_LOG_HANDOFF_RUN_ID_ENV, "")
    ).strip()
    context = None
    if (
        handoff_repo
        and handoff_run_id
        and _same_path_text(
            handoff_repo,
            str(repo_root),
        )
    ):
        try:
            context = run_logging_module.load_run_log_context(
                repo_root,
                run_id=handoff_run_id,
            )
        except ValueError:
            context = None
    if context is None:
        context = run_logging_module.create_run_log_context(
            repo_root,
            command,
            ["devcovenant", command, *command_args],
            cwd=Path.cwd(),
            metadata={
                "dispatch_source": "cli",
                "managed_reexec_source": os.environ.get(
                    _MANAGED_REEXEC_SOURCE_ENV, ""
                ),
            },
        )
    execution_runtime_module.set_active_run_log_context(context)
    execution_runtime_module.merge_active_run_log_metadata(
        {
            "invoked_python": (
                str(os.environ.get(_MANAGED_REEXEC_SOURCE_ENV, "")).strip()
                or sys.executable
            ),
            "effective_python": sys.executable,
            "managed_environment_active": bool(
                str(os.environ.get("DEVCOV_MANAGED_PYTHON", "")).strip()
                or str(os.environ.get("VIRTUAL_ENV", "")).strip()
            ),
            "managed_reexec_applied": bool(
                str(os.environ.get(_MANAGED_REEXEC_SOURCE_ENV, "")).strip()
            ),
        }
    )
    os.environ[_RUN_LOG_HANDOFF_REPO_ENV] = str(repo_root)
    os.environ[_RUN_LOG_HANDOFF_RUN_ID_ENV] = context.run_id
    return context


def _finalize_cli_run_logging(
    *,
    exit_code: int | None,
    status: str | None = None,
    metadata_updates: dict[str, object] | None = None,
) -> None:
    """Finalize active CLI run logging and clear handoff state."""
    execution_runtime_module = _execution_runtime()
    if metadata_updates:
        execution_runtime_module.merge_active_run_log_metadata(
            metadata_updates
        )
    error_like = (exit_code is not None and int(exit_code) != 0) or str(
        status or ""
    ).strip().lower() in {"failure", "exception", "interrupted"}
    pointer_stream = sys.stderr if error_like else None
    execution_runtime_module.emit_active_run_log_pointer(
        file=pointer_stream,
        once=True,
    )
    execution_runtime_module.finalize_active_run_log_context(
        exit_code=exit_code,
        status=status,
    )
    execution_runtime_module.clear_active_run_log_context()
    os.environ.pop(_RUN_LOG_HANDOFF_REPO_ENV, None)
    os.environ.pop(_RUN_LOG_HANDOFF_RUN_ID_ENV, None)


def _exit_code_from_system_exit(exc: SystemExit) -> int:
    """Normalize `SystemExit.code` to a process exit code integer."""
    code = exc.code
    if code is None:
        return 0
    if isinstance(code, int):
        return int(code)
    return 1


def _maybe_reexec_managed_environment(
    command: str,
    command_args: list[str],
) -> None:
    """Re-exec command in managed interpreter when policy is active."""
    if command in _MANAGED_REEXEC_BYPASS_COMMANDS:
        return
    if _should_skip_managed_reexec(command_args):
        return
    execution_runtime_module = _execution_runtime()
    repo_root = execution_runtime_module.find_git_root(Path.cwd())
    if repo_root is None:
        return
    if not _has_local_managed_environment_policy(repo_root):
        return
    stage = _managed_stage_for_command(command, command_args)
    managed_env: dict[str, str] | None = None
    managed_python: str | None = None
    managed_resolution_error: str | None = None
    try:
        managed_env, managed_python = (
            execution_runtime_module.resolve_managed_environment_for_stage(
                repo_root,
                stage,
                base_env=os.environ,
            )
        )
    except ValueError as exc:
        managed_resolution_error = str(exc)
    if managed_env is not None and managed_python:
        managed_python_display = display_path(
            Path(managed_python),
            repo_root=repo_root,
        )
        if not _managed_python_is_executable(managed_python):
            managed_resolution_error = (
                "Managed-environment interpreter is not executable: "
                f"`{managed_python_display}`."
            )
            managed_python = None
        if managed_python is None:
            pass
        elif _same_interpreter_path(sys.executable, managed_python):
            execution_runtime_module.merge_active_run_log_metadata(
                {
                    "invoked_python": sys.executable,
                    "effective_python": sys.executable,
                    "managed_environment_active": True,
                    "managed_reexec_applied": False,
                }
            )
            return
        elif os.environ.get(_MANAGED_REEXEC_GUARD_ENV) == "1":
            raise SystemExit(
                "Managed-environment auto-rerun did not converge to the "
                "expected interpreter."
            )
        else:
            rerun_message = (
                "Re-running DevCovenant from managed interpreter: "
                f"{managed_python_display}\n"
            )
            execution_runtime_module.runtime_print(
                rerun_message,
                end="",
                file=sys.stderr,
                flush=True,
            )
            env = _apply_run_log_handoff_env(dict(managed_env))
            env[_MANAGED_REEXEC_GUARD_ENV] = "1"
            env[_MANAGED_REEXEC_SOURCE_ENV] = sys.executable
            argv = [
                managed_python,
                "-m",
                "devcovenant",
                command,
                *command_args,
            ]
            try:
                # Re-exec uses explicit argv/env in the managed interpreter.
                os.execve(managed_python, argv, env)  # nosec B606
            except OSError as exc:
                managed_resolution_error = (
                    "Managed-environment interpreter exec failed: "
                    f"`{managed_python_display}` ({exc})."
                )
                managed_python = None

    if managed_resolution_error:
        raise SystemExit(managed_resolution_error)


def main(argv: list[str] | None = None) -> None:
    """Main CLI entry point."""
    parser = _build_parser()
    command_args = list(sys.argv[1:] if argv is None else argv)
    try:
        cli_output_override = cli_args_module.resolve_cli_output_mode_override(
            command_args
        )
    except ValueError as exc:
        parser.error(str(exc))
    command_args = cli_args_module.strip_leading_cli_output_mode_overrides(
        command_args
    )
    if not command_args:
        parser.print_help()
        raise SystemExit(0)

    first = command_args[0]
    if first in {"-h", "--help"}:
        parser.print_help()
        raise SystemExit(0)
    if first in {"-V", "--version"}:
        output_runtime_module.write_console_text(f"devcovenant {__version__}")
        raise SystemExit(0)
    if first not in _COMMAND_MODULES:
        parser.error(
            f"argument command: invalid choice: '{first}' "
            f"(choose from {', '.join(sorted(_COMMAND_MODULES))})"
        )
    os.environ[_TOP_LEVEL_COMMAND_ENV] = first
    subcommand_args = command_args[1:]
    skip_runtime_setup = first in _RUNTIME_SETUP_SKIP_COMMANDS or (
        _should_skip_managed_reexec(subcommand_args)
    )
    execution_runtime_module: ModuleType | None = None
    repo_root: Path | None = None
    if not skip_runtime_setup:
        execution_runtime_module = _execution_runtime()
        repo_root = execution_runtime_module.find_git_root(Path.cwd())
        if repo_root is not None:
            execution_runtime_module.cleanup_source_checkout_import_cache(
                repo_root
            )
            execution_runtime_module.configure_repo_pycache_prefix(repo_root)
            if cli_output_override is None:
                execution_runtime_module.configure_output_mode_from_config(
                    repo_root
                )
            else:
                execution_runtime_module.configure_output_mode(
                    cli_output_override
                )
            execution_runtime_module.configure_logs_keep_last_from_config(
                repo_root
            )
        elif cli_output_override is not None:
            execution_runtime_module.configure_output_mode(cli_output_override)
        _initialize_cli_run_logging(repo_root, first, subcommand_args)
        if cli_output_override is not None:
            execution_runtime_module.merge_active_run_log_metadata(
                {"cli_output_mode_override": cli_output_override}
            )
        _maybe_reexec_managed_environment(first, subcommand_args)
    try:
        module = _load_command_module(first)

        if not hasattr(module, "main"):
            raise SystemExit(
                f"Command module '{module.__name__}' is missing main()."
            )

        module.main(subcommand_args)
    except SystemExit as exc:
        exit_code = _exit_code_from_system_exit(exc)
        metadata_updates: dict[str, object] = {
            "exit_kind": "system_exit",
            "exit_code_normalized": exit_code,
        }
        if execution_runtime_module is not None and isinstance(exc.code, str):
            message = str(exc.code).strip()
            if message:
                execution_runtime_module.append_active_run_log_output(
                    "stderr",
                    message + "\n",
                )
                metadata_updates["system_exit_message"] = message
        if execution_runtime_module is not None:
            _finalize_cli_run_logging(
                exit_code=exit_code,
                status="success" if exit_code == 0 else "failure",
                metadata_updates=metadata_updates,
            )
        raise
    except KeyboardInterrupt:
        if execution_runtime_module is not None:
            execution_runtime_module.append_active_run_log_output(
                "stderr",
                traceback.format_exc(),
            )
            _finalize_cli_run_logging(
                exit_code=130,
                status="interrupted",
                metadata_updates={"exit_kind": "keyboard_interrupt"},
            )
        raise
    # DEVCOV_ALLOW_BROAD_ONCE CLI top-level normalization boundary.
    except Exception as exc:
        runtime_errors_module = _runtime_errors()
        normalized_error = runtime_errors_module.normalize_unhandled_exception(
            exc
        )
        if execution_runtime_module is None:
            execution_runtime_module = _execution_runtime()
        execution_runtime_module.append_active_run_log_output(
            "stderr",
            traceback.format_exc(),
        )
        execution_runtime_module.runtime_print(
            runtime_errors_module.render_error(normalized_error),
            file=sys.stderr,
            flush=True,
        )
        _finalize_cli_run_logging(
            exit_code=normalized_error.exit_code,
            status="exception",
            metadata_updates={
                "exit_kind": "normalized_exception",
                "error_code": normalized_error.code.value,
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            },
        )
        raise SystemExit(normalized_error.exit_code) from exc
    else:
        if execution_runtime_module is not None:
            _finalize_cli_run_logging(
                exit_code=0,
                status="success",
                metadata_updates={"exit_kind": "return"},
            )


if __name__ == "__main__":
    main()
