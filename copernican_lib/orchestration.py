# Rationale: Service orchestration lives here because both the CLI and GUI need
# one authoritative map of run-control hooks to avoid duplicating lifecycle
# logic.
"""Orchestration map and GUI-safe service interfaces.

The Copernican Suite now exposes a concise set of orchestration services that
are safe for GUI callers. The map below highlights the modules that already
coordinate configuration validation, manifest generation and run lifecycle
reporting so front-ends never have to duplicate logic borrowed from the CLI.
Protocols describe how GUIs can request runs, pause or resume execution and
stream live status or log lines while the underlying runner continues to use
the shared helpers imported here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable, Protocol, Sequence

from copernican_lib import result_writer, run_manifest
from copernican_lib.cli import dependencies
from copernican_lib.model_spec_validator import validate_and_cache_model


class LaunchMode(str, Enum):
    """Available launch surfaces supported by the orchestrator."""

    CLI = "cli"
    GUI = "gui"


@dataclass(frozen=True)
class ServiceDescriptor:
    """Summarise where orchestration responsibilities live."""

    name: str
    module: str
    entrypoints: tuple[str, ...]
    rationale: str


@dataclass(frozen=True)
class OrchestrationMap:
    """Describe the shared services a GUI is allowed to import directly."""

    config_validation: ServiceDescriptor
    manifest_generation: ServiceDescriptor
    run_control: ServiceDescriptor


@dataclass
class RunRequest:
    """Inputs required to start a Copernican run."""

    config: Path
    datasets: Sequence[str]
    engine: str
    seed: int | None = None
    mode: LaunchMode = LaunchMode.GUI


@dataclass
class RunHandle:
    """Token returned when a run is scheduled or started."""

    token: str
    mode: LaunchMode


@dataclass
class RunStatus:
    """Current status snapshot for a run in progress."""

    token: str
    stage: str
    progress: float
    message: str = ""


class RunController(Protocol):
    """Interface consumed by GUI clients to manage run lifecycle events."""

    def request_run(self, request: RunRequest) -> RunHandle:
        """Schedule a run described by *request* and return its handle."""

    def cancel(self, handle: RunHandle) -> None:
        """Cancel an in-flight run if it has not completed."""

    def pause(self, handle: RunHandle) -> None:
        """Pause sampling or analysis at the next checkpoint."""

    def resume(self, handle: RunHandle) -> None:
        """Resume a paused run."""

    def stream_status(self, handle: RunHandle) -> Iterable[RunStatus]:
        """Yield status snapshots that mirror the CLI progress feed."""

    def stream_logs(self, handle: RunHandle) -> Iterable[str]:
        """Yield log lines produced by the active run."""


@dataclass
class InProcessRunController:
    """Delegate lifecycle requests to shared helpers without duplication."""

    run_executor: Callable[[RunRequest], str]
    cancel_hook: Callable[[str], None] | None = None
    pause_hook: Callable[[str], None] | None = None
    resume_hook: Callable[[str], None] | None = None
    status_hook: Callable[[str], Iterable[RunStatus]] | None = None
    log_hook: Callable[[str], Iterable[str]] | None = None

    def request_run(self, request: RunRequest) -> RunHandle:
        token = self.run_executor(request)
        return RunHandle(token=token, mode=request.mode)

    def cancel(self, handle: RunHandle) -> None:
        if self.cancel_hook is None:
            raise RuntimeError(
                "Cancel hook is not configured for this runner."
            )
        self.cancel_hook(handle.token)

    def pause(self, handle: RunHandle) -> None:
        if self.pause_hook is None:
            raise RuntimeError("Pause hook is not configured for this runner.")
        self.pause_hook(handle.token)

    def resume(self, handle: RunHandle) -> None:
        if self.resume_hook is None:
            raise RuntimeError(
                "Resume hook is not configured for this runner."
            )
        self.resume_hook(handle.token)

    def stream_status(self, handle: RunHandle) -> Iterable[RunStatus]:
        if self.status_hook is None:
            return ()
        return self.status_hook(handle.token)

    def stream_logs(self, handle: RunHandle) -> Iterable[str]:
        if self.log_hook is None:
            return ()
        return self.log_hook(handle.token)


def describe_orchestration_services() -> OrchestrationMap:
    """Return the modules that drive config validation, manifests and runs."""

    return OrchestrationMap(
        config_validation=ServiceDescriptor(
            name="Configuration Validation",
            module="copernican_lib.model_spec_validator",
            entrypoints=(validate_and_cache_model.__name__,),
            rationale=(
                "Validates YAML model definitions and caches parsed callables "
                "for both CLI and GUI callers without importing menu helpers."
            ),
        ),
        manifest_generation=ServiceDescriptor(
            name="Manifest Generation",
            module="copernican_lib.run_manifest",
            entrypoints=(run_manifest.build_manifest.__name__,),
            rationale=(
                "Builds immutable run manifests that capture dataset hashes, "
                "engine metadata and Git state so GUIs can persist the same "
                "artifacts as the CLI."
            ),
        ),
        run_control=ServiceDescriptor(
            name="Run Controller",
            module="copernican_lib.result_writer",
            entrypoints=(
                result_writer.save_summary.__name__,
                dependencies.get_runtime_options.__name__,
            ),
            rationale=(
                "Keeps lifecycle hooks inside shared modules so GUI launchers "
                "can reuse the CLI logging and cache cleanup routines while "
                "remaining isolated from interactive menus."
            ),
        ),
    )
