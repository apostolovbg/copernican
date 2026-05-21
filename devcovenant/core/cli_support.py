"""CLI argument and output helpers for DevCovenant."""

from __future__ import annotations

import argparse
import importlib
import sys
from dataclasses import dataclass
from typing import Literal, Sequence, TextIO

OutputMode = Literal["normal", "verbose", "quiet"]
ChildOutputChannel = Literal[
    "gate_child",
    "workflow_child",
    "managed_child",
    "generic_child",
]

OUTPUT_MODE_DEFAULT: OutputMode = "verbose"
OUTPUT_MODE_ALLOWED = frozenset({"normal", "verbose", "quiet"})
WAIT_PROGRESS_MESSAGE = "Please wait. In progress..."
_NORMAL_MODE_SUPPRESSED_CHANNELS = frozenset(
    {"managed_child", "workflow_child"}
)
_QUIET_MODE_SUPPRESSED_CHANNELS = frozenset(
    {"gate_child", "workflow_child", "managed_child", "generic_child"}
)
_OUTPUT_MODE_OVERRIDE_DEST = "output_mode_override"
_OUTPUT_MODE_OVERRIDE_FLAG_MAP: dict[str, OutputMode] = {
    "--quiet": "quiet",
    "--normal": "normal",
    "--verbose": "verbose",
}


@dataclass(frozen=True)
class ChildOutputPlan:
    """Resolved child-command output behavior for one mode/channel pair."""

    emit_console: bool
    heartbeat_message: str | None

    @property
    def child_output_suppressed(self) -> bool:
        """Return True when child output is hidden from console."""
        return not self.emit_console


class DevCovenantArgumentParser(argparse.ArgumentParser):
    """Argument parser with shared DevCovenant output-mode flags."""


class _OutputModeOverrideAction(argparse.Action):
    """Argparse action that records and applies one output-mode override."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: object,
        option_string: str | None = None,
    ) -> None:
        """Store the override token and apply it immediately."""
        del parser, values, option_string
        setattr(namespace, self.dest, self.const)
        execution_runtime_module = importlib.import_module(
            "devcovenant.core.execution"
        )
        execution_runtime_module.configure_output_mode(self.const)


def normalize_output_mode(
    raw_value: str | None,
    *,
    default: OutputMode = OUTPUT_MODE_DEFAULT,
) -> OutputMode:
    """Normalize one output-mode token to an allowed runtime mode."""
    token = str(raw_value or "").strip().lower()
    if token in OUTPUT_MODE_ALLOWED:
        return token  # type: ignore[return-value]
    return default


def resolve_child_output_plan(
    output_mode: OutputMode,
    channel: ChildOutputChannel,
) -> ChildOutputPlan:
    """Resolve child-output emission behavior for one command channel."""
    normalized_mode = normalize_output_mode(output_mode)
    normalized_channel = str(channel or "").strip().lower()
    if normalized_mode == "quiet":
        emit_console = (
            normalized_channel not in _QUIET_MODE_SUPPRESSED_CHANNELS
        )
        return ChildOutputPlan(
            emit_console=emit_console,
            heartbeat_message=None,
        )
    if normalized_mode != "normal":
        return ChildOutputPlan(
            emit_console=True,
            heartbeat_message=None,
        )
    emit_console = normalized_channel not in _NORMAL_MODE_SUPPRESSED_CHANNELS
    return ChildOutputPlan(
        emit_console=emit_console,
        heartbeat_message=WAIT_PROGRESS_MESSAGE,
    )


def channel_suppresses_child_output(
    output_mode: OutputMode,
    channel: ChildOutputChannel,
) -> bool:
    """Return True when child output should be hidden for mode/channel."""
    return resolve_child_output_plan(
        output_mode, channel
    ).child_output_suppressed


def write_console_text(
    message: str,
    *,
    file: TextIO | None = None,
    end: str = "\n",
    flush: bool = False,
) -> None:
    """Write lightweight console text before full runtime init is needed."""
    stream = file if file is not None else sys.stdout
    stream.write(f"{message}{end}")
    if flush:
        stream.flush()


def add_output_mode_override_arguments(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Add universal per-invocation output-mode override flags."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--quiet",
        dest=_OUTPUT_MODE_OVERRIDE_DEST,
        action=_OutputModeOverrideAction,
        nargs=0,
        const="quiet",
        help="Suppress routine stdout for this invocation only.",
    )
    group.add_argument(
        "--normal",
        dest=_OUTPUT_MODE_OVERRIDE_DEST,
        action=_OutputModeOverrideAction,
        nargs=0,
        const="normal",
        help="Use concise progress output for this invocation only.",
    )
    group.add_argument(
        "--verbose",
        dest=_OUTPUT_MODE_OVERRIDE_DEST,
        action=_OutputModeOverrideAction,
        nargs=0,
        const="verbose",
        help="Stream fuller console detail for this invocation only.",
    )
    return parser


def output_mode_override_from_namespace(
    namespace: argparse.Namespace | object | None,
) -> OutputMode | None:
    """Return one parsed per-invocation output-mode override."""
    if namespace is None:
        return None
    raw_value = getattr(namespace, _OUTPUT_MODE_OVERRIDE_DEST, None)
    token = str(raw_value or "").strip().lower()
    if token in OUTPUT_MODE_ALLOWED:
        return token  # type: ignore[return-value]
    return None


def apply_output_mode_override_from_namespace(
    namespace: argparse.Namespace | object | None,
) -> OutputMode | None:
    """Apply one parsed per-invocation output-mode override, if present."""
    override = output_mode_override_from_namespace(namespace)
    if override is None:
        return None
    execution_runtime_module = importlib.import_module(
        "devcovenant.core.execution"
    )
    execution_runtime_module.configure_output_mode(override)
    return override


def resolve_cli_output_mode_override(argv: Sequence[str]) -> OutputMode | None:
    """Resolve a consistent CLI output-mode override before `--`."""
    override: OutputMode | None = None
    for token in argv:
        if token == "--":
            break
        candidate = _OUTPUT_MODE_OVERRIDE_FLAG_MAP.get(str(token).strip())
        if candidate is None:
            continue
        if override is not None and override != candidate:
            raise ValueError(
                "Output-mode overrides are mutually exclusive. Choose only "
                "one of `--quiet`, `--normal`, or `--verbose`."
            )
        override = candidate
    return override


def strip_leading_cli_output_mode_overrides(argv: Sequence[str]) -> list[str]:
    """Strip root-level leading output-mode flags before command dispatch."""
    remaining = list(argv)
    while remaining:
        token = str(remaining[0]).strip()
        if token not in _OUTPUT_MODE_OVERRIDE_FLAG_MAP:
            break
        remaining.pop(0)
    return remaining


def build_command_parser(
    command_name: str,
    description: str,
) -> argparse.ArgumentParser:
    """Build a command-scoped parser with stable usage text."""
    parser = DevCovenantArgumentParser(
        prog=f"devcovenant {command_name}",
        description=description,
    )
    return add_output_mode_override_arguments(parser)
