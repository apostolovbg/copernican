"""Copernican package public entrypoints and helpers."""

from importlib import import_module

_WORKFLOW_EXPORTS = {
    "COPERNICAN_VERSION",
    "LaunchRequest",
    "_parse_launch_args",
    "exit_clean",
    "launch_gui",
    "main",
    "main_workflow",
    "orchestration",
}


def __getattr__(name: str):
    """Load workflow symbols lazily so package imports stay lightweight."""

    if name not in _WORKFLOW_EXPORTS:
        raise AttributeError(f"module 'copernican' has no attribute {name!r}")
    if name == "orchestration":
        value = import_module("copernican.lib.orchestration")
        globals()[name] = value
        return value
    workflow = import_module(".workflow", __name__)
    value = getattr(workflow, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Advertise the lazily exported workflow symbols."""

    return sorted({*globals(), *_WORKFLOW_EXPORTS})
