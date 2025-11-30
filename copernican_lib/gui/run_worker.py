"""CLI run worker invoked by the GUI in a child process."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from copernican_lib import dataset_registry, utils
from copernican_lib.cli import menus as cli_menus


def _patch_cli_runtime(config: dict[str, Any]) -> list[tuple[Any, str, Any]]:
    """Apply deterministic selections for the CLI workflow."""

    import copernican  # Imported lazily inside the worker

    patches: list[tuple[Any, str, Any]] = []

    def _patch(obj: Any, attr: str, replacement: Any) -> None:
        patches.append((obj, attr, getattr(obj, attr)))
        setattr(obj, attr, replacement)

    seed_value = int(config.get("seed", 0))

    def _select_seed() -> int:
        utils.set_random_seed(seed_value)
        return seed_value

    _patch(cli_menus, "select_seed", _select_seed)

    model_filename = config.get("model_filename", "")
    engine_filename = config.get("engine_filename", "")

    def _select_from_list(options, prompt):
        prompt_lower = prompt.lower()
        if "cosmological model" in prompt_lower and model_filename in options:
            return model_filename
        if "computation engine" in prompt_lower and engine_filename in options:
            return engine_filename
        return options[0] if options else None

    _patch(cli_menus, "select_from_list", _select_from_list)

    datasets = config.get("datasets", {})
    original_prompt = dataset_registry.prompt_dataset_selection

    def _prompt_dataset_selection(parser_registry, label):
        key = label.lower()
        target = datasets.get(key)
        if target and target in parser_registry:
            return target
        if parser_registry:
            return next(iter(parser_registry.keys()))
        return original_prompt(parser_registry, label)

    _patch(
        dataset_registry,
        "prompt_dataset_selection",
        _prompt_dataset_selection,
    )

    sampling_plan = config.get("sampling_plan", {})

    def _sampling_plan(*_args, **_kwargs):
        return sampling_plan

    _patch(copernican, "prompt_sampling_configuration", _sampling_plan)
    return patches


def main(argv: list[str] | None = None) -> int:
    """Entry point for the GUI worker invoked via ``python -m``."""

    parser = argparse.ArgumentParser(
        description="Copernican GUI run worker"
    )
    parser.add_argument("config_path", help="Path to the JSON configuration")
    args = parser.parse_args(argv)
    config_path = Path(args.config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    progress_path = config.get("progress_path")
    if progress_path:
        os.environ.setdefault("COPERNICAN_GUI_PROGRESS_PATH", progress_path)
    patches = _patch_cli_runtime(config)
    try:
        import copernican

        os.environ.setdefault("COPERNICAN_DETACH_GUI", "0")
        try:
            copernican.main_workflow()
            return 0
        except SystemExit as exc:  # pragma: no cover - propagated status
            return int(exc.code or 0)
    finally:
        for obj, attr, original in reversed(patches):
            setattr(obj, attr, original)

if __name__ == "__main__":  # pragma: no cover - worker entry point
    raise SystemExit(main())
