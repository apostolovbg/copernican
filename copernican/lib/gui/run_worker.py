"""CLI run worker invoked by the GUI in a child process."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    """Entry point for the GUI worker invoked via ``python -m``."""

    parser = argparse.ArgumentParser(description="Copernican GUI run worker")
    parser.add_argument("config_path", help="Path to the JSON configuration")
    args = parser.parse_args(argv)
    config_path = Path(args.config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    manifest_path = config.get("manifest_path")
    if not manifest_path:
        logging.getLogger().error(
            "GUI worker configuration lacks a manifest_path entry."
        )
        return 1
    progress_path = config.get("progress_path")
    if progress_path:
        os.environ.setdefault("COPERNICAN_GUI_PROGRESS_PATH", progress_path)
    os.environ.setdefault("COPERNICAN_HEADLESS_RUN", "1")
    run_args = ["--manifest", manifest_path]
    output_dir = config.get("output_dir")
    if output_dir:
        run_args.extend(["--output-dir", output_dir])
    try:
        import copernican

        try:
            return copernican.main(run_args)
        except SystemExit as exc:  # pragma: no cover - propagated status
            return int(exc.code or 0)
        except (OSError, RuntimeError, TypeError, ValueError):
            logging.getLogger().critical(
                "GUI worker encountered an unhandled exception.",
                exc_info=True,
            )
            return 1
    except ImportError as exc:
        logging.getLogger().critical(
            "Failed to import Copernican entrypoint: %s", exc
        )
        return 1


if __name__ == "__main__":  # pragma: no cover - worker entry point
    raise SystemExit(main())
