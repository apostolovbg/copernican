"""CLI run worker invoked by the GUI in a child process."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from copernican.lib import logger as log_mod


def main(argv: list[str] | None = None) -> int:
    """Entry point for the GUI worker invoked via ``python -m``."""

    parser = argparse.ArgumentParser(description="Copernican GUI run worker")
    parser.add_argument("config_path", help="Path to the JSON configuration")
    args = parser.parse_args(argv)
    config_path = Path(args.config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    manifest_path = config.get("manifest_path")
    output_dir = config.get("output_dir")
    run_start_ts = config.get("run_start_ts")
    log_prefix = config.get("log_prefix")
    log_name = config.get("log_name")
    if not all(
        isinstance(value, str) and value
        for value in (
            manifest_path,
            output_dir,
            run_start_ts,
            log_prefix,
            log_name,
        )
    ):
        logging.getLogger().error(
            "GUI worker configuration lacks a complete run identity."
        )
        return 1
    expected_log_name = f"{log_prefix}_{run_start_ts}.txt"
    if log_name != expected_log_name:
        logging.getLogger().error(
            "GUI worker log identity does not match its timestamp."
        )
        return 1
    progress_path = config.get("progress_path")
    if progress_path:
        os.environ.setdefault("COPERNICAN_GUI_PROGRESS_PATH", progress_path)
    os.environ["COPERNICAN_GUI_EVENT_STREAM"] = "1"
    os.environ["COPERNICAN_RUN_START_TS"] = run_start_ts
    os.environ["COPERNICAN_RUN_LOG_PREFIX"] = log_prefix
    os.environ.setdefault("COPERNICAN_HEADLESS_RUN", "1")
    log_mod.setup_logging(
        log_dir=output_dir,
        base_dir=str(Path(__file__).resolve().parents[3]),
        log_tag=log_name,
    )
    run_args = ["--manifest", manifest_path]
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
