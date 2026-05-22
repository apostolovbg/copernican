"""Manifest-driven validation runner that reuses the standard pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Iterable

from copernican_lib import run_executor, run_manifest, utils

_LOGGER = logging.getLogger(__name__)
_VALIDATION_DIR = Path(__file__).resolve().parent
_MANIFEST_DIR = _VALIDATION_DIR / "manifests"
_OUTPUT_DIR = _VALIDATION_DIR / "output"


def discover_manifests() -> list[Path]:
    """Return all validation manifest files in canonical sort order."""

    if not _MANIFEST_DIR.exists():
        return []
    return sorted(_MANIFEST_DIR.glob("*.yml"))


def run_validation_suite(
    *,
    script_dir: Path | None = None,
    manifest_paths: Iterable[Path] | None = None,
    output_base: Path | None = None,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
    strict_warnings: bool = False,
) -> tuple[int, str]:
    """
    Run every validation manifest through the standard run executor.

    The golden manifests live under ``validation/manifests/`` and their outputs
    are written to
    ``validation/output/{manifest_stem}/validation_run_<timestamp>``.
    """

    manifest_list = (
        list(manifest_paths)
        if manifest_paths is not None
        else discover_manifests()
    )
    if not manifest_list:
        raise RuntimeError(
            f"No validation manifests found under {_MANIFEST_DIR}"
        )
    repo_root = Path(__file__).resolve().parents[1]
    script_dir = Path(script_dir or repo_root)
    output_base = Path(output_base or _OUTPUT_DIR)
    output_base.mkdir(parents=True, exist_ok=True)

    summary_lines: list[str] = ["Validation manifest results:"]
    exit_code = 0
    for manifest_path in manifest_list:
        manifest_path = manifest_path.resolve()
        run_root = output_base / manifest_path.stem
        timestamp = utils.get_timestamp()
        run_dir = run_root / f"validation_run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        _LOGGER.info("Running validation manifest %s", manifest_path.name)
        try:
            manifest = run_manifest.load_manifest(str(manifest_path))
            run_executor.execute_run_from_manifest(
                manifest,
                script_dir=script_dir,
                output_root=run_dir,
                progress_callback=progress_callback,
                strict_warnings=strict_warnings,
                run_start_ts=timestamp,
                log_prefix="validation_run",
            )
        except (OSError, RuntimeError, ValueError) as exc:
            exit_code = 1
            _LOGGER.exception(
                "Validation manifest %s failed", manifest_path.name
            )
            summary_lines.append(f"  {manifest_path.name}: FAILURE ({exc})")
        else:
            summary_lines.append(
                f"  {manifest_path.name}: PASS (outputs in {run_dir})"
            )
    return exit_code, "\n".join(summary_lines)
