"""Lightweight manifest-driven runner scaffolding."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Sequence

from copernican_lib import (
    console_output,
    dataset_registry,
    logger as log_mod,
    utils,
)
from copernican_lib.run_config import (
    DatasetDescriptor,
    build_config_from_manifest,
)


def execute_run_from_manifest(
    manifest: dict,
    *,
    script_dir: Path,
    output_root: Path,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
    strict_warnings: bool = False,
    run_start_ts: str | None = None,
) -> None:
    """Execute the run described by ``manifest``."""

    log = log_mod.get_logger()
    console_output.write("Manifest-driven run path invoked.")
    config = build_config_from_manifest(manifest)
    log.info(
        "Executing manifest run: seed=%s, models=%s, engine=%s",
        config.seed,
        config.models,
        config.engine.module_name,
    )
    if progress_callback is not None:
        progress_callback(
            {
                "status": "manifest_execution_started",
                "seed": config.seed,
            }
        )
    console_output.write(
        f"Manifest run targets models {config.models} with engine "
        f"{config.engine.module_name}."
    )
    if strict_warnings:
        log.info("Strict warnings enforced via manifest run.")
    actual_ts = _resolve_run_timestamp(output_root, run_start_ts)
    run_log = log_mod.setup_logging(
        log_dir=str(output_root),
        base_dir=str(script_dir),
        log_tag=f"copernican-run_{actual_ts}.txt",
    )
    console_output.write(
        f"Output directory: {output_root}",
    )
    log.info("Run log stored at %s", run_log)
    _describe_datasets(config.datasets)
    loaded_data: dict[str, Any] = {}
    for descriptor in config.datasets:
        frame = _load_dataset_from_descriptor(descriptor)
        if frame is not None:
            loaded_data[descriptor.dataset_type.lower()] = frame
            log_mod.get_logger().info(
                "Loaded dataset %s: %d entries",
                descriptor.dataset_id,
                len(frame),
            )


def _describe_datasets(datasets: Sequence[DatasetDescriptor]) -> None:
    log = log_mod.get_logger()
    for descriptor in datasets:
        log.info(
            "Dataset %s (%s): %s",
            descriptor.dataset_id,
            descriptor.dataset_type,
            descriptor.dataset_name,
        )


def _load_dataset_from_descriptor(
    descriptor: DatasetDescriptor,
):
    loader_map = {
        "sne": dataset_registry.load_sne_data,
        "bao": dataset_registry.load_bao_data,
        "cmb": dataset_registry.load_cmb_data,
    }
    loader = loader_map.get(descriptor.dataset_type.lower())
    if not loader:
        log_mod.get_logger().warning(
            "No loader configured for dataset type '%s'",
            descriptor.dataset_type,
        )
        return None
    try:
        return loader(dataset_id=descriptor.dataset_id)
    except Exception as exc:
        log_mod.get_logger().error(
            "Failed to load %s dataset '%s': %s",
            descriptor.dataset_type,
            descriptor.dataset_id,
            exc,
            exc_info=True,
        )
    return None


def _resolve_run_timestamp(output_root: Path, override: str | None) -> str:
    if override:
        return override
    name = output_root.name
    prefix = "copernican-run_"
    if name.startswith(prefix):
        return name[len(prefix) :]
    return utils.get_timestamp()
