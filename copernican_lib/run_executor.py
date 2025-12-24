"""Lightweight manifest-driven runner scaffolding."""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Sequence

import yaml

from copernican_lib import (
    console_output,
    dataset_registry,
    engine_plugin_validation,
)
from copernican_lib import logger as log_mod
from copernican_lib import (
    model_coder,
    model_spec_validator,
    run_manifest,
    run_pipeline,
    utils,
)
from copernican_lib.run_config import (
    DatasetDescriptor,
    build_config_from_manifest,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MODELS_DIR = _REPO_ROOT / "models"
_MODEL_CACHE_DIR = _MODELS_DIR / "cache"
_LCDM_MODEL_PATH = _MODELS_DIR / "cosmo_model_lcdm.yml"

_PLUGIN_CACHE: dict[str, Any] = {}


@lru_cache(maxsize=1)
def _model_name_index() -> dict[str, Path]:
    """Return a lookup table from model names to their YAML files."""

    index: dict[str, Path] = {}
    if not _MODELS_DIR.is_dir():
        return index
    for path in sorted(_MODELS_DIR.glob("*.yml")):
        if path.name.startswith("__"):
            continue
        try:
            raw = path.read_text(encoding="utf-8")
            model_metadata = yaml.safe_load(raw) or {}
        except Exception:
            model_metadata = {}
        model_name = str(model_metadata.get("model_name") or path.stem).strip()
        stems = {path.stem, model_name}
        if path.stem.startswith("cosmo_model_"):
            stems.add(path.stem.split("cosmo_model_", 1)[-1])
        for stem in stems:
            if not stem:
                continue
            index[stem.casefold()] = path
    return index


def _resolve_model_path(model_name: str | None) -> Path | None:
    """Return the YAML path associated with *model_name*."""

    if not model_name:
        return None
    candidate = _model_name_index().get(model_name.casefold())
    if candidate:
        return candidate
    fallback = (_MODELS_DIR / model_name).with_suffix(".yml")
    return fallback if fallback.is_file() else None


def _build_plugin_from_path(model_path: Path) -> Any:
    """Return an EnginePlugin built from *model_path*."""

    cache_key = str(model_path.resolve())
    if cache_key in _PLUGIN_CACHE:
        return _PLUGIN_CACHE[cache_key]
    cache_path = Path(
        model_spec_validator.validate_and_cache_model(
            model_path, str(_MODEL_CACHE_DIR)
        )
    )
    funcs, parsed = model_coder.generate_callables(cache_path)
    plugin = engine_plugin_validation.build_plugin(parsed, funcs)
    _PLUGIN_CACHE[cache_key] = plugin
    return plugin


def _load_model_plugin(model_name: str | None) -> Any:
    """Load the model declared by *model_name* or fail."""

    if not model_name:
        raise RuntimeError("Model selection must include a name.")
    model_path = _resolve_model_path(model_name)
    if model_path is None:
        raise RuntimeError(f"Model '{model_name}' could not be found.")
    return _build_plugin_from_path(model_path)


def execute_run_from_manifest(
    manifest: dict,
    *,
    script_dir: Path,
    output_root: Path,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
    strict_warnings: bool = False,
    run_start_ts: str | None = None,
    log_prefix: str = "copernican-run",
) -> None:
    """Execute the run described by ``manifest``.

    The ``log_prefix`` argument controls the prefix of the generated
    ``<prefix>_<timestamp>.txt`` file written inside ``output_root`` so
    validation runs can keep their own naming without altering the shared
    manifest executor.
    """

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
    output_root.mkdir(parents=True, exist_ok=True)
    actual_ts = _resolve_run_timestamp(output_root, run_start_ts)
    manifest_filename = f"run_manifest_{actual_ts}.yml"
    manifest_target = output_root / manifest_filename
    try:
        run_manifest.save_manifest(
            manifest,
            str(output_root),
            target_path=manifest_target,
        )
    except Exception as exc:
        log.warning(
            "Failed to copy manifest into %s: %s",
            manifest_target,
            exc,
        )
    else:
        log.info("Persisted manifest at %s", manifest_target)
    run_log = log_mod.setup_logging(
        log_dir=str(output_root),
        base_dir=str(script_dir),
        log_tag=f"{log_prefix}_{actual_ts}.txt",
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

    try:
        engine_module = import_module(config.engine.module_name)
    except Exception as exc:
        log.error(
            "Failed to import engine module %s: %s",
            config.engine.module_name,
            exc,
        )
        raise

    if not _LCDM_MODEL_PATH.is_file():
        raise RuntimeError("Missing required model cosmo_model_lcdm.yml.")
    lcdm_plugin = _build_plugin_from_path(_LCDM_MODEL_PATH)
    alt_name = config.models[0] if config.models else None
    alt_plugin = _load_model_plugin(alt_name) if alt_name else lcdm_plugin

    sampling_plan = dict(config.run_settings.settings or {})
    sampling_plan.setdefault("engine_kind", config.run_settings.engine_kind)
    display_progress = bool(sampling_plan.pop("display_progress", True))
    run_pipeline.execute_run_pipeline(
        lcdm=lcdm_plugin,
        alt_model_plugin=alt_plugin,
        engine_module=engine_module,
        sne_data_df=loaded_data.get("sne"),
        bao_data_df=loaded_data.get("bao"),
        cmb_data_df=loaded_data.get("cmb"),
        sampling_plan=sampling_plan,
        output_dir=str(output_root),
        run_start_ts=actual_ts,
        progress_callback=progress_callback,
        display_progress=display_progress,
        logger=log,
    )


def _describe_datasets(datasets: Sequence[DatasetDescriptor]) -> None:
    """Log the metadata for selected datasets before running."""
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
    """Load a dataset using its descriptor and registered loaders."""
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
    """Return either the override timestamp or derive one from the folder."""
    if override:
        return override
    name = output_root.name
    prefix = "copernican-run_"
    if name.startswith(prefix):
        slice_start = len(prefix)
        return name[slice_start:]
    return utils.get_timestamp()
