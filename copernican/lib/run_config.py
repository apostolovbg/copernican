"""Structured run configuration helpers derived from manifests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class DatasetDescriptor:
    """Store dataset metadata recorded in run manifests."""

    dataset_id: str
    dataset_name: str
    dataset_type: str
    version: str
    path: str
    hashes: dict[str, str] = field(default_factory=dict)
    independence: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class EngineDescriptor:
    """Describes the computational engine recorded in a manifest."""

    module_name: str
    version: str
    label: str | None = None


@dataclass(frozen=True)
class RunSettings:
    """Sampler or inference settings captured on the manifest."""

    engine_kind: str
    settings: dict[str, Any]


@dataclass(frozen=True)
class RunConfig:
    """High-level configuration assembled from a manifest or GUI builder."""

    seed: int
    models: Sequence[str]
    engine: EngineDescriptor
    datasets: Sequence[DatasetDescriptor]
    run_settings: RunSettings


def build_config_from_manifest(manifest: Mapping[str, Any]) -> RunConfig:
    """Translate ``manifest`` contents into a :class:`RunConfig`."""

    selection = manifest.get("selection", {})
    configuration = manifest.get("configuration", {})
    engine_meta = selection.get("engine", {})
    datasets_meta = manifest.get("datasets", {})
    run_settings = configuration.get("run_settings", {})
    settings = {
        key: value for key, value in run_settings.items() if value is not None
    }
    datasets: list[DatasetDescriptor] = []
    for dataset_id, dataset in datasets_meta.items():
        descriptor = DatasetDescriptor(
            dataset_id=dataset_id,
            dataset_name=dataset.get("name", dataset_id),
            dataset_type=dataset.get("type", "unknown"),
            version=dataset.get("version", "unknown"),
            path=dataset.get("path", ""),
            hashes=dataset.get("hashes", {}),
            independence=dataset.get("independence", []),
        )
        datasets.append(descriptor)
    return RunConfig(
        seed=int(manifest.get("seed", 0)),
        models=list(selection.get("models", [])),
        engine=EngineDescriptor(
            module_name=engine_meta.get(
                "name", "copernican.engines.engine_mcmc"
            ),
            version=engine_meta.get("version", "unknown"),
        ),
        datasets=datasets,
        run_settings=RunSettings(
            engine_kind=settings.get("engine_kind", "mcmc"),
            settings=settings,
        ),
    )
