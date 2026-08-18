"""Structured run configuration helpers derived from manifests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .cmb_identity import CCMBS_ID, CCMBS_LABEL
from .likelihoods.cmb.solvers.registry import (
    resolve_cmb_solver,
    solver_provenance,
)
from .model_selection import ComparisonRequest, comparison_from_manifest


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
class SamplerDescriptor:
    """Describes the sampler recorded in a manifest."""

    module_name: str
    version: str
    label: str | None = None


@dataclass(frozen=True)
class CMBSolverDescriptor:
    """Describe the CMB solver selected for a run."""

    solver_id: str
    label: str
    capabilities: Mapping[str, Any]


@dataclass(frozen=True)
class RunSettings:
    """Sampler or inference settings captured on the manifest."""

    sampler_kind: str
    settings: dict[str, Any]


@dataclass(frozen=True)
class RunConfig:
    """High-level configuration assembled from a manifest or GUI builder."""

    seed: int
    models: Sequence[str]
    sampler: SamplerDescriptor
    cmb_solver: CMBSolverDescriptor
    datasets: Sequence[DatasetDescriptor]
    run_settings: RunSettings
    comparison: ComparisonRequest

    @property
    def control_model(self) -> str:
        """Return the selected control model name."""

        return self.comparison.control_model.name

    @property
    def test_model(self) -> str:
        """Return the selected test model name."""

        return self.comparison.test_model.name


def build_config_from_manifest(manifest: Mapping[str, Any]) -> RunConfig:
    """Translate ``manifest`` contents into a :class:`RunConfig`."""

    selection = manifest.get("selection", {})
    comparison = comparison_from_manifest(manifest)
    selected_models = list(selection.get("models", []))
    if tuple(selected_models) != comparison.model_names:
        raise ValueError(
            "Run manifests must list exactly the declared control and test "
            "models in role order."
        )
    configuration = manifest.get("configuration", {})
    sampler_meta = selection.get("sampler", {})
    solver_meta = selection.get("cmb_solver", {})
    if not solver_meta:
        solver_meta = manifest.get("cmb_solver", {}) or {}
    selected_solver = resolve_cmb_solver(solver_meta or {"id": CCMBS_ID})
    solver_snapshot = solver_provenance(selected_solver)
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
        models=selected_models,
        sampler=SamplerDescriptor(
            module_name=sampler_meta.get(
                "name", "copernican.samplers.sampler_mcmc"
            ),
            version=sampler_meta.get("version", "unknown"),
        ),
        cmb_solver=CMBSolverDescriptor(
            solver_id=str(solver_snapshot["solver_id"]),
            label=str(solver_snapshot["solver_label"] or CCMBS_LABEL),
            capabilities=dict(solver_snapshot["capabilities"]),
        ),
        datasets=datasets,
        run_settings=RunSettings(
            sampler_kind=settings.get("sampler_kind", "mcmc"),
            settings=settings,
        ),
        comparison=comparison,
    )
