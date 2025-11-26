"""Integration test covering synthetic SNe, BAO and CMB stages."""

from __future__ import annotations

import importlib
import importlib.util as importlib_util

import numpy as np
import pytest
import yaml

from copernican_lib import dataset_registry, result_writer, run_manifest, utils
from engines import cosmo_engine_mcmc, cosmo_engine_nested
from tests.data.synthetic import model_plugin

# Restore ``importlib.util`` attribute removed by the frozen importlib shim.
setattr(importlib, "util", importlib_util)


@pytest.fixture(autouse=True)
def _temporary_fake_cmb(monkeypatch):
    """Isolate the synthetic CMB toggle to this module's execution.

    The synthetic dataset relies on ``COPERNICAN_FAKE_CMB`` to bypass CAMB so
    the integration suite runs quickly. Applying the flag globally polluted
    other tests with the stubbed spectra and background values, causing
    regressions across the BAO and CMB likelihood checks. Scoping the
    environment variables to each test keeps the optimisation fast without
    altering the remainder of the suite.
    """

    monkeypatch.setenv("COPERNICAN_FAKE_CMB", "1")
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    yield


_EXPECTED_HASHES = {
    "bao.csv": (
        "cc98874d217c1fb3a6f1a4acef2ea8bf3a513496bb7d1979b1e8cb949e551654"
    ),
    "cmb.csv": (
        "75eeaa66c50c836a6aa5b86294b6fee2bd5122efd7e019902b78d1ef1bfb6083"
    ),
    "model.yml": (
        "fb564437121906b249bf38c137209e672ee2e1f2d08de0baa7f1a6f3db448081"
    ),
    "metadata_synthetic.yml": (
        "61a268cc1df54bc1f901c13d4dc083d8c862977c2cd2fc199403a6d27daa2c47"
    ),
    "sne.csv": (
        "43be03513255fe62c358b19671c27918fb40fbb4bca89f39f8db914b3765831b"
    ),
}


@pytest.fixture(
    scope="module", params=[cosmo_engine_mcmc, cosmo_engine_nested]
)
def engine_module(request):
    return request.param


@pytest.fixture(scope="module")
def synthetic_plugin():
    plugin = model_plugin.build_plugin()
    utils.set_random_seed(4)
    return plugin


def _load_datasets():
    import tests.data.synthetic.cosmo_parser_synthetic  # noqa: F401

    sne_df = dataset_registry.load_sne_data("synthetic_integration")
    bao_df = dataset_registry.load_bao_data("synthetic_integration")
    cmb_df = dataset_registry.load_cmb_data("synthetic_integration")
    return sne_df, bao_df, cmb_df


def _dataset_entry(df):
    return {
        "id": df.attrs["dataset_id"],
        "name": df.attrs["dataset_name"],
        "version": df.attrs["dataset_version"],
        "path": df.attrs["data_path"],
        "hashes": df.attrs["file_hashes"],
        "independence": df.attrs.get("independence_assumptions", []),
    }


def _assert_hashes(df):
    hashes = df.attrs["file_hashes"]
    for key, digest in _EXPECTED_HASHES.items():
        assert hashes.get(key) == digest


def _assert_manifest(manifest, engine_name):
    assert manifest["seed"] == utils.get_random_seed()
    assert manifest["engine"]["name"].endswith(engine_name)
    datasets = manifest["datasets"]
    assert set(datasets.keys()) == {"synthetic_integration"}
    entry = datasets["synthetic_integration"]
    for key, digest in _EXPECTED_HASHES.items():
        assert entry["hashes"].get(key) == digest
    assert entry["independence"]


@pytest.mark.parametrize("timestamp", ["20000101_000000"])
def test_synthetic_pipeline(
    tmp_path, engine_module, synthetic_plugin, timestamp
):
    utils.set_random_seed(7)
    sne_df, bao_df, cmb_df = _load_datasets()
    for frame in (sne_df, bao_df, cmb_df):
        _assert_hashes(frame)

    if engine_module is cosmo_engine_mcmc:
        fit = engine_module.fit_cosmology_parameters(
            sne_df,
            synthetic_plugin,
            bao_data_df=bao_df,
            cmb_data_df=cmb_df,
            n_walkers=6,
            n_steps=6,
            burn_in_steps=2,
            progress_granularity=2,
            display_progress=False,
        )
    else:
        fit = engine_module.fit_cosmology_parameters(
            sne_df,
            synthetic_plugin,
            bao_data_df=bao_df,
            cmb_data_df=cmb_df,
            n_live_points=12,
            max_iterations=20,
            evidence_tolerance=1e-3,
            display_progress=False,
        )

    assert fit["success"]
    assert np.isfinite(fit.get("chi2_total", np.nan))
    assert fit.get("chi2_components", {})

    results = {synthetic_plugin.MODEL_NAME: fit}
    summary_paths = result_writer.save_summary(
        results,
        tmp_path,
        timestamp=timestamp,
    )
    for path in summary_paths:
        digest = utils.compute_sha256(path)
        assert digest

    manifest = run_manifest.build_manifest(
        models=[(synthetic_plugin, "0.1")],
        engine_module=engine_module,
        datasets=[_dataset_entry(sne_df)],
    )
    manifest_path = tmp_path / f"run_manifest_{timestamp}.yml"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(manifest, handle, sort_keys=False)

    with open(manifest_path, "r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    _assert_manifest(loaded, engine_module.__name__)

    manifest_hash = utils.compute_sha256(manifest_path)
    assert manifest_hash == utils.compute_sha256(manifest_path)
    assert manifest_hash
