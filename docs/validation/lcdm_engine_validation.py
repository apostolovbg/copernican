"""Lightweight ΛCDM regression test for both sampling engines.

The runner loads a trimmed Pantheon+SH0ES 2022 supernova slice and the full
BOSS DR12 BAO consensus table, evaluates the Planck 2018 base-ΛCDM parameters
and compares the resulting posterior summaries against lenient reference
thresholds. The goal is to catch regressions in likelihood construction or
engine wiring without running the full documentation build.
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from copernican_lib import (
    dataset_registry,
    engine_plugin_validation,
    model_coder,
    model_spec_validator,
)
from engines import cosmo_engine_mcmc, cosmo_engine_nested

LOGGER = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = REPO_ROOT / "models" / "cosmo_model_lcdm.yml"
CACHE_DIR = MODEL_PATH.parent / "cache"

SNE_DATASET_ID = "pantheon+sh0es_2022"
BAO_DATASET_ID = "boss_dr12_bao"
SNE_SUBSET_SIZE = 40

REFERENCE_PARAMETERS = {
    "H_0": 67.66,
    "Omega_m0": 0.3111,
    "Omega_b": 0.04897,
    "Omega_gamma": 5.38e-5,
    "z_rec": 1089.92,
}

REFERENCE_CHI2 = {
    "sne": 45.2998,
    "bao": 7.2630,
    "total": 52.5627,
}

MEAN_TOLERANCES = {
    "H_0": 4.0,
    "Omega_m0": 0.04,
    "Omega_b": 0.01,
    "Omega_gamma": 2.0e-5,
    "z_rec": 20.0,
}

CHI2_TOLERANCES = {
    "sne": 6.0,
    "bao": 1.5,
    "total": 7.0,
}

def _ensure_importlib_util() -> None:
    """Guarantee ``importlib.util`` is available before parser discovery."""

    if not hasattr(importlib, "util"):
        import importlib.util as importlib_util

        importlib.util = importlib_util

def _build_plugin():
    """Parse the ΛCDM YAML and return a validated engine plugin."""

    _ensure_importlib_util()
    CACHE_DIR.mkdir(exist_ok=True)
    cache_path = model_spec_validator.validate_and_cache_model(
        str(MODEL_PATH), str(CACHE_DIR)
    )
    func_dict, parsed = model_coder.generate_callables(cache_path)
    return engine_plugin_validation.build_plugin(parsed, func_dict)

def _trim_sne_dataset(full_df: pd.DataFrame) -> pd.DataFrame:
    """Return the first ``SNE_SUBSET_SIZE`` rows with diagonal covariance."""

    subset = full_df.head(SNE_SUBSET_SIZE).copy()
    trimmed_attrs = dict(full_df.attrs)
    trimmed_attrs.pop("covariance_matrix_inv", None)
    subset.attrs = trimmed_attrs
    return subset

def load_validation_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the trimmed Pantheon+SH0ES sample and full BOSS DR12 table."""

    _ensure_importlib_util()
    sne_df = dataset_registry.load_sne_data(SNE_DATASET_ID)
    bao_df = dataset_registry.load_bao_data(BAO_DATASET_ID)
    if sne_df is None or bao_df is None:
        msg = "Validation datasets could not be loaded."
        raise RuntimeError(msg)
    return _trim_sne_dataset(sne_df), bao_df

def _parameter_vector(
    plugin: Any, values: Mapping[str, float]
) -> tuple[float, ...]:
    """Order ``values`` according to the plugin parameter list."""

    ordered: list[float] = []
    for name in getattr(
        plugin, "PARAMETER_NAMES", ()
    ):  # type: ignore[attr-defined]
        ordered.append(values[name])
    return tuple(ordered)

def compute_reference_chi2(
    plugin: Any, sne_df: pd.DataFrame, bao_df: pd.DataFrame
) -> dict[str, float]:
    """Evaluate χ² for the Planck 2018 reference point."""

    _, loglike, joint_like = cosmo_engine_mcmc._build_joint_logposterior(
        plugin,
        sne_df,
        bao_df,
        None,
    )  # noqa: SLF001
    loglike(_parameter_vector(plugin, REFERENCE_PARAMETERS))
    state = joint_like.state
    components = state.get("metadata", {}).get("components", {})
    chi2_sne = float(components.get("sne", {}).get("chi2", np.inf))
    chi2_bao = float(components.get("bao", {}).get("chi2", np.inf))
    chi2_total = float(state.get("chi2", np.inf))
    return {
        "sne": chi2_sne,
        "bao": chi2_bao,
        "total": chi2_total,
    }

def _patch_cpu_count_for_tests() -> None:
    """Force sequential execution so notebook runs avoid subprocess errors."""

    cosmo_engine_mcmc.mp.cpu_count = lambda: 1  # type: ignore[attr-defined]

def run_mcmc_validation(
    plugin: Any, sne_df: pd.DataFrame, bao_df: pd.DataFrame
) -> dict[str, Any]:
    """Run the MCMC engine with conservative iteration counts."""

    _patch_cpu_count_for_tests()
    return cosmo_engine_mcmc.fit_cosmology_parameters(
        sne_df,
        plugin,
        bao_data_df=bao_df,
        n_walkers=16,
        n_steps=48,
        burn_in_steps=16,
        display_progress=False,
    )

def run_nested_validation(
    plugin: Any, sne_df: pd.DataFrame, bao_df: pd.DataFrame
) -> dict[str, Any]:
    """Run the nested sampler with limited live points for speed."""

    return cosmo_engine_nested.fit_cosmology_parameters(
        sne_df,
        plugin,
        bao_data_df=bao_df,
        n_live_points=18,
        max_iterations=60,
        evidence_tolerance=0.5,
        display_progress=False,
    )  # type: ignore[no-any-return]

def _assert_within_tolerance(
    observed: Mapping[str, float],
    reference: Mapping[str, float],
    tolerances: Mapping[str, float],
    label: str,
) -> None:
    """Raise ``AssertionError`` when ``observed`` drifts past tolerance."""

    for key, ref_val in reference.items():
        obs_val = observed.get(key)
        tol = tolerances.get(key, 0.0)
        if obs_val is None:
            msg = f"{label} missing expected entry '{key}'"
            raise AssertionError(msg)
        if abs(obs_val - ref_val) > tol:
            msg = (
                f"{label} for {key} drifted by {obs_val - ref_val:+.4f} "
                f"(tolerance ±{tol})."
            )
            raise AssertionError(msg)

def validate_engines() -> None:
    """Run both engines and compare against the stored ΛCDM references."""

    plugin = _build_plugin()
    sne_df, bao_df = load_validation_datasets()

    chi2_now = compute_reference_chi2(plugin, sne_df, bao_df)
    _assert_within_tolerance(
        chi2_now,
        REFERENCE_CHI2,
        {k: 0.1 for k in REFERENCE_CHI2},
        "Reference χ² drift",
    )

    mcmc_result = run_mcmc_validation(plugin, sne_df, bao_df)
    nested_result = run_nested_validation(plugin, sne_df, bao_df)

    _assert_within_tolerance(
        mcmc_result.get("posterior_mean_params", {}),
        REFERENCE_PARAMETERS,
        MEAN_TOLERANCES,
        "MCMC posterior mean",
    )
    _assert_within_tolerance(
        nested_result.get("posterior_mean_params", {}),
        REFERENCE_PARAMETERS,
        MEAN_TOLERANCES,
        "Nested posterior mean",
    )

    _assert_within_tolerance(
        mcmc_result.get("chi2_components", {}),
        REFERENCE_CHI2,
        CHI2_TOLERANCES,
        "MCMC χ²",
    )
    _assert_within_tolerance(
        nested_result.get("chi2_components", {}),
        REFERENCE_CHI2,
        CHI2_TOLERANCES,
        "Nested χ²",
    )

    LOGGER.info("Validation completed successfully for both engines.")

def _format_summary(label: str, result: Mapping[str, Any]) -> str:
    """Return a concise, human-readable engine summary."""

    means = result.get("posterior_mean_params", {})
    chi2 = result.get("chi2_components", {})
    lines = [f"{label} posterior means:"]
    for key in REFERENCE_PARAMETERS:
        val = means.get(key, float("nan"))
        lines.append(f"  {key}: {val:.4f}")
    lines.append("χ² contributions:")
    for key in ("sne", "bao", "total"):
        val = chi2.get(key, float("nan"))
        lines.append(f"  {key}: {val:.4f}")
    return "\n".join(lines)

def _main() -> int:
    """Entry point for CLI execution."""

    logging.basicConfig(level=logging.INFO)
    try:
        plugin = _build_plugin()
        sne_df, bao_df = load_validation_datasets()
        chi2_now = compute_reference_chi2(plugin, sne_df, bao_df)
        LOGGER.info("Reference χ²: %s", chi2_now)

        mcmc_result = run_mcmc_validation(plugin, sne_df, bao_df)
        nested_result = run_nested_validation(plugin, sne_df, bao_df)

        _assert_within_tolerance(
            chi2_now,
            REFERENCE_CHI2,
            {k: 0.1 for k in REFERENCE_CHI2},
            "Reference χ² drift",
        )
        _assert_within_tolerance(
            mcmc_result.get("posterior_mean_params", {}),
            REFERENCE_PARAMETERS,
            MEAN_TOLERANCES,
            "MCMC posterior mean",
        )
        _assert_within_tolerance(
            nested_result.get("posterior_mean_params", {}),
            REFERENCE_PARAMETERS,
            MEAN_TOLERANCES,
            "Nested posterior mean",
        )
        _assert_within_tolerance(
            mcmc_result.get("chi2_components", {}),
            REFERENCE_CHI2,
            CHI2_TOLERANCES,
            "MCMC χ²",
        )
        _assert_within_tolerance(
            nested_result.get("chi2_components", {}),
            REFERENCE_CHI2,
            CHI2_TOLERANCES,
            "Nested χ²",
        )
    except AssertionError as exc:
        LOGGER.error("Validation failed: %s", exc)
        return 1
    except Exception:
        LOGGER.exception("Unexpected error during validation run.")
        return 1

    print(_format_summary("MCMC", mcmc_result))
    print(_format_summary("Nested", nested_result))
    return 0

if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(_main())
