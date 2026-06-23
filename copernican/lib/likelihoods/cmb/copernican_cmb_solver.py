r"""Native declared-graph CMB solver orchestration helpers."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import numpy

from ...model_coder import validate_native_perturbation_execution
from .native_projection import _compute_custom_cmb_spectrum_data

_CMB_TEMPERATURE_SPECTRA = {"BB", "EE", "TE", "TT"}
_LENSED_NATIVE_SPECTRA = frozenset(
    {"lensed_BB", "lensed_EE", "lensed_TE", "lensed_TT"}
)
_TEMPERATURE_LIKE_OUTPUT_ROLES = {
    "polarization_b",
    "polarization_e",
    "temperature",
}


def _gaussian_smooth_spectrum(
    values: numpy.ndarray,
    *,
    sigma: float,
) -> numpy.ndarray:
    """Return ``values`` convolved with an index-space Gaussian kernel."""

    spectrum = numpy.asarray(values, dtype=float)
    if spectrum.size <= 2 or sigma <= 0.0:
        return spectrum.copy()
    radius = max(1, int(math.ceil(3.0 * float(sigma))))
    offsets = numpy.arange(-radius, radius + 1, dtype=float)
    kernel = numpy.exp(-0.5 * (offsets / float(sigma)) ** 2)
    kernel /= max(float(numpy.sum(kernel)), 1.0e-12)
    padded = numpy.pad(spectrum, radius, mode="edge")
    smoothed = numpy.convolve(padded, kernel, mode="valid")
    return numpy.asarray(smoothed, dtype=float)


def _assemble_approximate_lensed_spectra(
    scaled_spectra: Mapping[str, numpy.ndarray],
    ell_grid: numpy.ndarray,
) -> dict[str, numpy.ndarray]:
    """Return bounded approximate lensed spectra from unlensed inputs."""

    del ell_grid
    if "PP" not in scaled_spectra:
        raise ValueError(
            "Native lensed spectra require a declared PP spectrum."
        )
    if "EE" not in scaled_spectra:
        raise ValueError(
            "Native lensed spectra require a declared EE spectrum."
        )
    if "TT" not in scaled_spectra or "TE" not in scaled_spectra:
        raise ValueError(
            "Native lensed spectra require declared TT and TE spectra."
        )
    tt_spectrum = numpy.asarray(scaled_spectra["TT"], dtype=float)
    te_spectrum = numpy.asarray(scaled_spectra["TE"], dtype=float)
    ee_spectrum = numpy.asarray(scaled_spectra["EE"], dtype=float)
    bb_spectrum = numpy.asarray(
        scaled_spectra.get(
            "BB",
            numpy.zeros_like(ee_spectrum, dtype=float),
        ),
        dtype=float,
    )
    pp_spectrum = numpy.asarray(scaled_spectra["PP"], dtype=float)
    radiative_scale = max(
        float(numpy.max(numpy.abs(tt_spectrum))),
        float(numpy.max(numpy.abs(ee_spectrum))),
        float(numpy.max(numpy.abs(te_spectrum))),
        1.0,
    )
    potential_scale = max(
        float(numpy.max(numpy.abs(pp_spectrum))),
        1.0e-30,
    )
    coupling_strength = numpy.clip(
        2.5 * (potential_scale / radiative_scale) ** 0.2,
        0.0,
        0.65,
    )
    smoothing_sigma = 1.0 + 6.0 * float(coupling_strength)
    smoothed_tt = _gaussian_smooth_spectrum(
        tt_spectrum,
        sigma=smoothing_sigma,
    )
    smoothed_te = _gaussian_smooth_spectrum(
        te_spectrum,
        sigma=smoothing_sigma,
    )
    smoothed_ee = _gaussian_smooth_spectrum(
        ee_spectrum,
        sigma=smoothing_sigma,
    )
    ee_leakage = numpy.abs(smoothed_ee - ee_spectrum)
    lensed_tt = (
        1.0 - coupling_strength
    ) * tt_spectrum + coupling_strength * smoothed_tt
    lensed_te = (
        1.0 - coupling_strength
    ) * te_spectrum + coupling_strength * smoothed_te
    lensed_ee = (
        1.0 - coupling_strength
    ) * ee_spectrum + coupling_strength * smoothed_ee
    lensed_bb = bb_spectrum + (1.5 * coupling_strength * ee_leakage)
    return {
        "lensed_TT": numpy.asarray(lensed_tt, dtype=float),
        "lensed_TE": numpy.asarray(lensed_te, dtype=float),
        "lensed_EE": numpy.asarray(lensed_ee, dtype=float),
        "lensed_BB": numpy.asarray(lensed_bb, dtype=float),
    }


def _power_spectrum_scale_factor(
    perturbation_data: Any,
    spectrum_name: str,
    *,
    ell_factor: numpy.ndarray,
    t_cmb_muK: float,
) -> numpy.ndarray:
    """Return the output scaling applied to one native power spectrum."""

    if spectrum_name in _LENSED_NATIVE_SPECTRA:
        return ell_factor * (t_cmb_muK * t_cmb_muK)
    observable_entry = perturbation_data.observables.get(spectrum_name)
    if (
        observable_entry is None
        or observable_entry.kind != "angular_power_spectrum"
    ):
        return numpy.ones_like(ell_factor, dtype=float)
    primary_name = str(observable_entry.primary or "")
    secondary_name = str(observable_entry.secondary or "")
    primary_entry = perturbation_data.observables.get(primary_name)
    secondary_entry = perturbation_data.observables.get(secondary_name)
    if primary_entry is None or secondary_entry is None:
        return numpy.ones_like(ell_factor, dtype=float)
    temperature_like_count = sum(
        entry.output_role in _TEMPERATURE_LIKE_OUTPUT_ROLES
        for entry in (primary_entry, secondary_entry)
    )
    potential_count = sum(
        entry.output_role == "potential"
        for entry in (primary_entry, secondary_entry)
    )
    if temperature_like_count == 2 and potential_count == 0:
        return ell_factor * (t_cmb_muK * t_cmb_muK)
    if temperature_like_count == 1 and potential_count == 1:
        return ell_factor * t_cmb_muK
    return numpy.ones_like(ell_factor, dtype=float)


def _is_structured_camb_contract(
    contract_or_params: Mapping[str, Any],
) -> bool:
    """Return ``True`` for structured CMB contracts."""

    keys = {str(key) for key in contract_or_params.keys()}
    required = {
        "backend",
        "calls",
        "grids",
        "param_map",
        "perturbations",
        "values",
    }
    return required.issubset(keys)


def _combine_camb_contracts(
    background_contract: Mapping[str, Any],
    perturbation_contract: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return a structured CAMB contract with perturbation metadata."""

    combined = dict(background_contract)
    if perturbation_contract:
        combined["perturbations"] = dict(perturbation_contract)
    return combined


def _validate_camb_perturbation_execution(
    contract: Mapping[str, Any],
) -> None:
    """Reject unsupported perturbation declarations before CAMB runs."""

    perturbations = contract.get("perturbations")
    if perturbations is None:
        raise ValueError("Structured CAMB contract is missing perturbations")
    if not isinstance(perturbations, Mapping):
        raise ValueError("cmb.perturbations must be a mapping")

    model_name = contract.get("model_name", "unknown model")
    backend = str(contract.get("backend", "camb"))

    standard = perturbations.get("standard")
    if not isinstance(standard, bool):
        raise ValueError("cmb.perturbations.standard must be boolean")
    if standard:
        return

    backend_mapping = perturbations.get("backend_mapping", {})
    backend_entry = {}
    if isinstance(backend_mapping, Mapping):
        backend_entry = backend_mapping.get(backend, {}) or {}

    implemented = None
    if isinstance(backend_entry, Mapping):
        implemented = backend_entry.get("implemented")

    validate_native_perturbation_execution(
        model_name=str(model_name),
        backend=backend,
        standard=standard,
        implemented=implemented if isinstance(implemented, bool) else None,
    )


def _compute_declared_perturbation_spectrum(
    contract_or_params: Mapping[str, Any],
    ells: Iterable[int],
    *,
    spectra: Sequence[str] = ("TT",),
    background_payload: Mapping[str, Any] | None = None,
    background_provider: Any | None = None,
) -> numpy.ndarray | Mapping[str, numpy.ndarray]:
    """Return spectra from a declared non-standard perturbation contract."""

    del background_payload
    perturbation_data = contract_or_params.get("perturbation_data")
    if perturbation_data is None:
        raise ValueError(
            "Native CMB execution requires precompiled perturbation_data."
        )
    custom_data = _compute_custom_cmb_spectrum_data(
        contract_or_params,
        ells,
        background_provider=background_provider,
    )
    ell_factor = (
        custom_data.ell_grid.astype(float)
        * (custom_data.ell_grid.astype(float) + 1.0)
        / (2.0 * math.pi)
    )
    t_cmb_muK = 2.7255e6
    requested_spectra = tuple(str(name) for name in spectra)
    spectra_results: dict[str, numpy.ndarray] = {}
    for spectrum_name, spectrum_values in custom_data.spectra.items():
        raw_values = numpy.asarray(spectrum_values, dtype=float)
        scale = _power_spectrum_scale_factor(
            perturbation_data,
            str(spectrum_name),
            ell_factor=ell_factor,
            t_cmb_muK=t_cmb_muK,
        )
        spectra_results[str(spectrum_name)] = numpy.asarray(
            scale * raw_values,
            dtype=float,
        )
    if any(name in _LENSED_NATIVE_SPECTRA for name in requested_spectra):
        spectra_results.update(
            _assemble_approximate_lensed_spectra(
                spectra_results,
                custom_data.ell_grid,
            )
        )
    for spectrum_name, spectrum_values in spectra_results.items():
        if not numpy.all(numpy.isfinite(spectrum_values)):
            raise ValueError(
                "Custom CMB spectrum calculation produced non-finite "
                f"{spectrum_name} values"
            )
    result = {
        spec: numpy.asarray(spectra_results[spec], dtype=float)
        for spec in requested_spectra
        if spec in spectra_results
    }
    if len(result) != len(requested_spectra):
        missing = sorted(set(requested_spectra) - set(result))
        missing_str = ", ".join(missing)
        raise ValueError(
            "Declared CMB graph does not provide requested spectra: "
            f"{missing_str}"
        )
    if len(result) == 1:
        return next(iter(result.values()))
    return result
