r"""Native declared-graph CMB solver orchestration helpers."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import numpy

from ...model_coder import validate_native_perturbation_execution
from .native_lensing import lensed_cls as _lensed_cls
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
_SPECTRUM_ALIASES = {
    "EPHI": "EP",
    "PHIPHI": "PP",
    "TPHI": "TP",
}


def _safe_float_output(values: numpy.ndarray) -> numpy.ndarray:
    """Return ``values`` clipped into the finite float64 range."""

    long_values = numpy.asarray(values, dtype=numpy.longdouble)
    float_limits = numpy.finfo(float)
    clipped = numpy.clip(long_values, -float_limits.max, float_limits.max)
    return numpy.asarray(clipped, dtype=float)


def _canonical_spectrum_name(spectrum_name: str) -> str:
    """Return the canonical native-spectrum name for ``spectrum_name``."""

    name = str(spectrum_name)
    if name.lower().startswith("lensed_"):
        suffix = name.split("_", 1)[1].upper()
        return f"lensed_{suffix}"
    upper_name = name.upper()
    return _SPECTRUM_ALIASES.get(upper_name, upper_name)


def _lensing_potential_clpp(pp_spectrum: numpy.ndarray) -> numpy.ndarray:
    """Return the declared lensing-potential spectrum with low-ell guard."""

    spectrum = numpy.asarray(pp_spectrum, dtype=numpy.longdouble)
    clpp = numpy.zeros_like(spectrum, dtype=numpy.longdouble)
    if spectrum.size > 2:
        clpp[2:] = spectrum[2:]
    return clpp


def _assemble_exact_lensed_spectra(
    scaled_spectra: Mapping[str, numpy.ndarray],
    ell_grid: numpy.ndarray,
) -> dict[str, numpy.ndarray]:
    """Return exact curved-sky lensed spectra from unlensed inputs."""

    missing = sorted(
        required
        for required in ("PP", "TT", "TE", "EE")
        if required not in scaled_spectra
    )
    if missing:
        raise ValueError(
            "Native lensed spectra require declared TT, TE, EE, and PP "
            f"spectra: {', '.join(missing)}"
        )
    lmax = int(numpy.max(numpy.asarray(ell_grid, dtype=int)))
    tt_spectrum = numpy.asarray(scaled_spectra["TT"], dtype=numpy.longdouble)
    te_spectrum = numpy.asarray(scaled_spectra["TE"], dtype=numpy.longdouble)
    ee_spectrum = numpy.asarray(scaled_spectra["EE"], dtype=numpy.longdouble)
    bb_spectrum = numpy.asarray(
        scaled_spectra.get(
            "BB",
            numpy.zeros_like(tt_spectrum, dtype=numpy.longdouble),
        ),
        dtype=numpy.longdouble,
    )
    if (
        min(
            tt_spectrum.size,
            te_spectrum.size,
            ee_spectrum.size,
            bb_spectrum.size,
            numpy.asarray(scaled_spectra["PP"]).size,
        )
        <= lmax
    ):
        raise ValueError(
            "Native lensed spectra require unlensed spectra defined on the "
            "same ell grid as the remapping calculation."
        )
    if lmax < 2:
        return {
            "lensed_TT": numpy.asarray(tt_spectrum[: lmax + 1], dtype=float),
            "lensed_TE": numpy.asarray(te_spectrum[: lmax + 1], dtype=float),
            "lensed_EE": numpy.asarray(ee_spectrum[: lmax + 1], dtype=float),
            "lensed_BB": numpy.asarray(bb_spectrum[: lmax + 1], dtype=float),
        }
    base_cls = numpy.zeros((lmax + 1, 4), dtype=numpy.longdouble)
    base_cls[:, 0] = tt_spectrum[: lmax + 1]
    base_cls[:, 1] = ee_spectrum[: lmax + 1]
    base_cls[:, 2] = bb_spectrum[: lmax + 1]
    base_cls[:, 3] = te_spectrum[: lmax + 1]
    clpp = _lensing_potential_clpp(
        numpy.asarray(scaled_spectra["PP"], dtype=numpy.longdouble)[: lmax + 1]
    )
    lensed_cls = _lensed_cls(
        base_cls,
        clpp,
        lmax=lmax,
        lmax_lensed=lmax,
    )
    return {
        "lensed_TT": _safe_float_output(lensed_cls[:, 0]),
        "lensed_EE": _safe_float_output(lensed_cls[:, 1]),
        "lensed_BB": _safe_float_output(lensed_cls[:, 2]),
        "lensed_TE": _safe_float_output(lensed_cls[:, 3]),
    }


def _power_spectrum_scale_factor(
    perturbation_data: Any,
    spectrum_name: str,
    *,
    ell_factor: numpy.ndarray,
    t_cmb_muK: float,
    lensing_mode: bool,
) -> numpy.ndarray:
    """Return the physical normalization applied to one native spectrum."""

    del perturbation_data
    del lensing_mode
    name = str(spectrum_name).upper()
    if name in {"TT", "TE", "EE", "BB"}:
        return (
            ell_factor
            * numpy.longdouble(t_cmb_muK)
            * numpy.longdouble(t_cmb_muK)
        )
    if name in {"TP", "EP"}:
        return ell_factor * numpy.longdouble(t_cmb_muK)
    if name == "PP":
        return numpy.ones_like(ell_factor, dtype=numpy.longdouble)
    return (
        ell_factor * numpy.longdouble(t_cmb_muK) * numpy.longdouble(t_cmb_muK)
    )


def _normalize_lensing_input_spectra(
    spectra_results: Mapping[str, numpy.ndarray],
) -> dict[str, numpy.ndarray]:
    """Return the spectra passed into the exact lensing remapper."""

    normalized_spectra = {
        name: numpy.asarray(values, dtype=numpy.longdouble)
        for name, values in spectra_results.items()
    }
    return normalized_spectra


def _requested_base_spectra(
    canonical_requested_spectra: Sequence[str],
) -> tuple[str, ...]:
    """Return the non-lensed spectra needed to satisfy one request set."""

    base_spectra: set[str] = set()
    for spectrum_name in canonical_requested_spectra:
        if spectrum_name in _LENSED_NATIVE_SPECTRA:
            base_spectra.update({"TT", "TE", "EE", "BB", "PP"})
            continue
        base_spectra.add(spectrum_name)
    return tuple(sorted(base_spectra))


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
    requested_ell_grid = numpy.asarray(tuple(ells), dtype=int)
    if requested_ell_grid.size == 0:
        raise ValueError("ells must not be empty")
    requested_spectra = tuple(str(name) for name in spectra)
    canonical_requested_spectra = tuple(
        _canonical_spectrum_name(name) for name in requested_spectra
    )
    needs_lensing = any(
        spectrum_name in _LENSED_NATIVE_SPECTRA
        for spectrum_name in canonical_requested_spectra
    )
    if needs_lensing:
        analysis_ell_grid = numpy.arange(
            int(requested_ell_grid.max()) + 1,
            dtype=int,
        )
        output_indices = requested_ell_grid
    else:
        analysis_ell_grid = requested_ell_grid
        output_indices = numpy.arange(requested_ell_grid.size, dtype=int)
    base_requested_spectra = _requested_base_spectra(
        canonical_requested_spectra
    )
    custom_data = _compute_custom_cmb_spectrum_data(
        contract_or_params,
        analysis_ell_grid,
        background_provider=background_provider,
        requested_spectra=base_requested_spectra,
    )
    ell_factor = (
        numpy.asarray(custom_data.ell_grid, dtype=numpy.longdouble)
        * (numpy.asarray(custom_data.ell_grid, dtype=numpy.longdouble) + 1.0)
        / (2.0 * math.pi)
    )
    t_cmb_muK = numpy.longdouble("2.7255e6")
    requested_spectra = tuple(str(name) for name in spectra)
    spectra_results: dict[str, numpy.ndarray] = {}
    for spectrum_name, spectrum_values in custom_data.spectra.items():
        raw_values = numpy.asarray(spectrum_values, dtype=numpy.longdouble)
        canonical_name = _canonical_spectrum_name(spectrum_name)
        scale = numpy.asarray(
            _power_spectrum_scale_factor(
                perturbation_data,
                canonical_name,
                ell_factor=ell_factor,
                t_cmb_muK=t_cmb_muK,
                lensing_mode=needs_lensing,
            ),
            dtype=numpy.longdouble,
        )
        spectra_results[canonical_name] = scale * raw_values
    if needs_lensing:
        lensing_inputs = _normalize_lensing_input_spectra(spectra_results)
        spectra_results.update(
            _assemble_exact_lensed_spectra(
                lensing_inputs,
                custom_data.ell_grid,
            )
        )
    for spectrum_name, spectrum_values in spectra_results.items():
        if not numpy.all(numpy.isfinite(spectrum_values)):
            raise ValueError(
                "Custom CMB spectrum calculation produced non-finite "
                f"{spectrum_name} values"
            )
    result = {}
    for original_name, canonical_name in zip(
        requested_spectra,
        canonical_requested_spectra,
    ):
        if canonical_name not in spectra_results:
            continue
        result[original_name] = _safe_float_output(
            spectra_results[canonical_name]
        )[output_indices]
    if len(result) != len(requested_spectra):
        missing = sorted(
            original_name
            for original_name, canonical_name in zip(
                requested_spectra,
                canonical_requested_spectra,
            )
            if canonical_name not in spectra_results
        )
        missing_str = ", ".join(missing)
        raise ValueError(
            "Declared CMB graph does not provide requested spectra: "
            f"{missing_str}"
        )
    if len(result) == 1:
        return next(iter(result.values()))
    return result
