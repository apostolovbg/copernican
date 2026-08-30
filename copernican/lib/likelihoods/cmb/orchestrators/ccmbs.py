r"""Declared-graph CMB solver orchestration helpers."""

from __future__ import annotations

import math
from contextvars import ContextVar
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence

import numpy

from ....cmb_contract import audit_cmb_capabilities, require_cmb_capability
from ....cmb_output import (
    canonical_cmb_spectrum_name as _canonical_spectrum_name,
)
from ....cmb_output import (
    compose_cmb_spectrum_name as _compose_canonical_spectrum_name,
)
from ....cmb_output import (
    split_cmb_spectrum_name as _split_canonical_spectrum_name,
)
from ..errors import classify_exception, failure_context
from ..runtime import cache
from ..runtime.lensing import lensed_cls as _lensed_cls
from ..runtime.performance import PhaseTimer
from ..runtime.projection import _compute_custom_cmb_spectrum_data

_TEMPERATURE_LIKE_OUTPUT_ROLES = {
    "polarization_b",
    "polarization_e",
    "temperature",
}

_LAST_DECLARED_RAW_SPECTRA: ContextVar[Mapping[str, numpy.ndarray] | None] = (
    ContextVar(
        "last_declared_raw_spectra",
        default=None,
    )
)


def last_declared_raw_spectra() -> Mapping[str, numpy.ndarray] | None:
    """Return raw unscaled spectra from the most recent declared solve."""

    return _LAST_DECLARED_RAW_SPECTRA.get()


def _safe_float_output(values: numpy.ndarray) -> numpy.ndarray:
    """Return ``values`` clipped into the finite float64 range."""

    long_values = numpy.asarray(values, dtype=numpy.longdouble)
    float_limits = numpy.finfo(float)
    clipped = numpy.clip(long_values, -float_limits.max, float_limits.max)
    return numpy.asarray(clipped, dtype=float)


def _base_spectrum_name(spectrum_name: str) -> str:
    """Return the unprefixed unlensed spectrum token for ``spectrum_name``."""

    _, _, base_name = _split_canonical_spectrum_name(spectrum_name)
    return base_name


def _is_lensed_requested_spectrum(spectrum_name: str) -> bool:
    """Return ``True`` when ``spectrum_name`` requests exact lensing."""

    lensed, _, _ = _split_canonical_spectrum_name(spectrum_name)
    return lensed


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
    *,
    sampling_factor: float = 1.4,
) -> dict[str, numpy.ndarray]:
    """Return exact curved-sky lensed spectra from unlensed inputs."""

    ell_values = numpy.asarray(ell_grid, dtype=int)
    if ell_values.ndim != 1 or ell_values.size == 0:
        raise ValueError(
            "Declared lensed spectra require a one-dimensional ell grid"
        )
    lmax = int(numpy.max(ell_values))
    expected_ell_grid = numpy.arange(lmax + 1, dtype=int)
    if not numpy.array_equal(ell_values[: lmax + 1], expected_ell_grid):
        raise ValueError(
            "Declared lensed spectra require a contiguous ell grid beginning "
            "at zero"
        )
    missing = sorted(
        required
        for required in ("PP", "TT", "TE", "EE")
        if required not in scaled_spectra
    )
    if missing:
        raise ValueError(
            "Declared lensed spectra require declared TT, TE, EE, and PP "
            f"spectra: {', '.join(missing)}"
        )
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
            "Declared lensed spectra require unlensed spectra defined on the "
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
        sampling_factor=sampling_factor,
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
    """Return the physical normalization applied to one declared spectrum."""

    del perturbation_data
    del lensing_mode
    name = _base_spectrum_name(spectrum_name)
    if name in {"TT", "TE", "EE", "BB"}:
        return (
            ell_factor
            * numpy.longdouble(t_cmb_muK)
            * numpy.longdouble(t_cmb_muK)
        )
    if name in {"TP", "EP"}:
        return ell_factor
    if name == "PP":
        return 2.0 * numpy.longdouble(math.pi) * ell_factor * ell_factor
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
    *,
    perturbation_data: Any | None = None,
) -> tuple[str, ...]:
    """Return the non-lensed spectra needed to satisfy one request set."""

    declared_spectra = {
        _canonical_spectrum_name(name)
        for name, entry in (
            getattr(perturbation_data, "observables", {}) or {}
        ).items()
        if getattr(entry, "kind", None) == "angular_power_spectrum"
    }
    base_spectra: set[str] = set()
    for spectrum_name in canonical_requested_spectra:
        if _is_lensed_requested_spectrum(spectrum_name):
            base_spectra.update({"TT", "TE", "EE", "BB", "PP"})
            continue
        if spectrum_name in declared_spectra:
            base_spectra.add(spectrum_name)
        else:
            base_spectra.add(_base_spectrum_name(spectrum_name))
    return tuple(sorted(base_spectra))


def _resolve_available_spectrum_name(
    requested_name: str,
    *,
    perturbation_data: Any,
    available_spectra: Mapping[str, numpy.ndarray],
) -> str | None:
    """Return the available internal spectrum backing ``requested_name``."""

    canonical_requested = _canonical_spectrum_name(requested_name)
    if canonical_requested in available_spectra:
        return canonical_requested
    lensed, component, base_name = _split_canonical_spectrum_name(
        canonical_requested
    )
    fallback_name = _compose_canonical_spectrum_name(
        lensed=lensed,
        component=None,
        base_name=base_name,
    )
    if fallback_name not in available_spectra:
        return None
    if component is None:
        return fallback_name
    sector_names = tuple(
        str(name)
        for name in perturbation_data.manifest_summary.get("sector_names", ())
    )
    observable_entry = perturbation_data.observables.get(
        _base_spectrum_name(fallback_name)
    )
    if component == "total":
        if len(sector_names) == 1:
            return fallback_name
        observable_sector = str(
            getattr(observable_entry, "sector", None) or ""
        )
        return (
            fallback_name
            if observable_sector in {"", "mixed", "total"}
            else None
        )
    if len(sector_names) == 1 and sector_names[0] == component:
        return fallback_name
    if (
        observable_entry is not None
        and str(observable_entry.sector or "") == component
    ):
        return fallback_name
    return None


def _preflight_requested_capabilities(
    perturbation_data: Any,
    requested_spectra: Sequence[str],
) -> None:
    """Reject unsupported public spectra before background construction."""

    audit = audit_cmb_capabilities(perturbation_data)
    declared_spectra = {
        _canonical_spectrum_name(name)
        for name, entry in perturbation_data.observables.items()
        if entry.kind == "angular_power_spectrum"
    }
    for requested_name in requested_spectra:
        lensed, _component, base_name = _split_canonical_spectrum_name(
            requested_name
        )
        if lensed:
            for dependency in ("TT", "TE", "EE", "PP"):
                require_cmb_capability(audit, dependency)
            continue
        if requested_name in declared_spectra or base_name in declared_spectra:
            continue
        require_cmb_capability(audit, base_name)


def _compute_declared_perturbation_spectrum_impl(
    contract_or_params: Mapping[str, Any],
    ells: Iterable[int],
    *,
    spectra: Sequence[str] = ("TT",),
    background_payload: Mapping[str, Any] | None = None,
    background_provider: Any | None = None,
    workload: str = "full_spectrum",
) -> numpy.ndarray | Mapping[str, numpy.ndarray]:
    """Return spectra from a declared perturbation contract."""

    del background_payload
    perturbation_data = contract_or_params.get("perturbation_data")
    if perturbation_data is None:
        raise ValueError(
            "Declared CMB execution requires precompiled perturbation_data."
        )
    requested_ell_grid = numpy.asarray(tuple(ells), dtype=int)
    if requested_ell_grid.size == 0:
        raise ValueError("ells must not be empty")
    requested_spectra = tuple(str(name) for name in spectra)
    if not requested_spectra:
        raise ValueError("Requested CMB spectra must not be empty")
    canonical_requested_spectra = tuple(
        _canonical_spectrum_name(name) for name in requested_spectra
    )
    _preflight_requested_capabilities(
        perturbation_data,
        canonical_requested_spectra,
    )
    needs_lensing = any(
        _is_lensed_requested_spectrum(spectrum_name)
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
        canonical_requested_spectra,
        perturbation_data=perturbation_data,
    )
    custom_data = _compute_custom_cmb_spectrum_data(
        contract_or_params,
        analysis_ell_grid,
        background_provider=background_provider,
        requested_spectra=base_requested_spectra,
        workload=workload,
    )
    _LAST_DECLARED_RAW_SPECTRA.set(
        {
            str(name): numpy.asarray(values, dtype=numpy.longdouble).copy()
            for name, values in custom_data.spectra.items()
        }
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
        lensing_started = perf_counter()
        try:
            spectra_results.update(
                _assemble_exact_lensed_spectra(
                    lensing_inputs,
                    custom_data.ell_grid,
                    sampling_factor=float(
                        custom_data.runtime_envelope.get(
                            "lensing_sampling_factor",
                            1.4,
                        )
                    ),
                )
            )
        # DEVCOV_ALLOW_BROAD_ONCE lensing adapter normalization boundary.
        except Exception as exc:
            raise classify_exception(
                exc,
                context={"stop_phase": "lensing"},
            ) from exc
        finally:
            cache.extend_latest_cmb_request_phase(
                "lensing",
                perf_counter() - lensing_started,
            )
    for spectrum_name, spectrum_values in spectra_results.items():
        if not numpy.all(numpy.isfinite(spectrum_values)):
            raise ValueError(
                "Declared CMB spectrum calculation produced non-finite "
                f"{spectrum_name} values"
            )
    result = {}
    for original_name, canonical_name in zip(
        requested_spectra,
        canonical_requested_spectra,
    ):
        available_name = _resolve_available_spectrum_name(
            canonical_name,
            perturbation_data=perturbation_data,
            available_spectra=spectra_results,
        )
        if available_name is None:
            continue
        result[original_name] = _safe_float_output(
            spectra_results[available_name]
        )[output_indices]
    if len(result) != len(requested_spectra):
        missing = sorted(
            original_name
            for original_name, canonical_name in zip(
                requested_spectra,
                canonical_requested_spectra,
            )
            if _resolve_available_spectrum_name(
                canonical_name,
                perturbation_data=perturbation_data,
                available_spectra=spectra_results,
            )
            is None
        )
        missing_str = ", ".join(missing)
        raise ValueError(
            "Declared CMB graph does not provide requested spectra: "
            f"{missing_str}"
        )
    if len(result) == 1:
        return next(iter(result.values()))
    return result


def _compute_declared_perturbation_spectrum(
    contract_or_params: Mapping[str, Any],
    ells: Iterable[int],
    *,
    spectra: Sequence[str] = ("TT",),
    background_payload: Mapping[str, Any] | None = None,
    background_provider: Any | None = None,
    workload: str = "full_spectrum",
) -> numpy.ndarray | Mapping[str, numpy.ndarray]:
    """Return typed, fully accounted declared spectra."""

    started = perf_counter()
    previous_record = cache.latest_cmb_performance_record()
    previous_index = (
        0 if previous_record is None else int(previous_record["request_index"])
    )
    context = failure_context(
        contract_or_params,
        workload=workload,
        spectra=spectra,
    )
    try:
        result = _compute_declared_perturbation_spectrum_impl(
            contract_or_params,
            ells,
            spectra=spectra,
            background_payload=background_payload,
            background_provider=background_provider,
            workload=workload,
        )
        latest = cache.latest_cmb_performance_record()
        if (
            latest is not None
            and int(latest["request_index"]) > previous_index
        ):
            accounted = float(
                latest["phase_seconds"].get("total_seconds", 0.0)
            )
            assembly = max(perf_counter() - started - accounted, 0.0)
            cache.extend_latest_cmb_request_phase(
                "likelihood_assembly",
                assembly,
            )
        return result
    # DEVCOV_ALLOW_BROAD_ONCE public declared solver normalization boundary.
    except Exception as exc:
        typed_error = classify_exception(exc, context=context)
        latest = cache.latest_cmb_performance_record()
        if latest is None or int(latest["request_index"]) <= previous_index:
            timer = PhaseTimer(failed_phase="likelihood_assembly")
            record = cache.record_cmb_performance(
                timer.snapshot(total_seconds=perf_counter() - started),
                workload=workload,
                outcome="failure",
                stop_phase="likelihood_assembly",
                failure=typed_error.diagnostic(),
                context=context,
            )
        else:
            cache.fail_latest_cmb_request(
                typed_error.diagnostic(),
                stop_phase=(
                    typed_error.context.get("stop_phase")
                    or latest.get("stop_phase")
                    or "likelihood_assembly"
                ),
            )
            record = cache.latest_cmb_performance_record()
        typed_error.add_context(performance_record=record)
        if typed_error is exc:
            raise
        raise typed_error from exc
