"""Independent CAMB reference helpers for scientific tests.

Nothing in the production package imports this module. Its calculations
provide an external comparison surface for declared solver acceptance tests.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import camb
import numpy

_C_LIGHT_KM_S = 299_792.458
_LMAX_PADDING = 300
_LENS_POTENTIAL_ACCURACY = 0
_MNU_PATTERN = re.compile(r"^mnu(\d+)$")
CAMB_REFERENCE_IDENTITY = f"camb:{camb.__version__}"

# This contract is intentionally frozen in the test-owned reference surface.
# It mirrors the bundled LambdaCDM initial point but never enters production
# CCMBS code or becomes a backend fallback.
FIXED_LCDM_REFERENCE_CONTRACT = {
    "backend": "camb",
    "param_map": {
        "H0": 75.0,
        "ombh2": 0.0309375,
        "omch2": 0.18,
        "tau": 0.054,
        "As": 2.1e-9,
        "ns": 0.965,
        "Neff": 3.0,
        "YHe": 0.245,
    },
    "grids": {},
    "values": {},
    "calls": [],
}
FIXED_LCDM_REFERENCE_ELL_VALUES = (2, 20, 100, 200, 500, 1000, 1500, 2000)
FIXED_LCDM_REFERENCE_SPECTRA = ("TT", "TE", "EE")
FIXED_LCDM_FULL_REFERENCE_SPECTRA = (
    "TT",
    "TE",
    "EE",
    "BB",
    "PP",
    "TP",
    "EP",
    "lensed_TT",
    "lensed_TE",
    "lensed_EE",
    "lensed_BB",
)
FIXED_LCDM_REFERENCE_TOLERANCES = {
    "TT": 0.02,
    "TE": 0.03,
    "EE": 0.02,
}
FIXED_LCDM_FULL_REFERENCE_TOLERANCES = {
    "TT": 0.02,
    "TE": 0.03,
    "EE": 0.02,
    "BB": 0.02,
    "PP": 0.03,
    "TP": 0.05,
    "EP": 0.05,
    "lensed_TT": 0.02,
    "lensed_TE": 0.03,
    "lensed_EE": 0.02,
    "lensed_BB": 0.05,
}
FULL_LCDM_REFERENCE_FIXTURE_PATH = (
    Path(__file__).resolve().parent.parent
    / "fixtures"
    / "camb_lcdm_reference.json"
)
_CMB_SPECTRUM_COLUMNS = {"TT": 0, "EE": 1, "BB": 2, "TE": 3}
_LENSING_SPECTRUM_COLUMNS = {"PP": 0, "TP": 1, "EP": 2}


def _coerce_numeric_scalar(value: Any, *, name: str) -> float:
    """Return ``value`` as a finite scalar float."""

    array_value = numpy.asarray(value, dtype=float)
    if array_value.ndim != 0:
        raise ValueError(f"{name} must evaluate to a scalar")
    scalar = float(array_value)
    if not numpy.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _coerce_numeric_array(value: Any, *, name: str) -> numpy.ndarray:
    """Return ``value`` as a finite one-dimensional array."""

    array_value = numpy.asarray(value, dtype=float)
    if array_value.ndim != 1:
        raise ValueError(f"{name} must evaluate to a one-dimensional array")
    if array_value.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not numpy.all(numpy.isfinite(array_value)):
        raise ValueError(f"{name} must contain only finite values")
    return array_value


def _is_structured_camb_background_contract(
    contract_or_params: Mapping[str, Any],
) -> bool:
    """Return ``True`` when ``contract_or_params`` uses the CAMB adapter."""

    keys = {str(key) for key in contract_or_params.keys()}
    required = {"backend", "calls", "grids", "param_map", "values"}
    return required.issubset(keys)


def _make_camb_params(
    contract_or_params: Mapping[str, Any], *, lmax: int | None = None
) -> camb.CAMBparams:
    """Return CAMB parameters from a structured reference contract."""

    contract = contract_or_params
    if not _is_structured_camb_background_contract(contract):
        raise ValueError(
            "Structured CAMB reference contracts must include backend, "
            "param_map, grids, values and calls"
        )
    if contract.get("backend") != "camb":
        raise ValueError("Only the CAMB backend is supported")

    param_map = contract.get("param_map", {})
    if not isinstance(param_map, Mapping):
        raise ValueError("cmb.param_map must be a mapping")

    params = camb.CAMBparams()
    model_kwargs: dict[str, Any] = {}
    consumed_keys: set[str] = set()

    def _use_scalar(key: str) -> float:
        """Return one scalar CAMB parameter and mark it as consumed."""

        value = _coerce_numeric_scalar(param_map[key], name=key)
        consumed_keys.add(key)
        return value

    if "H0" in param_map:
        model_kwargs["H0"] = _use_scalar("H0")
    if "ombh2" in param_map:
        model_kwargs["ombh2"] = _use_scalar("ombh2")
    if "omch2" in param_map:
        model_kwargs["omch2"] = _use_scalar("omch2")
    if "omk" in param_map:
        model_kwargs["omk"] = _use_scalar("omk")
    if "tau" in param_map:
        model_kwargs["tau"] = _use_scalar("tau")
    if "YHe" in param_map:
        model_kwargs["YHe"] = _use_scalar("YHe")
    if "theta_H0_range" in param_map:
        theta_range = _coerce_numeric_array(
            param_map["theta_H0_range"], name="theta_H0_range"
        )
        if theta_range.size < 2:
            raise ValueError("theta_H0_range must contain at least two values")
        model_kwargs["theta_H0_range"] = tuple(
            float(value) for value in theta_range[:2]
        )
        consumed_keys.add("theta_H0_range")

    if "Neff" in param_map:
        model_kwargs["nnu"] = _use_scalar("Neff")
    if "standard_neutrino_neff" in param_map:
        model_kwargs["standard_neutrino_neff"] = _use_scalar(
            "standard_neutrino_neff"
        )
    if "num_massive_neutrinos" in param_map:
        model_kwargs["num_massive_neutrinos"] = int(
            _use_scalar("num_massive_neutrinos")
        )
    if "neutrino_hierarchy" in param_map:
        model_kwargs["neutrino_hierarchy"] = param_map["neutrino_hierarchy"]
        consumed_keys.add("neutrino_hierarchy")

    dynamic_mass_keys = [
        key for key in param_map if _MNU_PATTERN.match(str(key))
    ]
    if dynamic_mass_keys:
        ordered = sorted(
            dynamic_mass_keys,
            key=lambda item: int(_MNU_PATTERN.match(str(item)).group(1)),
        )
        masses = [
            _coerce_numeric_scalar(param_map[key], name=str(key))
            for key in ordered
        ]
        model_kwargs.setdefault("num_massive_neutrinos", len(masses))
        model_kwargs["mnu"] = float(numpy.sum(masses))
        consumed_keys.update(ordered)
    if "sum_mnu" in param_map:
        model_kwargs["mnu"] = _use_scalar("sum_mnu")
    elif "mnu" in param_map:
        model_kwargs["mnu"] = _use_scalar("mnu")

    if "Alens" in param_map:
        model_kwargs["Alens"] = _use_scalar("Alens")

    params.set_cosmology(**model_kwargs)

    if "omnuh2" in param_map:
        params.omnuh2 = _use_scalar("omnuh2")

    accuracy = getattr(params, "Accuracy", None)
    if accuracy is not None:
        if "AccuracyBoost" in param_map:
            accuracy.AccuracyBoost = _use_scalar("AccuracyBoost")
        if "lAccuracyBoost" in param_map:
            accuracy.LAccuracyBoost = _use_scalar("lAccuracyBoost")
        if "kAccuracyBoost" in param_map:
            accuracy.KAccuracyBoost = _use_scalar("kAccuracyBoost")

    if lmax is not None:
        params.set_for_lmax(
            int(lmax) + _LMAX_PADDING,
            lens_potential_accuracy=_LENS_POTENTIAL_ACCURACY,
        )

    power_kwargs: dict[str, Any] = {}
    if "As" in param_map:
        power_kwargs["As"] = _use_scalar("As")
    if "ns" in param_map:
        power_kwargs["ns"] = _use_scalar("ns")
    if "nrun" in param_map:
        power_kwargs["nrun"] = _use_scalar("nrun")
    if "nrunrun" in param_map:
        power_kwargs["nrunrun"] = _use_scalar("nrunrun")
    if "r" in param_map:
        power_kwargs["r"] = _use_scalar("r")
    if power_kwargs:
        params.InitPower.set_params(**power_kwargs)

    for call in contract.get("calls", []) or []:
        method = call.get("method")
        if method == "set_dark_energy":
            call_kwargs = dict(call.get("kwargs", {}) or {})
            call_args = call.get("args", {}) or {}
            if call_args:
                raise ValueError("set_dark_energy does not accept args")
            if "w0" in call_kwargs and "w" not in call_kwargs:
                call_kwargs["w"] = call_kwargs.pop("w0")
            elif "w0" in call_kwargs and "w" in call_kwargs:
                raise ValueError(
                    "set_dark_energy cannot receive both w and w0"
                )
            for numeric_key in ("w", "wa", "cs2"):
                if numeric_key in call_kwargs:
                    call_kwargs[numeric_key] = _coerce_numeric_scalar(
                        call_kwargs[numeric_key], name=numeric_key
                    )
            params.set_dark_energy(**call_kwargs)
        elif method == "set_dark_energy_w_a":
            call_args = dict(call.get("args", {}) or {})
            call_kwargs = dict(call.get("kwargs", {}) or {})
            if set(call_args) != {"a", "w"}:
                raise ValueError(
                    "set_dark_energy_w_a requires args 'a' and 'w'"
                )
            a_array = _coerce_numeric_array(call_args["a"], name="a")
            w_array = _coerce_numeric_array(call_args["w"], name="w")
            if a_array.shape != w_array.shape:
                raise ValueError("set_dark_energy_w_a arrays must match")
            if not numpy.all(numpy.diff(a_array) > 0.0):
                raise ValueError(
                    "set_dark_energy_w_a scale-factor array must be "
                    "strictly increasing"
                )
            params.set_dark_energy_w_a(
                a=a_array,
                w=w_array,
                **call_kwargs,
            )
        else:
            raise ValueError(f"Unsupported CAMB call method: {method!r}")

    unused_keys = sorted(
        str(key) for key in param_map if key not in consumed_keys
    )
    if unused_keys:
        raise ValueError(
            "Unconsumed scalar CAMB parameter(s): " + ", ".join(unused_keys)
        )

    return params


def _compute_cmb_spectrum_direct(
    contract_or_params: Mapping[str, Any],
    ells: Iterable[int],
    *,
    spectra: Sequence[str] = ("TT",),
) -> numpy.ndarray | Mapping[str, numpy.ndarray]:
    """Return spectra directly without routing through the scalar cache."""

    ell_arr = numpy.asarray(list(ells), dtype=int)
    if ell_arr.size == 0:
        raise ValueError("ells must not be empty")
    lmax = int(ell_arr.max())
    params = _make_camb_params(contract_or_params, lmax=lmax)
    results = camb.get_results(params)
    cls = results.get_unlensed_scalar_cls(lmax=lmax, CMB_unit="muK")
    lensed_cls = results.get_lensed_scalar_cls(lmax=lmax, CMB_unit="muK")
    lensing_cls = results.get_lens_potential_cls(lmax=lmax, CMB_unit="muK")
    out: dict[str, numpy.ndarray] = {}
    for spectrum_name in spectra:
        name = str(spectrum_name)
        if name in _CMB_SPECTRUM_COLUMNS:
            out[name] = cls[:, _CMB_SPECTRUM_COLUMNS[name]]
        elif name.startswith("lensed_") and name[7:] in _CMB_SPECTRUM_COLUMNS:
            out[name] = lensed_cls[:, _CMB_SPECTRUM_COLUMNS[name[7:]]]
        elif name in _LENSING_SPECTRUM_COLUMNS:
            out[name] = lensing_cls[:, _LENSING_SPECTRUM_COLUMNS[name]]
        else:
            raise ValueError(f"Unsupported CAMB reference spectrum: {name}")
    result = {spec: out[str(spec)][ell_arr] for spec in spectra}
    if len(result) == 1:
        return next(iter(result.values()))
    return result


def _compute_camb_background_direct(
    contract_or_params: Mapping[str, Any],
    redshifts: Sequence[float],
) -> dict[str, numpy.ndarray]:
    """Return background observables directly without cached scalars."""

    z_arr = numpy.asarray(redshifts, dtype=float)
    params = _make_camb_params(contract_or_params, lmax=None)
    results = camb.get_results(params)
    derived = results.get_derived_params()
    rs_drag = float(derived.get("rdrag", float("nan")))

    comoving_distances: list[float] = []
    angular_distance_values: list[float] = []
    hubble_parameters: list[float] = []
    for z_val in z_arr:
        comoving_distances.append(
            float(results.comoving_radial_distance(float(z_val)))
        )
        angular_distance_values.append(
            float(results.angular_diameter_distance(float(z_val)))
        )
        hubble_parameters.append(float(results.hubble_parameter(float(z_val))))

    comoving_distance_array = numpy.asarray(comoving_distances, dtype=float)
    angular_distance_array = numpy.asarray(
        angular_distance_values, dtype=float
    )
    hubble_parameter_array = numpy.asarray(hubble_parameters, dtype=float)
    with numpy.errstate(divide="ignore", invalid="ignore"):
        hubble_distance_array = numpy.where(
            numpy.abs(hubble_parameter_array) > 1e-12,
            _C_LIGHT_KM_S / hubble_parameter_array,
            numpy.nan,
        )
    term = comoving_distance_array * comoving_distance_array
    term *= z_arr
    with numpy.errstate(divide="ignore", invalid="ignore"):
        term = term * hubble_distance_array
    volume_average_distance_array = numpy.full_like(
        term, numpy.nan, dtype=float
    )
    mask = numpy.isfinite(term) & (term >= 0.0)
    volume_average_distance_array[mask] = numpy.power(term[mask], 1.0 / 3.0)
    zero = numpy.isfinite(term) & (z_arr == 0.0)
    volume_average_distance_array[zero] = 0.0

    return {
        "rs_drag": rs_drag,
        "DM": comoving_distance_array,
        "DH": hubble_distance_array,
        "DA": angular_distance_array,
        "DV": volume_average_distance_array,
        "Hz": hubble_parameter_array,
        "z": z_arr.copy(),
    }


def compute_cmb_spectrum_from_camb_contract(
    contract_or_params: Mapping[str, Any],
    ells: Iterable[int],
    *,
    spectra: Sequence[str] = ("TT",),
) -> numpy.ndarray | Mapping[str, numpy.ndarray]:
    """Return standard CAMB spectra from a structured contract."""

    logger = logging.getLogger()
    try:
        return _compute_cmb_spectrum_direct(
            contract_or_params,
            ells,
            spectra=spectra,
        )
    except (
        AttributeError,
        ImportError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        logger.error("(compute_cmb_spectrum_from_camb_contract): %s", exc)
        raise


def compute_camb_background_observables(
    contract_or_params: Mapping[str, Any], redshifts: Sequence[float]
) -> dict[str, numpy.ndarray]:
    """Return CAMB background quantities for ``redshifts``."""

    if not _is_structured_camb_background_contract(contract_or_params):
        raise ValueError(
            "Structured CAMB background contracts must include backend, "
            "param_map, grids, values and calls"
        )

    return _compute_camb_background_direct(contract_or_params, redshifts)


def describe_camb_configuration() -> dict[str, Any]:
    """Return the default CAMB configuration used by the CMB helpers."""

    params = camb.CAMBparams()
    accuracy = getattr(params, "Accuracy", None)
    accuracy_info: dict[str, float] = {}
    if accuracy is not None:
        accuracy_info = {
            "AccuracyBoost": float(getattr(accuracy, "AccuracyBoost", 1.0)),
            "LAccuracyBoost": float(getattr(accuracy, "LAccuracyBoost", 1.0)),
            "KAccuracyBoost": float(getattr(accuracy, "KAccuracyBoost", 1.0)),
        }

    return {
        "reference_identity": CAMB_REFERENCE_IDENTITY,
        "lmax_padding": _LMAX_PADDING,
        "lens_potential_accuracy": _LENS_POTENTIAL_ACCURACY,
        "reionization_model": "optical_depth_tau",
        "accuracy": accuracy_info,
    }


def _reference_jsonable(value: Any) -> Any:
    """Convert NumPy values into deterministic fixture JSON values."""

    if isinstance(value, numpy.ndarray):
        return value.tolist()
    if isinstance(value, numpy.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {
            str(key): _reference_jsonable(item)
            for key, item in sorted(
                value.items(), key=lambda pair: str(pair[0])
            )
        }
    if isinstance(value, (tuple, list)):
        return [_reference_jsonable(item) for item in value]
    return value


def reference_fixture_sha256(fixture: Mapping[str, Any]) -> str:
    """Return the deterministic digest for one frozen reference fixture."""

    payload = json.dumps(
        _reference_jsonable(fixture), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_lcdm_reference_fixture(
    ells: Iterable[int] = FIXED_LCDM_REFERENCE_ELL_VALUES,
    *,
    spectra: Sequence[str] = FIXED_LCDM_REFERENCE_SPECTRA,
) -> dict[str, Any]:
    """Return the fixed CAMB LCDM spectra and auditable provenance."""

    ell_values = tuple(int(value) for value in ells)
    requested_spectra = tuple(str(value).upper() for value in spectra)
    computed = compute_cmb_spectrum_from_camb_contract(
        FIXED_LCDM_REFERENCE_CONTRACT,
        ell_values,
        spectra=requested_spectra,
    )
    if not isinstance(computed, Mapping):
        computed = {requested_spectra[0]: computed}
    fixture: dict[str, Any] = {
        "schema_version": 1,
        "reference_identity": CAMB_REFERENCE_IDENTITY,
        "normalization": "unlensed_scalar_D_ell_microkelvin_squared",
        "ell_values": ell_values,
        "spectra": {
            name: numpy.asarray(computed[name], dtype=float)
            for name in requested_spectra
        },
        "contract": FIXED_LCDM_REFERENCE_CONTRACT,
        "tolerances": {
            name: float(FIXED_LCDM_REFERENCE_TOLERANCES[name])
            for name in requested_spectra
            if name in FIXED_LCDM_REFERENCE_TOLERANCES
        },
        "provenance": describe_camb_configuration(),
    }
    fixture["fixture_sha256"] = reference_fixture_sha256(fixture)
    return fixture


def build_lcdm_full_reference_fixture(
    ells: Iterable[int] = FIXED_LCDM_REFERENCE_ELL_VALUES,
) -> dict[str, Any]:
    """Return the complete fixed scalar CAMB comparison fixture.

    The fixture stores both raw ``C_ell`` and CAMB-native ``D_ell`` values.
    CAMB's lens-potential spectra use their documented deflection-potential
    convention; the per-observable conversion metadata makes that distinction
    explicit instead of treating lensing outputs as CMB temperature spectra.
    """

    ell_values = tuple(int(value) for value in ells)
    if not ell_values or any(value < 2 for value in ell_values):
        raise ValueError("ells must contain values at or above 2")
    if tuple(sorted(set(ell_values))) != ell_values:
        raise ValueError("ells must be sorted and unique")
    requested = FIXED_LCDM_FULL_REFERENCE_SPECTRA
    ell_array = numpy.asarray(ell_values, dtype=int)
    lmax = int(ell_array.max())
    params = _make_camb_params(FIXED_LCDM_REFERENCE_CONTRACT, lmax=lmax)
    results = camb.get_results(params)
    unlensed_d = numpy.asarray(
        results.get_unlensed_scalar_cls(lmax=lmax, CMB_unit="muK"),
        dtype=float,
    )
    unlensed_c = numpy.asarray(
        results.get_unlensed_scalar_cls(
            lmax=lmax, CMB_unit="muK", raw_cl=True
        ),
        dtype=float,
    )
    lensed_d = numpy.asarray(
        results.get_lensed_scalar_cls(lmax=lmax, CMB_unit="muK"),
        dtype=float,
    )
    lensed_c = numpy.asarray(
        results.get_lensed_scalar_cls(lmax=lmax, CMB_unit="muK", raw_cl=True),
        dtype=float,
    )
    lensing_d = numpy.asarray(
        results.get_lens_potential_cls(lmax=lmax, CMB_unit="muK"),
        dtype=float,
    )
    lensing_c = numpy.asarray(
        results.get_lens_potential_cls(lmax=lmax, CMB_unit="muK", raw_cl=True),
        dtype=float,
    )
    spectra: dict[str, dict[str, list[float]]] = {}
    for name in requested:
        if name in _CMB_SPECTRUM_COLUMNS:
            column = _CMB_SPECTRUM_COLUMNS[name]
            d_values = unlensed_d[ell_array, column]
            c_values = unlensed_c[ell_array, column]
        elif name.startswith("lensed_"):
            base_name = name[7:]
            column = _CMB_SPECTRUM_COLUMNS[base_name]
            d_values = lensed_d[ell_array, column]
            c_values = lensed_c[ell_array, column]
        else:
            column = _LENSING_SPECTRUM_COLUMNS[name]
            d_values = lensing_d[ell_array, column]
            c_values = lensing_c[ell_array, column]
        if not numpy.all(numpy.isfinite(d_values)) or not numpy.all(
            numpy.isfinite(c_values)
        ):
            raise ValueError(f"CAMB returned non-finite {name} reference")
        spectra[name] = {
            "C_ell": [float(value) for value in c_values],
            "D_ell": [float(value) for value in d_values],
        }
    fixture: dict[str, Any] = {
        "schema_version": 2,
        "reference_identity": CAMB_REFERENCE_IDENTITY,
        "backend": "camb",
        "model": "LambdaCDM",
        "sector": "scalar",
        "ell_values": ell_values,
        "spectra": spectra,
        "declared_observables": requested,
        "applicability": {
            "scalar": {
                "included": requested,
                "omitted": (),
                "reason": "All scalar CMB and lensing observables requested",
            },
            "vector": {
                "included": (),
                "omitted": requested,
                "reason": "No vector sector is declared by bundled models",
            },
            "tensor": {
                "included": (),
                "omitted": requested,
                "reason": (
                    "Tensor sectors require a separate nonzero-r fixture"
                ),
            },
        },
        "conventions": {
            "cmb": "D_ell = ell*(ell+1)*C_ell/(2*pi)",
            "PP": "D_ell = ell^2*(ell+1)^2*C_ell/(2*pi)",
            "TP_EP": ("D_ell = [ell*(ell+1)]^(3/2)*C_ell/(2*pi)"),
            "temperature_polarization_unit": "microkelvin_squared",
            "lensing_potential_unit": "dimensionless_PP_and_microkelvin_cross",
            "source": "CAMB get_*_cls raw_cl and native D_ell outputs",
        },
        "contract": FIXED_LCDM_REFERENCE_CONTRACT,
        "tolerances": {
            name: float(FIXED_LCDM_FULL_REFERENCE_TOLERANCES[name])
            for name in requested
        },
        "provenance": describe_camb_configuration(),
    }
    fixture["fixture_sha256"] = reference_fixture_sha256(fixture)
    return fixture


def write_lcdm_full_reference_fixture(
    path: str | Path = FULL_LCDM_REFERENCE_FIXTURE_PATH,
) -> Path:
    """Write one canonical full LCDM fixture and return its path."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fixture = build_lcdm_full_reference_fixture()
    destination.write_text(
        json.dumps(_reference_jsonable(fixture), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return destination


def load_lcdm_full_reference_fixture(
    path: str | Path = FULL_LCDM_REFERENCE_FIXTURE_PATH,
) -> dict[str, Any]:
    """Load and integrity-check the tracked full LCDM fixture."""

    source = Path(path)
    fixture = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(fixture, dict):
        raise ValueError("CAMB fixture root must be an object")
    stored_digest = fixture.get("fixture_sha256")
    if not isinstance(stored_digest, str):
        raise ValueError("CAMB fixture is missing fixture_sha256")
    actual_digest = reference_fixture_sha256(
        {
            key: value
            for key, value in fixture.items()
            if key != "fixture_sha256"
        }
    )
    if stored_digest != actual_digest:
        raise ValueError(
            "CAMB fixture digest mismatch: "
            f"{stored_digest} != {actual_digest}"
        )
    return fixture


def compare_lcdm_reference_spectra(
    actual: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    representation: str = "D_ell",
) -> dict[str, dict[str, float]]:
    """Compare aligned fixture spectra without interpolation or fallback."""

    if representation not in {"C_ell", "D_ell"}:
        raise ValueError("representation must be C_ell or D_ell")
    metrics: dict[str, dict[str, float]] = {}
    for name, reference_entry in reference.items():
        if name not in actual:
            raise KeyError(f"Missing spectrum '{name}' for comparison")
        if not isinstance(reference_entry, Mapping):
            raise ValueError(f"Reference spectrum '{name}' is not structured")
        expected = numpy.asarray(reference_entry[representation], dtype=float)
        observed = numpy.asarray(actual[name], dtype=float)
        if expected.shape != observed.shape:
            raise ValueError(
                f"Spectrum '{name}' shape mismatch: "
                f"{observed.shape} != {expected.shape}"
            )
        if not numpy.all(numpy.isfinite(expected)) or not numpy.all(
            numpy.isfinite(observed)
        ):
            raise ValueError(f"Spectrum '{name}' contains non-finite values")
        delta = numpy.abs(observed - expected)
        scale = numpy.maximum(numpy.abs(expected), 1.0e-30)
        metrics[name] = {
            "max_absolute": float(numpy.max(delta, initial=0.0)),
            "max_fractional": float(numpy.max(delta / scale, initial=0.0)),
            "sample_count": float(expected.size),
        }
    return metrics


__all__ = [
    "CAMB_REFERENCE_IDENTITY",
    "FIXED_LCDM_REFERENCE_CONTRACT",
    "FIXED_LCDM_REFERENCE_ELL_VALUES",
    "FIXED_LCDM_REFERENCE_SPECTRA",
    "FIXED_LCDM_REFERENCE_TOLERANCES",
    "FIXED_LCDM_FULL_REFERENCE_SPECTRA",
    "FIXED_LCDM_FULL_REFERENCE_TOLERANCES",
    "FULL_LCDM_REFERENCE_FIXTURE_PATH",
    "build_lcdm_reference_fixture",
    "build_lcdm_full_reference_fixture",
    "compare_lcdm_reference_spectra",
    "compute_camb_background_observables",
    "compute_cmb_spectrum_from_camb_contract",
    "describe_camb_configuration",
    "load_lcdm_full_reference_fixture",
    "reference_fixture_sha256",
    "write_lcdm_full_reference_fixture",
]
