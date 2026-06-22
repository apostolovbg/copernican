"""CMB likelihood package."""

from .cmb import (
    CMBLike,
    compute_camb_background_observables,
    compute_cmb_spectrum,
    compute_cmb_spectrum_cached,
    compute_cmb_spectrum_from_contract,
    compute_cmb_spectrum_from_legacy_params_for_tests,
    describe_camb_configuration,
)

__all__ = [
    "CMBLike",
    "compute_camb_background_observables",
    "compute_cmb_spectrum",
    "compute_cmb_spectrum_cached",
    "compute_cmb_spectrum_from_contract",
    "compute_cmb_spectrum_from_legacy_params_for_tests",
    "describe_camb_configuration",
]
