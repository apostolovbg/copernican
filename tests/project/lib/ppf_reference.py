"""Generate independent stock-CAMB PPF history samples for focused tests.

Run explicitly with ``python -m tests.project.lib.ppf_reference``. CAMB's
symbolic custom-source compiler needs a Fortran compiler during generation;
reading the frozen fixture does not. No experimental CAMB source is used.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import camb
import camb.symbolic as symbolic
import numpy
import scipy


def build_reference():
    """Sample stock CAMB before and after a phantom-divide crossing."""
    other_velocity = (
        symbolic.q - symbolic.rho_de * (1 + symbolic.w_de) * symbolic.v_de
    ) / (symbolic.rho + symbolic.P - symbolic.rho_de * (1 + symbolic.w_de))
    density_ratio = symbolic.rho_de / (
        symbolic.rho + symbolic.P - symbolic.rho_de * (1 + symbolic.w_de)
    )
    fields = [
        "a",
        symbolic.Delta_de,
        (1 + symbolic.w_de) * symbolic.v_de,
        other_velocity,
        symbolic.w_de,
        symbolic.H,
        density_ratio,
    ]
    expansion_today = 70.0
    gravity = (expansion_today / 299792.458) ** 2
    waves = numpy.asarray((0.0001, 0.001))
    redshifts = (0.25, 0.9, 1.1, 3.0)
    step = 0.0001
    rows = []
    for equation_today, equation_running in ((-0.9, 0.0), (-1.1, 0.2)):
        params = camb.CAMBparams()
        params.set_cosmology(
            H0=expansion_today,
            ombh2=0.022,
            omch2=0.12,
            mnu=0.0,
            tau=0.06,
        )
        params.set_dark_energy(
            w=equation_today,
            wa=equation_running,
            cs2=1.0,
            dark_energy_model="ppf",
        )
        params.set_for_lmax(20)
        result = camb.get_results(params)
        centers = numpy.asarray(
            result.conformal_time(numpy.asarray(redshifts))
        )
        times = (
            centers[:, None] * numpy.asarray((1 - step, 1, 1 + step))
        ).ravel()
        values = result.get_time_evolution(
            waves, times, fields, frame="Newtonian"
        ).reshape(len(waves), len(redshifts), 3, len(fields))
        for wave_index, wave in enumerate(waves):
            for redshift_index, redshift in enumerate(redshifts):
                samples = values[wave_index, redshift_index]
                (
                    scale_factor,
                    delta,
                    velocity,
                    other_velocity_value,
                    eos,
                    hubble,
                    ratio,
                ) = samples.T
                sample_z = 1 / scale_factor - 1
                physical_hubble = numpy.asarray(
                    result.hubble_parameter(sample_z)
                )
                fraction = numpy.asarray(
                    [
                        result.get_Omega("de", float(value))
                        for value in sample_z
                    ]
                )
                density = (
                    scale_factor
                    * scale_factor
                    * fraction
                    * (physical_hubble / expansion_today) ** 2
                )
                gamma = (
                    -1.5
                    * gravity
                    * density
                    * (delta + 3 * hubble * velocity / wave)
                    / wave**2
                )
                dt = times.reshape(-1, 3)[redshift_index]
                derivative = (gamma[2] - gamma[0]) / (dt[2] - dt[0])
                enthalpy = density[1] / ratio[1]
                rows.append(
                    {
                        "w0": equation_today,
                        "wa": equation_running,
                        "redshift": redshift,
                        "wave": float(wave),
                        "a": float(scale_factor[1]),
                        "hubble": float(hubble[1]),
                        "gravity": gravity,
                        "eos": float(eos[1]),
                        "density_background": float(density[1]),
                        "other_enthalpy": float(enthalpy),
                        "other_momentum": float(
                            wave * other_velocity_value[1] * enthalpy
                        ),
                        "gamma": float(gamma[1]),
                        "gamma_derivative": float(derivative),
                        "density_source": float(density[1] * delta[1]),
                        "momentum_source": float(
                            density[1] * wave * velocity[1]
                        ),
                    }
                )
    library = Path(camb.__file__).resolve().parent / "camblib.so"
    return {
        "schema_version": 1,
        "reference": "stock CAMB PPF; conformal Newtonian frame",
        "camb_version": camb.__version__,
        "camb_library_sha256": hashlib.sha256(
            library.read_bytes()
        ).hexdigest(),
        "numpy_version": numpy.__version__,
        "scipy_version": scipy.__version__,
        "fractional_derivative_step": step,
        "notes": "Gamma is reconstructed from the CAMB rest density.",
        "rows": rows,
    }


def main():
    """Write the independently generated reference beside other fixtures."""
    old_tempdir = tempfile.tempdir
    try:
        with tempfile.TemporaryDirectory(
            prefix="ccmbs-ppf-reference-"
        ) as work:
            tempfile.tempdir = work
            fixture = build_reference()
    finally:
        tempfile.tempdir = old_tempdir
    target = (
        Path(__file__).resolve().parents[1] / "fixtures/ccmbs_ppf_camb.json"
    )
    lines = []
    for line in json.dumps(fixture, indent=2, sort_keys=True).splitlines():
        if len(line) > 79 and '": ' in line:
            key, value = line.split('": ', 1)
            indentation = len(line) - len(line.lstrip())
            lines.extend((key + '":', " " * (indentation + 2) + value))
        else:
            lines.append(line)
    if any(len(line) > 79 for line in lines):
        raise ValueError("PPF reference cannot be formatted within 79 columns")
    target.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
