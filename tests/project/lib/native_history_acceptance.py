"""Bounded native scalar diagnosis before any angular-spectrum acceptance.

This explicit component acceptance runs the ordinary declared executor up to
its real hierarchy/source boundary. It stops before angular projection and
never supplies reference opacity, states, amplitudes, or metric corrections.
"""

import copy
from dataclasses import fields, replace
from unittest import mock

import numpy

from copernican.lib.likelihoods.cmb.runtime import execution
from tests.copernican.lib.likelihoods.cmb.test_cmb import (
    _prepare_declared_contract,
)
from tests.copernican.lib.test_model_adapter import DeclaredLCDMModelTestCase
from tests.project.lib import camb_reference, scientific_acceptance


class _ComponentBoundary(BaseException):
    """Stop after the ordinary evolution owner, before angular work."""


def measure_native_histories(filename, *, refinement=1):
    """Return independently matched native histories over eta 10--1100 Mpc.

    Truncating this test's physical time domain does not thin its mesh or
    change equations. Full late-time/angular certification remains separate.
    """

    plugin = DeclaredLCDMModelTestCase._build_plugin(filename)
    raw = plugin.get_cmb_contract(plugin.INITIAL_GUESSES)
    raw["perturbations"] = copy.deepcopy(plugin.CMB_PERTURBATION_CONTRACT)
    raw["numerical"] = {
        "ell_min": 2,
        "ell_max": 20,
        "k_min": 1e-4,
        "k_max": 0.04,
        "k_sample_count": 2,
        "eta_sample_count": 1024 * refinement,
        "evolution_eta_sample_count": 2048 * refinement,
        "evolution_phase_step": 0.05 / refinement,
        "source_grid_multiplier": 1,
        "photon_hierarchy_l_max": 24 * refinement,
        "photon_polarization_hierarchy_l_max": 24 * refinement,
        "neutrino_hierarchy_l_max": 32 * refinement,
        "massive_neutrino_hierarchy_l_max": 32 * refinement,
        "initial_redshift": 1e6,
        "a_min": 1e-8,
        "tight_coupling_ratio": 2000,
        "ode_rtol": 1e-7,
        "ode_atol": 1e-10,
    }
    raw["perturbations"]["numerics"] = dict(raw["numerical"])
    contract = _prepare_declared_contract(raw)
    reference = scientific_acceptance.completion_reference_contract(contract)
    reference_background = camb_reference.camb.get_background(
        camb_reference._make_camb_params(reference)
    )
    eta = numpy.unique(
        numpy.r_[
            numpy.geomspace(10.0, 200.0, 15),
            numpy.linspace(200.0, 1100.0, 91),
        ]
    )
    names = (
        "H",
        "etak",
        "Weyl",
        "v_newtonian_cdm",
        "v_newtonian_baryon",
        "delta_cdm",
        "delta_baryon",
        "delta_photon",
        "delta_neutrino",
        "delta_nu",
        "pi_photon",
        "E_2",
        "E_source",
    )
    density_name = (
        "delta_nu_massive"
        if reference_background.Params.num_nu_massive > 0
        else "delta_nu"
    )
    report = {
        "model": filename,
        "eta": eta.tolist(),
        "cases": {},
        "active_neutrino_density": density_name,
        "numerical": raw["numerical"],
        "reference_contract": reference,
        "momentum_grids": {
            name: dict(definition)
            for name, definition in contract["numerical"]
            .get("momentum_grids", {})
            .items()
        },
    }
    original_build = execution.build_declared_mode_evolution

    def collect(inputs):
        """Use the real executor with a declared test time-domain boundary."""

        mesh = inputs.background.eta_grid
        selected = mesh <= 1250.0
        bounded_background = replace(
            inputs.background,
            **{
                field.name: getattr(inputs.background, field.name)[selected]
                for field in fields(inputs.background)
                if isinstance(
                    getattr(inputs.background, field.name), numpy.ndarray
                )
                and getattr(inputs.background, field.name).shape == mesh.shape
            },
        )
        replacements = {
            "background": bounded_background,
            "shared_generated_mode_grids": None,
        }
        for prefix in ("active", "source"):
            grid = getattr(inputs, prefix + "_grids")["eta"]
            for suffix in (
                "_grids",
                "_declared_background_histories",
                "_coordinate_rate_histories",
            ):
                replacements[prefix + suffix] = {
                    key: numpy.asarray(value)[grid <= 1250.0]
                    for key, value in getattr(inputs, prefix + suffix).items()
                }
        runtime = original_build(replace(inputs, **replacements))
        report["state_count"] = len(inputs.runtime_spec.state_slots)
        report["continuous_collision_solver"] = bool(
            inputs.runtime_envelope.get("continuous_collision_solver", False)
        )
        report["source_history_reconstruction"] = bool(
            inputs.source_history_reconstruction_enabled
        )
        for k_value in (1e-4, 0.02):
            history = {}
            runtime.evolve_declared_mode(
                k_value,
                history_sink=history,
                collect_diagnostics=False,
            )
            native = camb_reference.compute_camb_scalar_time_evolution(
                reference,
                k_value,
                eta,
                variables=names,
                accuracy_boost=4,
            )
            expected = camb_reference.camb_histories_in_newtonian_gauge(
                native,
                k_value,
                curvature_sign=-1,
            )
            if reference_background.Params.num_nu_massless == 0:
                expected.pop("delta_nu")
            if reference_background.Params.num_nu_massive > 0:
                # CAMB supplies a^4*rho_nu. Its logarithmic slope determines
                # w_nu by energy conservation, independently of CCMBS q data.
                scale_factor = 1.0 / (
                    1.0 + reference_background.redshift_at_conformal_time(eta)
                )
                step = 1e-3
                densities = [
                    reference_background.get_background_densities(
                        scale_factor * numpy.exp(sign * step), vars=["nu"]
                    )["nu"]
                    for sign in (-1, 1)
                ]
                equation_of_state = (
                    1.0 - numpy.log(densities[1] / densities[0]) / (2 * step)
                ) / 3.0
                shift = (
                    numpy.square(native["H"] / k_value)
                    * native["v_newtonian_cdm"]
                )
                expected["delta_nu_massive"] = -(
                    native["delta_nu"] + 3 * (1 + equation_of_state) * shift
                )
            actual = {
                name: numpy.interp(
                    eta,
                    history["evolution_eta"],
                    values,
                )
                for name, values in history["native_context"].items()
                if name in expected
            }
            chi = reference_background.conformal_time(0) - eta
            expected["polarization_source"] = -native[
                "E_source"
            ] * numpy.square(k_value * chi)
            expected["lensing_potential"] = expected["Phi"] + expected["Psi"]
            redshift = reference_background.redshift_at_conformal_time(eta)
            thermal = reference_background.get_background_redshift_evolution(
                redshift,
                vars=["visibility", "opacity"],
            )
            visibility = thermal["visibility"]
            polarization = expected["theta_gamma2"] + 6 * expected["e_gamma2"]
            expected["temperature_monopole"] = visibility * (
                expected["theta_gamma0"] + expected["Psi"]
            )
            expected["temperature_doppler"] = (
                visibility * expected["theta_b"] / k_value
            )
            expected["temperature_quadrupole"] = (
                0.25 * visibility * polarization
            )
            # The Weyl combination needs no gauge-shift differentiation.
            offset = 0.5
            derivative_mesh = numpy.sort(numpy.r_[eta - offset, eta + offset])
            derivative_native = (
                camb_reference.compute_camb_scalar_time_evolution(
                    reference,
                    k_value,
                    derivative_mesh,
                    variables=("Weyl",),
                    accuracy_boost=4,
                )["Weyl"]
            )
            derivative = (
                numpy.interp(eta + offset, derivative_mesh, derivative_native)
                - numpy.interp(
                    eta - offset, derivative_mesh, derivative_native
                )
            ) / (2 * offset)
            expected["temperature_isw"] = (
                -2 * derivative / k_value**2 * visibility / thermal["opacity"]
            )
            source_names = (
                "polarization_source",
                "lensing_potential",
                "temperature_monopole",
                "temperature_doppler",
                "temperature_quadrupole",
                "temperature_isw",
            )
            for name in source_names:
                actual[name] = numpy.interp(
                    eta,
                    history["source_eta"],
                    history["source_arrays"][name],
                )
            errors = {
                name: float(numpy.max(numpy.abs(actual[name] - values)))
                / max(float(numpy.max(numpy.abs(values))), 1e-30)
                for name, values in expected.items()
            }
            report["cases"][str(k_value)] = {
                "errors": errors,
                "actual": {
                    key: values.tolist() for key, values in actual.items()
                },
                "reference": {
                    key: values.tolist() for key, values in expected.items()
                },
            }
        raise _ComponentBoundary()

    try:
        with mock.patch.object(
            execution,
            "build_declared_mode_evolution",
            side_effect=collect,
        ):
            execution._compute_custom_cmb_spectrum_data(
                contract,
                numpy.asarray((20,)),
                requested_spectra=("TT", "EE", "PP"),
            )
    except _ComponentBoundary:
        return report
    raise AssertionError("Native component boundary was not reached")


def assess_native_histories(report, *, bound=0.02):
    """Check complete, finite histories against the component bound."""

    if not numpy.isfinite(bound) or not 0.0 < bound <= 0.02:
        raise ValueError("Native component bounds must be positive and <=2%")
    required = {
        "Phi",
        "Psi",
        "theta_gamma0",
        "delta_b",
        "delta_c",
        "theta_b",
        "theta_c",
        "theta_gamma2",
        "e_gamma2",
        "polarization_source",
        "lensing_potential",
        "temperature_monopole",
        "temperature_doppler",
        "temperature_quadrupole",
        "temperature_isw",
    }
    density_name = report.get("active_neutrino_density")
    if density_name not in {"delta_nu", "delta_nu_massive"}:
        return False
    required.add(density_name)
    eta = numpy.asarray(report.get("eta", ()), dtype=float)
    if (
        eta.ndim != 1
        or eta.size < 2
        or not numpy.all(numpy.isfinite(eta))
        or numpy.any(numpy.diff(eta) <= 0.0)
    ):
        return False
    cases = report.get("cases", {})
    if set(cases) != {"0.0001", "0.02"}:
        return False
    for case in cases.values():
        actual, reference = case.get("actual", {}), case.get("reference", {})
        if not required.issubset(actual) or not required.issubset(reference):
            return False
        for name, expected in reference.items():
            values = numpy.asarray(actual.get(name, ()), dtype=float)
            expected = numpy.asarray(expected, dtype=float)
            if (
                values.shape != eta.shape
                or expected.shape != eta.shape
                or not numpy.all(numpy.isfinite(values))
                or not numpy.all(numpy.isfinite(expected))
            ):
                return False
            scale = max(float(numpy.max(numpy.abs(expected))), 1e-30)
            if numpy.max(numpy.abs(values - expected)) > bound * scale:
                return False
    return True


def main(argv=None):
    """Retain the bounded native verdict without launching angular work."""

    import argparse
    import hashlib
    import json
    import time
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--refine", action="store_true")
    args = parser.parse_args(argv)
    start = time.monotonic()
    runner_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    source_identity = (
        scientific_acceptance.diagnostics.cmb_completion_source_identity()
    )
    reports = [
        measure_native_histories(filename)
        for filename in (
            "model_lcdm.yml",
            "model_ref_planck2018.yml",
        )
    ]
    accepted = all(assess_native_histories(report) for report in reports)
    refined = []
    refinement_accepted = None
    if args.refine and accepted:
        refined = [
            measure_native_histories(report["model"], refinement=2)
            for report in reports
        ]
        refinement_accepted = all(
            assess_native_histories(fine)
            and assess_native_refinement(coarse, fine)
            for coarse, fine in zip(reports, refined)
        )
        accepted = accepted and refinement_accepted
    unchanged_sources = (
        source_identity
        == scientific_acceptance.diagnostics.cmb_completion_source_identity()
        and runner_sha256
        == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    )
    accepted = bool(accepted and unchanged_sources)
    payload = {
        "accepted": accepted,
        "scientific_acceptance": False,
        "scope": "native_components_before_angular_projection",
        "elapsed_seconds": time.monotonic() - start,
        "models": reports,
        "refined_models": refined,
        "refinement_accepted": refinement_accepted,
        "unchanged_sources": unchanged_sources,
        "source_identity": source_identity,
        "component_runner_sha256": runner_sha256,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "accepted": accepted,
                "path": str(args.output),
                "scope": payload["scope"],
            }
        )
    )
    return 0 if accepted else 1


def assess_native_refinement(coarse, fine):
    """Require the same physical point and <=0.5% measured component change."""

    for name in (
        "model",
        "reference_contract",
        "active_neutrino_density",
        "eta",
    ):
        if coarse.get(name) != fine.get(name):
            return False
    if not assess_native_histories(coarse) or not assess_native_histories(
        fine
    ):
        return False
    for key in ("0.0001", "0.02"):
        lower = coarse["cases"][key]["actual"]
        upper = fine["cases"][key]["actual"]
        if set(lower) != set(upper):
            return False
        for name in upper:
            values = numpy.asarray(upper[name], dtype=float)
            scale = max(float(numpy.max(numpy.abs(values))), 1e-30)
            if (
                numpy.max(numpy.abs(numpy.asarray(lower[name]) - values))
                > 0.005 * scale
            ):
                return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
