"""Focused tests for the independent CAMB reference helper."""

import gzip
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy

from copernican.lib.likelihoods.cmb import diagnostics as cmb_diagnostics
from tests.project.lib import camb_reference


class CambReferenceModuleTestCase(unittest.TestCase):
    """Exercise the independent CAMB reference helper surface."""

    def test_newtonian_history_mapping_preserves_entropy_and_curvature(self):
        """Gauge changes retain entropy and the fixed seed convention."""

        k = 1.0e-4
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
            "pi_photon",
            "E_2",
        )
        native = camb_reference.compute_camb_scalar_time_evolution(
            camb_reference.FIXED_LCDM_REFERENCE_CONTRACT,
            k,
            (1.0, 2.0, 5.0),
            variables=names,
        )
        mapped = camb_reference.camb_histories_in_newtonian_gauge(
            native, k, curvature_sign=-1
        )
        numpy.testing.assert_allclose(
            mapped["delta_b"] - 3 * mapped["theta_gamma0"],
            -(native["delta_baryon"] - 0.75 * native["delta_photon"]),
            rtol=0.0,
            atol=2e-15,
        )
        numpy.testing.assert_allclose(
            mapped["Phi"] - mapped["theta_gamma0"],
            1.0,
            rtol=1e-6,
        )
        numpy.testing.assert_allclose(
            mapped["Phi"] + mapped["Psi"],
            -2 * native["Weyl"] / k**2,
            rtol=1e-14,
        )
        numpy.testing.assert_array_equal(
            mapped["theta_gamma2"], -native["pi_photon"] / 4
        )
        with self.assertRaisesRegex(ValueError, "grids must match"):
            camb_reference.camb_histories_in_newtonian_gauge(
                {**native, "H": native["H"][:1]}, k
            )
        with self.assertRaisesRegex(ValueError, "sign"):
            camb_reference.camb_histories_in_newtonian_gauge(
                native, k, curvature_sign=0
            )

    def test_bundled_reionization_and_drag_match_independent_backgrounds(self):
        """Both declarations match full late ionization and drag histories."""

        from copernican.lib import (
            model_adapter,
            model_coder,
            model_spec_validator,
        )
        from copernican.lib.likelihoods.cmb.runtime import background, cache

        self.addCleanup(cache.clear_cmb_caches)
        model_root = Path(__file__).resolve().parents[3] / "copernican/models"
        for filename in ("model_lcdm.yml", "model_ref_planck2018.yml"):
            with self.subTest(model=filename):
                with tempfile.TemporaryDirectory() as directory:
                    validated = model_spec_validator.validate_and_cache_model(
                        model_root / filename, directory
                    )
                    functions, model_data = model_coder.generate_callables(
                        validated
                    )
                plugin = model_adapter.build_plugin(model_data, functions)
                contract = plugin.get_cmb_declared_runtime(
                    plugin.INITIAL_GUESSES
                )
                physical = background._resolve_custom_cmb_physical_parameters(
                    contract
                )
                summary = (
                    background._summarize_declared_background_manifest_summary(
                        contract
                    )["recombination_runtime"]
                )
                self.assertFalse(summary["reionization_ode"])
                self.assertEqual(
                    summary["reionization_history"],
                    "declared_electron_fraction",
                )
                actual = background._build_custom_cmb_background_impl(
                    contract,
                    physical,
                    background._CustomCMBNumerics(eta_sample_count=2048),
                )
                parameters = dict(contract["param_map"])
                parameters.setdefault("mnu", 0.0)
                parameters.setdefault("num_massive_neutrinos", 0)
                parameters["TCMB"] = physical.Tcmb_K
                reference = camb_reference.camb.get_background(
                    camb_reference._make_camb_params(
                        {
                            "backend": "camb",
                            "param_map": parameters,
                            "grids": {},
                            "values": {},
                            "calls": [],
                        }
                    )
                )
                redshift = numpy.linspace(0.0, 25.0, 251)
                thermal = reference.get_background_redshift_evolution(
                    redshift,
                    vars=("x_e", "opacity"),
                )
                eta = actual.eta_of_a(1.0 / (1.0 + redshift))
                electrons = actual.x_e_of_eta(eta)
                numpy.testing.assert_allclose(
                    electrons,
                    thermal["x_e"],
                    rtol=0.0,
                    atol=0.01,
                    err_msg=filename,
                )
                opacity = -actual.tau_dot_of_eta(eta)
                ionized = thermal["x_e"] > 0.02
                numpy.testing.assert_allclose(
                    opacity[ionized],
                    thermal["opacity"][ionized],
                    rtol=0.01,
                    atol=0.0,
                    err_msg=filename,
                )
                self.assertAlmostEqual(actual.reionization_tau, 0.054, 10)
                self.assertGreater(
                    actual.resolution_evidence[
                        "reionization_calibration_value"
                    ],
                    0.0,
                )
                # These are native histories: no external opacity or metric
                # is supplied to the engine. Bounds precede angular parity.
                redshift_early = numpy.geomspace(100.0, 5000.0, 151)
                scale = 1.0 / (1.0 + redshift_early)
                early = reference.get_background_redshift_evolution(
                    redshift_early,
                    vars=("x_e", "opacity", "visibility", "T_b"),
                )
                eta_early = actual.eta_of_a(scale)
                expansion = background._resolve_declared_background_context(
                    contract, a_values=scale, z_values=redshift_early
                )["H"]
                numpy.testing.assert_allclose(
                    expansion,
                    reference.hubble_parameter(redshift_early),
                    rtol=2e-4,
                    atol=0.0,
                )
                numpy.testing.assert_allclose(
                    eta_early,
                    reference.conformal_time(redshift_early),
                    rtol=5e-4,
                    atol=0.0,
                )
                numpy.testing.assert_allclose(
                    actual.x_e_of_eta(eta_early),
                    early["x_e"],
                    rtol=1e-3,
                    atol=1e-7,
                )
                numpy.testing.assert_allclose(
                    actual.matter_temperature_of_eta(eta_early),
                    early["T_b"],
                    rtol=5e-3,
                    atol=0.0,
                )
                native_visibility = actual.visibility_of_eta(eta_early)
                self.assertLess(
                    float(
                        numpy.max(
                            numpy.abs(native_visibility - early["visibility"])
                        )
                    )
                    / float(numpy.max(early["visibility"])),
                    0.005,
                )
                finer = background._build_custom_cmb_background_impl(
                    contract,
                    physical,
                    background._CustomCMBNumerics(eta_sample_count=4096),
                )
                finer_eta = finer.eta_of_a(scale)
                numpy.testing.assert_allclose(
                    actual.x_e_of_eta(eta_early),
                    finer.x_e_of_eta(finer_eta),
                    rtol=5e-4,
                    atol=1e-7,
                )
                numpy.testing.assert_allclose(
                    actual.matter_temperature_of_eta(eta_early),
                    finer.matter_temperature_of_eta(finer_eta),
                    rtol=5e-4,
                    atol=0.0,
                )
                reference = reference.get_derived_params()
                self.assertLess(
                    abs(actual.sound_horizon_mpc / reference["rstar"] - 1),
                    0.002,
                )
                self.assertGreater(
                    actual.drag_sound_horizon_mpc, actual.sound_horizon_mpc
                )
                self.assertLess(actual.drag_redshift, actual.z_rec)
                self.assertLess(
                    abs(
                        actual.drag_sound_horizon_mpc / reference["rdrag"] - 1
                    ),
                    0.002,
                )
                self.assertLess(
                    abs(actual.drag_redshift / reference["zdrag"] - 1),
                    0.005,
                )
                self.assertAlmostEqual(
                    actual.resolution_evidence[
                        "drag_remaining_depth_at_transition"
                    ],
                    1.0,
                    places=10,
                )

    def test_camb_reference_uses_explicit_nonstandard_temperature(self):
        """An independent reference cannot silently revert to default T."""

        contract = dict(camb_reference.FIXED_LCDM_REFERENCE_CONTRACT)
        contract["param_map"] = dict(contract["param_map"], TCMB=3.0)
        parameters = camb_reference._make_camb_params(contract)
        self.assertEqual(parameters.TCMB, 3.0)

    def test_compact_histories_keep_native_gauges_and_adiabatic_entropy(self):
        """The frozen reference checks invariant initial ratios per mode."""

        fixture = camb_reference.load_lcdm_history_reference_fixture()
        self.assertEqual(fixture["reference_identity"], "camb:1.6.0")
        self.assertEqual(
            fixture["variables"]["delta_photon"]["gauge"],
            "CDM_comoving_synchronous",
        )
        self.assertEqual(fixture["resolved_parameters"]["num_nu_massive"], 0)
        histories = fixture["histories"]
        photons = numpy.asarray(histories["delta_photon"])
        baryons = numpy.asarray(histories["delta_baryon"])
        numpy.testing.assert_allclose(
            baryons[:, :4], 0.75 * photons[:, :4], rtol=1e-6, atol=1e-13
        )
        self.assertEqual(photons.shape, (2, 10))
        self.assertGreater(float(numpy.ptp(photons[-1])), 0.01)
        self.assertTrue(numpy.all(numpy.diff(histories["a"], axis=1) > 0))
        scale_factor = 1 / (1 + numpy.asarray(fixture["redshift"]))
        numpy.testing.assert_allclose(
            histories["a"][0], scale_factor, rtol=1e-7
        )
        numpy.testing.assert_allclose(
            histories["H"][0],
            scale_factor * numpy.asarray(fixture["H_km_s_Mpc"]) / 299792.458,
            rtol=1e-7,
        )
        self.assertIn(
            "theta_gamma2=pi_photon/4", fixture["mapping"]["polarization"]
        )

    def test_selected_modes_match_frozen_histories_without_angular_spectra(
        self,
    ):
        """Compare exact k/eta points with independent thermal/mode data."""

        fixture = camb_reference.load_lcdm_history_reference_fixture()
        indices = numpy.asarray((0, 4, 7, 9))
        eta = numpy.asarray(fixture["eta_mpc"])[indices]
        with mock.patch.object(
            camb_reference.camb,
            "get_results",
            side_effect=AssertionError(
                "No angular spectra in recurring history checks"
            ),
        ):
            for index, k_value in enumerate(fixture["k_mpc_inv"]):
                actual = camb_reference.compute_camb_scalar_time_evolution(
                    fixture["contract"],
                    k_value,
                    eta,
                    variables=("delta_photon", "pi_photon", "E_2"),
                )
                for name, values in actual.items():
                    with self.subTest(k=k_value, variable=name):
                        self._assert_rebuilt_spectrum_matches(
                            values,
                            numpy.asarray(fixture["histories"][name])[
                                index, indices
                            ],
                        )

    def test_compact_history_loader_rejects_corrupted_data(self):
        """Stored history integrity stays exact across numerical rebuilds."""

        path = camb_reference.LCDM_HISTORY_REFERENCE_FIXTURE_PATH
        content = path.read_text()
        corrupted = content.replace(
            '"l_accuracy_boost": 4', '"l_accuracy_boost": 5'
        )
        self.assertNotEqual(corrupted, content)
        with mock.patch.object(Path, "read_text", return_value=corrupted):
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                camb_reference.load_lcdm_history_reference_fixture()

    def test_history_wave_controls_expose_sign_phase_and_spike_errors(self):
        """Diagnostic rescaling cannot turn a corrupted wave into parity."""

        eta = numpy.linspace(0, 30, 301)
        reference = numpy.sin(eta)
        spike = reference.copy()
        spike[157] += 20
        for actual in (-reference, numpy.sin(eta + 0.8), spike):
            report = camb_reference.compare_scalar_history_waves(
                {"wave": actual}, {"wave": reference}, eta
            )
            metric = report["variables"]["wave"]
            self.assertTrue(
                metric["phase_correlation"] < 0.9
                or metric["normalized_rms_residual"] > 0.1
            )

    def _assert_rebuilt_spectrum_matches(self, actual, expected):
        """Compare compiled CAMB outputs with a strict numerical envelope.

        Different Fortran builds need not reproduce floating-point bytes.
        Retain one-part-per-million relative agreement and a smaller
        peak-scaled absolute floor for cross spectra near zero. These
        bounds are separate from the CCMBS scientific parity tolerances.
        """

        expected = numpy.asarray(expected, dtype=float)
        numpy.testing.assert_allclose(
            actual,
            expected,
            rtol=1.0e-6,
            atol=1.0e-8 * float(numpy.max(numpy.abs(expected))),
        )

    def test_describe_camb_configuration_reports_expected_defaults(self):
        """The reference helper should report its default CAMB settings."""

        configuration = camb_reference.describe_camb_configuration()

        self.assertEqual(
            configuration["reionization_model"], "optical_depth_tau"
        )
        self.assertIn("lmax_padding", configuration)
        self.assertIn("lens_potential_accuracy", configuration)
        self.assertIn("accuracy", configuration)
        self.assertEqual(
            configuration["reference_identity"],
            f"camb:{camb_reference.camb.__version__}",
        )

    def test_fixed_lcdm_resolves_massless_neutrino_physics(self):
        """The fixed reference must not inherit CAMB's massive default."""

        resolved = camb_reference.resolve_camb_parameters(
            camb_reference.FIXED_LCDM_REFERENCE_CONTRACT
        )

        self.assertEqual(resolved["num_nu_massive"], 0)
        self.assertEqual(resolved["num_nu_massless"], 3.0)
        self.assertEqual(resolved["omnuh2"], 0.0)
        self.assertEqual(
            resolved["H0"],
            camb_reference.FIXED_LCDM_REFERENCE_CONTRACT["param_map"]["H0"],
        )

    def test_reference_helper_is_test_owned(self):
        """The CAMB builder should remain outside the production package."""

        helper_path = Path(camb_reference.__file__).resolve()
        self.assertTrue(helper_path.is_relative_to(Path("tests").resolve()))
        self.assertEqual(
            camb_reference.CAMB_REFERENCE_IDENTITY,
            f"camb:{camb_reference.camb.__version__}",
        )

    def test_reference_symbols_are_exposed(self):
        """The test module should expose independent reference entrypoints."""

        self.assertIn(
            "compute_cmb_spectrum_from_camb_contract", camb_reference.__all__
        )
        self.assertIn(
            "compute_camb_background_observables", camb_reference.__all__
        )
        self.assertIn("describe_camb_configuration", camb_reference.__all__)
        self.assertIn("CAMB_REFERENCE_IDENTITY", camb_reference.__all__)
        self.assertIn("build_lcdm_reference_fixture", camb_reference.__all__)
        self.assertIn(
            "build_lcdm_full_reference_fixture", camb_reference.__all__
        )
        self.assertIn(
            "build_camb_full_reference_fixture", camb_reference.__all__
        )
        self.assertIn(
            "build_camb_parity_reference_set", camb_reference.__all__
        )
        self.assertIn(
            "build_fixed_lcdm_camb_parity_row", camb_reference.__all__
        )
        self.assertIn("compare_lcdm_reference_spectra", camb_reference.__all__)
        self.assertIn("compare_scalar_history_waves", camb_reference.__all__)
        self.assertIn(
            "load_lcdm_full_reference_fixture", camb_reference.__all__
        )
        self.assertIn("reference_fixture_sha256", camb_reference.__all__)

    def test_scalar_time_evolution_returns_aligned_finite_histories(self):
        """The independent scalar-history reference preserves grid shape."""

        eta_values = numpy.asarray((10.0, 100.0, 1000.0), dtype=float)
        histories = camb_reference.compute_camb_scalar_time_evolution(
            camb_reference.FIXED_LCDM_REFERENCE_CONTRACT,
            0.02,
            eta_values,
            variables=("delta_photon", "v_photon", "T_source"),
        )

        self.assertEqual(
            set(histories),
            {"delta_photon", "v_photon", "T_source"},
        )
        for values in histories.values():
            self.assertEqual(values.shape, eta_values.shape)
            self.assertTrue(numpy.all(numpy.isfinite(values)))

    def test_scalar_history_wave_comparator_records_phase_evidence(self):
        """Wave comparison retains finite phase and residual diagnostics."""

        eta_values = numpy.linspace(0.0, 40.0, 401)
        reference = {"reference_wave": numpy.sin(eta_values)}
        actual = {"declared_wave": 2.5 * numpy.sin(eta_values + 0.1) + 3.0}
        report = camb_reference.compare_scalar_history_waves(
            actual,
            reference,
            eta_values,
            variable_map={"declared_wave": "reference_wave"},
            k_value=0.02,
        )

        self.assertTrue(report["available"])
        self.assertTrue(report["finite"])
        self.assertTrue(report["phase_coherent"])
        self.assertEqual(report["wave_count"], 1)
        metric = report["variables"]["declared_wave"]
        self.assertGreater(metric["actual_zero_crossings"], 0)
        self.assertGreater(metric["reference_zero_crossings"], 0)
        self.assertGreater(metric["phase_correlation"], 0.9)
        self.assertLess(metric["normalized_rms_residual"], 0.1)
        self.assertEqual(len(report["eta_sha256"]), 64)

    def test_scalar_history_wave_comparator_rejects_unaligned_grids(self):
        """Wave comparison must not interpolate silently between grids."""

        eta_values = numpy.linspace(1.0, 4.0, 16)
        with self.assertRaises(ValueError):
            camb_reference.compare_scalar_history_waves(
                {"actual": numpy.sin(eta_values[:-1])},
                {"reference": numpy.sin(eta_values)},
                eta_values,
                variable_map={"actual": "reference"},
            )

    def test_full_fixture_is_frozen_complete_and_reversible(self):
        """The tracked fixture covers scalar observables in both units."""

        fixture = camb_reference.load_lcdm_full_reference_fixture()
        spectra = fixture["spectra"]
        expected_names = set(camb_reference.FIXED_LCDM_FULL_REFERENCE_SPECTRA)
        self.assertEqual(set(spectra), expected_names)
        self.assertEqual(
            tuple(fixture["declared_observables"]),
            camb_reference.FIXED_LCDM_FULL_REFERENCE_SPECTRA,
        )
        self.assertEqual(
            tuple(fixture["ell_values"]),
            (2, 20, 100, 200, 500, 1000, 1500, 2000),
        )
        self.assertEqual(fixture["resolved_parameters"]["num_nu_massive"], 0)
        self.assertEqual(fixture["resolved_parameters"]["omnuh2"], 0.0)
        self.assertEqual(fixture["applicability"]["scalar"]["omitted"], [])
        self.assertEqual(fixture["applicability"]["vector"]["included"], [])
        self.assertEqual(fixture["applicability"]["tensor"]["included"], [])
        ell = numpy.asarray(fixture["ell_values"], dtype=float)
        for name, values in spectra.items():
            self.assertEqual(set(values), {"C_ell", "D_ell"})
            raw = numpy.asarray(values["C_ell"], dtype=float)
            native = numpy.asarray(values["D_ell"], dtype=float)
            self.assertTrue(numpy.all(numpy.isfinite(raw)))
            self.assertTrue(numpy.all(numpy.isfinite(native)))
            if name == "PP":
                factor = ell**2 * (ell + 1.0) ** 2 / (2.0 * numpy.pi)
            elif name in {"TP", "EP"}:
                factor = (ell * (ell + 1.0)) ** 1.5 / (2.0 * numpy.pi)
            else:
                factor = ell * (ell + 1.0) / (2.0 * numpy.pi)
            numpy.testing.assert_allclose(
                native,
                raw * factor,
                rtol=1.0e-12,
                atol=1.0e-30,
            )

    def test_full_fixture_rejects_content_with_a_stale_digest(self):
        """Numerical rebuild tolerance must not weaken stored integrity."""

        fixture_path = camb_reference.FULL_LCDM_REFERENCE_FIXTURE_PATH
        content = fixture_path.read_text(encoding="utf-8")
        corrupted = content.replace('"LambdaCDM"', '"ChangedModel"', 1)
        self.assertNotEqual(corrupted, content)
        with mock.patch.object(Path, "read_text", return_value=corrupted):
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                camb_reference.load_lcdm_full_reference_fixture()

    def test_rebuild_comparison_rejects_spectrum_drift_and_nonzero_bb(self):
        """The platform envelope still catches physical spectrum changes."""

        for expected in (
            numpy.asarray((1000.0, 250.0, 50.0)),
            numpy.asarray((1.0e-6, -1.0e-10, 1.0e-9)),
            numpy.zeros(3),
        ):
            with self.subTest(expected=expected):
                drifted = expected.copy()
                scale = float(numpy.max(numpy.abs(expected)))
                drifted[0] += 1.0e-3 * scale if scale else 1.0e-30
                with self.assertRaises(AssertionError):
                    self._assert_rebuilt_spectrum_matches(drifted, expected)

    def test_full_comparator_requires_aligned_finite_arrays(self):
        """Comparison reports enforce shape and finite aligned arrays."""

        frozen = camb_reference.load_lcdm_full_reference_fixture()
        actual = {
            name: values["D_ell"] for name, values in frozen["spectra"].items()
        }
        metrics = camb_reference.compare_lcdm_reference_spectra(
            actual,
            frozen["spectra"],
        )
        self.assertEqual(
            set(metrics), set(camb_reference.FIXED_LCDM_FULL_REFERENCE_SPECTRA)
        )
        self.assertTrue(
            all(metric["max_fractional"] == 0.0 for metric in metrics.values())
        )
        with self.assertRaises(ValueError):
            camb_reference.compare_lcdm_reference_spectra(
                {"TT": [1.0]},
                {"TT": frozen["spectra"]["TT"]},
            )

    def test_bounded_reference_report_rejects_changed_raw_surface(self):
        """A changed array cannot pass the raw CAMB parity comparator."""

        fixture = camb_reference.load_lcdm_full_reference_fixture()
        actual = {
            "sector": "scalar",
            "ell_values": fixture["ell_values"],
            "spectra": {
                name: {
                    "C_ell": values["C_ell"],
                    "D_ell": list(values["D_ell"]),
                }
                for name, values in fixture["spectra"].items()
            },
        }
        actual["spectra"]["TT"]["D_ell"][2] *= 1.01
        report = cmb_diagnostics.compare_full_cmb_observable_parity(
            actual,
            fixture,
            refinement={"converged": True},
            relative_tolerances={"TT": 1.0e-6},
            fixture_digest=fixture["fixture_sha256"],
            require_fixture_digest=True,
        )

        self.assertFalse(report["accepted"])
        tt_row = next(
            row for row in report["rows"] if row["row"] == "scalar:TT"
        )
        self.assertFalse(tt_row["metric"]["converged"])
        self.assertIn("raw-array parity tolerance failed", tt_row["issues"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()


class CompletionReferenceWindowTestCase(unittest.TestCase):
    """Freeze window selection independently from the engine residuals."""

    def test_portable_references_preserve_all_frozen_numeric_products(self):
        import json

        root = Path(__file__).resolve().parents[1] / "fixtures"
        contract = json.loads(
            (root / "ccmbs_completion_contract.json").read_text()
        )
        archive = json.loads(
            gzip.decompress(
                (root / "ccmbs_comparable_camb.json.gz").read_bytes()
            )
        )
        required = {
            name: value
            for name, value in contract["cases"].items()
            if value["reference_kind"] == "camb"
        }
        self.assertEqual(set(archive["references"]), set(required))
        self.assertEqual(len(required), 12)
        for name, requirement in required.items():
            reference = archive["references"][name]
            body = {
                key: value
                for key, value in reference.items()
                if key != "fixture_sha256"
            }
            self.assertEqual(
                camb_reference.reference_fixture_sha256(body),
                requirement["reference_sha256"],
            )
            self.assertEqual(reference["ell_values"], requirement["ells"])
            self.assertEqual(
                set(reference["spectra"]), set(requirement["surfaces"])
            )
            self.assertEqual(reference["features"], requirement["features"])
            self.assertEqual(
                reference["resolved_parameters"],
                requirement["resolved_reference_inputs"],
            )
            for surface in reference["spectra"].values():
                for values in surface.values():
                    self.assertEqual(len(values), len(requirement["ells"]))
                    self.assertTrue(numpy.all(numpy.isfinite(values)))

    def test_planck_density_matches_fermi_dirac_and_unmodified_camb(self):
        import json
        import math

        from scipy import constants, special

        path = (
            Path(__file__).resolve().parents[1]
            / "fixtures/ccmbs_completion_contract.json"
        )
        fixture = json.loads(path.read_text())
        inputs = fixture["cases"]["model_ref_planck2018.yml:initial"][
            "resolved_reference_inputs"
        ]
        params = camb_reference.camb.CAMBparams()
        params.set_cosmology(
            H0=inputs["H0"],
            ombh2=inputs["ombh2"],
            omch2=inputs["omch2"],
            mnu=0.06,
            nnu=3.044,
            num_massive_neutrinos=3,
            standard_neutrino_neff=3.044,
            TCMB=2.7255,
        )
        self.assertEqual(params.omnuh2, inputs["omnuh2"])
        temperature = 2.7255 * (4 / 11) ** (1 / 3) * (3.044 / 3) ** 0.25
        number = (
            3
            * special.zeta(3)
            / (2 * math.pi**2)
            * (constants.k * temperature / (constants.hbar * constants.c)) ** 3
        )
        hubble = 100000 / (constants.parsec * 1e6)
        critical = 3 * hubble**2 / (8 * math.pi * constants.G)
        independent = (
            number * 0.06 * constants.electron_volt / constants.c**2 / critical
        )
        self.assertLess(abs(independent / inputs["omnuh2"] - 1), 1e-8)

    def test_reference_refresh_preserves_explicit_windows_and_surfaces(self):
        import copy

        requirement = {
            "ells": [2, 20, 21],
            "surfaces": ["EE", "TT"],
            "features": [
                {
                    "kind": "peak",
                    "surface": "TT",
                    "ell": 20,
                    "lower": 20,
                    "upper": 21,
                }
            ],
            "damping_windows": [[20, 21]],
        }
        dense = {"ell_values": requirement["ells"], "fixture_sha256": "old"}
        with mock.patch.object(
            camb_reference,
            "build_camb_full_reference_fixture",
            return_value=copy.deepcopy(dense),
        ) as build:
            reference = camb_reference.build_completion_camb_reference(
                camb_reference.FIXED_LCDM_REFERENCE_CONTRACT,
                model_name="declared",
                fixed_point="initial",
                requirement=requirement,
            )
        self.assertEqual(build.call_args.args[1], requirement["ells"])
        self.assertEqual(
            set(build.call_args.kwargs["spectra"]),
            set(requirement["surfaces"]),
        )
        self.assertEqual(reference["features"], requirement["features"])
        self.assertEqual(
            reference["damping_windows"], requirement["damping_windows"]
        )
        self.assertNotEqual(reference["fixture_sha256"], "old")

    def test_completion_windows_require_dense_reference_and_cover_features(
        self,
    ):
        import numpy

        from tests.project.lib import camb_reference

        ell = numpy.arange(2, 2501)
        fixture = {
            "ell_values": ell,
            "spectra": {
                "TT": {"D_ell": 2 + numpy.cos(ell / 40)},
                "EE": {"D_ell": 2 - numpy.cos(ell / 40)},
                "TE": {"D_ell": numpy.sin(ell / 40)},
            },
        }
        features = camb_reference.completion_reference_windows(fixture)
        self.assertEqual(len(features), 12)
        self.assertTrue(
            all(row["upper"] - row["lower"] == 16 for row in features)
        )
        fixture["ell_values"] = [2, 20, 100]
        with self.assertRaisesRegex(ValueError, "dense"):
            camb_reference.completion_reference_windows(fixture)

    def test_frozen_completion_references_resolve_all_required_physics_points(
        self,
    ):
        import json
        from pathlib import Path

        path = (
            Path(__file__).resolve().parents[1]
            / "fixtures"
            / "ccmbs_completion_contract.json"
        )
        fixture = json.loads(path.read_text())
        cases = fixture["cases"]
        self.assertEqual(len(cases), 23)
        for label in (
            "model_lcdm.yml:initial",
            "model_lcdm_mnu.yml:mass_zero",
        ):
            self.assertEqual(
                cases[label]["resolved_reference_inputs"]["num_nu_massive"], 0
            )
        for row in cases.values():
            if row["reference_kind"] != "camb":
                continue
            self.assertEqual(len(row["reference_sha256"]), 64)
            self.assertEqual(len(row["surfaces"]), 11)
            self.assertEqual(row["ells"][-1], 2500)
            for feature in row["features"]:
                self.assertTrue(
                    set(
                        range(feature["lower"], feature["upper"] + 1)
                    ).issubset(row["ells"])
                )


if __name__ == "__main__":
    unittest.main()
