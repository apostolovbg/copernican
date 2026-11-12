"""Tests for the explicit scientific acceptance handoff runner."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tests.project.lib import scientific_acceptance


class ScientificAcceptanceHandoffTestCase(unittest.TestCase):
    """Keep the handoff wiring durable without running the scientific solve."""

    def test_reference_preflight_distinguishes_physical_change_from_tamper(
        self,
    ):
        """Retain differences and reject altered arrays."""
        from copernican.lib.likelihoods.cmb.errors import ContractError

        camb = scientific_acceptance.camb_reference
        contract = {"param_map": {"omnuh2": 0.001}, "calls": []}
        body = {
            "contract": contract,
            "reference_identity": camb.CAMB_REFERENCE_IDENTITY,
            "spectra": {"TT": [1.0]},
        }
        digest = camb.reference_fixture_sha256(body)
        reference = {**body, "fixture_sha256": digest}
        verify = scientific_acceptance._validate_completion_reference
        verify(reference, digest, contract, "example:initial")
        with self.assertRaises(ContractError) as mismatch:
            verify(
                reference,
                digest,
                {"param_map": {"omnuh2": 0.002}, "calls": []},
                "example:initial",
            )
        self.assertEqual(
            mismatch.exception.context["changed_parameters"]["omnuh2"],
            {"reference": 0.001, "current": 0.002},
        )
        self.assertIn("physical inputs changed", str(mismatch.exception))
        with self.assertRaisesRegex(ContractError, "integrity"):
            verify(
                {**reference, "spectra": {"TT": [2.0]}},
                digest,
                contract,
                "example:initial",
            )

    def test_experimental_reference_requires_matching_source_identity(self):
        """A version label cannot masquerade as a validated variant."""
        from copernican.lib.likelihoods.cmb.errors import ContractError

        camb = scientific_acceptance.camb_reference
        diagnostics = scientific_acceptance.diagnostics
        sources = {
            name: "a" * 64
            for name in (
                "results.f90",
                "reionization.f90",
                "recfast.f90",
                "cmbmain.f90",
            )
        }
        scripts = {
            name: f"# independent {name}\n"
            for name in (
                "ccmbs_experimental_helper.py",
                "ccmbs-recombination-case.py",
                "ccmbs-reionization-case.py",
                "ccmbs-experimental-reference-case.py",
            )
        }
        sources.update(
            {
                name: hashlib.sha256(text.encode()).hexdigest()
                for name, text in scripts.items()
            }
        )
        source_digest = diagnostics._canonical_sha256(sources)
        identity = camb.experimental_reference_identity(source_digest)
        contract = {"param_map": {"H0": 70.0}}
        body = {
            "contract": contract,
            "reference_identity": identity,
            "experimental_inputs": {
                "source_sha256": sources,
                "source_identity_sha256": source_digest,
                "independent_scripts": scripts,
            },
        }
        digest = camb.reference_fixture_sha256(body)
        reference = {**body, "fixture_sha256": digest}
        scientific_acceptance._validate_completion_reference(
            reference,
            digest,
            contract,
            "example:initial",
            expected_identity=identity,
        )
        changed = copy.deepcopy(body)
        changed["experimental_inputs"]["independent_scripts"][
            "ccmbs_experimental_helper.py"
        ] += "# changed\n"
        changed_digest = camb.reference_fixture_sha256(changed)
        with self.assertRaisesRegex(ContractError, "source identity"):
            scientific_acceptance._validate_completion_reference(
                {**changed, "fixture_sha256": changed_digest},
                changed_digest,
                contract,
                "example:initial",
                expected_identity=identity,
            )
        body["experimental_inputs"]["source_sha256"] = {"wrong": "a" * 64}
        digest = camb.reference_fixture_sha256(body)
        with self.assertRaisesRegex(ContractError, "source identity"):
            scientific_acceptance._validate_completion_reference(
                {**body, "fixture_sha256": digest},
                digest,
                contract,
                "example:initial",
                expected_identity=identity,
            )

    def test_screen_completion_photon_density_uses_reference_background(self):
        """Photon density is measured rather than inferred from temperature."""

        physical = SimpleNamespace(Omega_gamma0=5.5e-5)
        reference = SimpleNamespace(get_Omega=lambda name: 4.4e-5)
        with (
            mock.patch.object(
                scientific_acceptance.background,
                "_resolve_custom_cmb_physical_parameters",
                return_value=physical,
            ),
            mock.patch.object(
                scientific_acceptance, "completion_reference_contract"
            ),
            mock.patch.object(
                scientific_acceptance.camb_reference, "_make_camb_params"
            ),
            mock.patch.object(
                scientific_acceptance.camb_reference.camb,
                "get_background",
                return_value=reference,
            ),
        ):
            screen = scientific_acceptance._screen_completion_photon_density(
                {}
            )
        self.assertAlmostEqual(screen["relative_difference"], 0.25)
        self.assertFalse(screen["certifies_physical_matching"])

    def test_experimental_screens_use_retained_independent_histories(self):
        """A temporary reference must never be silently replaced by stock."""
        import numpy

        redshift = numpy.linspace(0.0, 25.0, 251)
        reference = {
            "reference_identity": "camb:experimental:test",
            "experimental_inputs": {
                "photon_density": 5.5e-5,
                "background_samples": {
                    "redshift": redshift.tolist(),
                    "x_e": numpy.ones_like(redshift).tolist(),
                },
            },
        }
        history = SimpleNamespace(
            eta_of_a=lambda scale: scale,
            x_e_of_eta=lambda eta: numpy.ones_like(eta),
        )
        with (
            mock.patch.object(
                scientific_acceptance.background,
                "_resolve_custom_cmb_physical_parameters",
                return_value=SimpleNamespace(Omega_gamma0=5.5e-5),
            ),
            mock.patch.object(
                scientific_acceptance.background,
                "_build_custom_cmb_background_impl",
                return_value=history,
            ),
            mock.patch.object(
                scientific_acceptance.camb_reference.camb,
                "get_background",
                side_effect=AssertionError("stock CAMB must not be used"),
            ),
        ):
            photon = scientific_acceptance._screen_completion_photon_density(
                {}, reference
            )
            reionization = (
                scientific_acceptance._screen_completion_reionization(
                    {}, reference
                )
            )
        self.assertEqual(photon["relative_difference"], 0.0)
        self.assertEqual(reionization["maximum_relative_difference"], 0.0)
        self.assertFalse(reionization["certifies_physical_matching"])

    def test_experimental_background_screen_rejects_wrong_neutrino_density(
        self,
    ):
        """Check independently retained neutrino histories."""
        import numpy

        from copernican.lib.likelihoods.cmb.errors import ContractError
        from copernican.lib.likelihoods.cmb.runtime import evolution

        redshift = numpy.asarray([0.0, 25.0, 2500.0])
        density = 0.1 * (1.0 + redshift) ** 4
        reference = {
            "reference_identity": "camb:experimental:test",
            "experimental_inputs": {
                "background_samples": {
                    "redshift": redshift.tolist(),
                    "H": [70.0] * 3,
                    "x_e": [1.0] * 3,
                    "T_b": [10.0] * 3,
                    "neutrino_density": density.tolist(),
                }
            },
        }
        native = SimpleNamespace(
            eta_of_a=lambda scale: scale,
            x_e_of_eta=lambda eta: numpy.ones_like(eta),
            matter_temperature_of_eta=lambda eta: numpy.full_like(eta, 10.0),
        )
        with (
            mock.patch.object(
                scientific_acceptance.background,
                "_resolve_custom_cmb_physical_parameters",
            ),
            mock.patch.object(
                scientific_acceptance.background,
                "_resolve_declared_background_context",
                return_value={"H": numpy.full(3, 70.0)},
            ),
            mock.patch.object(
                scientific_acceptance.background,
                "_build_custom_cmb_background_impl",
                return_value=native,
            ),
            mock.patch.object(
                evolution,
                "_declared_momentum_grid_context",
                return_value={"massive_neutrino_density_fraction": density},
            ),
        ):
            runtime = {"perturbation_data": object()}
            screen = scientific_acceptance._screen_experimental_background(
                runtime, reference
            )
            self.assertEqual(screen["maximum_relative_differences"]["H"], 0.0)
            reference["experimental_inputs"]["background_samples"][
                "neutrino_density"
            ] = (density * 2).tolist()
            with self.assertRaises(ContractError):
                scientific_acceptance._screen_experimental_background(
                    runtime, reference
                )

    def test_screen_completion_reionization_uses_both_resolutions(self):
        """The screen retains samples and does not claim physical proof."""

        import numpy

        background = scientific_acceptance.background
        histories = iter((0.9, 0.91))

        def make_background(*args, **kwargs):
            value = next(histories)
            return SimpleNamespace(
                eta_of_a=lambda scale_factor: scale_factor,
                x_e_of_eta=lambda eta: numpy.full_like(eta, value),
            )

        reference = SimpleNamespace(
            get_background_redshift_evolution=lambda z, vars: {
                "x_e": numpy.ones_like(z)
            }
        )
        with (
            mock.patch.object(
                background, "_resolve_custom_cmb_physical_parameters"
            ),
            mock.patch.object(
                background,
                "_build_custom_cmb_background_impl",
                side_effect=make_background,
            ),
            mock.patch.object(
                scientific_acceptance, "completion_reference_contract"
            ),
            mock.patch.object(
                scientific_acceptance.camb_reference, "_make_camb_params"
            ),
            mock.patch.object(
                scientific_acceptance.camb_reference.camb,
                "get_background",
                return_value=reference,
            ),
        ):
            screen = scientific_acceptance._screen_completion_reionization({})
        self.assertAlmostEqual(screen["maximum_relative_difference"], 0.09)
        self.assertAlmostEqual(
            screen["background_resolution_difference"], 0.01
        )
        self.assertEqual(len(screen["redshift"]), 251)
        self.assertEqual(screen["fine_x_e"][0], 0.91)
        self.assertFalse(screen["certifies_physical_matching"])

    def test_matching_quadrature_requires_the_actual_projected_transfers(self):
        """An equal integral from another source cannot replace this one."""

        import numpy

        from copernican.lib.likelihoods.cmb import evidence_assets
        from copernican.lib.likelihoods.cmb.runtime import spectrum_projection

        waves = numpy.array([0.01, 0.02, 0.03])
        first = numpy.array([[1.0, -2.0, 3.0]])
        second = numpy.array([[2.0, 1.0, -1.0]])
        captured = []
        spectrum = spectrum_projection._integrate_power_spectrum(
            primordial_grid=numpy.ones(3),
            log_k_values=numpy.log(waves),
            primary=first,
            secondary=second,
            product_observer=lambda kind, payload: captured.append(payload),
            product_identity={
                "observable": "TT",
                "primary_component": "T",
                "secondary_component": "U",
            },
        )
        projection = {
            "k": waves,
            "spectra": {"TT": spectrum},
            "transfers": {"T": first, "U": second},
        }
        with tempfile.TemporaryDirectory() as directory:
            good = evidence_assets.write_array_products(
                directory, "good", captured[0]
            )
            unrelated = evidence_assets.write_array_products(
                directory,
                "unrelated",
                {
                    **captured[0],
                    "weighted_transfer_products": numpy.zeros((1, 3)),
                },
            )

            def candidate(artifact, **changes):
                return {
                    "artifact": artifact,
                    "observable": "TT",
                    "primary": "T",
                    "secondary": "U",
                    "log_k": captured[0]["log_k"],
                    "C_ell": captured[0]["C_ell"],
                    "primordial": captured[0]["primordial"],
                    **changes,
                }

            selected = scientific_acceptance._match_projection_quadratures(
                directory,
                projection,
                [candidate(good), candidate(unrelated)],
            )
            self.assertEqual(selected["TT"]["artifact"], good)
            self.assertEqual(
                scientific_acceptance._match_projection_quadratures(
                    directory, projection, [candidate(unrelated)]
                ),
                {},
            )
            self.assertEqual(
                scientific_acceptance._match_projection_quadratures(
                    directory,
                    projection,
                    [candidate(good, log_k=captured[0]["log_k"] + 1)],
                ),
                {},
            )

    def test_named_point_retains_the_accepted_projection_chain(self):
        """A later refinement cannot displace the returned base product."""

        import numpy

        from copernican.lib.likelihoods.cmb import (
            cmb,
            diagnostics,
            evidence_assets,
        )
        from copernican.lib.likelihoods.cmb.runtime import (
            evidence,
            spectrum_projection,
        )

        wave = numpy.array([0.01, 0.02, 0.03])
        first = numpy.array([[1.0, -2.0, 3.0]])
        second = numpy.array([[2.0, 1.0, -1.0]])
        observed = []
        raw = spectrum_projection._integrate_power_spectrum(
            primordial_grid=numpy.ones(3),
            log_k_values=numpy.log(wave),
            primary=first,
            secondary=second,
            product_observer=lambda kind, payload: observed.append(payload),
            product_identity={
                "observable": "TT",
                "primary_component": "T",
                "secondary_component": "U",
            },
        )
        source = {
            "k": wave,
            "eta": numpy.arange(3.0),
            "eta_weights": numpy.ones(3),
            "sources_by_k": {
                str(float(k)): {"s": numpy.ones(3)} for k in wave
            },
        }
        projection = {
            "k": wave,
            "ell": numpy.array([2]),
            "eta": numpy.arange(3.0),
            "eta_weights": spectrum_projection._simpson_weights(
                numpy.arange(3.0)
            ),
            "chi": numpy.asarray((2.0, 1.0, 0.0)),
            "source_chi": 2.0,
            "transfers": {"T": first, "U": second},
            "spectra": {"TT": raw},
        }
        result = SimpleNamespace(
            solver_id="ccmbs",
            spectra={"TT": raw},
            raise_for_failure=lambda: None,
        )
        actual = {"spectra": {"TT": {"C_ell": raw, "D_ell": raw}}}
        runtime = {"model_name": "fixture"}
        inputs = scientific_acceptance.completion_physical_inputs(runtime)
        request = {
            "ells": [2],
            "spectra": ["TT"],
            "numerical_overrides": {},
            "workload": "full_spectrum",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = evidence_assets.write_json_artifact(
                root, "reference.json", {}
            )
            requirement = {
                "reference_kind": "camb",
                "ells": [2],
                "surfaces": ["TT"],
                "reference_identity": "fixture",
                "reference_sha256": "fixture",
            }
            contract = {
                "contract_sha256": "fixture",
                "cases": {"fixture:initial": requirement},
                "bindings": {
                    "fixture:initial": {
                        "reference_artifact": reference,
                        "model_filename": "fixture",
                        "point": "initial",
                        "physical_inputs": inputs,
                        "request": request,
                        "declaration_sha256": "fixture",
                        "numerical_plan": {},
                    }
                },
            }
            calls = []

            def execute(*args, **kwargs):
                calls.append(1)
                cmb._LAST_CMB_RESULT.set(result)
                if len(calls) == 1:
                    evidence.publish_completed_cmb_products("sources", source)
                    evidence.publish_completed_cmb_products(
                        "spectrum_quadrature", observed[0]
                    )
                    evidence.publish_completed_cmb_products(
                        "projection", projection
                    )
                    # A second calculation completes later but is not returned.
                    other = {**projection, "k": wave * 2}
                    evidence.publish_completed_cmb_products(
                        "sources", {**source, "k": wave * 2}
                    )
                    evidence.publish_completed_cmb_products(
                        "projection", other
                    )
                    evidence.publish_completed_cmb_products(
                        "accepted_numerical_spectrum",
                        {
                            "k": wave,
                            "ell": numpy.array([2]),
                            "spectra": {"TT": raw},
                        },
                    )
                return result.spectra

            with (
                mock.patch.object(
                    scientific_acceptance,
                    "completion_point_contract",
                    return_value=runtime,
                ),
                mock.patch.object(
                    scientific_acceptance,
                    "_screen_completion_photon_density",
                    return_value={
                        "relative_difference": 0.0,
                        "screen_threshold": 0.005,
                    },
                ),
                mock.patch.object(
                    scientific_acceptance,
                    "_screen_completion_reionization",
                    return_value={
                        "maximum_relative_difference": 0.0,
                        "background_resolution_difference": 0.0,
                        "screen_threshold": 0.005,
                    },
                ),
                mock.patch.object(
                    cmb,
                    "compute_cmb_spectrum_from_contract",
                    side_effect=execute,
                ),
                mock.patch.object(
                    diagnostics,
                    "_ordinary_parity_payload",
                    return_value=actual,
                ),
                mock.patch.object(
                    diagnostics,
                    "compare_cmb_completion_spectra",
                    return_value={"accepted": True},
                ),
            ):
                report = scientific_acceptance.run_completion_point(
                    root,
                    "fixture:initial",
                    contract=contract,
                    model_records=[
                        SimpleNamespace(
                            model_filename="fixture", plugin=object()
                        )
                    ],
                )
            self.assertTrue(report["parity_accepted"])
            self.assertFalse(report["completion_accepted"])
            self.assertNotIn(
                "accepted ordinary full-window physical comparison",
                report["incomplete_requirements"],
            )
            self.assertIn(
                "independent physical-domain and numerical-axis verification",
                report["incomplete_requirements"],
            )
            self.assertEqual(report["repeat"]["new_numerical_products"], 0)
            loaded = diagnostics.read_cmb_completion_products(
                root, report["raw_products"]
            )
            numpy.testing.assert_array_equal(loaded["source_k"], wave)
            numpy.testing.assert_array_equal(loaded["transfer_k"], wave)
            diagnostics.validate_cmb_completion_products(loaded)

    def test_named_point_failure_preserves_actual_observed_arrays(self):
        """A failed numerical call retains its partial evidence honestly."""

        import numpy

        from copernican.lib.likelihoods.cmb import cmb, evidence_assets
        from copernican.lib.likelihoods.cmb.runtime import evidence

        runtime = {"model_name": "failure_fixture"}
        physical_inputs = scientific_acceptance.completion_physical_inputs(
            runtime
        )
        request = {
            "ells": [2],
            "spectra": ["TT"],
            "numerical_overrides": {},
            "workload": "full_spectrum",
        }
        case_id = "fixture:initial"
        payload = {"state": numpy.arange(4.0)}

        def failing_solve(*args, **kwargs):
            evidence.publish_completed_cmb_products(
                "native_refinement", payload
            )
            raise ValueError("independent numerical rejection")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = evidence_assets.write_json_artifact(
                root, "reference.json", {"not_a_scientific_reference": True}
            )
            contract = {
                "contract_sha256": "fixture-contract",
                "cases": {
                    case_id: {
                        "reference_kind": "camb",
                        "ells": [2],
                        "surfaces": ["TT"],
                        "reference_identity": "fixture",
                        "reference_sha256": reference["sha256"],
                    }
                },
                "bindings": {
                    case_id: {
                        "reference_artifact": reference,
                        "model_filename": "fixture",
                        "point": "initial",
                        "physical_inputs": physical_inputs,
                        "request": request,
                        "declaration_sha256": "fixture-declaration",
                        "numerical_plan": {"policy": "ordinary_automatic"},
                    }
                },
            }
            with (
                mock.patch.object(
                    scientific_acceptance,
                    "completion_point_contract",
                    return_value=runtime,
                ),
                mock.patch.object(
                    scientific_acceptance,
                    "_screen_completion_photon_density",
                    return_value={
                        "relative_difference": 0.0,
                        "screen_threshold": 0.005,
                    },
                ),
                mock.patch.object(
                    scientific_acceptance,
                    "_screen_completion_reionization",
                    return_value={
                        "maximum_relative_difference": 0.0,
                        "background_resolution_difference": 0.0,
                        "screen_threshold": 0.005,
                    },
                ),
                mock.patch.object(
                    cmb,
                    "compute_cmb_spectrum_from_contract",
                    side_effect=failing_solve,
                ),
            ):
                result = scientific_acceptance.run_completion_point(
                    root,
                    case_id,
                    contract=contract,
                    model_records=[
                        SimpleNamespace(
                            model_filename="fixture", plugin=object()
                        )
                    ],
                )
            self.assertFalse(result["accepted"])
            self.assertFalse(result["completion_accepted"])
            self.assertIn(
                "independent numerical rejection", result["failure"]["message"]
            )
            self.assertEqual(len(result["observations"]), 1)
            retained = evidence_assets.read_array_products(
                root, result["observations"][0]["artifact"], materialize=True
            )
            numpy.testing.assert_array_equal(
                retained["state"], payload["state"]
            )
            producer = next(root.glob("cases/*/*/producer.json"))
            self.assertEqual(
                json.loads(producer.read_text())["observations"], []
            )
            self.assertEqual(
                json.loads(producer.read_text())["source_identity"],
                result["source_identity"],
            )
            with (
                mock.patch.object(
                    scientific_acceptance,
                    "completion_point_contract",
                    return_value=runtime,
                ),
                mock.patch.object(
                    scientific_acceptance,
                    "_screen_completion_photon_density",
                    return_value={
                        "relative_difference": 0.0,
                        "screen_threshold": 0.005,
                    },
                ),
                mock.patch.object(
                    scientific_acceptance,
                    "_screen_completion_reionization",
                    return_value={
                        "maximum_relative_difference": 0.12,
                        "background_resolution_difference": 0.001,
                        "screen_threshold": 0.005,
                    },
                ),
                mock.patch.object(
                    cmb, "compute_cmb_spectrum_from_contract"
                ) as solve,
            ):
                screened = scientific_acceptance.run_completion_point(
                    root,
                    case_id,
                    contract=contract,
                    model_records=[
                        SimpleNamespace(
                            model_filename="fixture", plugin=object()
                        )
                    ],
                )
            solve.assert_not_called()
            self.assertFalse(screened["completion_accepted"])
            self.assertIn(
                "reionization histories differ", screened["failure"]["message"]
            )
            self.assertEqual(screened["observations"], [])
            with (
                mock.patch.object(
                    scientific_acceptance,
                    "completion_point_contract",
                    return_value=runtime,
                ),
                mock.patch.object(
                    scientific_acceptance,
                    "_screen_completion_photon_density",
                    return_value={
                        "relative_difference": 0.25,
                        "screen_threshold": 0.005,
                    },
                ),
                mock.patch.object(
                    scientific_acceptance, "_screen_completion_reionization"
                ) as reionization,
                mock.patch.object(
                    cmb, "compute_cmb_spectrum_from_contract"
                ) as solve,
            ):
                photon_mismatch = scientific_acceptance.run_completion_point(
                    root,
                    case_id,
                    contract=contract,
                    model_records=[
                        SimpleNamespace(
                            model_filename="fixture", plugin=object()
                        )
                    ],
                )
            reionization.assert_not_called()
            solve.assert_not_called()
            self.assertFalse(photon_mismatch["completion_accepted"])
            self.assertIn(
                "photon densities differ",
                photon_mismatch["failure"]["message"],
            )

    def test_physical_binding_keeps_math_and_excludes_compiled_handles(
        self,
    ):
        runtime = {
            "model_name": "independent_declaration",
            "background": {"H": "H0/a"},
            "perturbations": {"equations": ["x'=k*x"]},
            "background_runtime": object(),
            "perturbation_data": object(),
            "compile_diagnostics": object(),
            "_engine_numerical_plan": {"nodes": 1024},
        }
        inputs = scientific_acceptance.completion_physical_inputs(runtime)
        self.assertEqual(inputs["background"], runtime["background"])
        self.assertEqual(inputs["perturbations"], runtime["perturbations"])
        self.assertNotIn("background_runtime", inputs)
        self.assertNotIn("_engine_numerical_plan", inputs)
        self.assertEqual(json.loads(json.dumps(inputs)), inputs)

    def test_completion_reference_retains_declared_temperature(self):
        """Reference input resolution follows the actual physical contract."""

        from tests.copernican.lib.likelihoods.cmb import test_cmb
        from tests.project.lib import camb_reference

        raw = test_cmb._declared_scalar_hierarchy_contract()
        raw["model_parameters"]["Tcmb_K"] = 3.0
        contract = test_cmb._prepare_declared_contract(raw)
        reference = scientific_acceptance.completion_reference_contract(
            contract
        )
        self.assertEqual(reference["param_map"]["TCMB"], 3.0)
        self.assertEqual(camb_reference._make_camb_params(reference).TCMB, 3.0)

    def test_canonical_default_temperature_keeps_resolved_physics(self):
        from tests.copernican.lib.likelihoods.cmb import test_cmb
        from tests.project.lib import camb_reference

        runtime = test_cmb._prepare_declared_contract(
            test_cmb._declared_scalar_hierarchy_contract()
        )
        explicit = scientific_acceptance.completion_reference_contract(runtime)
        canonical = scientific_acceptance.completion_reference_contract(
            runtime, canonical_defaults=True
        )
        self.assertEqual(explicit["param_map"]["TCMB"], 2.7255)
        self.assertNotIn("TCMB", canonical["param_map"])
        self.assertEqual(
            str(camb_reference._make_camb_params(explicit)),
            str(camb_reference._make_camb_params(canonical)),
        )
        runtime["model_parameters"]["Tcmb_K"] = 3.0
        canonical = scientific_acceptance.completion_reference_contract(
            runtime, canonical_defaults=True
        )
        self.assertEqual(canonical["param_map"]["TCMB"], 3.0)

    def test_completion_preflight_persists_failures_without_solver_execution(
        self,
    ):
        from copernican.lib.likelihoods.cmb import cmb, evidence_assets

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                mock.patch.object(
                    scientific_acceptance,
                    "materialize_completion_contract",
                    side_effect=ValueError(
                        "Frozen independent reference differs"
                    ),
                ),
                mock.patch.object(
                    cmb,
                    "compute_cmb_spectrum_from_contract",
                    side_effect=AssertionError("preflight must not solve"),
                ) as solve,
                mock.patch("builtins.print"),
            ):
                status = scientific_acceptance.main(
                    ["--completion-preflight", "--evidence-root", directory]
                )
            report = json.loads(
                (root / "completion-preflight.json").read_text()
            )
            self.assertEqual(status, 1)
            self.assertFalse(report["preflight_ready"])
            self.assertFalse(report["accepted"])
            self.assertFalse(report["scientific_acceptance"])
            self.assertIn("Frozen independent", report["failure"]["message"])
            retained = evidence_assets.read_json_artifact(
                root, report["attempt"]
            )
            self.assertEqual(retained["failure"], report["failure"])
            solve.assert_not_called()

    def test_preflight_readiness_does_not_claim_scientific_acceptance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scientific_acceptance._retain(
                root / "completion-contract.json", {}
            )
            with mock.patch.object(
                scientific_acceptance,
                "materialize_completion_contract",
                return_value={
                    "contract_sha256": "contract",
                    "cases": {"case": {}},
                },
            ):
                report = scientific_acceptance.run_completion_preflight(root)
            self.assertTrue(report["preflight_ready"])
            self.assertFalse(report["accepted"])
            self.assertFalse(report["scientific_acceptance"])

    def test_development_cli_reports_cost_without_rejecting_passing_tests(
        self,
    ):
        """A passing test retains cost telemetry and CLI success."""

        selection = (
            "tests.copernican.lib.test_cmb_output.CMBOutputTestCase."
            "test_public_units_follow_temperature_and_potential_dimensions",
        )
        with tempfile.TemporaryDirectory() as directory:
            with (
                mock.patch.dict(
                    scientific_acceptance.DEVELOPMENT_WORKLOADS,
                    {"focused": selection},
                ),
                mock.patch.dict(
                    scientific_acceptance.DEVELOPMENT_COST_TARGETS,
                    {"focused": 0.0},
                ),
                mock.patch("builtins.print"),
            ):
                status = scientific_acceptance.main(
                    [
                        "--development-checks",
                        "focused",
                        "--evidence-root",
                        directory,
                    ]
                )
            report = json.loads(
                (Path(directory) / "development-focused.json").read_text()
            )
            self.assertEqual(status, 0)
            self.assertTrue(report["passed"])
            self.assertFalse(report["within_cost_target"])
            self.assertGreater(report["elapsed_seconds"], 0.0)
            self.assertEqual(report["tests_run"], 1)
            self.assertFalse(report["scientific_acceptance"])

    def test_development_runner_measures_tests_without_scientific_solves(
        self,
    ):
        """A measured unit-test workload retains per-test cost and logs."""

        selection = (
            "tests.copernican.lib.test_cmb_output.CMBOutputTestCase."
            "test_public_units_follow_temperature_and_potential_dimensions",
        )
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "development.json"
            with mock.patch.dict(
                scientific_acceptance.DEVELOPMENT_WORKLOADS,
                {"focused": selection},
            ):
                report = scientific_acceptance.run_development_checks(
                    "focused", destination
                )
            self.assertTrue(report["passed"])
            self.assertEqual(report["tests_run"], 1)
            self.assertEqual(report["tests"][0]["test"], selection[0])
            self.assertGreater(report["elapsed_seconds"], 0)
            self.assertTrue((destination.parent / report["test_log"]).exists())
            self.assertFalse(report["scientific_acceptance"])
            self.assertIn("numpy", report["environment"])

    def test_development_runner_retains_an_actual_failed_negative_control(
        self,
    ):
        """An isolated faulty probe produces a failed development report."""

        class FaultyProbe(unittest.TestCase):
            def runTest(self):
                self.assertEqual(1, 2, "manufactured negative control")

        with tempfile.TemporaryDirectory() as directory:
            with (
                mock.patch.object(
                    unittest.defaultTestLoader,
                    "loadTestsFromNames",
                    return_value=unittest.TestSuite((FaultyProbe(),)),
                ),
                mock.patch("builtins.print"),
            ):
                status = scientific_acceptance.main(
                    [
                        "--development-checks",
                        "focused",
                        "--evidence-root",
                        directory,
                    ]
                )
            report = json.loads(
                (Path(directory) / "development-focused.json").read_text()
            )
            self.assertEqual(status, 1)
            self.assertFalse(report["passed"])
            self.assertEqual(report["failures"], 1)
            self.assertEqual(report["tests"][0]["outcome"], "failed")

    def test_history_regeneration_command_cannot_launch_scientific_acceptance(
        self,
    ):
        with mock.patch.object(
            scientific_acceptance.camb_reference,
            "write_lcdm_history_reference_fixture",
            return_value=Path.cwd() / "reference.json",
        ) as write:
            with mock.patch.object(
                scientific_acceptance,
                "run_completion_baseline",
                side_effect=AssertionError("No baseline execution"),
            ):
                with mock.patch("builtins.print"):
                    self.assertEqual(
                        scientific_acceptance.main(
                            ["--freeze-development-reference"]
                        ),
                        0,
                    )
        write.assert_called_once_with()

    def test_handoff_reloads_the_report_written_by_the_runner(self):
        """The explicit handoff must compare writer and verifier outputs."""

        plugin = SimpleNamespace(
            INITIAL_GUESSES=(75.0,),
            get_cmb_declared_runtime=mock.Mock(
                return_value={"param_map": {"H0": 75.0}}
            ),
        )
        record = SimpleNamespace(
            model_filename="model_lcdm.yml",
            ready=True,
            plugin=plugin,
            failure=None,
            status="ready",
        )
        report = {
            "accepted": True,
            "quantitative_decision": {"status": "accepted"},
            "report_sha256": "a" * 64,
        }
        with tempfile.TemporaryDirectory() as output_directory:
            destination = Path(output_directory) / "fixed-lcdm.json"
            with (
                mock.patch.object(
                    scientific_acceptance,
                    "discover_cmb_model_records",
                    return_value=(record,),
                ),
                mock.patch.object(
                    scientific_acceptance.camb_reference,
                    "build_fixed_lcdm_camb_parity_row",
                    return_value={"reference": "camb"},
                ),
                mock.patch.object(
                    scientific_acceptance,
                    "run_fixed_lcdm_cmb_parity",
                    return_value=report,
                ) as run,
                mock.patch.object(
                    scientific_acceptance,
                    "read_cmb_parity_matrix_report",
                    return_value=report,
                ) as read,
            ):
                result = (
                    scientific_acceptance.run_fixed_lcdm_scientific_acceptance(
                        destination
                    )
                )

        self.assertEqual(result, report)
        run.assert_called_once()
        read.assert_called_once_with(destination)
        plugin.get_cmb_declared_runtime.assert_called_once_with((75.0,))

    def test_baseline_command_keeps_rejection_explicit(self):
        with mock.patch.object(
            scientific_acceptance,
            "run_completion_baseline",
            return_value={"accepted": False},
        ) as run:
            with mock.patch("builtins.print"):
                result = scientific_acceptance.main(["--baseline"])
        self.assertEqual(result, 1)
        self.assertIn("ccmbs-acceptance", str(run.call_args.args[0]))

    def test_checkout_provenance_records_current_uncommitted_content(self):
        root = Path(__file__).resolve().parents[3]
        recorded = scientific_acceptance.checkout_provenance(root)
        self.assertEqual(len(recorded["source_revision"]), 40)
        self.assertEqual(
            recorded["sha256"],
            scientific_acceptance.diagnostics._canonical_sha256(
                {
                    name: recorded[name]
                    for name in ("source_revision", "uncommitted")
                }
            ),
        )


if __name__ == "__main__":
    unittest.main()
