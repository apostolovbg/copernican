"""Explicit, user-invoked fixed-LCDM scientific handoff."""

from __future__ import annotations

import argparse
import contextlib
import copy
import gzip
import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess  # nosec B404 - reviewed read-only Git provenance queries.
import sys
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy

from copernican.lib.likelihoods.cmb import diagnostics, evidence_assets
from copernican.lib.likelihoods.cmb.diagnostics import (
    discover_cmb_model_records,
    read_cmb_parity_matrix_report,
    run_fixed_lcdm_cmb_parity,
)
from copernican.lib.likelihoods.cmb.runtime import background, cache
from copernican.workflow import _output_root
from tests.project.lib import camb_reference
from tests.project.lib.ccmbs_production_graph import _read_source_revision

_MODEL_FILENAME = "model_lcdm.yml"
_CMB_BEHAVIOR_OWNER = (
    "tests.copernican.lib.likelihoods.cmb.test_cmb."
    "CMBCustomRuntimeBehaviorTestCase."
)

# These are explicit unittest selections, not replacement acceptance claims.
DEVELOPMENT_WORKLOADS = {
    "focused": (
        "tests.copernican.lib.test_perturbation_contract",
        "tests.copernican.lib.likelihoods.cmb.runtime.test_collisions",
        "tests.copernican.lib.likelihoods.cmb.runtime.test_evolution",
        "tests.copernican.lib.likelihoods.cmb.runtime."
        "test_spectrum_projection",
        "tests.copernican.lib.likelihoods.cmb.runtime.test_cache",
        "tests.copernican.lib.likelihoods.cmb.runtime.test_adaptive",
        "tests.copernican.lib.likelihoods.cmb.runtime.test_convergence",
        "tests.copernican.lib.likelihoods.cmb.test_diagnostics."
        "CompletionContractTestCase",
        "tests.copernican.lib.likelihoods.cmb.test_diagnostics."
        "CompletionPersistenceTestCase",
        "tests.copernican.lib.likelihoods.cmb.test_diagnostics."
        "CCMBSDiagnosticTestCase."
        "test_source_history_audit_recomputes_all_declared_closures",
        "tests.copernican.lib.likelihoods.cmb.test_diagnostics."
        "CCMBSDiagnosticTestCase."
        "test_fixed_lcdm_parity_rejects_mismatched_physical_inputs",
        "tests.copernican.lib.likelihoods.cmb.test_diagnostics."
        "CCMBSDiagnosticTestCase."
        "test_fixed_lcdm_parity_rejects_massive_camb_defaults",
        _CMB_BEHAVIOR_OWNER
        + "test_declared_scalar_hierarchy_materializes_generated_hierarchy",
        "tests.copernican.lib.test_cmb_output",
        "tests.copernican.lib.test_run_pipeline",
        "tests.copernican.lib.test_run_executor",
        "tests.copernican.lib.likelihoods.test_likelihoods",
        "tests.copernican.samplers.test_sampler_mcmc",
        "tests.project.lib.test_camb_reference",
        "tests.project.lib.test_scientific_acceptance",
    ),
    "bounded": (
        "tests.copernican.lib.likelihoods.cmb.runtime.test_mode_evolution",
        _CMB_BEHAVIOR_OWNER
        + "test_generated_scalar_superhorizon_metric_seed_stays_regular",
        _CMB_BEHAVIOR_OWNER
        + "test_declared_scalar_hierarchy_synchronous_matches_newtonian",
        _CMB_BEHAVIOR_OWNER
        + "test_declared_scalar_hierarchy_depth_converges_at_anchor_surface",
        _CMB_BEHAVIOR_OWNER
        + "test_declared_vector_spectra_remain_stable_under_phase_refinement",
        _CMB_BEHAVIOR_OWNER
        + "test_declared_spectrum_cache_identity_covers_runtime_inputs",
        _CMB_BEHAVIOR_OWNER
        + "test_runtime_cache_state_requires_matching_request_shape",
        _CMB_BEHAVIOR_OWNER
        + "test_batched_declared_evolution_matches_scalar_modes",
    ),
}
DEVELOPMENT_COST_TARGETS = {"focused": 60.0, "bounded": 300.0}


class _TimedDevelopmentResult(unittest.TextTestResult):
    """Record each real unittest outcome and observed parent-process work."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rows = []
        self.requests = []

    def startTest(self, test):
        self.started = time.perf_counter()
        self.request_start = len(self.requests)
        self.current_outcome = "passed"
        super().startTest(test)

    def addFailure(self, test, error):
        self.current_outcome = "failed"
        super().addFailure(test, error)

    def addError(self, test, error):
        self.current_outcome = "failed"
        super().addError(test, error)

    def addSubTest(self, test, subtest, error):
        if error is not None:
            self.current_outcome = "failed"
        super().addSubTest(test, subtest, error)

    def addSkip(self, test, reason):
        self.current_outcome = "skipped"
        super().addSkip(test, reason)

    def addExpectedFailure(self, test, error):
        self.current_outcome = "expected_failure"
        super().addExpectedFailure(test, error)

    def addUnexpectedSuccess(self, test):
        self.current_outcome = "unexpected_success"
        super().addUnexpectedSuccess(test)

    def stopTest(self, test):
        self.rows.append(
            {
                "test": test.id(),
                "seconds": time.perf_counter() - self.started,
                "outcome": self.current_outcome,
                "ccmbs_requests": self.requests[self.request_start :],
            }
        )
        super().stopTest(test)


def run_development_checks(workload: str, output_path: str | Path) -> dict:
    """Measure selected development tests without starting scientific runs.

    Cost overruns are development blockers, not physical theory rejections.
    Spawned-worker work remains covered by its owning tests; telemetry here
    records parent-process requests, including cold/warm states and phases.
    """

    import numpy
    import scipy

    if workload not in DEVELOPMENT_WORKLOADS:
        raise ValueError(f"Unknown development workload: {workload}")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    log_path = destination.with_suffix(".log")
    result_holder = []

    def make_result(*args, **kwargs):
        result = _TimedDevelopmentResult(*args, **kwargs)
        result_holder.append(result)
        return result

    original_record = cache.record_cmb_performance

    def observe_request(*args, **kwargs):
        record = original_record(*args, **kwargs)
        if result_holder:
            result_holder[-1].requests.append(
                {
                    key: record[key]
                    for key in (
                        "workload",
                        "cache_state",
                        "phase_seconds",
                        "work_units",
                        "outcome",
                        "context",
                    )
                }
            )
        return record

    cache.clear_cmb_caches()
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as stream:
        with (
            contextlib.ExitStack() as handlers,
            contextlib.redirect_stdout(stream),
            contextlib.redirect_stderr(stream),
        ):
            suite = unittest.defaultTestLoader.loadTestsFromNames(
                DEVELOPMENT_WORKLOADS[workload]
            )
            # Existing console handlers retain their original stream across
            # redirect_stderr. Keep their output in the same retained log
            # without disabling logging or interfering with assertLogs.
            loggers = [logging.getLogger()] + [
                logger
                for logger in logging.Logger.manager.loggerDict.values()
                if isinstance(logger, logging.Logger)
            ]
            for logger in loggers:
                for handler in logger.handlers:
                    if isinstance(handler, logging.StreamHandler) and (
                        handler.stream is sys.__stderr__
                        or handler.stream is sys.__stdout__
                    ):
                        previous_stream = handler.stream
                        handler.setStream(stream)
                        handlers.callback(handler.setStream, previous_stream)
            cache.record_cmb_performance = observe_request
            try:
                result = unittest.TextTestRunner(
                    stream=stream, verbosity=2, resultclass=make_result
                ).run(suite)
            finally:
                cache.record_cmb_performance = original_record
    elapsed = time.perf_counter() - started
    try:
        import resource
    except ImportError:
        peak_rss_bytes = None
    else:
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_rss_bytes = int(
            peak_rss if sys.platform == "darwin" else peak_rss * 1024
        )
    reference_fixture = camb_reference.load_lcdm_history_reference_fixture()
    report = {
        "schema_version": 1,
        "kind": "ccmbs_development_checks",
        "workload": workload,
        "selections": list(DEVELOPMENT_WORKLOADS[workload]),
        "scientific_acceptance": False,
        "source_revision": _read_source_revision(),
        "source_identity": diagnostics.cmb_completion_source_identity(),
        "reference_fixture_sha256": reference_fixture["fixture_sha256"],
        "environment": {
            "system": platform.system(),
            "machine": platform.machine(),
            "cpu": platform.processor(),
            "cpu_count": os.cpu_count(),
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "camb": camb_reference.camb.__version__,
            "threads": {
                key: os.environ.get(key)
                for key in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
        },
        "initial_ccmbs_cache_state": "cleared",
        "telemetry_scope": (
            "observed parent-process requests; tests may clear caches"
        ),
        "phase_timing_scope": (
            "raw executor counters; use elapsed_seconds for wall costs"
        ),
        "elapsed_seconds": elapsed,
        "peak_process_rss_bytes": peak_rss_bytes,
        "cost_target_seconds": DEVELOPMENT_COST_TARGETS[workload],
        "within_cost_target": elapsed <= DEVELOPMENT_COST_TARGETS[workload],
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "expected_failures": len(result.expectedFailures),
        "unexpected_successes": len(result.unexpectedSuccesses),
        "passed": result.wasSuccessful()
        and result.testsRun > 0
        and not result.skipped
        and not result.expectedFailures,
        "tests": result.rows,
        "test_log": log_path.name,
    }
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def run_fixed_lcdm_scientific_acceptance(
    output_path: str | Path,
    *,
    model_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Run and reload one durable fixed-LCDM acceptance handoff.

    The CAMB reference is built by the test-owned independent surface.  The
    resulting report is written to ``output_path`` and immediately reloaded
    through the production digest verifier.  A separate process can verify
    the retained artifact with this module's ``--verify`` mode.
    """

    records = discover_cmb_model_records(model_directory)
    record = next(
        (
            candidate
            for candidate in records
            if candidate.model_filename == _MODEL_FILENAME
        ),
        None,
    )
    if record is None:
        raise RuntimeError(
            f"Required model was not discovered: {_MODEL_FILENAME}"
        )
    if not record.ready or record.plugin is None:
        raise RuntimeError(
            f"Required model is not ready: {record.failure or record.status}"
        )
    plugin = record.plugin
    declared_contract = plugin.get_cmb_declared_runtime(plugin.INITIAL_GUESSES)
    reference_row = camb_reference.build_fixed_lcdm_camb_parity_row()
    report = run_fixed_lcdm_cmb_parity(
        reference_row,
        declared_contract,
        model_directory=model_directory,
        output_path=output_path,
    )
    reloaded = read_cmb_parity_matrix_report(output_path)
    if reloaded != report:
        raise ValueError(
            "Reloaded fixed-LCDM scientific report differs from its writer "
            "output"
        )
    return reloaded


def completion_point_contract(plugin: Any, point: str) -> dict[str, Any]:
    """Resolve each frozen physical point through its declaration compiler."""

    values = list(plugin.INITIAL_GUESSES)
    updates = {
        "mass_zero": {"sum_mnu": 0.0},
        "mass_006": {"sum_mnu": 0.06},
        "mass_050": {"sum_mnu": 0.5},
        "w_minus_09": {"w_de": -0.9},
        "w0_minus_09": {"w0": -0.9},
        "wa_plus_02": {"wa": 0.2},
    }.get(point, {})
    names = list(plugin.PARAMETER_NAMES)
    for name, value in updates.items():
        values[names.index(name)] = value
    contract = copy.deepcopy(plugin.get_cmb_declared_runtime(values))
    if point == "amplitude_up":
        contract["param_map"]["As"] *= 1.1
    return contract


def completion_reference_contract(
    contract: dict[str, Any], *, canonical_defaults: bool = False
) -> dict:
    """Declare all CAMB physical inputs, including massless and DE limits."""

    params = dict(contract["param_map"])
    mass = float(params.get("sum_mnu", params.get("mnu", 0.0)))
    if mass == 0.0:
        params.update(mnu=0.0, num_massive_neutrinos=0)
    physical = background._resolve_custom_cmb_physical_parameters(contract)
    params["TCMB"] = physical.Tcmb_K
    if canonical_defaults and params["TCMB"] == float(
        camb_reference.camb.CAMBparams().TCMB
    ):
        # The frozen references serialize this resolved library default by
        # omission. Its physical value is checked against the frozen case;
        # retaining that canonical spelling preserves the original hashes.
        params.pop("TCMB")
    model = contract.get("model_parameters", {})
    calls = [
        {
            "method": "set_dark_energy",
            "kwargs": {
                "w": float(model.get("w_de", model.get("w0", -1.0))),
                "wa": float(model.get("wa", 0.0)),
                "cs2": 1.0,
            },
        }
    ]
    return {
        "backend": "camb",
        "param_map": params,
        "calls": calls,
        "grids": {},
        "values": {},
    }


def completion_physical_inputs(runtime):
    """Retain declaration mathematics without serializing compiled handles.

    Background/perturbation runtime objects and compile diagnostics are
    derived cache assets. Their declaration inputs and compiler source
    identity, rather than machine-local object representations, identify
    the physical calculation. Numerical planning has its separate identity.
    """

    return diagnostics._jsonable(
        {
            key: runtime.get(key, {})
            for key in (
                "model_name",
                "model_parameters",
                "param_map",
                "background",
                "grids",
                "values",
                "calls",
                "perturbations",
            )
        }
    )


def checkout_provenance(repository_root):
    """Record a commit and every uncommitted change, including new files."""

    root = Path(repository_root)
    git_executable = shutil.which("git")
    if git_executable is None:
        raise FileNotFoundError(
            "Git is required to record checkout provenance"
        )

    def git(*arguments):
        """Read checkout identity without modifying its index or branch."""

        # Call sites below contain only fixed read-only Git arguments. The
        # executable is resolved explicitly; no shell expansion is involved.
        return subprocess.check_output(  # nosec B603
            [git_executable, *arguments], cwd=root
        ).decode("utf-8")

    revision = git("rev-parse", "HEAD").strip()
    names = set(git("diff", "--name-only", "-z", "HEAD").split("\0"))
    names.update(
        git("ls-files", "--others", "--exclude-standard", "-z").split("\0")
    )
    changes = {}
    for name in sorted(names - {""}):
        path = root / name
        changes[name] = (
            hashlib.sha256(path.read_bytes()).hexdigest()
            if path.is_file()
            else None
        )
    payload = {"source_revision": revision, "uncommitted": changes}
    return {**payload, "sha256": diagnostics._canonical_sha256(payload)}


def _retain(path: Path, payload: dict) -> dict:
    """Persist JSON with a portable relative artifact identity."""

    return evidence_assets.write_json_artifact(
        path.parent, path.name, diagnostics._jsonable(payload)
    )


def materialize_completion_contract(root: Path, *, model_records=None) -> dict:
    """Freeze exact points, independent references, and applicability.

    Independent reference generation is explicit, never ordinary discovery.
    Frozen scientific requirements and historical materialization hashes are
    preserved verbatim. Current declarations and artifact locations are
    separate bindings, rather than a new identity pasted onto old payloads.
    """

    repository = Path(__file__).resolve().parents[3]
    frozen_path = (
        repository / "tests/project/fixtures/ccmbs_completion_contract.json"
    )
    frozen = json.loads(frozen_path.read_text())
    independent_references = json.loads(
        gzip.decompress(
            (
                repository
                / "tests/project/fixtures/ccmbs_comparable_camb.json.gz"
            ).read_bytes()
        )
    )["references"]
    if frozen["metrics"] != diagnostics._jsonable(
        diagnostics.CMB_COMPLETION_METRICS
    ):
        raise ValueError("Frozen scientific metrics differ from the evaluator")
    records = (
        discover_cmb_model_records()
        if model_records is None
        else model_records
    )
    cases = {}
    for record in records:
        if record.model_filename not in diagnostics.CMB_COMPLETION_POINTS:
            continue
        if not record.ready:
            raise ValueError(f"Required declaration is unavailable: {record}")
        plugin = record.plugin
        for point in diagnostics.CMB_COMPLETION_POINTS[record.model_filename]:
            runtime = completion_point_contract(plugin, point)
            physical = completion_physical_inputs(runtime)
            case = {
                "model_filename": record.model_filename,
                "point": point,
                "physical_inputs": physical,
                "declaration_sha256": hashlib.sha256(
                    (
                        repository
                        / "copernican/models"
                        / record.model_filename
                    ).read_bytes()
                ).hexdigest(),
                "sectors": ["scalar"],
                "surfaces": list(
                    diagnostics.declared_cmb_spectrum_names(plugin)
                ),
                "ells": list(camb_reference.CAMB_PARITY_ELL_VALUES),
                "axis_non_applicability": {},
                "reference_kind": "theory_invariants",
                "required_invariants": [
                    "equation_residuals",
                    "conservation",
                    "initial_constraints",
                    "limiting_case",
                ],
                "graph_required": record.model_filename
                in ("model_lcdm.yml", "model_ref_planck2018.yml")
                and point == "initial",
            }
            if float(runtime["param_map"].get("sum_mnu", 0.0)) == 0.0:
                no_massive_species = (
                    "No massive species in this physical point."
                )
                case["axis_non_applicability"].update(
                    momentum_q=no_massive_species,
                )
            if (
                record.model_filename
                in diagnostics.CAMB_COMPARABLE_CMB_MODEL_FILENAMES
            ):
                reference_contract = completion_reference_contract(
                    runtime, canonical_defaults=True
                )
                requirement = frozen["cases"][
                    f"{record.model_filename}:{point}"
                ]
                reference_digest = requirement["reference_sha256"]
                relative = f"references/{reference_digest}.json"
                path = evidence_assets.contained_path(root, relative)
                reference = None
                if path.exists():
                    candidate = json.loads(path.read_text())
                    saved = candidate.pop("fixture_sha256", None)
                    if not (
                        saved
                        == reference_digest
                        == camb_reference.reference_fixture_sha256(candidate)
                        and candidate.get("contract") == reference_contract
                        and candidate.get("reference_identity")
                        == camb_reference.CAMB_REFERENCE_IDENTITY
                    ):
                        raise ValueError(
                            "Retained independent reference is stale or "
                            f"tampered: {record.model_filename}:{point}"
                        )
                    candidate["fixture_sha256"] = saved
                    reference = candidate
                if reference is None:
                    # Reopen the exact independently frozen arrays. Rebuilding
                    # on another compiler/architecture cannot be required to
                    # reproduce floating-point bytes before a solve can run.
                    reference = copy.deepcopy(
                        independent_references[
                            f"{record.model_filename}:{point}"
                        ]
                    )
                    body = {
                        key: value
                        for key, value in reference.items()
                        if key != "fixture_sha256"
                    }
                    if (
                        camb_reference.reference_fixture_sha256(body)
                        != reference_digest
                        or reference.get("contract") != reference_contract
                        or reference.get("reference_identity")
                        != camb_reference.CAMB_REFERENCE_IDENTITY
                    ):
                        raise ValueError(
                            "Canonical independent reference is stale "
                            "or tampered"
                        )
                if reference["fixture_sha256"] != reference_digest:
                    raise ValueError(
                        "Frozen independent reference differs: "
                        f"{record.model_filename}:{point}; "
                        f"expected={reference_digest}; "
                        f"actual={reference['fixture_sha256']}; "
                        "frozen physical inputs="
                        f"{requirement['resolved_reference_inputs']}; "
                        "current physical inputs="
                        f"{reference['resolved_parameters']}"
                    )
                for field, actual in (
                    ("ells", reference["ell_values"]),
                    ("features", reference["features"]),
                    ("damping_windows", reference["damping_windows"]),
                    (
                        "resolved_reference_inputs",
                        reference["resolved_parameters"],
                    ),
                ):
                    if actual != requirement[field]:
                        raise ValueError(f"Frozen reference {field} differs")
                actual_temperature = float(
                    camb_reference._make_camb_params(reference_contract).TCMB
                )
                if actual_temperature != requirement["temperature_K"]:
                    raise ValueError("Frozen reference temperature differs")
                artifact = evidence_assets.write_json_artifact(
                    root, relative, reference
                )
                case.update(
                    reference_kind="camb",
                    temperature_K=float(
                        camb_reference._make_camb_params(
                            reference_contract
                        ).TCMB
                    ),
                    reference_artifact=artifact,
                    reference_sha256=reference["fixture_sha256"],
                    reference_identity=reference["reference_identity"],
                    resolved_reference_inputs=reference["resolved_parameters"],
                    resolved_reference_defaults=reference["resolved_defaults"],
                    surfaces=list(reference["spectra"]),
                    ells=reference["ell_values"],
                    features=reference["features"],
                    damping_windows=reference["damping_windows"],
                    required_physics_matching=[
                        "species_distributions",
                        "primordial",
                        "recombination",
                        "reionization",
                        "gauge_sources",
                        "units",
                        "lensing",
                        "sectors",
                    ],
                )
            theory_checks = {
                "model_qauc.yml": [
                    "qauc_attractor_envelope_A_zero",
                    "qauc_dark_energy_normalization",
                ],
                "model_qrsf.yml": [
                    "qrsf_baryon_locked_density",
                    "qrsf_baryon_locked_momentum",
                    "qrsf_inertial_euler",
                ],
                "model_tog.yml": [
                    "tog_temporal_factor_today_unity",
                    "tog_early_activation_limit",
                ],
                "model_torg.yml": [
                    "torg_baryon_locked_density",
                    "torg_baryon_locked_momentum",
                    "torg_diluted_photon_drag",
                ],
                "model_usmf2.yml": [
                    "shrink_field_equation",
                    "shrink_field_initial_constraints",
                    "scalar_B_parity_zero",
                ],
            }
            case["required_invariants"].extend(
                theory_checks.get(record.model_filename, ())
            )
            cases[f"{record.model_filename}:{point}"] = case
    auxiliary = {
        "novel:renamed_lcdm": (
            "_declared_scalar_hierarchy_contract",
            {},
            {"rename": {"theta_gamma0": "novel_temperature_monopole"}},
        ),
        "novel:recombination_opacity": (
            "_generic_background_custom_contract",
            {},
            {
                "hydrogen_temperature_K": "2.7255*(1+z)",
                "hydrogen_alpha_B": (
                    "1e-19*(hydrogen_temperature_K/3000)**(-0.5)"
                ),
                "beta_continuum": "5e-20*(hydrogen_temperature_K/3000)**0.5",
                "peebles_c": "0.9",
            },
        ),
        "novel:extra_fluid_interaction": ("_split_collision_contract", {}, {}),
        "sector:nonzero_vector": (
            "_declared_vector_hierarchy_contract",
            {},
            {},
        ),
        "sector:nonzero_tensor": (
            "_declared_tensor_hierarchy_contract",
            {},
            {},
        ),
        "sector:nonzero_lensing": (
            "_declared_scalar_hierarchy_contract",
            {"include_lensing": True},
            {},
        ),
    }
    for name, (builder, kwargs, mutation) in auxiliary.items():
        cases[name] = {
            "reference_kind": "theory_invariants",
            "ells": [20, 50, 80, 120],
            "physical_inputs": {
                "module": "tests.copernican.lib.likelihoods.cmb.test_cmb",
                "builder": builder,
                "kwargs": kwargs,
                "mutation": mutation,
            },
            "surfaces": ["TT", "TE", "EE", "BB"],
            "sectors": [
                (
                    name.rsplit("_", 1)[-1]
                    if "vector" in name or "tensor" in name
                    else "scalar"
                )
            ],
            "axis_non_applicability": {
                "momentum_q": "Fixture declares no massive species."
            },
            "required_invariants": [
                "equation_residuals",
                "conservation",
                "initial_constraints",
                "limiting_case",
            ],
            "nonzero_required": name.startswith("sector:"),
            "graph_required": False,
        }
    cases["sector:nonzero_lensing"]["surfaces"] = list(
        camb_reference.CAMB_PARITY_SPECTRA
    )
    if set(cases) != set(frozen["cases"]):
        raise ValueError(
            "Completion declaration inventory differs from the frozen cases"
        )
    for case_id, binding in cases.items():
        requirement = frozen["cases"][case_id]
        binding["request"] = {
            "ells": requirement["ells"],
            "spectra": requirement["surfaces"],
            "workload": "full_spectrum",
            "numerical_overrides": {},
        }
        # This identifies the requested planning policy; the actual selected
        # grids and controls belong to the completed execution artifact.
        binding["numerical_plan"] = {"policy": "ordinary_automatic"}
    contract = {
        "schema_version": 2,
        "cases": frozen["cases"],
        "bindings": cases,
        "frozen_fixture_sha256": hashlib.sha256(
            frozen_path.read_bytes()
        ).hexdigest(),
        "frozen_materialization_sha256": frozen["case_sha256"],
        "checkout_provenance": checkout_provenance(repository),
        "source_identity": diagnostics.cmb_completion_source_identity(),
        "source_revision": _read_source_revision(),
        "metrics": diagnostics.CMB_COMPLETION_METRICS,
        "metric_rationale": {
            "cross_floor": (
                "0.1% of the matching auto covariance scale; "
                "no TT-only floor."
            ),
            "auto_floor": (
                "64 binary64 epsilons of each surface; "
                "exact zero stays zero."
            ),
            "numerical_budget": (
                "Combined 0.5% budget leaves margin "
                "within the strictest 2% parity ceiling."
            ),
            "units": (
                "CAMB native muK C/D for temperature and polarization; "
                "dimensionless potential with native mixed units."
            ),
            "features": (
                "Unit ell spacing, +/-8 windows, location <=2 ell; "
                "no phase or amplitude fits."
            ),
        },
    }
    contract["contract_sha256"] = diagnostics._canonical_sha256(contract)
    evidence_assets.write_json_artifact(
        root, f"contracts/{contract['contract_sha256']}.json", contract
    )
    _retain(root / "completion-contract.json", contract)
    return contract


def run_completion_preflight(root: Path) -> dict:
    """Validate every reference before allowing expensive CCMBS execution.

    This is a prerequisite report, never scientific acceptance. Failed
    attempts remain durable and retain their original checkout provenance.
    """

    root = Path(root)
    repository = Path(__file__).resolve().parents[3]
    started = time.monotonic()
    report = {
        "purpose": "completion_reference_preflight",
        "accepted": False,
        "scientific_acceptance": False,
        "preflight_ready": False,
        "checkout_provenance": checkout_provenance(repository),
        "source_identity": diagnostics.cmb_completion_source_identity(),
    }
    try:
        contract = materialize_completion_contract(root)
        report["contract_artifact"] = {
            "path": f"contracts/{contract['contract_sha256']}.json",
            "sha256": evidence_assets._file_sha256(
                root / "completion-contract.json"
            ),
        }
        report["preflight_ready"] = True
        report["case_count"] = len(contract["cases"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        report["failure"] = diagnostics._diagnostic_failure(error)
    report["elapsed_seconds"] = time.monotonic() - started
    attempt = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    artifact = evidence_assets.write_json_artifact(
        root, f"preflight/{attempt}.json", report
    )
    evidence_assets.write_json_artifact(
        root, "completion-preflight.json", {**report, "attempt": artifact}
    )
    return report


def run_completion_point(
    root: Path, case_id: str, *, contract=None, model_records=None
) -> dict:
    """Execute one exact frozen point and retain every completed raw product.

    This reports quantitative full-window parity separately from complete
    engine certification. Missing axis/consumer evidence never becomes a
    passing completion decision, even when the angular comparison passes.
    """

    from copernican.lib.likelihoods.cmb import cmb
    from copernican.lib.likelihoods.cmb.runtime import evidence

    root = Path(root)
    records = (
        discover_cmb_model_records()
        if model_records is None
        else model_records
    )
    contract = (
        materialize_completion_contract(root, model_records=records)
        if contract is None
        else contract
    )
    requirement = contract["cases"][case_id]
    if requirement.get("reference_kind") != "camb":
        raise ValueError(
            "This execution requires a frozen comparable physical point"
        )
    binding = contract["bindings"][case_id]
    reference = evidence_assets.read_json_artifact(
        root, binding["reference_artifact"]
    )
    record = next(
        row
        for row in records
        if row.model_filename == binding["model_filename"]
    )
    runtime = completion_point_contract(record.plugin, binding["point"])
    if completion_physical_inputs(runtime) != binding["physical_inputs"]:
        raise ValueError(
            "Current physical declaration differs from its preflight binding"
        )
    request = binding["request"]
    if (
        request["numerical_overrides"]
        or request["ells"] != requirement["ells"]
        or request["spectra"] != requirement["surfaces"]
    ):
        raise ValueError(
            "Completion execution must retain the exact frozen request"
        )
    repository = Path(__file__).resolve().parents[3]
    dependency = diagnostics.cmb_completion_dependency_identity(
        {
            "runtime": binding["physical_inputs"],
            "declaration_sha256": binding["declaration_sha256"],
        },
        request,
        binding["numerical_plan"],
        {
            "reference_identity": requirement["reference_identity"],
            "reference_sha256": requirement["reference_sha256"],
        },
        requirement,
        repository_root=repository,
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    prefix = f"cases/{diagnostics._canonical_sha256(case_id)}/{stamp}"
    report = {
        "purpose": "ordinary_completion_point",
        "case_id": case_id,
        "accepted": False,
        "completion_accepted": False,
        "numerical_accepted": False,
        "parity_accepted": False,
        "source_identity": diagnostics.cmb_completion_source_identity(),
        "checkout_provenance": checkout_provenance(repository),
        "dependency_identity": dependency,
        "request": request,
        "requirement_sha256": diagnostics._canonical_sha256(requirement),
        "reference_artifact": binding["reference_artifact"],
        "physical_inputs": binding["physical_inputs"],
        "observations": [],
        "completion_contract_sha256": contract["contract_sha256"],
    }
    evidence_assets.write_json_artifact(
        root, f"{prefix}/producer.json", report
    )

    def retain(kind, products):
        """Persist borrowed arrays synchronously, before their owner exits."""

        entry = evidence_assets.write_array_products(
            root,
            f"{prefix}/products/{len(report['observations']):05d}-{kind}",
            products,
        )
        report["observations"].append({"kind": kind, "artifact": entry})
        if (
            kind != "spectrum_quadrature"
            or len(report["observations"]) % 64 == 0
        ):
            evidence_assets.write_json_artifact(
                root,
                f"{prefix}/observations.json",
                {
                    "products": report["observations"],
                },
            )

    started = time.perf_counter()
    try:
        with evidence.capture_completed_cmb_products(retain):
            cmb.compute_cmb_spectrum_from_contract(
                runtime,
                request["ells"],
                spectra=request["spectra"],
                workload="full_spectrum",
                solver="ccmbs",
            )
        result = cmb._LAST_CMB_RESULT.get()
        result.raise_for_failure()
        if result.solver_id != "ccmbs":
            raise ValueError(
                "An independent reference cannot substitute for CCMBS"
            )
        actual = diagnostics._ordinary_parity_payload(
            result,
            ell_values=request["ells"],
            spectra=request["spectra"],
        )
        actual["raw_units"] = "dimensionless"
        report["actual"] = diagnostics._jsonable(actual)
        report["output_artifact"] = evidence_assets.write_array_products(
            root,
            f"{prefix}/output",
            {
                "ell": numpy.asarray(request["ells"]),
                "raw_spectra": {
                    name: numpy.asarray(values["C_ell"])
                    for name, values in actual["spectra"].items()
                },
                "public_spectra": dict(result.spectra),
            },
        )
        report["numerical_accepted"] = True
        report["comparison"] = diagnostics.compare_cmb_completion_spectra(
            actual,
            reference,
            requirement,
        )
        report["parity_accepted"] = report["comparison"]["accepted"]
        previous_count = len(report["observations"])
        with evidence.capture_completed_cmb_products(retain):
            repeated = cmb.compute_cmb_spectrum_from_contract(
                runtime,
                request["ells"],
                spectra=request["spectra"],
                workload="full_spectrum",
                solver="ccmbs",
            )
        report["repeat"] = {
            "identical": all(
                numpy.array_equal(result.spectra[name], repeated[name])
                for name in request["spectra"]
            ),
            "new_numerical_products": len(report["observations"])
            - previous_count,
        }
        if (
            not report["repeat"]["identical"]
            or report["repeat"]["new_numerical_products"]
        ):
            report["parity_accepted"] = False
    except (Exception, KeyboardInterrupt) as error:  # DEVCOV_ALLOW_BROAD_ONCE
        # This explicit evidence boundary retains typed solver failure and
        # incomplete cancellation, together with all completed raw products.
        report["failure"] = diagnostics._diagnostic_failure(error)
        report["cancelled"] = isinstance(error, KeyboardInterrupt)
    # Failed numerical attempts may still contain useful completed products.
    # Keep their chain without manufacturing a completion certificate.
    sources = [
        row["artifact"]
        for row in report["observations"]
        if row["kind"] == "sources"
    ]
    projections = [
        row["artifact"]
        for row in report["observations"]
        if row["kind"] == "projection"
    ]
    quadratures = [
        row["artifact"]
        for row in report["observations"]
        if row["kind"] == "spectrum_quadrature"
    ]
    if sources and projections and quadratures:
        report["raw_products"] = evidence_assets.write_json_artifact(
            root,
            f"{prefix}/raw-products.json",
            {
                "schema_version": 2,
                "source": sources[-1],
                "projection": projections[-1],
                "quadratures": quadratures,
            },
        )
    report["incomplete_requirements"] = [
        "accepted ordinary full-window physical comparison",
        "independent physical-domain and numerical-axis verification",
        "durable graph and real-consumer verification",
    ]
    report["elapsed_seconds"] = time.perf_counter() - started
    report["performance"] = diagnostics._jsonable(
        cache.latest_cmb_performance_record()
    )
    path = root / prefix / "report.json"
    diagnostics.write_cmb_parity_matrix_report(report, path)
    retained = read_cmb_parity_matrix_report(path)
    entry = {
        "path": f"{prefix}/report.json",
        "sha256": evidence_assets._file_sha256(path),
    }
    evidence_assets.write_json_artifact(
        root,
        f"cases/{diagnostics._canonical_sha256(case_id)}/latest.json",
        entry,
    )
    return retained


def run_completion_baseline(root: Path) -> dict:
    """Retain a smallest ordinary LCDM request, including typed rejection."""

    from copernican.lib.likelihoods.cmb import cmb as cmb_api

    contract = materialize_completion_contract(root)
    case = contract["cases"]["model_lcdm.yml:initial"]
    binding = contract["bindings"]["model_lcdm.yml:initial"]
    reference = diagnostics._completion_artifact(
        root, binding["reference_artifact"]
    )
    ells = [2, 20, 100]
    indices = [reference["ell_values"].index(ell) for ell in ells]
    selected = {
        "ell_values": ells,
        "spectra": {
            name: {
                unit: [values[unit][index] for index in indices]
                for unit in ("C_ell", "D_ell")
            }
            for name, values in reference["spectra"].items()
            if name in ("TT", "TE", "EE")
        },
    }
    record = next(
        row
        for row in discover_cmb_model_records()
        if row.model_filename == _MODEL_FILENAME
    )
    runtime = completion_point_contract(record.plugin, "initial")
    started = time.monotonic()
    report = {
        "purpose": "diagnostic_baseline",
        "accepted": False,
        "engine_accepted": False,
        "completion_contract_sha256": contract["contract_sha256"],
        "source_identity": contract["source_identity"],
        "source_revision": contract["source_revision"],
        "reference": selected,
        "physical_inputs": binding["physical_inputs"],
        "physical_matching": (
            "Parameter defaults frozen; history "
            "equivalence must pass Slice Three."
        ),
        "request": {
            "ells": ells,
            "spectra": ["TT", "TE", "EE"],
            "workload": "full_spectrum",
            "numerical_overrides": {},
        },
        "contract_artifact": {
            "path": f"contracts/{contract['contract_sha256']}.json",
            "sha256": hashlib.sha256(
                (root / "completion-contract.json").read_bytes()
            ).hexdigest(),
        },
    }
    try:
        cmb_api.compute_cmb_spectrum_from_contract(
            runtime, ells, spectra=("TT", "TE", "EE"), workload="full_spectrum"
        )
        result = cmb_api._LAST_CMB_RESULT.get()
        if result is None or not result.success:
            raise RuntimeError(
                "Ordinary request did not retain a successful result"
            )
        report["actual"] = diagnostics._ordinary_parity_payload(
            result, ell_values=ells, spectra=("TT", "TE", "EE")
        )
        report["actual"]["raw_units"] = "dimensionless"
        report["comparison"] = diagnostics.compare_cmb_completion_spectra(
            report["actual"],
            selected,
            {
                "ells": ells,
                "surfaces": ["TT", "TE", "EE"],
                "temperature_K": case["temperature_K"],
            },
        )
        report["quantitative_decision"] = {
            "status": (
                "baseline_only"
                if report["comparison"]["accepted"]
                else "baseline_rejected"
            )
        }
    except Exception as error:
        report["failure"] = diagnostics._diagnostic_failure(error)
        report["quantitative_decision"] = {"status": "execution_failed"}
    report["elapsed_seconds"] = time.monotonic() - started
    diagnostics.write_cmb_parity_matrix_report(report, root / "baseline.json")
    return read_completion_baseline(root / "baseline.json")


def read_completion_baseline(path: Path) -> dict:
    """Reopen baseline and every frozen reference after producer exit."""

    report = read_cmb_parity_matrix_report(path)
    contract = diagnostics._completion_artifact(
        path.parent, report["contract_artifact"]
    )
    if (
        contract["source_identity"]
        != diagnostics.cmb_completion_source_identity()
    ):
        raise ValueError("Baseline source identity is stale")
    for case in contract["bindings"].values():
        if case.get("reference_kind") == "camb":
            diagnostics._completion_artifact(
                path.parent, case["reference_artifact"]
            )
    return report


def _summary(report: dict[str, Any], path: str | Path) -> str:
    """Return a compact machine-readable handoff result."""

    decision = report.get("quantitative_decision", {})
    return json.dumps(
        {
            "accepted": bool(report.get("accepted")),
            "decision": decision.get("status", "unknown"),
            "path": str(path),
            "report_sha256": report.get("report_sha256"),
        },
        sort_keys=True,
    )


def main(argv: list[str] | None = None) -> int:
    """Run or independently verify the explicit scientific handoff."""

    parser = argparse.ArgumentParser(
        description="Run or verify the fixed-LCDM CCMBS/CAMB handoff."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--development-checks",
        choices=tuple(DEVELOPMENT_WORKLOADS),
        help="run measured focused or bounded development tests",
    )
    group.add_argument(
        "--freeze-development-reference",
        action="store_true",
        help="explicitly regenerate compact CAMB background/history data",
    )
    group.add_argument(
        "--output",
        type=Path,
        help="write a new fixed-LCDM acceptance report",
    )
    group.add_argument(
        "--verify",
        type=Path,
        help="reload and verify an existing acceptance report",
    )
    group.add_argument(
        "--completion-point",
        help=(
            "execute one exact frozen comparable point "
            "and retain raw observations"
        ),
    )
    group.add_argument(
        "--completion-preflight",
        action="store_true",
        help="validate frozen case/reference bindings without a CCMBS solve",
    )
    group.add_argument(
        "--baseline",
        action="store_true",
        help="freeze completion references and run a diagnostic baseline",
    )
    group.add_argument(
        "--verify-baseline",
        type=Path,
        help="verify retained baseline and all independent references",
    )
    parser.add_argument(
        "--evidence-root",
        type=Path,
        default=_output_root() / "ccmbs-acceptance" / "completion-evidence",
    )
    args = parser.parse_args(argv)
    if args.completion_point:
        report = run_completion_point(
            args.evidence_root, args.completion_point
        )
        print(
            json.dumps(
                {
                    key: report.get(key)
                    for key in (
                        "case_id",
                        "numerical_accepted",
                        "parity_accepted",
                        "completion_accepted",
                        "elapsed_seconds",
                        "failure",
                    )
                }
            )
        )
        return 0 if report["completion_accepted"] else 1
    if args.completion_preflight:
        report = run_completion_preflight(args.evidence_root)
        print(
            json.dumps(
                {
                    "preflight_ready": report["preflight_ready"],
                    "scientific_acceptance": False,
                    "elapsed_seconds": report["elapsed_seconds"],
                    "failure": report.get("failure"),
                }
            )
        )
        return 0 if report["preflight_ready"] else 1
    if args.freeze_development_reference:
        path = camb_reference.write_lcdm_history_reference_fixture()
        print(path.relative_to(Path.cwd()))
        return 0
    if args.development_checks:
        report = run_development_checks(
            args.development_checks,
            args.evidence_root / f"development-{args.development_checks}.json",
        )
        print(
            json.dumps(
                {
                    key: report[key]
                    for key in (
                        "workload",
                        "passed",
                        "tests_run",
                        "elapsed_seconds",
                        "within_cost_target",
                    )
                }
            )
        )
        return 0 if report["passed"] else 1
    if args.baseline or args.verify_baseline:
        path = args.verify_baseline or args.evidence_root / "baseline.json"
        report = (
            read_completion_baseline(path)
            if args.verify_baseline
            else run_completion_baseline(args.evidence_root)
        )
        print(_summary(report, path))
        return 0 if args.verify_baseline else 1
    path = args.output or args.verify
    if args.verify is not None:
        report = read_cmb_parity_matrix_report(args.verify)
    else:
        report = run_fixed_lcdm_scientific_acceptance(args.output)
    print(_summary(report, path))
    return 0 if bool(report.get("accepted")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
