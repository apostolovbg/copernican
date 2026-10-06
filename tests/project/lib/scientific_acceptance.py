"""Explicit, user-invoked fixed-LCDM scientific handoff."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from copernican.lib.likelihoods.cmb import diagnostics
from copernican.lib.likelihoods.cmb.diagnostics import (
    discover_cmb_model_records,
    read_cmb_parity_matrix_report,
    run_fixed_lcdm_cmb_parity,
)
from copernican.workflow import _output_root
from tests.project.lib import camb_reference
from tests.project.lib.test_ccmbs_production_graph import _read_source_revision

_MODEL_FILENAME = "model_lcdm.yml"


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


def completion_reference_contract(contract: dict[str, Any]) -> dict:
    """Declare all CAMB physical inputs, including massless and DE limits."""

    params = dict(contract["param_map"])
    mass = float(params.get("sum_mnu", params.get("mnu", 0.0)))
    if mass == 0.0:
        params.update(mnu=0.0, num_massive_neutrinos=0)
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


def _retain(path: Path, payload: dict) -> dict:
    """Persist JSON with a portable relative artifact identity."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(diagnostics._jsonable(payload), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return {
        "path": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def materialize_completion_contract(root: Path) -> dict:
    """Freeze exact points, independent references, and applicability.

    Independent reference generation is explicit, never ordinary discovery.
    Resolved histories remain a mandatory comparison in Slices Three/Four;
    equality of cosmological parameter maps cannot establish that equality.
    """

    records = discover_cmb_model_records()
    cases = {}
    for record in records:
        if record.model_filename not in diagnostics.CMB_COMPLETION_POINTS:
            continue
        if not record.ready:
            raise ValueError(f"Required declaration is unavailable: {record}")
        plugin = record.plugin
        for point in diagnostics.CMB_COMPLETION_POINTS[record.model_filename]:
            runtime = completion_point_contract(plugin, point)
            physical = diagnostics._jsonable(
                {
                    key: runtime.get(key, {})
                    for key in (
                        "model_parameters",
                        "param_map",
                        "background",
                        "perturbations",
                    )
                }
            )
            case = {
                "model_filename": record.model_filename,
                "point": point,
                "physical_inputs": physical,
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
                reference_contract = completion_reference_contract(runtime)
                path = root / f"reference-{record.model_filename}-{point}.json"
                reference = None
                if path.exists():
                    candidate = json.loads(path.read_text())
                    saved = candidate.pop("fixture_sha256", None)
                    if (
                        saved
                        == camb_reference.reference_fixture_sha256(candidate)
                        and candidate.get("contract") == reference_contract
                        and candidate.get("reference_identity")
                        == camb_reference.CAMB_REFERENCE_IDENTITY
                    ):
                        candidate["fixture_sha256"] = saved
                        reference = candidate
                if reference is None:
                    reference = camb_reference.build_completion_camb_reference(
                        reference_contract,
                        model_name=record.model_filename,
                        fixed_point=point,
                    )
                artifact = _retain(path, reference)
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
    contract = {
        "schema_version": 1,
        "cases": cases,
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
    _retain(root / "completion-contract.json", contract)
    return contract


def run_completion_baseline(root: Path) -> dict:
    """Retain a smallest ordinary LCDM request, including typed rejection."""

    from copernican.lib.likelihoods.cmb import cmb as cmb_api

    contract = materialize_completion_contract(root)
    case = contract["cases"]["model_lcdm.yml:initial"]
    reference = diagnostics._completion_artifact(
        root, case["reference_artifact"]
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
        "physical_inputs": case["physical_inputs"],
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
            "path": "completion-contract.json",
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
    for case in contract["cases"].values():
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
