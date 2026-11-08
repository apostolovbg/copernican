"""Explicit ordinary-route numerical cost/reuse evidence, without CAMB parity.

This module is not a discovery test. It runs real final-accuracy LCDM and
Planck requests, retaining every outcome rather than hiding failed attempts.
Scientific reference agreement remains the separate scientific acceptance.
"""

import argparse
import hashlib
import json
import logging
import platform
import time
from collections.abc import Mapping
from pathlib import Path
from unittest import mock

import numpy

from copernican.lib.likelihoods.cmb import cmb, diagnostics
from copernican.lib.likelihoods.cmb.orchestrators import ccmbs
from copernican.lib.likelihoods.cmb.runtime import cache
from copernican.workflow import _output_root
from tests.copernican.lib.test_model_adapter import DeclaredLCDMModelTestCase


def _compare_observations(reference, candidate):
    """Compare common multipoles within both propagated absolute budgets.

    Adaptive requests may accept different meshes. The triangle bound is
    the scientific consistency criterion; equal discretizations additionally
    agree to roundoff. Check both raw and public products so serialization
    or conversion drift cannot hide behind one of the two representations.
    """

    common = sorted(set(reference["ells"]) & set(candidate["ells"]))
    first_indices = [reference["ells"].index(ell) for ell in common]
    second_indices = [candidate["ells"].index(ell) for ell in common]
    if not common:
        return ["no_common_multipoles"]
    problems = []
    for name in ("TT", "TE", "EE"):
        try:
            first = numpy.asarray(
                reference["raw_spectra"][name], dtype=numpy.longdouble
            )[first_indices]
            second = numpy.asarray(
                candidate["raw_spectra"][name], dtype=numpy.longdouble
            )[second_indices]
            public_first = numpy.asarray(reference["products"][name])[
                first_indices
            ]
            public_second = numpy.asarray(candidate["products"][name])[
                second_indices
            ]
            bound = sum(
                float(
                    row["runtime_envelope"]["observable_error_budget"][
                        "products"
                    ][name]["absolute_error_bound"]
                )
                for row in (reference, candidate)
            )
            factor = numpy.ones_like(first)
            numpy.divide(
                abs(public_first), abs(first), out=factor, where=first != 0
            )
            numpy.divide(
                abs(public_second),
                abs(second),
                out=factor,
                where=(first == 0) & (second != 0),
            )
            if (
                not numpy.isfinite(bound)
                or bound < 0
                or not numpy.all(
                    numpy.isfinite(first) & numpy.isfinite(second)
                )
                or numpy.any(abs(first - second) > bound)
                or numpy.any(
                    abs(public_first - public_second) > bound * factor
                )
            ):
                problems.append(f"{name}:disagreement")
        except (KeyError, TypeError, ValueError, IndexError):
            problems.append(f"{name}:unreadable_comparison")
    return problems


def assess_numerical_workload(operations):
    """Require complete products, numerical closure and ordered reuse proof."""

    required = {
        "cold",
        "exact_repeat",
        "warm",
        "changed_request",
        "subset",
        "changed_parameter",
        "batch",
    }
    rows = {row["name"]: row for row in operations}
    problems = []
    if set(rows) != required or len(rows) != len(operations):
        problems.append("missing_or_duplicate_operations")
    for name, row in rows.items():
        if not row.get("completed"):
            problems.append(f"{name}:failed")
            continue
        if name == "batch":
            if row.get("ordered_success") != [True, False, True]:
                problems.append("batch:failure_isolation")
            if row.get("indices") != [0, 1, 2]:
                problems.append("batch:ordering")
            try:
                for index in (0, 2):
                    for product in ("TT", "TE", "EE"):
                        reference = numpy.asarray(
                            rows["cold"]["raw_spectra"][product],
                            dtype=numpy.longdouble,
                        )
                        actual = numpy.asarray(
                            row["results"][index]["raw_spectra"][product],
                            dtype=numpy.longdouble,
                        )
                        if not numpy.array_equal(reference, actual):
                            problems.append(
                                f"batch:{index}:{product}:disagreement"
                            )
            except (KeyError, TypeError, ValueError, IndexError):
                problems.append("batch:unreadable_products")
            continue
        products = row.get("products", {})
        if set(products) != {"TT", "TE", "EE"} or any(
            not numpy.all(numpy.isfinite(values))
            for values in products.values()
        ):
            problems.append(f"{name}:invalid_products")
        envelope = row.get("runtime_envelope", {})
        budget = envelope.get("observable_error_budget", {})
        if not budget.get("converged"):
            problems.append(f"{name}:observable_budget")
        axes = {
            "background",
            "hierarchy_depth",
            "evolution",
            "source",
            "projection",
            "k",
        }
        if envelope.get("momentum_point_count", 0):
            axes.add("momentum_q")
        if not budget.get("products") or any(
            not axes.issubset(product.get("axis_relative_bounds", {}))
            for product in budget.get("products", {}).values()
        ):
            problems.append(f"{name}:missing_error_axes")
        if not envelope.get("production_scalar_k_convergence", {}).get(
            "converged"
        ):
            problems.append(f"{name}:k_refinement")
    if "cold" in rows and rows["cold"].get("completed"):
        reference = rows["cold"]["products"]
        for name in ("exact_repeat", "warm", "changed_request", "subset"):
            if name not in rows or not rows[name].get("completed"):
                continue
            problems.extend(
                f"{name}:{problem}"
                for problem in _compare_observations(rows["cold"], rows[name])
            )
        if "changed_parameter" in rows and rows["changed_parameter"].get(
            "completed"
        ):
            if all(
                numpy.array_equal(reference[product], values)
                for product, values in rows["changed_parameter"][
                    "products"
                ].items()
            ):
                problems.append("changed_parameter:unchanged_products")
    if "exact_repeat" in rows and rows["exact_repeat"].get("completed"):
        if (
            rows["exact_repeat"].get("performance", {}).get("cache_state")
            != "exact_cache_hit"
        ):
            problems.append("exact_repeat:cache_miss")
    return {"passed": not problems, "problems": problems}


def _encode(value):
    """Serialize retained numerical arrays and immutable mappings exactly."""

    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, numpy.ndarray):
        return value.tolist()
    if isinstance(value, numpy.generic):
        scalar = value.item()
        # Extended precision has no equivalent built-in Python scalar.
        # Preserve its decimal digits instead of recursively returning it.
        return str(value) if isinstance(scalar, numpy.generic) else scalar
    raise TypeError(f"Unsupported evidence type: {type(value).__name__}")


def run_numerical_workload(model_filename, persist):
    """Measure ordinary requests, cache pressure and invalidation."""

    cache.clear_cmb_caches()
    plugin = DeclaredLCDMModelTestCase._build_plugin(model_filename)
    parameters = tuple(plugin.INITIAL_GUESSES)
    changed = (parameters[0] * 1.002, *parameters[1:])
    operations = []
    for name, ells, values in (
        ("cold", (2, 10, 20), parameters),
        ("exact_repeat", (2, 10, 20), parameters),
        ("warm", (2, 10, 20), parameters),
        ("changed_request", (2, 3, 10, 20), parameters),
        ("subset", (20, 2), parameters),
        ("changed_parameter", (2, 10, 20), changed),
    ):
        if name == "warm":
            cache.clear_cmb_result_caches()
        started = time.perf_counter()
        row = {"name": name, "ells": ells, "parameters": values}
        try:
            # Pressure the global LRU during the actual cold refinement.
            # Request-owned retention must keep the necessary raw work.
            limit = (
                16 if name == "cold" else cache._CMB_SOURCE_HISTORY_CACHE.limit
            )
            with mock.patch.object(
                cache._CMB_SOURCE_HISTORY_CACHE, "limit", limit
            ):
                products = cmb.compute_cmb_spectrum_cached(
                    plugin,
                    values,
                    ells,
                    spectra=("TT", "TE", "EE"),
                    workload="full_spectrum",
                )
            result = cmb._LAST_CMB_RESULT.get()
            row.update(
                completed=True,
                products=products,
                raw_spectra=result.raw_spectra,
                cache_provenance=result.cache_provenance,
                phase_timings=result.phase_timings,
                performance=cache.latest_cmb_performance_record(),
                runtime_envelope=ccmbs._LAST_DECLARED_RUNTIME_EVIDENCE.get(),
                cache_stats=cache.cmb_cache_stats(),
            )
        except Exception as error:  # Preserve a failed explicit attempt.
            row.update(
                completed=False,
                failure=str(error),
                failure_type=type(error).__name__,
            )
        row["elapsed_seconds"] = time.perf_counter() - started
        operations.append(row)
        persist(operations)
        if not row["completed"]:
            return operations
    contract = plugin.get_cmb_declared_runtime(parameters)
    contract = {**contract, "_background_provider": plugin}
    started = time.perf_counter()
    batch = cmb.compute_cmb_spectrum_batch(
        (contract, {}, contract),
        (2, 10, 20),
        background_provider=plugin,
        requested_spectra=("TT", "TE", "EE"),
    )
    operations.append(
        {
            "name": "batch",
            "completed": True,
            "indices": [item.index for item in batch],
            "ordered_success": [item.failure is None for item in batch],
            "results": [item.to_dict() for item in batch],
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    persist(operations)
    return operations


def main(argv=None):
    """Persist each completed operation and report numerical closure only."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assess-only",
        type=Path,
        help="Read retained measurements without rerunning the solver",
    )
    parser.add_argument(
        "--model",
        action="append",
        choices=("model_lcdm.yml", "model_ref_planck2018.yml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(_output_root())
        / "ccmbs-acceptance/numerical-runtime.json",
    )
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.assess_only is not None:
        if args.assess_only.resolve() == args.output.resolve():
            parser.error("Readback output must not replace producer evidence")
        contents = args.assess_only.read_bytes()
        producer = json.loads(contents)
        current = diagnostics.cmb_completion_source_identity()
        models = {
            name: assess_numerical_workload(row.get("operations", []))
            for name, row in producer.get("models", {}).items()
        }
        valid = (
            bool(models)
            and producer.get("kind") == "ordinary_numerical_runtime_evidence"
            and (
                producer.get("source_identity")
                == producer.get("final_source_identity")
                == current
            )
        )
        result = {
            "kind": "numerical_runtime_readback",
            "producer_file": args.assess_only.name,
            "producer_sha256": hashlib.sha256(contents).hexdigest(),
            "producer_assessment": producer.get("passed"),
            "producer_runner_sha256": producer.get("runner_sha256"),
            "reader_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "source_sha256": current["sha256"],
            "source_current": valid,
            "models": models,
            "passed": valid and all(row["passed"] for row in models.values()),
        }
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(
            json.dumps(
                {"passed": result["passed"], "output": str(args.output)}
            )
        )
        return 0 if result["passed"] else 1
    report = {
        "kind": "ordinary_numerical_runtime_evidence",
        "raw_numeric_encoding": "extended_precision_decimal_strings",
        "source_identity": diagnostics.cmb_completion_source_identity(),
        "runner_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "machine": {
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "models": {},
    }

    def persist():
        """Atomically retain each operation, including failures."""

        report["final_source_identity"] = (
            diagnostics.cmb_completion_source_identity()
        )
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(report, default=_encode, indent=2) + "\n"
        )
        temporary.replace(args.output)

    for filename in args.model or (
        "model_lcdm.yml",
        "model_ref_planck2018.yml",
    ):
        report["models"][filename] = {}

        def retain(operations):
            """Retain the currently evaluated model's actual outcomes."""

            report["models"][filename]["operations"] = operations
            persist()

        operations = run_numerical_workload(filename, retain)
        report["models"][filename]["assessment"] = assess_numerical_workload(
            operations
        )
        persist()
    report["passed"] = all(
        row["assessment"]["passed"] for row in report["models"].values()
    ) and (report["source_identity"] == report["final_source_identity"])
    persist()
    print(json.dumps({"passed": report["passed"], "output": str(args.output)}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
