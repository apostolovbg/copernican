"""Explicit, user-invoked fixed-LCDM scientific handoff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from copernican.lib.likelihoods.cmb.diagnostics import (
    discover_cmb_model_records,
    read_cmb_parity_matrix_report,
    run_fixed_lcdm_cmb_parity,
)
from tests.project.lib import camb_reference

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
    args = parser.parse_args(argv)
    path = args.output or args.verify
    if args.verify is not None:
        report = read_cmb_parity_matrix_report(args.verify)
    else:
        report = run_fixed_lcdm_scientific_acceptance(args.output)
    print(_summary(report, path))
    return 0 if bool(report.get("accepted")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
