# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.

"""Deterministic CMB solver fixture for recurring non-physics tests."""

from __future__ import annotations

import numpy

from copernican.lib.likelihoods.cmb.contracts import CMBResult


class SyntheticCmbSolver:
    """Exercise CMB likelihood plumbing without a full CCMBS evolution."""

    solver_id = "synthetic_project_cmb"
    solver_label = "Synthetic project CMB solver"

    def __init__(self):
        self.evaluate_calls = 0

    def capabilities(self):
        """Return minimal deterministic solver capabilities."""

        return {"implementation": "test_double", "accuracy_tiers": ()}

    def prepare(self, contract):
        """Retain the declared contract for the test evaluation."""

        return contract

    def evaluate(self, prepared, ells, *, spectra, workload):
        """Return finite deterministic spectra on the requested ell grid."""

        del prepared, workload
        self.evaluate_calls += 1
        ell_values = numpy.asarray(tuple(ells), dtype=float)
        values = {
            str(name): 1200.0 - 1.25 * (ell_values - 20.0) for name in spectra
        }
        return CMBResult(
            spectra=values[str(spectra[0])] if len(spectra) == 1 else values,
            requested_ells=tuple(int(value) for value in ell_values),
            requested_spectra=tuple(str(name) for name in spectra),
            diagnostics={"synthetic": True},
            solver_id=self.solver_id,
            solver_label=self.solver_label,
        )

    def evaluate_batch(self, prepared, ells, *, spectra, workload):
        """Evaluate prepared contracts in input order."""

        return tuple(
            self.evaluate(
                item,
                ells,
                spectra=spectra,
                workload=workload,
            )
            for item in prepared
        )

    def cleanup(self):
        """Release no resources because the solver is in memory only."""

        return None
