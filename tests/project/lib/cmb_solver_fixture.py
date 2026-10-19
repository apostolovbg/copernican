# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.

"""Deterministic CMB solver fixture for recurring non-physics tests."""

from __future__ import annotations

import numpy
import pandas

from copernican.lib.likelihoods.cmb.contracts import CMBResult


class SyntheticCmbSolver:
    """Exercise CMB likelihood plumbing without a full CCMBS evolution."""

    solver_id = "synthetic_project_cmb"
    solver_label = "Synthetic project CMB solver"

    def __init__(self, *, spectrum_offsets=None):
        self.evaluate_calls = 0
        self.requests = []
        self.spectrum_offsets = dict(spectrum_offsets or {})

    def capabilities(self):
        """Return minimal deterministic solver capabilities."""

        return {"implementation": "test_double", "accuracy_tiers": ()}

    def prepare(self, contract):
        """Retain the declared contract for the test evaluation."""

        return contract

    def evaluate(self, prepared, ells, *, spectra, workload):
        """Return finite deterministic spectra on the requested ell grid."""

        del prepared
        self.evaluate_calls += 1
        requested_ells = tuple(ells)
        ell_values = numpy.asarray(requested_ells, dtype=float)
        self.requests.append((requested_ells, tuple(spectra), workload))
        values = {
            str(name): 1200.0
            - 1.25 * (ell_values - 20.0)
            + self.spectrum_offsets.get(str(name), 0.0)
            for name in spectra
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


def explicit_cmb_observations():
    """Independent four-row data with repeated ells and cross covariance.

    With offsets TT=0, TE=-1205, EE=-1190, expected theory in table order
    is [-17.5, 1200, -2.5, 1187.5]; residuals are [1, -2, 3, 0.5].
    """

    observations = pandas.DataFrame(
        {
            "ell": [30, 20, 30, 30],
            "spectrum": ["TE", "TT", "EE", "TT"],
            "Dl_obs": [-16.5, 1198.0, 0.5, 1188.0],
        }
    )
    observations.attrs.update(
        {
            "dataset_id": "independent-cmb-four-rows",
            "dataset_name": "Small CMB",
            "covariance_matrix_inv": numpy.asarray(
                [
                    [2, 0.25, 0, 0.1],
                    [0.25, 3, 0.4, 0],
                    [0, 0.4, 4, 0.2],
                    [0.1, 0, 0.2, 2],
                ]
            ),
        }
    )
    return observations
