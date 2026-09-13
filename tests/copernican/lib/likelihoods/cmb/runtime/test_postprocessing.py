"""Focused acceptance tests for declared CMB post-processing evidence."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy

from copernican.lib.likelihoods.cmb.orchestrators import ccmbs
from copernican.lib.likelihoods.cmb.runtime import cache
from copernican.lib.likelihoods.cmb.runtime.postprocessing import (
    build_postprocessing_evidence,
)


class PostProcessingEvidenceTestCase(unittest.TestCase):
    """Validate complete surface accounting without a solver integration."""

    def test_public_builder_is_exposed(self) -> None:
        """The post-processing boundary is part of the runtime API."""

        self.assertTrue(callable(build_postprocessing_evidence))

    def _evidence(self, *, lensed: bool = False):
        """Build one deterministic complete observable surface."""

        ell = numpy.arange(4, dtype=int)
        k_grid = numpy.asarray((1.0e-4, 1.0e-3, 1.0e-2), dtype=float)
        transfer = {
            "temperature": numpy.ones((4, 3)),
            "polarization_e": numpy.full((4, 3), 0.5),
        }
        base = {
            "TT": numpy.asarray((0.0, 1.0, 2.0, 1.5)),
            "TE": numpy.asarray((0.0, 0.2, -0.1, 0.05)),
            "EE": numpy.asarray((0.0, 0.4, 0.3, 0.2)),
            "PP": numpy.asarray((0.0, 0.01, 0.02, 0.03)),
        }
        outputs = dict(base)
        requested = tuple(base)
        if lensed:
            outputs.update(
                {
                    "lensed_TT": base["TT"],
                    "lensed_TE": base["TE"],
                    "lensed_EE": base["EE"],
                    "lensed_BB": numpy.asarray((0.0, 0.01, 0.02, 0.03)),
                }
            )
            requested = tuple(outputs)
        return build_postprocessing_evidence(
            transfer_components=transfer,
            unlensed_spectra=base,
            output_spectra=outputs,
            requested_spectra=requested,
            spectrum_availability={"BB": "physical_zero"},
            ell_grid=ell,
            k_grid=k_grid,
            lensed=lensed,
            lensing_cache={"reused": False},
        )

    def test_complete_surface_dependencies_and_digests(self) -> None:
        """Every requested surface has dependencies and raw hashes."""

        evidence = self._evidence(lensed=True)
        self.assertTrue(evidence["accepted"])
        self.assertTrue(evidence["complete"])
        self.assertEqual(
            evidence["surface_dependencies"]["lensed_BB"],
            ("TT", "TE", "EE", "BB", "PP"),
        )
        self.assertEqual(evidence["surface_units"]["TT"], "muK^2")
        self.assertEqual(evidence["surface_units"]["PP"], "dimensionless")
        self.assertEqual(
            set(evidence["raw_transfer_components"]),
            {
                "temperature",
                "polarization_e",
            },
        )
        self.assertRegex(evidence["sha256"], r"^[0-9a-f]{64}$")
        for record in evidence["surface_records"].values():
            self.assertTrue(record["finite"])
            self.assertTrue(record["accepted"])

    def test_digest_is_stable_and_cross_signs_are_retained(self) -> None:
        """The evidence hash is repeatable without erasing cross signs."""

        first = self._evidence()
        second = self._evidence()
        self.assertEqual(first["sha256"], second["sha256"])
        self.assertTrue(first["surface_records"]["TE"]["sign_changes"])

    def test_negative_auto_surface_is_rejected(self) -> None:
        """A materially negative auto spectrum cannot pass post-processing."""

        evidence = self._evidence()
        bad = dict(evidence)
        outputs = {
            "TT": numpy.asarray((0.0, -2.0, 1.0, 1.0)),
        }
        bad = build_postprocessing_evidence(
            transfer_components={"temperature": numpy.ones((4, 3))},
            unlensed_spectra=outputs,
            output_spectra=outputs,
            requested_spectra=("TT",),
            spectrum_availability={},
            ell_grid=numpy.arange(4),
            k_grid=numpy.asarray((1.0e-4, 1.0e-3)),
            lensed=False,
        )
        self.assertFalse(bad["accepted"])
        self.assertIn("TT", bad["issues"][0])

    def test_exact_lensing_repeat_reuses_remapping(self) -> None:
        """An identical lensed request reuses one remapping computation."""

        cache.clear_cmb_result_caches()
        ell = numpy.arange(0, 8, dtype=int)
        inputs = {
            "TT": numpy.asarray((0.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.5)),
            "TE": numpy.asarray(
                (0.0, 0.2, -0.1, 0.1, -0.05, 0.03, -0.02, 0.01)
            ),
            "EE": numpy.asarray((0.0, 0.5, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1)),
            "BB": numpy.zeros(8),
            "PP": numpy.asarray(
                (0.0, 0.01, 0.02, 0.03, 0.02, 0.01, 0.005, 0.002)
            ),
        }
        with mock.patch.object(
            ccmbs,
            "_lensed_cls",
            wraps=ccmbs._lensed_cls,
        ) as remapper:
            first = ccmbs._assemble_exact_lensed_spectra(inputs, ell)
            second = ccmbs._assemble_exact_lensed_spectra(inputs, ell)

        self.assertEqual(remapper.call_count, 1)
        self.assertEqual(
            cache.cmb_cache_stats()["declared_lensing"]["hits"],
            1,
        )
        for name in first:
            numpy.testing.assert_array_equal(first[name], second[name])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
