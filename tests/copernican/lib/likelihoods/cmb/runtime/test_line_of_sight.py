"""Direct tests for line-of-sight grid ownership."""

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy

from copernican.lib.likelihoods.cmb.runtime import line_of_sight


class LineOfSightTestCase(unittest.TestCase):
    """Exercise nested phase-aware line-of-sight grids directly."""

    def test_projection_ladder_is_nested_and_materially_refined(self):
        """Fine projection nodes retain every coarse physical node."""

        background = SimpleNamespace(
            visibility_of_eta=lambda eta: numpy.exp(
                -numpy.square(numpy.asarray(eta, dtype=float) - 0.5)
            )
        )
        fine, coarse = line_of_sight.build_phase_aware_projection_ladder(
            numpy.linspace(0.0, 1.0, 8),
            background=background,
            k_max=2.0,
            phase_points_per_cycle=4.0,
            minimum_nodes=8,
            maximum_nodes=16,
        )
        self.assertGreater(fine.size, coarse.size)
        self.assertEqual(
            line_of_sight.build_phase_aware_projection_ladder(
                numpy.linspace(0.0, 1.0, 8),
                background=background,
                k_max=2.0,
                phase_points_per_cycle=4.0,
                minimum_nodes=8,
                maximum_nodes=16,
            )[0].size,
            fine.size,
        )
        self.assertTrue(numpy.all(numpy.isin(coarse, fine)))

    def test_background_sampling_returns_aligned_coordinate_histories(self):
        """Sample line-of-sight backgrounds on aligned coordinate grids."""

        eta = numpy.asarray((1.0, 2.0, 3.0), dtype=float)
        background = SimpleNamespace(
            sample=lambda values: {
                "a": numpy.asarray((0.1, 0.2, 0.3)),
                "z": numpy.asarray((9.0, 4.0, 7.0 / 3.0)),
                "H": numpy.asarray((10.0, 8.0, 6.0)),
                "tau": numpy.asarray((3.0, 2.0, 1.0)),
                "tau_dot": numpy.asarray((-2.0, -1.0, -0.5)),
                "visibility": numpy.asarray((0.1, 0.3, 0.2)),
                "chi": numpy.asarray((30.0, 20.0, 10.0)),
                "angular_diameter_distance": numpy.asarray((3.0, 4.0, 3.0)),
                "sound_speed": numpy.asarray((0.5, 0.4, 0.3)),
                "baryon_sound_speed_sq": numpy.asarray((0.1, 0.2, 0.3)),
            }
        )
        physical_params = SimpleNamespace(Omega_b0=0.05, Omega_gamma0=1.0e-4)
        with mock.patch.object(
            line_of_sight,
            "_resolve_declared_background_context",
            return_value={},
        ):
            grids, declared, rates = line_of_sight.sample_eta_background_grids(
                eta,
                background=background,
                physical_params=physical_params,
                contract_or_params={},
            )
            self.assertEqual(declared, {})
            self.assertEqual(
                line_of_sight.sample_eta_background_grids(
                    eta,
                    background=background,
                    physical_params=physical_params,
                    contract_or_params={},
                )[1],
                declared,
            )
            self.assertTrue(numpy.array_equal(grids["eta"], eta))
            self.assertTrue(set(rates).issubset(set(grids) | {"eta"}))
            required_rates = {"a", "eta", "Hconf", "visibility"}
            self.assertTrue(required_rates.issubset(rates))
            self.assertTrue(
                all(values.shape == eta.shape for values in rates.values())
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
