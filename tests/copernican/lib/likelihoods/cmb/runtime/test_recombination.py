"""Independent physical limits for the native effective atom adapter."""

import unittest
from unittest import mock

import numpy

from copernican.lib.likelihoods.cmb.runtime import recombination


class RecombinationTestCase(unittest.TestCase):
    """Check thermal limits and reject invalid physical histories."""

    def test_sobolev_escape_has_thin_and_thick_limits(self):
        self.assertEqual(recombination._escape_probability(0.0), 1.0)
        self.assertAlmostEqual(
            recombination._escape_probability(1e-10), 1.0, places=9
        )
        self.assertAlmostEqual(
            recombination._escape_probability(1e5), 1e-5, places=14
        )
        for invalid in (-1.0, numpy.nan, numpy.inf):
            with self.assertRaises(ValueError):
                recombination._escape_probability(invalid)

    def test_neutral_gas_obeys_exact_adiabatic_cooling(self):
        """With no free electrons, T_m scales as a^-2 rather than a^-1."""

        scale = numpy.geomspace(0.01, 1.0, 65)
        actual = recombination.evolve_thermal_history(
            scale,
            numpy.full(scale.shape, 1e-13),
            hydrogen_today=0.0,
            temperature_today=2.7255,
            helium_ratio=0.0,
            initial_hydrogen=0.0,
            initial_helium=0.0,
            rtol=1e-9,
        )
        self.assertIsInstance(actual, recombination.ThermalHistory)
        numpy.testing.assert_allclose(
            actual.matter_temperature,
            2.7255 * scale[0] / scale**2,
            rtol=3e-7,
        )
        # Blackbody photoionization is exponentially suppressed, not zero.
        numpy.testing.assert_allclose(
            actual.hydrogen_fraction, 0.0, atol=1e-30
        )
        numpy.testing.assert_array_equal(actual.helium_electrons, 0.0)

    def test_compton_coupling_retains_the_radiation_temperature(self):
        """Strong scattering has its independent equilibrium limit."""

        scale = numpy.geomspace(1e-4, 1e-3, 33)
        options = dict(
            hydrogen_today=0.0,
            temperature_today=2.7255,
            helium_ratio=0.0,
            initial_hydrogen=1.0,
            initial_helium=1.0,
        )
        coupled = recombination.evolve_thermal_history(
            scale,
            numpy.full(scale.shape, 1e-15),
            **options,
        )
        numpy.testing.assert_allclose(
            coupled.matter_temperature, 2.7255 / scale, rtol=2e-5
        )
        with mock.patch.object(recombination, "_THOMSON", 0.0):
            free = recombination.evolve_thermal_history(
                scale,
                numpy.full(scale.shape, 1e-15),
                **options,
            )
        self.assertLess(
            free.matter_temperature[-1] / coupled.matter_temperature[-1],
            0.11,
        )

    def test_thermal_evolution_rejects_invalid_grids(self):
        for scale, hubble in (
            ([0.1, 0.1], [1.0, 1.0]),
            ([0.2, 0.1], [1.0, 1.0]),
            ([0.1, 0.2], [1.0, 0.0]),
            ([0.1, 0.2], [1.0, numpy.nan]),
            ([0.1], [1.0]),
        ):
            with self.assertRaisesRegex(ValueError, "ordered grids"):
                recombination.evolve_thermal_history(
                    scale,
                    hubble,
                    hydrogen_today=0.1,
                    temperature_today=2.7255,
                    helium_ratio=0.08,
                    initial_hydrogen=1.0,
                    initial_helium=1.0,
                )

    def test_discontinuous_hydrogen_switch_crosses_in_both_directions(self):
        """Each branch is integrated to the exact switch without smoothing."""
        log_time = numpy.linspace(0.0, 0.1, 101)
        scale = numpy.exp(log_time - 0.1)

        def derivative(
            scale_factor, _state, *, hydrogen_early_branch, **_options
        ):
            time = numpy.log(scale_factor) + 0.1
            speed = 2.0 if hydrogen_early_branch else 1.0
            return numpy.asarray((-speed if time < 0.05 else speed, 0.0, 0.0))

        with mock.patch.object(
            recombination, "_thermal_derivative", side_effect=derivative
        ):
            result = recombination.evolve_thermal_history(
                scale,
                numpy.ones_like(scale),
                hydrogen_today=1.0,
                temperature_today=1.0,
                helium_ratio=0.0,
                initial_hydrogen=0.99,
                initial_helium=0.0,
                rtol=1e-9,
            )
        expected = numpy.where(
            log_time <= 0.0025,
            0.99 - 2.0 * log_time,
            numpy.where(
                log_time <= 0.05,
                0.985 - (log_time - 0.0025),
                numpy.where(
                    log_time <= 0.0975,
                    0.9375 + (log_time - 0.05),
                    0.985 + 2.0 * (log_time - 0.0975),
                ),
            ),
        )
        numpy.testing.assert_allclose(
            result.hydrogen_fraction,
            expected,
            rtol=0,
            atol=2e-7,
            err_msg="The transition must preserve both physical branches.",
        )
        transition = 0.985
        self.assertGreater(result.hydrogen_fraction[2], transition)
        self.assertLess(result.hydrogen_fraction[3], transition)
        self.assertLess(result.hydrogen_fraction[97], transition)
        self.assertGreater(result.hydrogen_fraction[98], transition)

    def test_ionized_initial_state_resolves_sub_ulp_relaxation(self):
        """An early neutral fraction can relax faster than absolute log(a)."""
        scale = numpy.geomspace(0.01, 0.02, 9)
        equilibrium_neutral = 1e-8

        def derivative(_scale_factor, _state, *, hydrogen_neutral, **_options):
            return numpy.asarray(
                (1e16 * (hydrogen_neutral - equilibrium_neutral), 0.0, 0.0)
            )

        with mock.patch.object(
            recombination, "_thermal_derivative", side_effect=derivative
        ):
            result = recombination.evolve_thermal_history(
                scale,
                numpy.ones_like(scale),
                hydrogen_today=1.0,
                temperature_today=1.0,
                helium_ratio=0.0,
                initial_hydrogen=1.0,
                initial_helium=0.0,
            )
        self.assertEqual(result.hydrogen_fraction[0], 1.0)
        numpy.testing.assert_allclose(
            result.hydrogen_fraction[1:],
            1.0 - equilibrium_neutral,
            rtol=0,
            atol=1e-11,
        )

    def test_neutral_coordinate_preserves_atomic_derivatives(self):
        """The conditioned coordinate evaluates the same physical rates."""
        neutral = 2.0**-34
        state = (1.0 - neutral, 0.5, 0.99)
        options = {
            "hubble_si": 1e-13,
            "hydrogen_today": 0.26,
            "temperature_today": 2.7255,
            "helium_ratio": 0.08,
        }
        original = recombination._thermal_derivative(0.0003, state, **options)
        conditioned = recombination._thermal_derivative(
            0.0003, state, hydrogen_neutral=neutral, **options
        )
        numpy.testing.assert_array_equal(conditioned, original)

    def test_derivative_uses_logarithmic_expansion_time(self):
        """At T_m=T_gamma, expansion alone gives d(T_m/T_gamma)/d ln a=-1."""

        original_integrate = recombination.solve_ivp

        def integrate(derivative, interval, initial, **options):
            self.assertAlmostEqual(
                float(derivative(interval[0], initial)[2]), -1.0
            )
            return original_integrate(derivative, interval, initial, **options)

        scale = numpy.geomspace(1e-4, 1e-3, 33)
        with mock.patch.object(
            recombination, "solve_ivp", side_effect=integrate
        ):
            result = recombination.evolve_thermal_history(
                scale,
                numpy.full_like(scale, 1e-15),
                hydrogen_today=0.0,
                temperature_today=2.7255,
                helium_ratio=0.0,
                initial_hydrogen=1.0,
                initial_helium=0.0,
            )
        numpy.testing.assert_allclose(
            result.matter_temperature, 2.7255 / scale, rtol=2e-5
        )


if __name__ == "__main__":
    unittest.main()
