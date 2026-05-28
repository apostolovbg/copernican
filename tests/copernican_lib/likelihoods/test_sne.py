"""Smoke tests for copernican_lib.likelihoods.sne."""

import unittest

import numpy
import pandas

from copernican_lib.likelihoods import sne as module


class TestImportModule(unittest.TestCase):
    """Exercise the module import path."""

    def test_import_module(self) -> None:
        self.assertEqual(module.__name__, "copernican_lib.likelihoods.sne")


class PublicSymbolCoverageTestCase(unittest.TestCase):
    """Expose the SNe helper API to the coverage policy."""

    def test_public_symbols_are_exposed(self) -> None:
        self.assertTrue(hasattr(module, "SNeLike"))
        self.assertTrue(callable(module.SNeLike))
        self.assertTrue(hasattr(module, "compute_sne_intercept_delta"))
        self.assertTrue(callable(module.compute_sne_intercept_delta))

    def test_loglike_and_state_symbols_are_exposed(self) -> None:
        loglike = module.SNeLike.loglike
        state = module.SNeLike.state
        self.assertTrue(callable(loglike))
        self.assertTrue(hasattr(state, "__get__"))


class SNeInterceptMarginalizationTestCase(unittest.TestCase):
    """Exercise additive intercept marginalization behaviour."""

    @staticmethod
    def _make_observations(
        *,
        mu_obs: list[float] | tuple[float, ...],
        diag_errors: list[float] | tuple[float, ...] | None = None,
        covariance_matrix_inv: numpy.ndarray | None = None,
        requires_intercept: bool = False,
    ) -> pandas.DataFrame:
        zcmb = numpy.array([0.1, 0.2, 0.3], dtype=float)
        observations = pandas.DataFrame(
            {
                "zcmb": zcmb,
                "mu_obs": numpy.asarray(mu_obs, dtype=float),
                "e_mu_obs": (
                    numpy.asarray(diag_errors, dtype=float)
                    if diag_errors is not None
                    else numpy.array([0.1, 0.2, 0.3], dtype=float)
                ),
            }
        )
        if covariance_matrix_inv is not None:
            observations.attrs["covariance_matrix_inv"] = numpy.asarray(
                covariance_matrix_inv, dtype=float
            )
        if requires_intercept:
            observations.attrs["requires_sne_intercept_marginalization"] = True
            observations.attrs["sne_intercept_name"] = "Delta_mu"
        return observations

    @staticmethod
    def _constant_offset_model(z_values, *params):
        values = numpy.array([10.2, 11.2, 12.2], dtype=float)
        return values[: z_values.shape[0]].copy()

    def test_compute_sne_intercept_delta_with_full_covariance(self) -> None:
        residuals = numpy.array([-0.2, -0.2, -0.2], dtype=float)
        delta_mu = module.compute_sne_intercept_delta(
            residuals,
            covariance_matrix_inv=numpy.eye(3, dtype=float),
        )

        self.assertAlmostEqual(delta_mu, 0.2, places=8)
        self.assertTrue(numpy.allclose(residuals + delta_mu, 0.0))

    def test_compute_sne_intercept_delta_with_diagonal_errors(self) -> None:
        residuals = numpy.array([-0.2, -0.2, -0.2], dtype=float)
        delta_mu = module.compute_sne_intercept_delta(
            residuals,
            diag_errors=numpy.array([0.1, 0.2, 0.3], dtype=float),
        )

        self.assertAlmostEqual(delta_mu, 0.2, places=8)
        self.assertTrue(numpy.allclose(residuals + delta_mu, 0.0))

    def test_compute_sne_intercept_delta_rejects_invalid_inputs(self) -> None:
        with self.subTest("non-1d residuals"):
            with self.assertRaises(ValueError):
                module.compute_sne_intercept_delta(
                    numpy.array([[-0.2, -0.2, -0.2]], dtype=float),
                    covariance_matrix_inv=numpy.eye(3, dtype=float),
                )

        with self.subTest("covariance mismatch"):
            with self.assertRaises(ValueError):
                module.compute_sne_intercept_delta(
                    numpy.array([-0.2, -0.2, -0.2], dtype=float),
                    covariance_matrix_inv=numpy.eye(2, dtype=float),
                )

        with self.subTest("missing diagonal errors"):
            with self.assertRaises(ValueError):
                module.compute_sne_intercept_delta(
                    numpy.array([-0.2, -0.2, -0.2], dtype=float),
                )

    def test_sne_like_full_covariance_intercept_marginalization(self) -> None:
        observations = self._make_observations(
            mu_obs=[10.0, 11.0, 12.0],
            covariance_matrix_inv=numpy.eye(3, dtype=float),
            requires_intercept=True,
        )
        like = module.SNeLike(self._constant_offset_model, observations)

        loglike = like.loglike(())
        state = like.state

        self.assertAlmostEqual(loglike, 0.0, places=8)
        self.assertAlmostEqual(state["chi2"], 0.0, places=8)
        self.assertTrue(state["metadata"]["sne_intercept_marginalized"])
        self.assertAlmostEqual(
            state["metadata"]["sne_intercept_delta_mu"],
            0.2,
            places=8,
        )
        self.assertEqual(state["metadata"]["sne_intercept_name"], "Delta_mu")
        self.assertEqual(state["metadata"]["covariance"], "full")

    def test_sne_like_diagonal_fallback_intercept_marginalization(
        self,
    ) -> None:
        observations = self._make_observations(
            mu_obs=[10.0, 11.0, 12.0],
            diag_errors=[0.1, 0.2, 0.3],
            requires_intercept=True,
        )
        like = module.SNeLike(self._constant_offset_model, observations)

        loglike = like.loglike(())
        state = like.state

        self.assertAlmostEqual(loglike, 0.0, places=8)
        self.assertAlmostEqual(state["chi2"], 0.0, places=8)
        self.assertTrue(state["metadata"]["sne_intercept_marginalized"])
        self.assertAlmostEqual(
            state["metadata"]["sne_intercept_delta_mu"],
            0.2,
            places=8,
        )
        self.assertEqual(state["metadata"]["sne_intercept_name"], "Delta_mu")
        self.assertEqual(state["metadata"]["covariance"], "diagonal")

    def test_ordinary_sne_like_does_not_marginalize_intercept(self) -> None:
        observations = self._make_observations(
            mu_obs=[10.0, 11.0, 12.0],
            diag_errors=[0.1, 0.2, 0.3],
            requires_intercept=False,
        )
        like = module.SNeLike(self._constant_offset_model, observations)

        loglike = like.loglike(())

        self.assertTrue(numpy.isfinite(loglike))
        self.assertFalse(like.state["metadata"]["sne_intercept_marginalized"])
        self.assertGreater(like.state["chi2"], 0.0)

    def test_shape_residuals_survive_intercept_marginalization(self) -> None:
        observations = self._make_observations(
            mu_obs=[10.0, 11.0, 12.0],
            covariance_matrix_inv=numpy.eye(3, dtype=float),
            requires_intercept=True,
        )

        def shaped_model(z_values, *params):
            values = numpy.array([10.2, 11.2, 12.4], dtype=float)
            return values[: z_values.shape[0]].copy()

        like = module.SNeLike(shaped_model, observations)

        loglike = like.loglike(())

        self.assertTrue(numpy.isfinite(loglike))
        self.assertTrue(like.state["metadata"]["sne_intercept_marginalized"])
        self.assertGreater(like.state["chi2"], 0.0)


if __name__ == "__main__":
    unittest.main()
