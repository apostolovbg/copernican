"""Compare scheduled vector backgrounds with the independent scalar owner."""

import inspect
import unittest
from unittest import mock

import numpy

from copernican.lib.likelihoods.cmb import cmb
from copernican.lib.likelihoods.cmb.runtime import execution, mode_evolution
from copernican.lib.likelihoods.cmb.runtime.stage_background import (
    sample_declared_stage_contexts,
)
from tests.copernican.lib import test_model_adapter


class StageBackgroundTestCase(unittest.TestCase):
    """Retain actual graphs and compare every field at the same coordinates."""

    def test_scheduled_background_matches_all_scalar_fields(self):
        class Captured(BaseException):
            """Finish after real background comparisons, before any ODE."""

        build = mode_evolution.build_declared_mode_evolution

        def compare(inputs):
            runtime = build(inputs)
            scalar = inspect.getclosurevars(
                inspect.unwrap(runtime.evolve_declared_modes_batched)
            ).nonlocals["_scalar_background_context"]
            count = len(inputs.active_grids["eta"])
            indices = numpy.asarray(
                [0, 1, count // 4, count // 2, count - 2, count - 1]
            )
            blends = numpy.asarray([0, 0.25, 0.5, 1, 0.75, 0])
            wave = float(inputs.k_values[-1])
            arguments = dict(
                background=inputs.background,
                grids=inputs.active_grids,
                declared_histories=inputs.active_declared_background_histories,
                contract=inputs.contract_or_params,
                physical=inputs.physical_params,
                numerics=inputs.numerics,
                indices=indices,
                blends=blends,
                wave=wave,
            )
            self.assertIsInstance(
                sample_declared_stage_contexts(**arguments), tuple
            )
            epochs, columns = sample_declared_stage_contexts(**arguments)
            for row, (index, blend) in enumerate(zip(indices, blends)):
                expected_eta, expected = scalar(
                    int(index), float(blend), k_value=wave
                )
                self.assertEqual(float(epochs[row]), expected_eta)
                self.assertEqual(set(columns), set(expected))
                for name, value in expected.items():
                    numpy.testing.assert_allclose(
                        columns[name][row],
                        value,
                        rtol=8 * numpy.finfo(float).eps,
                        atol=0,
                        err_msg=name,
                    )
            for change in (
                {"indices": indices[:-1]},
                {"blends": numpy.full(indices.size, numpy.nan)},
                {"blends": numpy.full(indices.size, 1.01)},
                {"indices": numpy.full(indices.size, count)},
            ):
                with self.assertRaises(ValueError):
                    sample_declared_stage_contexts(**{**arguments, **change})
            raise Captured()

        self.addCleanup(execution.cache.clear_cmb_caches)
        for filename in ("model_lcdm.yml", "model_ref_planck2018.yml"):
            with self.subTest(model=filename):
                execution.cache.clear_cmb_caches()
                plugin = (
                    test_model_adapter.DeclaredLCDMModelTestCase._build_plugin(
                        filename
                    )
                )
                with (
                    mock.patch.object(
                        execution,
                        "build_declared_mode_evolution",
                        side_effect=compare,
                    ),
                    self.assertRaises(Captured),
                ):
                    cmb.compute_cmb_spectrum_cached(
                        plugin,
                        tuple(plugin.INITIAL_GUESSES),
                        (2, 10, 20),
                        spectra=("TT", "TE", "EE"),
                    )


if __name__ == "__main__":
    unittest.main()
