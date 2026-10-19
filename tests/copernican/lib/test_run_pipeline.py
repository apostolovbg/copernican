"""Behavior and smoke tests for the shared application run pipeline."""

import contextlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy
import pandas

from copernican.lib import run_pipeline as module
from copernican.lib.model_selection import build_comparison_request
from copernican.lib.utils import set_random_seed
from copernican.samplers import sampler_mcmc
from tests.copernican.samplers.test_sampler_mcmc import _build_model_plugin
from tests.project.lib.cmb_solver_fixture import (
    SyntheticCmbSolver,
    explicit_cmb_observations,
)


def bounded_pipeline_inputs():
    """Return real free/fixed plugins and independent tiny observations."""

    control = _build_model_plugin("model_lcdm.yml")
    reference = _build_model_plugin("model_ref_planck2018.yml")
    control.MODEL_FILENAME = "model_lcdm.yml"
    reference.MODEL_FILENAME = "model_ref_planck2018.yml"
    sne = pandas.DataFrame(
        {"zcmb": [0.01, 0.02], "mu_obs": [33.0, 35.0], "e_mu_obs": [0.2, 0.2]}
    )
    sne.attrs.update(
        dataset_id="independent-sne-two-rows", dataset_name="Small SNe"
    )
    comparison = build_comparison_request(
        control.MODEL_NAME,
        reference.MODEL_NAME,
        control_filename=control.MODEL_FILENAME,
        test_filename=reference.MODEL_FILENAME,
    )
    plan = {
        "sampler_kind": "mcmc",
        "n_steps": 2,
        "burn_in_steps": 1,
        "n_walkers": 4,
        "pool_size": 1,
    }
    return control, reference, sne, comparison, plan


def patch_unrelated_pipeline_plots(stack):
    """Keep CMB plotting real while avoiding unrelated posterior figures."""

    for name in ("_maybe_plot_corner", "_maybe_plot_parameter_histograms"):
        stack.enter_context(mock.patch.object(module, name))
    stack.enter_context(
        mock.patch.object(module.plotter, "plot_hubble_diagram")
    )
    stack.enter_context(mock.patch.object(module.console_output, "write"))


class RunPipelineBehaviorTestCase(unittest.TestCase):
    """Run the actual sampler/likelihood/output composition on tiny data."""

    def test_real_pipeline_preserves_free_fixed_roles_and_cmb_outputs(self):
        control, reference, sne, comparison, plan = bounded_pipeline_inputs()
        observations = explicit_cmb_observations()
        solver = SyntheticCmbSolver(
            spectrum_offsets={"TE": -1205, "EE": -1190}
        )
        captured = []
        savefig = module.plotter.plt.savefig

        def capture_figure(path, **kwargs):
            captured.extend(
                (
                    axis.get_title(),
                    [line.get_ydata().copy() for line in axis.get_lines()],
                )
                for axis in module.plotter.plt.gcf().axes
            )
            savefig(path, **kwargs)

        with (
            tempfile.TemporaryDirectory() as directory,
            contextlib.ExitStack() as stack,
        ):
            patch_unrelated_pipeline_plots(stack)
            stack.enter_context(
                mock.patch.object(
                    module.plotter.plt, "savefig", side_effect=capture_figure
                )
            )
            set_random_seed(123)
            result = module.execute_run_pipeline(
                control_model_plugin=control,
                test_model_plugin=reference,
                sampler_module=sampler_mcmc,
                sne_data_df=sne,
                bao_data_df=None,
                cmb_data_df=observations,
                sampling_plan=plan,
                output_dir=directory,
                run_start_ts="20261018_000000",
                display_progress=False,
                comparison=comparison,
                cmb_solver=solver,
            )
            for fit in result[:2]:
                self.assertTrue(fit["success"])
                self.assertAlmostEqual(fit["chi2_cmb"], 45.4)
            self.assertGreater(
                float(numpy.max(numpy.ptp(result[0]["samples"], axis=(0, 1)))),
                0.0,
            )
            numpy.testing.assert_allclose(
                result[1]["samples"],
                numpy.broadcast_to(
                    reference.INITIAL_GUESSES, result[1]["samples"].shape
                ),
            )
            (summary_path,) = Path(directory).glob("parameter-summary*.json")
            summary = json.loads(summary_path.read_text())
            self.assertEqual(summary["control"]["model"], control.MODEL_NAME)
            self.assertEqual(summary["test"]["model"], reference.MODEL_NAME)
            (csv_path,) = Path(directory).glob("cmb-data*.csv")
            output = pandas.read_csv(csv_path)
            self.assertEqual(
                output["spectrum"].tolist(), ["TE", "TT", "EE", "TT"]
            )
            predictions = [
                name
                for name in output
                if name.startswith("Dl_") and name != "Dl_obs"
            ]
            self.assertEqual(len(predictions), 2)
            for name in predictions:
                numpy.testing.assert_allclose(
                    output[name], [-17.5, 1200, -2.5, 1187.5]
                )
            self.assertTrue(list(Path(directory).glob("cmb-plot*.png")))
            for name in ("TE", "TT", "EE"):
                self.assertTrue(
                    any(name in title for title, lines in captured)
                )
            for name, expected in {
                "TE": [-17.5],
                "TT": [1200, 1187.5],
                "EE": [-2.5],
            }.items():
                lines = [
                    line
                    for title, curves in captured
                    if name in title
                    for line in curves
                ]
                self.assertTrue(
                    any(
                        numpy.shape(line) == numpy.shape(expected)
                        and numpy.allclose(line, expected)
                        for line in lines
                    )
                )
            self.assertEqual(
                set(result[4]["theory_spectrum"]), {"TT", "TE", "EE"}
            )
            self.assertTrue(
                any(
                    workload == "joint_mcmc"
                    for _, _, workload in solver.requests
                )
            )


class TestImportModule(unittest.TestCase):
    """Exercise the module import path."""

    def test_import_module(self) -> None:
        self.assertEqual(module.__name__, "copernican.lib.run_pipeline")


class PublicSymbolCoverageTestCase(unittest.TestCase):
    """Expose the run pipeline API to the coverage policy."""

    def test_public_symbols_are_exposed(self) -> None:
        self.assertTrue(callable(module.execute_run_pipeline))
        self.assertTrue(callable(module.extract_model_param_vector))
        self.assertTrue(callable(module.resolve_sampler_function))


if __name__ == "__main__":
    unittest.main()
