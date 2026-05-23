"""Tests for the `copernican_lib.analysis` plot helpers."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from copernican_lib import analysis, chain_io


def _build_run_result(run_dir: Path) -> analysis.RunAnalysisResult:
    summary = analysis.ModelSummary(
        name="LambdaCDM",
        parameters={},
        errors_1sigma=None,
        covariance_matrix=None,
        sampling=None,
        chi2={},
        acceptance=None,
        bao_rs=None,
    )
    return analysis.RunAnalysisResult(
        run_dir=run_dir,
        model_summaries={"LambdaCDM": summary},
        manifest=None,
        manifest_path=None,
        parameter_summary_path=None,
        datasets={"union3_2025": {"name": "Union sample"}},
        dataset_counts={},
        diagnostics=analysis.RunDiagnostics(rhat=None, ess=None),
        start_time=None,
        end_time=None,
        duration_seconds=None,
        log_path=None,
    )


class TestAnalysis(unittest.TestCase):
    """Exercise posterior plotting helpers."""

    def test_plot_posterior_generates_corner_histogram_and_overview(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "posterior-run"
            run_dir.mkdir()
            posterior_path = run_dir / "posterior-0001.nc"
            chain = np.random.default_rng(0).normal(size=(4, 3, 2))
            chain_io.save_posterior(
                chain,
                ["omega_c", "H0"],
                str(posterior_path),
                metadata={"dataset_id": "union3_2025"},
            )

            plots_dir = run_dir / "plots"
            result = _build_run_result(run_dir)

            saved = analysis.plot_posterior(
                run_dir,
                output_dir=plots_dir,
                posterior_file=posterior_path,
                kinds=("overview", "corner", "histograms"),
                result=result,
            )

            self.assertTrue(saved["corner"].exists())
            self.assertTrue(saved["histograms"].exists())
            self.assertTrue(saved["overview"].exists())


if __name__ == "__main__":
    unittest.main()
