"""Tests for the `copernican.lib.analysis` plot helpers."""

import tempfile
import unittest
from pathlib import Path

import numpy

from copernican.lib import analysis, chain_io
from copernican.lib.model_selection import build_comparison_request

_TEST_COMPARISON = build_comparison_request("LambdaCDM", "LambdaCDM")


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
        manifest={
            "selection": {
                "comparison": _TEST_COMPARISON.as_manifest(),
            },
        },
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
            chain = numpy.random.default_rng(0).normal(size=(4, 3, 2))
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


class PublicSymbolCoverageTestCase(unittest.TestCase):
    """Expose the analysis API to the coverage policy."""

    def test_public_symbols_are_exposed(self) -> None:
        result = _build_run_result(Path("."))
        self.assertTrue(callable(analysis.analyze_run))
        self.assertTrue(callable(analysis.compare_run_dirs))
        self.assertTrue(callable(analysis.compare_runs))
        self.assertTrue(callable(analysis.format_run_summary_text))
        self.assertTrue(callable(analysis.load_manifest))
        self.assertTrue(callable(analysis.load_parameter_summary))
        self.assertTrue(callable(analysis.parse_log))
        self.assertTrue(callable(analysis.plot_posterior))
        self.assertTrue(callable(analysis.save_comparison_summary))
        self.assertTrue(callable(analysis.save_run_summary))
        self.assertIsInstance(result, analysis.RunAnalysisResult)
        self.assertIsInstance(result.diagnostics, analysis.RunDiagnostics)
        self.assertIsInstance(
            result.model_summaries["LambdaCDM"], analysis.ModelSummary
        )
        self.assertIsInstance(result.to_dict(), dict)


if __name__ == "__main__":
    unittest.main()
