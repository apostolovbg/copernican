"""Tests for the lightweight CLI helper utilities exposed in copernican."""

import os
import time

os.environ.setdefault("COPERNICAN_ALLOW_DIRECT", "1")

import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402
import yaml  # noqa: E402

import copernican  # noqa: E402  pylint: disable=wrong-import-position


def _create_run_dir(tmp_path, name, chi2_total, rows, h0_value):
    run_dir = tmp_path / name
    run_dir.mkdir()
    manifest = {
        "datasets": {"union3_2025": {"name": "Union sample", "type": "sne"}}
    }
    (run_dir / "run_manifest_20250101.yml").write_text(
        yaml.safe_dump(manifest), encoding="utf-8"
    )
    summary = {
        "LambdaCDM": {
            "parameters": {"H_0": h0_value},
            "errors_1sigma": {"H_0": 0.4},
            "covariance_matrix": {"param_names": ["H_0"], "matrix": [[0.16]]},
            "sampling": {"production_steps": 200},
        }
    }
    (run_dir / "parameter-summary_20250101.yml").write_text(
        yaml.safe_dump(summary), encoding="utf-8"
    )
    log_lines = "\n".join(
        [
            "2025-12-08 01:09:21,563 - INFO - --- ΛCDM Fit Report ---",
            f"2025-12-08 01:09:21,564 - INFO -   χ²_Total = {chi2_total}",
            "2025-12-08 01:09:21,565 - INFO -   χ²_BAO = 4.50",
            (
                "2025-12-08 01:09:21,566 - INFO - LambdaCDM BAO: "
                "r_s = 146.12 Mpc, χ²_BAO = 4.50"
            ),
            (
                "2025-12-08 01:09:21,566 - INFO - Loaded dataset "
                f"union3_2025: {rows} entries"
            ),
            "2025-12-08 01:09:21,568 - INFO - Evaluation complete.",
        ]
    )
    (run_dir / "copernican-run_20250101.txt").write_text(
        log_lines + "\n", encoding="utf-8"
    )
    return run_dir


def _add_posterior_file(run_dir):
    dataset = xr.Dataset({"H0": (("draw",), np.linspace(65, 70, 10))})
    posterior_path = run_dir / "posterior-0001.nc"
    dataset.to_netcdf(posterior_path)
    return posterior_path


def test_catalogue_summary_reports_counts():
    summary = copernican._gather_catalogue_summary()  # pylint: disable=protected-access
    assert summary["dataset_count"] > 0
    assert summary["type_counter"]


def test_model_engine_summary_reports_counts():
    stats = copernican._gather_model_engine_summary()  # pylint: disable=protected-access
    assert stats["model_count"] > 0
    assert stats["engine_count"] > 0


def test_manifest_discovery_sorts_by_mtime(tmp_path):
    first = tmp_path / "copernican-run_20240101_000000"
    second = tmp_path / "copernican-run_20240102_000000"
    first.mkdir()
    second.mkdir()
    manifest_first = first / "run_manifest_20240101.yml"
    manifest_second = second / "run_manifest_20240102.yml"
    manifest_first.write_text("seed: 0\n", encoding="utf-8")
    manifest_second.write_text("seed: 1\n", encoding="utf-8")
    os.utime(manifest_first, (time.time() - 100, time.time() - 100))
    os.utime(manifest_second, (time.time(), time.time()))
    records = copernican._discover_manifest_records(tmp_path)  # pylint: disable=protected-access
    assert [directory.name for directory, _ in records] == [
        second.name,
        first.name,
    ]


def test_cli_revalidate_dataset_reports_missing():
    assert copernican._cli_revalidate_dataset("missing-dataset") is False  # pylint: disable=protected-access


def test_cli_revalidate_dataset_known_dataset():
    assert (
        copernican._cli_revalidate_dataset(  # pylint: disable=protected-access
            "planck_2018_lite"
        )
    )


def test_analysis_summary_cli_exports(tmp_path):
    run_dir = _create_run_dir(tmp_path, "analysis-summary", 360.11, 5, 67.2)
    output_dir = tmp_path / "analysis-summary-output"
    assert copernican._run_analysis_summary_cli(run_dir, output_dir, ("yml",))
    summary_files = list(output_dir.glob("analysis-summary_*.yml"))
    assert summary_files


def test_analysis_compare_cli_exports(tmp_path):
    base_dir = _create_run_dir(tmp_path, "base", 360.11, 5, 67.2)
    alt_dir = _create_run_dir(tmp_path, "alt", 362.22, 4, 67.8)
    output_dir = tmp_path / "comparison-output"
    assert copernican._run_analysis_compare_cli(
        base_dir, alt_dir, output_dir, ("yml",)
    )
    comparison_files = list(output_dir.glob("analysis-comparison_*.yml"))
    assert comparison_files


def test_analysis_posterior_cli_creates_plot(tmp_path):
    run_dir = _create_run_dir(tmp_path, "posterior", 360.11, 5, 67.2)
    _add_posterior_file(run_dir)
    output_file = tmp_path / "posterior.png"
    assert copernican._run_analysis_posterior_cli(run_dir, None, output_file)
    assert output_file.exists()
    corner_files = list(tmp_path.glob("corner-plot-*.png"))
    hist_files = list(tmp_path.glob("parameter-histograms-*.png"))
    assert corner_files
    assert hist_files
