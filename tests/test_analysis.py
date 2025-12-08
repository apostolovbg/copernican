# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.

from __future__ import annotations

import datetime
import textwrap

import yaml

from copernican_lib import analysis


def _write_log(tmp_path, content: str, name: str) -> None:
    path = tmp_path / name
    path.write_text(textwrap.dedent(content).strip() + "\n")
    return path


def test_parse_log_extracts_metrics(tmp_path):
    log_lines = """
        2025-12-08 01:09:21,563 - INFO - --- ΛCDM Fit Report ---
        2025-12-08 01:09:21,564 - INFO -   χ²_Total = 352.47
        2025-12-08 01:09:21,565 - INFO -   χ²_SNe = 29.44
        2025-12-08 01:09:21,565 - INFO -   χ²_BAO = 5.03
        2025-12-08 01:09:21,566 - INFO -   χ²_CMB = 318.00
        2025-12-08 01:09:21,567 - INFO - LambdaCDM BAO: r_s = 145.89 Mpc, χ²_BAO = 5.03
        2025-12-08 01:09:21,567 - INFO - Rank-normalised R-hat summary: min=1.317 median=1.464 max=1.524
        2025-12-08 01:09:21,567 - INFO - Effective sample sizes: bulk median=64.0 tail median=155.5
        2025-12-08 01:09:21,567 - INFO - MCMC acceptance for ΛCDM: mean=0.457, min=0.390, max=0.560
        2025-12-08 01:09:21,568 - INFO - Evaluation complete.
    """
    log_path = _write_log(tmp_path, log_lines, "validation_run_test.txt")
    parsed = analysis.parse_log(log_path)
    assert parsed["diagnostics"]["rhat"]["median"] == 1.464
    assert parsed["diagnostics"]["ess"]["tail_median"] == 155.5
    assert parsed["acceptance"]["lambdacdm"]["mean"] == 0.457
    assert parsed["models"]["lambdacdm"]["chi2"]["chi2_total"] == 352.47
    assert parsed["models"]["lambdacdm"]["bao_rs"] == 145.89
    assert parsed["end_time"] == datetime.datetime(
        2025, 12, 8, 1, 9, 21, 568000
    )


def test_analyze_run_merges_sources(tmp_path):
    run_dir = tmp_path / "run_root"
    run_dir.mkdir()
    manifest = {
        "datasets": {
            "union3_2025": {
                "name": "Union sample",
                "version": "2025.1",
                "path": "/tmp/data/sne",
                "hashes": {},
                "type": "sne",
            }
        }
    }
    manifest_path = run_dir / "run_manifest_20250101_000000.yml"
    manifest_path.write_text(yaml.safe_dump(manifest))

    summary = {
        "LambdaCDM": {
            "parameters": {"H_0": 67.2},
            "errors_1sigma": {"H_0": 0.4},
            "covariance_matrix": {"param_names": ["H_0"], "matrix": [[0.16]]},
            "sampling": {"production_steps": 200},
        }
    }
    summary_path = run_dir / "parameter-summary_20250101_000000.yml"
    summary_path.write_text(yaml.safe_dump(summary))

    log_lines = """
        2025-12-08 01:09:21,563 - INFO - --- ΛCDM Fit Report ---
        2025-12-08 01:09:21,564 - INFO -   χ²_Total = 360.11
        2025-12-08 01:09:21,565 - INFO -   χ²_BAO = 4.50
        2025-12-08 01:09:21,566 - INFO - LambdaCDM BAO: r_s = 146.12 Mpc, χ²_BAO = 4.50
        2025-12-08 01:09:21,566 - INFO - Loaded dataset union3_2025: 5 entries
        2025-12-08 01:09:21,568 - INFO - Evaluation complete.
    """
    log_file = run_dir / "copernican-run_20250101_000000.txt"
    log_file.write_text(textwrap.dedent(log_lines).strip() + "\n")

    result = analysis.analyze_run(run_dir)
    summary_entry = result.model_summaries["LambdaCDM"]
    assert summary_entry.chi2["chi2_total"] == 360.11
    assert summary_entry.bao_rs == 146.12
    assert result.datasets["union3_2025"]["name"] == "Union sample"
    assert result.dataset_counts["union3_2025"] == 5
    assert result.log_path == log_file
    assert result.parameter_summary_path == summary_path
    assert result.manifest_path == manifest_path


def test_save_run_summary_creates_serialised_files(tmp_path):
    run_dir = tmp_path / "run_root"
    run_dir.mkdir()
    manifest = {
        "datasets": {
            "union3_2025": {
                "name": "Union sample",
                "version": "2025.1",
                "path": "/tmp/data/sne",
                "hashes": {},
                "type": "sne",
            }
        }
    }
    (run_dir / "run_manifest_20250101_000000.yml").write_text(
        yaml.safe_dump(manifest)
    )

    summary = {
        "LambdaCDM": {
            "parameters": {"H_0": 67.2},
            "errors_1sigma": {"H_0": 0.4},
            "covariance_matrix": {"param_names": ["H_0"], "matrix": [[0.16]]},
            "sampling": {"production_steps": 200},
        }
    }
    (run_dir / "parameter-summary_20250101_000000.yml").write_text(
        yaml.safe_dump(summary)
    )

    log_lines = """
        2025-12-08 01:09:21,563 - INFO - --- ΛCDM Fit Report ---
        2025-12-08 01:09:21,564 - INFO -   χ²_Total = 360.11
        2025-12-08 01:09:21,565 - INFO -   χ²_BAO = 4.50
        2025-12-08 01:09:21,566 - INFO -   LambdaCDM BAO: r_s = 146.12 Mpc, χ²_BAO = 4.50
        2025-12-08 01:09:21,566 - INFO - Loaded dataset union3_2025: 5 entries
        2025-12-08 01:09:21,568 - INFO - Evaluation complete.
    """
    log_file = run_dir / "copernican-run_20250101_000000.txt"
    log_file.write_text(textwrap.dedent(log_lines).strip() + "\n")

    output_dir = tmp_path / "analysis_summary"
    saved = analysis.save_run_summary(run_dir, output_dir)
    assert "yml" in saved and "json" in saved

    loaded = yaml.safe_load(saved["yml"].read_text())
    assert loaded["datasets"]["union3_2025"]["name"] == "Union sample"
    assert "LambdaCDM" in loaded["model_summaries"]
    assert loaded["model_summaries"]["LambdaCDM"]["chi2"]["chi2_total"] == 360.11
    assert saved["json"].exists()
