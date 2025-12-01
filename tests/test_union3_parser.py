# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.

"""Basic smoke test for the Union3 parser."""

import numpy as np

from copernican_lib import dataset_registry


def test_union3_loader_returns_compressed_sample():
    """Ensure the Union3 parser returns distances, redshifts and covariance."""
    df = dataset_registry.load_sne_data("union3_2025")
    assert not df.empty
    assert df.shape[0] == 22
    assert "zcmb" in df.columns
    assert "mu_obs" in df.columns
    assert "covariance_matrix_inv" in df.attrs
    inv_cov = df.attrs["covariance_matrix_inv"]
    assert inv_cov.shape == (22, 22)
    diag_errors = df.attrs.get("diag_errors_for_plot")
    assert diag_errors is not None
    assert np.allclose(df["e_mu_obs"].values, diag_errors)
