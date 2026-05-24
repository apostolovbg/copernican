"""Deterministic parsers for the synthetic integration datasets."""

import os

import numpy as np
import pandas as pd

from copernican_lib.dataset_registry import (
    register_bao_parser,
    register_cmb_parser,
    register_sne_parser,
)

DATA_DIR = os.path.dirname(__file__)


@register_sne_parser(name="synthetic_integration", data_dir=DATA_DIR)
def parse_sne(data_dir: str):
    sne_path = os.path.join(data_dir, "sne.csv")
    df = pd.read_csv(sne_path)
    df.attrs["dataset_name"] = "Synthetic Integration Suite"
    df.attrs["dataset_id"] = "synthetic_integration"
    df.attrs["dataset_version"] = "0.1"
    diag = np.square(df["e_mu_obs"].to_numpy(dtype=float))
    df.attrs["covariance_matrix_inv"] = np.diag(1.0 / diag)
    return df


@register_bao_parser(name="synthetic_integration", data_dir=DATA_DIR)
def parse_bao(data_dir: str):
    bao_path = os.path.join(data_dir, "bao.csv")
    df = pd.read_csv(bao_path, comment="#")
    df.attrs["dataset_name"] = "Synthetic Integration Suite"
    df.attrs["dataset_id"] = "synthetic_integration"
    df.attrs["dataset_version"] = "0.1"
    df.attrs["covariance_matrix_inv"] = np.diag(
        1.0 / np.square(df["error"].to_numpy(dtype=float))
    )
    return df


@register_cmb_parser(name="synthetic_integration", data_dir=DATA_DIR)
def parse_cmb(data_dir: str):
    cmb_path = os.path.join(data_dir, "cmb.csv")
    df = pd.read_csv(cmb_path)
    df.attrs["dataset_name"] = "Synthetic Integration Suite"
    df.attrs["dataset_id"] = "synthetic_integration"
    df.attrs["dataset_version"] = "0.1"
    cov = np.diag(1.0 / np.square(df["Dl_err"].to_numpy(dtype=float)))
    df.attrs["covariance_matrix_inv"] = cov
    return df
