"""Deterministic parsers for the synthetic integration datasets."""

import os

import numpy as numpy_module
import pandas as pandas_module

from copernican.lib.dataset_registry import (
    register_bao_parser,
    register_cmb_parser,
    register_sne_parser,
)

DATA_DIR = os.path.dirname(__file__)


@register_sne_parser(name="synthetic_integration", data_dir=DATA_DIR)
def parse_sne(data_dir: str):
    sne_path = os.path.join(data_dir, "sne.csv")
    sne_dataframe = pandas_module.read_csv(sne_path)
    sne_dataframe.attrs["dataset_name"] = "Synthetic Integration Suite"
    sne_dataframe.attrs["dataset_id"] = "synthetic_integration"
    sne_dataframe.attrs["dataset_version"] = "0.1"
    sne_errors_squared = numpy_module.square(
        sne_dataframe["e_mu_obs"].to_numpy(dtype=float)
    )
    sne_dataframe.attrs["covariance_matrix_inv"] = numpy_module.diag(
        1.0 / sne_errors_squared
    )
    return sne_dataframe


@register_bao_parser(name="synthetic_integration", data_dir=DATA_DIR)
def parse_bao(data_dir: str):
    bao_path = os.path.join(data_dir, "bao.csv")
    bao_dataframe = pandas_module.read_csv(bao_path)
    bao_dataframe.attrs["dataset_name"] = "Synthetic Integration Suite"
    bao_dataframe.attrs["dataset_id"] = "synthetic_integration"
    bao_dataframe.attrs["dataset_version"] = "0.1"
    bao_dataframe.attrs["covariance_matrix_inv"] = numpy_module.diag(
        1.0 / numpy_module.square(bao_dataframe["error"].to_numpy(dtype=float))
    )
    return bao_dataframe


@register_cmb_parser(name="synthetic_integration", data_dir=DATA_DIR)
def parse_cmb(data_dir: str):
    cmb_path = os.path.join(data_dir, "cmb.csv")
    cmb_dataframe = pandas_module.read_csv(cmb_path)
    cmb_dataframe.attrs["dataset_name"] = "Synthetic Integration Suite"
    cmb_dataframe.attrs["dataset_id"] = "synthetic_integration"
    cmb_dataframe.attrs["dataset_version"] = "0.1"
    cmb_covariance = numpy_module.diag(
        1.0
        / numpy_module.square(cmb_dataframe["Dl_err"].to_numpy(dtype=float))
    )
    cmb_dataframe.attrs["covariance_matrix_inv"] = cmb_covariance
    return cmb_dataframe
