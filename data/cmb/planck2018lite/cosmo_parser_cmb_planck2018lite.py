"""Parse the Planck 2018 lite TT/TE/EE spectra with covariance."""
# The Planck team provides the data in a simple text format accompanied by a
# Fortran-style binary covariance matrix. This parser converts those files into
# a convenient Pandas DataFrame with the inverse covariance stored in
# ``.attrs``.

import os
import logging
import numpy as np
import pandas as pd

from copernican_lib.data_loaders import register_cmb_parser
from copernican_lib.utils import load_metadata_from_dir

DATA_DIR = os.path.dirname(__file__)
META = load_metadata_from_dir(DATA_DIR)

DATASET_NAME = META.get("dataset_name", "Planck 2018 Lite TT/TE/EE").replace("\\", "")
DESCRIPTION = META.get(
    "description",
    "Planck 2018 lite TT/TE/EE spectra.",
)


@register_cmb_parser(
    DATASET_NAME,
    DESCRIPTION,
    data_dir=DATA_DIR,
)
def parse_planck2018lite(data_dir, **kwargs):
    """Parse Planck 2018 lite power spectrum and covariance."""

    logger = logging.getLogger()
    cl_path = os.path.join(data_dir, "cl_cmb_plik_v22.dat")
    cov_path = os.path.join(data_dir, "c_matrix_plik_v22.dat")

    try:
        raw = pd.read_csv(cl_path, sep=r"\s+", header=None)

        # The file contains three blocks: TT, TE and EE. Each block lists the
        # power spectrum followed by its diagonal error. Identify the block
        # boundaries by the drop in the \ell column.
        drops = np.where(np.diff(raw[0]) < 0)[0]
        if len(drops) != 2:
            logger.error("Unexpected Planck2018lite file format")
            return None

        idx_tt = drops[0] + 1
        idx_te = drops[1] + 1

        block_tt = raw.iloc[:idx_tt].reset_index(drop=True)
        block_te = raw.iloc[idx_tt:idx_te].reset_index(drop=True)
        block_ee = raw.iloc[idx_te:].reset_index(drop=True)

        ell_tt = block_tt[0].values.astype(int)
        df = pd.DataFrame({"ell": ell_tt})
        df["Dl_obs"] = ell_tt * (ell_tt + 1) * block_tt[1].values / (2 * np.pi)

        # Build TE and EE columns aligned to the TT ell grid. Where the TE/EE
        # block does not provide a value (the high-\ell tail), NaN is used.
        ell_te = block_te[0].values.astype(int)
        ell_ee = block_ee[0].values.astype(int)
        te_map = ell_te * (ell_te + 1) * block_te[1].values / (2 * np.pi)
        te_err = ell_te * (ell_te + 1) * block_te[2].values / (2 * np.pi)
        ee_map = ell_ee * (ell_ee + 1) * block_ee[1].values / (2 * np.pi)
        ee_err = ell_ee * (ell_ee + 1) * block_ee[2].values / (2 * np.pi)

        df["Dl_te_obs"] = np.full_like(ell_tt, np.nan, dtype=float)
        df["Dl_ee_obs"] = np.full_like(ell_tt, np.nan, dtype=float)
        df["e_te_obs"] = np.full_like(ell_tt, np.nan, dtype=float)
        df["e_ee_obs"] = np.full_like(ell_tt, np.nan, dtype=float)

        idx_te = np.searchsorted(ell_tt, ell_te)
        idx_ee = np.searchsorted(ell_tt, ell_ee)
        df.loc[idx_te, "Dl_te_obs"] = te_map
        df.loc[idx_te, "e_te_obs"] = te_err
        df.loc[idx_ee, "Dl_ee_obs"] = ee_map
        df.loc[idx_ee, "e_ee_obs"] = ee_err

        n = len(ell_tt)

        # The covariance matrix file is stored as a Fortran unformatted binary
        # record. Determine the endianness from the leading 4-byte header and
        # validate that the trailer matches. The matrix entries cover the full
        # TT/TE/EE dataset, so read the entire matrix then slice the TT block.
        with open(cov_path, "rb") as fh:
            hdr_bytes = fh.read(4)
            if len(hdr_bytes) != 4:
                logger.error("Planck2018lite covariance matrix missing header")
                return None
            header_le = np.frombuffer(hdr_bytes, dtype="<i4")[0]
            header_be = np.frombuffer(hdr_bytes, dtype=">i4")[0]
            if header_le > 0 and int(np.sqrt(header_le / 8)) ** 2 * 8 == header_le:
                endian = "<"
                header = header_le
            elif header_be > 0 and int(np.sqrt(header_be / 8)) ** 2 * 8 == header_be:
                endian = ">"
                header = header_be
            else:
                logger.error(
                    "Planck2018lite covariance matrix header mismatch or size error."
                )
                return None
            n_full = int(np.sqrt(header / 8))
            cov_arr = np.fromfile(fh, dtype=f"{endian}f8", count=n_full * n_full)
            trailer = np.fromfile(fh, dtype=f"{endian}i4", count=1)[0]

        if cov_arr.size != n_full * n_full or trailer != header:
            logger.error(
                "Planck2018lite covariance matrix trailer mismatch or incomplete read."
            )
            return None

        cov_matrix = cov_arr.reshape(n_full, n_full)[:n, :n]
        # Convert covariance from $C_\ell$ to $D_\ell$ in $\mu K^2$.
        factors = ell_tt * (ell_tt + 1) / (2 * np.pi)
        cov_matrix = cov_matrix * np.outer(factors, factors)

        # Pre-compute diagonal errors for plotting or fallback usage
        diag_errors = np.sqrt(np.diag(cov_matrix))

        try:
            cov_inv = np.linalg.inv(cov_matrix)
            # Check for NaNs or infinities after inversion
            if not np.all(np.isfinite(cov_inv)):
                raise ValueError(
                    "Inverted Planck2018lite covariance contains non-finite values."
                )

            cond_num = np.linalg.cond(cov_matrix)
            if not np.isfinite(cond_num) or cond_num > 1e12:
                raise ValueError(
                    f"Planck2018lite covariance matrix ill-conditioned (cond={cond_num:.2e})."
                )
        except (np.linalg.LinAlgError, ValueError) as e:
            # Fall back to diagonal errors if inversion fails or matrix is bad
            logger.warning(f"{e} Falling back to diagonal errors.")
            cov_inv = None

        df.attrs["covariance_matrix_inv"] = cov_inv
        df.attrs["diag_errors_for_plot"] = diag_errors
        dataset_long = META.get("dataset_name", "CMB_Planck2018lite")
        df.attrs["dataset_long_name"] = dataset_long
        df.attrs["dataset_name_attr"] = dataset_long.replace(" ", "_")
        df.attrs["citation"] = META.get("citation", "")
        df.attrs["notes"] = META.get("notes", "")
        df.attrs["description"] = META.get("description", "")
        df.attrs["is_cmb"] = True
        # Map the order of CAMB parameters used by the engine
        df.attrs["param_names"] = [
            "H0",
            "ombh2",
            "omch2",
            "omnuh2",
            "tau",
            "As",
            "ns",
        ]
        return df
    except Exception as e:
        logger.error(f"Error parsing Planck2018lite data: {e}", exc_info=True)
        return None
