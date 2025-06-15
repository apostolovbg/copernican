"""OpenCL/CUDA accelerated engine for the Copernican Suite."""
# DEV NOTE (v1.5e): Experimental GPU backend using PyOpenCL with CPU fallback.

import numpy as np
from . import cosmo_engine_1_4b as base_engine

try:
    import pyopencl as cl
    import pyopencl.array as cl_array
    _HAS_OPENCL = True
except Exception:
    cl = None
    cl_array = None
    _HAS_OPENCL = False


def fit_sne_parameters(sne_data_df, model_plugin):
    """Use base SciPy engine for SNe fitting."""
    return base_engine.fit_sne_parameters(sne_data_df, model_plugin)


def calculate_bao_observables(bao_data_df, model_plugin, cosmo_params, z_smooth=None):
    """Use base SciPy engine for BAO predictions."""
    return base_engine.calculate_bao_observables(bao_data_df, model_plugin, cosmo_params, z_smooth=z_smooth)


def _chi2_bao_opencl(obs_vals, obs_err, pred_vals):
    """Compute chi-squared on the GPU if PyOpenCL is available."""
    if not _HAS_OPENCL:
        diff = (obs_vals - pred_vals) / obs_err
        return float(np.sum(diff ** 2))

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    obs_buf = cl_array.to_device(queue, obs_vals)
    err_buf = cl_array.to_device(queue, obs_err)
    pred_buf = cl_array.to_device(queue, pred_vals)
    out_buf = cl_array.empty(queue, obs_vals.shape, dtype=np.float64)

    prg = cl.Program(ctx, """
    __kernel void chi2(__global const double *obs, __global const double *err,
                       __global const double *pred, __global double *out)
    {
        int i = get_global_id(0);
        double diff = (obs[i] - pred[i]) / err[i];
        out[i] = diff * diff;
    }
    """).build()

    prg.chi2(queue, obs_vals.shape, None, obs_buf.data, err_buf.data, pred_buf.data, out_buf.data)
    result = cl_array.sum(out_buf).get()
    return float(result)


def chi_squared_bao(bao_data_df, model_plugin, cosmo_params, model_rs_Mpc):
    """BAO chi-squared using GPU acceleration when available."""
    pred_df, _, _ = calculate_bao_observables(bao_data_df, model_plugin, cosmo_params)
    if pred_df is None or pred_df.empty:
        return np.inf
    obs_vals = bao_data_df['value'].to_numpy(dtype=float)
    obs_err = bao_data_df['error'].to_numpy(dtype=float)
    pred_vals = pred_df['model_prediction'].to_numpy(dtype=float)
    if obs_vals.size != pred_vals.size:
        return np.inf
    return _chi2_bao_opencl(obs_vals, obs_err, pred_vals)

