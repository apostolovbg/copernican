"""Shared run pipeline extracted from the Copernican workflow."""

from __future__ import annotations

import copy
import os
from typing import Any, Callable, Mapping, Sequence

import numpy

from copernican.lib import chain_io, console_output, csv_writer, diagnostics
from copernican.lib import logger as log_mod
from copernican.lib import plotter, result_writer, utils
from copernican.lib.cmb_identity import CCMBS_LABEL
from copernican.lib.cmb_output import (
    describe_cmb_spectrum,
    observed_cmb_spectrum_names,
)
from copernican.lib.likelihoods import compute_cmb_spectrum_cached
from copernican.lib.model_selection import ComparisonRequest


# Fallback logger retrieved lazily so the helper works when called before
# the global logger is available.
def _get_logger(provided: Any | None = None):
    if provided is not None:
        return provided
    return log_mod.get_logger()


def resolve_sampler_function(sampler_module):
    """Return the canonical sampler callable."""

    fit_fn = getattr(sampler_module, "sample_parameters", None)
    if fit_fn is None:
        raise AttributeError("Sampler lacks sample_parameters")
    return fit_fn, "sample_parameters"


def extract_model_param_vector(
    fit_results: Mapping[str, Any] | None,
    model_plugin: Any,
    *,
    logger: Any | None = None,
) -> Sequence[float] | None:
    """Return the fitted cosmological parameters in declared order."""

    if not isinstance(fit_results, Mapping):
        return None
    if not fit_results.get("success"):
        return None
    params = fit_results.get("fitted_model_params")
    if not isinstance(params, Mapping):
        if logger is not None:
            model_name = getattr(model_plugin, "MODEL_NAME", "model")
            logger.warning(
                "%s fit results missing 'fitted_model_params'.",
                model_name,
            )
        return None

    names = list(getattr(model_plugin, "PARAMETER_NAMES", []))
    if not names:
        return list(params.values())

    missing = [name for name in names if name not in params]
    if missing:
        if logger is not None:
            model_name = getattr(model_plugin, "MODEL_NAME", "model")
            joined = ", ".join(missing)
            logger.warning(
                "%s fit missing values for %s; skipping dependent "
                "analysis.",
                model_name,
                joined,
            )
        return None

    return [params[name] for name in names]


def _posterior_metadata(sne_data_df: Any) -> dict[str, str]:
    """Return metadata used for posterior-related plots."""

    return {
        "dataset_id": (
            f"{sne_data_df.attrs.get('dataset_id', 'joint')}-posterior"
        ),
        "dataset_name": (
            f"{sne_data_df.attrs.get('dataset_name', 'Joint dataset')} "
            "Posterior Samples"
        ),
        "description": "Posterior summary for corner/histogram plots.",
        "citation": sne_data_df.attrs.get("citation", ""),
        "notes": sne_data_df.attrs.get("notes", ""),
    }


def _maybe_plot_corner(
    fit_results: Mapping[str, Any],
    plugin: Any,
    sne_data_df: Any,
    output_dir: str,
    timestamp: str,
    comparison: ComparisonRequest,
) -> None:
    """Render a corner plot when samples exist for ``fit_results``."""
    samples = fit_results.get("samples")
    if samples is None:
        return
    param_names = fit_results.get("param_names")
    posterior_attrs = _posterior_metadata(sne_data_df)
    plotter.plot_corner(
        samples,
        plugin,
        posterior_attrs,
        plot_dir=output_dir,
        parameter_names=param_names,
        timestamp=timestamp,
        comparison=comparison,
    )


def _maybe_plot_parameter_histograms(
    fit_results: Mapping[str, Any],
    plugin: Any,
    sne_data_df: Any,
    output_dir: str,
    timestamp: str,
    comparison: ComparisonRequest,
) -> None:
    """Render histogram plots when posterior samples are available."""
    samples = fit_results.get("samples")
    if samples is None:
        return
    param_names = fit_results.get("param_names")
    plotter.plot_parameter_histograms(
        samples,
        plugin,
        _posterior_metadata(sne_data_df),
        plot_dir=output_dir,
        parameter_names=param_names,
        timestamp=timestamp,
        comparison=comparison,
    )


def execute_run_pipeline(
    *,
    control_model_plugin: Any,
    test_model_plugin: Any,
    sampler_module: Any,
    sne_data_df: Any,
    bao_data_df: Any | None,
    cmb_data_df: Any | None,
    sampling_plan: Mapping[str, Any],
    output_dir: str,
    run_start_ts: str,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
    display_progress: bool = True,
    logger: Any | None = None,
    comparison: ComparisonRequest,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the sampling/diagnostics pipeline and persist outputs."""

    logger = _get_logger(logger)
    control_name, test_name = comparison.model_names
    plugin_names = (
        getattr(control_model_plugin, "MODEL_NAME", ""),
        getattr(test_model_plugin, "MODEL_NAME", ""),
    )
    if plugin_names != comparison.model_names:
        raise ValueError(
            "Pipeline model plugins must match the declared control/test "
            "comparison."
        )
    sampler_label = getattr(
        sampler_module,
        "SAMPLER_LABEL",
        getattr(sampler_module, "__name__", "Sampler"),
    )
    console_output.write(f"\n--- Sampling with {sampler_label} ---\n")
    if cmb_data_df is not None:
        console_output.write(f"CMB execution: {CCMBS_LABEL}.")
    console_output.write("")

    plan_kind = str(sampling_plan.get("sampler_kind", "mcmc")).lower()
    if plan_kind == "nested":
        sampling_live = int(sampling_plan["n_live_points"])
        sampling_max_iter = int(sampling_plan["max_iterations"])
        sampling_tol = float(sampling_plan["evidence_tolerance"])
        sampling_enlarge = float(sampling_plan["enlargement_fraction"])
        logger.info(
            "Nested sampler configuration: live=%d, max_iter=%d, tol=%g, "
            "enlarge=%.2f",
            sampling_live,
            sampling_max_iter,
            sampling_tol,
            sampling_enlarge,
        )
        console_output.write(
            f"Configured nested sampler: live points {sampling_live}, "
            f"max iterations {sampling_max_iter}."
        )
        console_output.write(
            f"Evidence tolerance {sampling_tol:g}, enlargement "
            f"fraction {sampling_enlarge:g}."
        )
    else:
        sampling_steps = int(sampling_plan["n_steps"])
        sampling_burn_in = int(sampling_plan["burn_in_steps"])
        sampling_walkers = int(sampling_plan["n_walkers"])
        sampling_pool = sampling_plan.get("pool_size")
        sampling_cmb_batch = int(sampling_plan.get("cmb_batch_size", 0))
        pool_label = sampling_pool if sampling_pool is not None else "auto"
        logger.info(
            "Sampler configuration: steps=%d, burn-in=%d, walkers=%d, "
            "pool=%s, cmb_batch=%d",
            sampling_steps,
            sampling_burn_in,
            sampling_walkers,
            pool_label,
            sampling_cmb_batch,
        )
        console_output.write(
            f"Configured sampler: steps {sampling_steps}, burn-in "
            f"{sampling_burn_in}."
        )
        console_output.write(
            f"Walker ensemble {sampling_walkers}, pool {pool_label}."
        )
        console_output.write(
            f"CMB batch size {sampling_cmb_batch or 'disabled'}."
        )

    fit_fn, _ = resolve_sampler_function(sampler_module)

    console_output.write(f"{control_name} control chain")
    if plan_kind == "nested":
        console_output.write(f"  Live points: {sampling_live}")
        console_output.write(f"  Max iterations: {sampling_max_iter}")
        console_output.write(f"  Evidence tolerance: {sampling_tol:g}")
        console_output.write(f"  Enlargement fraction: {sampling_enlarge:g}")
        console_output.write(f"  Starting {control_name} sampler...")
        console_output.write("")
        control_fit_results = fit_fn(
            sne_data_df,
            control_model_plugin,
            bao_data_df=bao_data_df,
            cmb_data_df=cmb_data_df,
            n_live_points=sampling_live,
            max_iterations=sampling_max_iter,
            evidence_tolerance=sampling_tol,
            enlargement_fraction=sampling_enlarge,
            display_progress=display_progress,
            progress_callback=progress_callback,
        )
    else:
        console_output.write(f"  Burn-in steps: {sampling_burn_in}")
        console_output.write(f"  Production steps: {sampling_steps}")
        console_output.write(f"  Walkers: {sampling_walkers}")
        console_output.write(f"  Worker pool: {pool_label}")
        console_output.write(f"  Starting {control_name} sampler...")
        console_output.write("")
        control_fit_results = fit_fn(
            sne_data_df,
            control_model_plugin,
            bao_data_df=bao_data_df,
            cmb_data_df=cmb_data_df,
            n_walkers=sampling_walkers,
            n_steps=sampling_steps,
            pool_size=sampling_pool,
            burn_in_steps=sampling_burn_in,
            display_progress=display_progress,
            progress_callback=progress_callback,
            cmb_batch_size=sampling_cmb_batch,
        )

    control_file = getattr(control_model_plugin, "MODEL_FILENAME", "")
    test_file = getattr(test_model_plugin, "MODEL_FILENAME", "")
    same_name = (
        getattr(control_model_plugin, "MODEL_NAME", "").casefold()
        == getattr(test_model_plugin, "MODEL_NAME", "").casefold()
    )
    if (
        same_name
        and control_file == test_file
        and type(control_model_plugin) is type(test_model_plugin)
    ):
        logger.info("Test model matches control model; reusing SNe chain.")
        console_output.write(
            "Test model matches control model; reusing the completed chain."
        )
        console_output.write("")
        test_fit_results = copy.deepcopy(control_fit_results)
    else:
        console_output.write("")
        console_output.write(f"Test model: {test_model_plugin.MODEL_NAME}")
        if plan_kind == "nested":
            console_output.write(f"  Live points: {sampling_live}")
            console_output.write(f"  Max iterations: {sampling_max_iter}")
            console_output.write(f"  Evidence tolerance: {sampling_tol:g}")
            console_output.write(
                f"  Enlargement fraction: {sampling_enlarge:g}"
            )
        if plan_kind == "nested":
            console_output.write("  Starting test-model sampler...")
            console_output.write("")
            test_fit_results = fit_fn(
                sne_data_df,
                test_model_plugin,
                bao_data_df=bao_data_df,
                cmb_data_df=cmb_data_df,
                n_live_points=sampling_live,
                max_iterations=sampling_max_iter,
                evidence_tolerance=sampling_tol,
                enlargement_fraction=sampling_enlarge,
                display_progress=display_progress,
                progress_callback=progress_callback,
            )
        else:
            console_output.write(f"  Burn-in steps: {sampling_burn_in}")
            console_output.write(f"  Production steps: {sampling_steps}")
            console_output.write(f"  Walkers: {sampling_walkers}")
            console_output.write(f"  Worker pool: {pool_label}")
            console_output.write("  Starting test-model sampler...")
            console_output.write("")
            test_fit_results = fit_fn(
                sne_data_df,
                test_model_plugin,
                bao_data_df=bao_data_df,
                cmb_data_df=cmb_data_df,
                n_walkers=sampling_walkers,
                n_steps=sampling_steps,
                pool_size=sampling_pool,
                burn_in_steps=sampling_burn_in,
                display_progress=display_progress,
                progress_callback=progress_callback,
                cmb_batch_size=sampling_cmb_batch,
            )
        console_output.write(
            f"Completed test-model sampling for "
            f"{test_model_plugin.MODEL_NAME}."
        )
        console_output.write("")

    console_output.write("Sampling complete.")
    console_output.write("")

    result_writer.save_summary(
        control_fit_results,
        test_fit_results,
        output_dir,
        timestamp=run_start_ts,
        comparison=comparison,
    )

    # BAO diagnostics.
    BAO_DIAG = diagnostics.bao_residual_diagnostics

    def _component_enabled(fit_results, component):
        state = fit_results.get("likelihood_state", {}) if fit_results else {}
        metadata = state.get("metadata", {})
        components = metadata.get("components", {})
        entry = components.get(component, {})
        flag = entry.get("metadata", {}).get("enabled")
        if flag is not None:
            return bool(flag)
        enabled_components = metadata.get("enabled_components", ())
        return component in enabled_components

    z_plot_smooth = (
        numpy.geomspace(
            max(bao_data_df["redshift"].min() * 0.8, 0.01),
            bao_data_df["redshift"].max() * 1.2,
            100,
        )
        if bao_data_df is not None
        else None
    )

    def _run_bao_analysis(
        model_plugin,
        fit_results,
        z_smooth_arr: numpy.ndarray | None,
    ):
        """Produce BAO predictions and diagnostics for ``model_plugin``."""
        summary = {
            "sne_fit_results": fit_results,
            "pred_df": None,
            "rs_Mpc": numpy.nan,
            "chi2_bao": float(
                (fit_results or {}).get("chi2_bao", float("inf"))
            ),
            "smooth_predictions": None,
        }
        if not (fit_results and fit_results.get("success")):
            logger.warning(
                "%s fit failed; skipping BAO analysis.",
                model_plugin.MODEL_NAME,
            )
            return summary
        if not _component_enabled(fit_results, "bao"):
            logger.info(
                "%s BAO likelihood disabled; skipping predictions.",
                model_plugin.MODEL_NAME,
            )
            summary["chi2_bao"] = float("inf")
            return summary
        fitted = extract_model_param_vector(
            fit_results, model_plugin, logger=logger
        )
        if fitted is None:
            logger.warning(
                "%s fit lacks cosmological parameters; skipping BAO.",
                model_plugin.MODEL_NAME,
            )
            summary["chi2_bao"] = float("inf")
            return summary
        pred_df, rs_Mpc, smooth_preds = (
            sampler_module.calculate_bao_observables(
                bao_data_df,
                model_plugin,
                fitted,
                z_smooth=z_smooth_arr,
            )
        )
        summary.update(
            {
                "pred_df": pred_df,
                "rs_Mpc": rs_Mpc,
                "smooth_predictions": smooth_preds,
            }
        )
        for line in BAO_DIAG(
            pred_df,
            model_name=model_plugin.MODEL_NAME,
        ):
            logger.info(line)
        chi2_bao = summary["chi2_bao"]
        if pred_df is not None and numpy.isfinite(rs_Mpc):
            if numpy.isfinite(chi2_bao):
                logger.info(
                    "%s BAO: r_s = %.2f Mpc, χ²_BAO = %.2f",
                    model_plugin.MODEL_NAME,
                    rs_Mpc,
                    chi2_bao,
                )
            else:
                logger.warning(
                    "%s BAO predictions available but χ² is non-finite.",
                    model_plugin.MODEL_NAME,
                )
        else:
            logger.warning(
                "%s BAO calculation failed or returned invalid r_s.",
                model_plugin.MODEL_NAME,
            )
        return summary

    control_bao_summary = _run_bao_analysis(
        control_model_plugin,
        control_fit_results,
        z_plot_smooth,
    )
    test_bao_summary = _run_bao_analysis(
        test_model_plugin, test_fit_results, z_plot_smooth
    )

    # CMB diagnostics.
    CMB_DIAG = diagnostics.cmb_residual_diagnostics

    def _run_cmb_analysis(model_plugin, fit_results):
        summary = {
            "chi2_cmb": float(
                (fit_results or {}).get("chi2_cmb", float("inf"))
            ),
            "theory_spectrum": None,
            "spectrum_metadata": {},
        }
        if not _component_enabled(fit_results, "cmb"):
            return summary
        if cmb_data_df is None or getattr(cmb_data_df, "empty", True):
            return summary
        if getattr(model_plugin, "valid_for_cmb", True) is False:
            logger.info(
                "%s does not support CMB; skipping analysis.",
                model_plugin.MODEL_NAME,
            )
            summary["chi2_cmb"] = float("inf")
            return summary
        model_params = extract_model_param_vector(
            fit_results, model_plugin, logger=logger
        )
        if model_params is None:
            logger.warning(
                "%s fit lacks parameters; skipping CMB.",
                model_plugin.MODEL_NAME,
            )
            summary["chi2_cmb"] = float("inf")
            return summary
        components = observed_cmb_spectrum_names(cmb_data_df) or ("TT",)
        try:
            theory = compute_cmb_spectrum_cached(
                model_plugin,
                model_params,
                cmb_data_df["ell"].values,
                spectra=components,
            )
        except (
            AttributeError,
            ImportError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            logger.warning(
                "%s failed to compute CMB spectrum: %s",
                model_plugin.MODEL_NAME,
                exc,
            )
            summary["chi2_cmb"] = float("inf")
            return summary
        summary["theory_spectrum"] = theory
        summary["spectrum_metadata"] = {
            name: describe_cmb_spectrum(name).as_mapping()
            for name in components
        }
        for line in CMB_DIAG(
            cmb_data_df,
            theory,
            model_name=model_plugin.MODEL_NAME,
        ):
            logger.info(line)
        chi2_cmb = summary["chi2_cmb"]
        if numpy.isfinite(chi2_cmb):
            logger.info(
                "%s CMB χ² = %.2f",
                model_plugin.MODEL_NAME,
                chi2_cmb,
            )
        else:
            logger.info(
                "%s CMB likelihood disabled or non-finite χ².",
                model_plugin.MODEL_NAME,
            )
        return summary

    control_cmb_summary = _run_cmb_analysis(
        control_model_plugin,
        control_fit_results,
    )
    test_cmb_summary = _run_cmb_analysis(
        test_model_plugin,
        test_fit_results,
    )

    logger.info("\n--- Generating outputs ---")
    logger.info(
        "%s CMB χ² = %.2f",
        control_model_plugin.MODEL_NAME,
        control_cmb_summary["chi2_cmb"],
    )
    logger.info(
        "%s CMB χ² = %.2f",
        test_model_plugin.MODEL_NAME,
        test_cmb_summary["chi2_cmb"],
    )

    plotter.plot_hubble_diagram(
        sne_data_df,
        control_fit_results,
        test_fit_results,
        control_model_plugin,
        test_model_plugin,
        plot_dir=output_dir,
        timestamp=run_start_ts,
        comparison=comparison,
    )
    if bao_data_df is not None:
        plotter.plot_bao_observables(
            bao_data_df,
            control_bao_summary,
            test_bao_summary,
            control_model_plugin,
            test_model_plugin,
            sne_data_df,
            plot_dir=output_dir,
            timestamp=run_start_ts,
            comparison=comparison,
        )
    if cmb_data_df is not None:
        plotter.plot_cmb_spectrum(
            cmb_data_df,
            control_cmb_summary,
            test_cmb_summary,
            control_fit_results,
            test_fit_results,
            control_model_plugin,
            test_model_plugin,
            plot_dir=output_dir,
            timestamp=run_start_ts,
            comparison=comparison,
        )

    _maybe_plot_corner(
        control_fit_results,
        control_model_plugin,
        sne_data_df,
        output_dir,
        run_start_ts,
        comparison,
    )
    _maybe_plot_parameter_histograms(
        control_fit_results,
        control_model_plugin,
        sne_data_df,
        output_dir,
        run_start_ts,
        comparison,
    )
    _maybe_plot_corner(
        test_fit_results,
        test_model_plugin,
        sne_data_df,
        output_dir,
        run_start_ts,
        comparison,
    )
    _maybe_plot_parameter_histograms(
        test_fit_results,
        test_model_plugin,
        sne_data_df,
        output_dir,
        run_start_ts,
        comparison,
    )

    console_output.write("\n--- Theory Abstracts ---\n")
    console_output.write(
        f"{control_name} Abstract:\n{control_model_plugin.MODEL_ABSTRACT}\n"
    )
    console_output.write(
        f"{test_model_plugin.MODEL_NAME} Abstract:\n"
        f"{test_model_plugin.MODEL_ABSTRACT}\n"
    )

    def _print_fit(
        label,
        fit_res,
        bao_res,
        cmb_res,
        plugin,
    ):
        """Dump a summary report of the latest fit."""
        console_output.write(f"--- {label} Fit Report ---\n")
        chi2_sne = float("nan")
        chi2_total = float("nan")
        if fit_res:
            from copernican.lib import latex_utils

            p_names = getattr(plugin, "PARAMETER_NAMES", [])
            p_latex = getattr(plugin, "PARAMETER_LATEX_NAMES", [])
            fitted = fit_res.get("fitted_model_params", {})
            printed_any = False
            for name, latex_name in zip(p_names, p_latex):
                param_value = fitted.get(name)
                if param_value is None:
                    continue
                disp = latex_utils.latex_to_unicode(latex_name)
                console_output.write(f"  {disp} = {param_value:.5g}")
                printed_any = True
            if not printed_any:
                console_output.write(
                    "  Parameters unavailable in fit results."
                )
            chi2_sne = fit_res.get(
                "chi2_sne", fit_res.get("chi2_min", float("nan"))
            )
            chi2_total = fit_res.get("chi2_total", float("nan"))
        else:
            console_output.write(
                "  Fit results unavailable (fixed parameters?)."
            )
        console_output.write(f"  χ²_Total = {chi2_total:.2f}")
        console_output.write(f"  χ²_SNe = {chi2_sne:.2f}")
        if bao_res:
            console_output.write(
                f"  χ²_BAO = {bao_res.get('chi2_bao', float('nan')):.2f}"
            )
        if cmb_res:
            console_output.write(
                f"  χ²_CMB = {cmb_res.get('chi2_cmb', float('nan')):.2f}"
            )
        console_output.write("")

    _print_fit(
        control_name,
        control_fit_results,
        control_bao_summary,
        control_cmb_summary,
        control_model_plugin,
    )
    _print_fit(
        test_model_plugin.MODEL_NAME,
        test_fit_results,
        test_bao_summary,
        test_cmb_summary,
        test_model_plugin,
    )

    csv_writer.save_sne_results_detailed_csv(
        sne_data_df,
        control_fit_results,
        test_fit_results,
        control_model_plugin,
        test_model_plugin,
        csv_dir=output_dir,
        timestamp=run_start_ts,
        comparison=comparison,
    )

    if bao_data_df is not None:
        csv_writer.save_bao_results_csv(
            bao_data_df,
            control_bao_summary,
            test_bao_summary,
            csv_dir=output_dir,
            timestamp=run_start_ts,
            comparison=comparison,
        )
    if cmb_data_df is not None:
        csv_writer.save_cmb_results_csv(
            cmb_data_df,
            control_cmb_summary,
            test_cmb_summary,
            csv_dir=output_dir,
            timestamp=run_start_ts,
            comparison=comparison,
        )

    if control_fit_results.get("samples") is not None:
        fname = utils.generate_filename(
            "posterior",
            sne_data_df.attrs.get("dataset_id", "sne_data"),
            "nc",
            model_name=control_model_plugin.MODEL_NAME.replace(" ", "_"),
            timestamp=run_start_ts,
        )
        chain_io.save_posterior(
            control_fit_results["samples"],
            control_fit_results.get(
                "param_names", control_model_plugin.PARAMETER_NAMES
            ),
            os.path.join(output_dir, fname),
            metadata={
                "model": control_model_plugin.MODEL_NAME,
                "dataset": sne_data_df.attrs.get("dataset_id", ""),
            },
        )
    if test_fit_results.get("samples") is not None:
        fname = utils.generate_filename(
            "posterior",
            sne_data_df.attrs.get("dataset_id", "sne_data"),
            "nc",
            model_name=test_model_plugin.MODEL_NAME.replace(" ", "_"),
            timestamp=run_start_ts,
        )
        chain_io.save_posterior(
            test_fit_results["samples"],
            test_fit_results.get(
                "param_names", test_model_plugin.PARAMETER_NAMES
            ),
            os.path.join(output_dir, fname),
            metadata={
                "model": test_model_plugin.MODEL_NAME,
                "dataset": sne_data_df.attrs.get("dataset_id", ""),
            },
        )

    console_output.write("Evaluation complete.")
    console_output.write("")
    return (
        control_fit_results,
        test_fit_results,
        control_bao_summary,
        test_bao_summary,
        control_cmb_summary,
        test_cmb_summary,
    )
