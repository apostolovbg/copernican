# copernican_suite/output_manager.py
"""
Output Manager for the Copernican Suite.
"""
import logging, os, time, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_timestamp(): return time.strftime("%Y%m%d_%H%M%S")

def _ensure_dir_exists(directory):
    """Creates the specified directory if it does not already exist."""
    os.makedirs(directory, exist_ok=True)

def setup_logging(run_name="copernican_run", log_level=logging.INFO, log_dir="."):
    """Initializes logging to both console and a file."""
    _ensure_dir_exists(log_dir)
    logger = logging.getLogger(); logger.setLevel(log_level)
    for handler in logger.handlers[:]: logger.removeHandler(handler)
    log_filename = os.path.join(log_dir, f"{run_name}_{get_timestamp()}.log")
    fh = logging.FileHandler(log_filename); fh.setLevel(log_level)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout); ch.setLevel(log_level)
    ch.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(ch)
    logging.info(f"Logging initialized. Log file: {log_filename}")
    return log_filename

def get_logger(): return logging.getLogger()

def format_model_summary_text(model_plugin, is_sne_summary, fit_results, **kwargs):
    """Formats a text block with model details, parameters, and fit statistics."""
    lines = [];
    model_name_raw = getattr(model_plugin, 'MODEL_NAME', 'N/A')
    model_name_latex = model_name_raw.replace('_', r'\_')
    lines.append(fr"**Model: {model_name_latex}**")

    eq_attr = 'MODEL_EQUATIONS_LATEX_SN' if is_sne_summary else 'MODEL_EQUATIONS_LATEX_BAO'
    if hasattr(model_plugin, eq_attr):
        lines.append(r"$\bf{Mathematical\ Form:}$")
        for eq_line in getattr(model_plugin, eq_attr): lines.append(f"  {eq_line}")

    lines.append(r"$\bf{Cosmological\ Parameters:}$")
    param_names = getattr(model_plugin, 'PARAMETER_NAMES', [])
    param_latex_names = getattr(model_plugin, 'PARAMETER_LATEX_NAMES', param_names)
    fitted_cosmo_params = fit_results.get('fitted_cosmological_params')

    if fitted_cosmo_params:
        for i, name in enumerate(param_names):
            val = fitted_cosmo_params.get(name)
            latex_name = param_latex_names[i] if i < len(param_latex_names) else name
            lines.append(fr"  {latex_name} = ${val:.4g}$" if val is not None else f"  {latex_name} = N/A")
    else:
        lines.append("  (Fit failed or parameters unavailable)")

    if is_sne_summary and fit_results.get('fitted_nuisance_params'):
        lines.append(r"$\bf{SNe\ Nuisance\ Parameters:}$")
        for name, val in fit_results['fitted_nuisance_params'].items():
            name_latex = {"M_B": r"M_B", "alpha_salt2": r"\alpha", "beta_salt2": r"\beta"}.get(name, name)
            lines.append(fr"  ${name_latex}$ = ${val:.4g}$")

    if is_sne_summary:
        lines.append(r"$\bf{SNe\ Fit\ Statistics:}$")
        lines.append(fr"  $\chi^2_{{SNe}}$ = {fit_results.get('chi2_min', np.nan):.2f}")
    else:
        lines.append(r"$\bf{BAO\ Test\ Results:}$")
        lines.append(fr"  $r_s$ = {kwargs.get('model_rs_Mpc', np.nan):.2f} Mpc")
        lines.append(fr"  $\chi^2_{{BAO}}$ = {kwargs.get('chi2_bao', np.nan):.2f}")

    return "\n".join(lines)

def plot_hubble_diagram(sne_data_df, lcdm_fit_results, alt_model_fit_results, lcdm_plugin, alt_model_plugin, plot_dir=".", base_filename="plot_hubble"):
    """Generates and saves a Hubble diagram and residuals plot."""
    _ensure_dir_exists(plot_dir)
    logger = get_logger()
    dataset_name = sne_data_df.attrs.get('dataset_name_attr', 'SNe_data')
    logger.info(f"Generating Hubble Diagram for {dataset_name}...")
    
    font_sizes = {'title': 22, 'label': 18, 'legend': 14, 'infobox': 12, 'ticks': 12}

    if 'mu_obs' not in sne_data_df.columns:
        fit_res_for_mu = alt_model_fit_results if alt_model_fit_results and alt_model_fit_results.get('fitted_nuisance_params') else lcdm_fit_results
        if sne_data_df.attrs.get('fit_style') == 'h2_fit_nuisance' and fit_res_for_mu and fit_res_for_mu.get('fitted_nuisance_params'):
            nuisance = fit_res_for_mu['fitted_nuisance_params']
            M_B, alpha, beta = nuisance['M_B'], nuisance['alpha_salt2'], nuisance['beta_salt2']
            sne_data_df['mu_obs'] = sne_data_df['mb'] - M_B + alpha * sne_data_df['x1'] - beta * sne_data_df['c']
        else:
            logger.error("Cannot plot Hubble Diagram: 'mu_obs' column missing and could not be calculated."); return

    mu_obs_data = sne_data_df['mu_obs'].values
    z_data = sne_data_df['zcmb'].values
    diag_errors_plot = sne_data_df.attrs.get('diag_errors_for_plot', np.ones_like(z_data) * 0.2)
    z_plot_smooth = np.geomspace(max(np.min(z_data) * 0.9, 0.001), np.max(z_data) * 1.05, 200)

    def get_binned_average(z, residuals, n_bins=20):
        if len(z) < n_bins: return np.array([]), np.array([])
        try:
            from scipy.stats import binned_statistic
            mean_stat, bin_edges, _ = binned_statistic(z, residuals, statistic='mean', bins=n_bins)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            valid_indices = ~np.isnan(mean_stat)
            return bin_centers[valid_indices], mean_stat[valid_indices]
        except ImportError:
            logger.warning("Scipy not found, cannot plot binned residual averages."); return np.array([]), np.array([])
        except Exception as e:
            logger.warning(f"Could not calculate binned average due to an error: {e}"); return np.array([]), np.array([])

    fig, axs = plt.subplots(2, 1, figsize=(17, 12), sharex=True, gridspec_kw={'height_ratios':[3,1.5],'hspace':0.05})
    # --- LAYOUT MODIFICATION: Widen plot by changing the 'right' parameter ---
    plt.subplots_adjust(left=0.08, bottom=0.08, right=0.75, top=0.92)
    try: plt.style.use('seaborn-v0_8-darkgrid')
    except Exception: logger.warning("Seaborn-v0_8-darkgrid style not found, using default.")

    axs[0].errorbar(z_data, mu_obs_data, yerr=diag_errors_plot, fmt='.', color='darkgray', alpha=0.6, label=f"{dataset_name}", elinewidth=1, capsize=2, ms=5, ecolor='lightgray', zorder=1)

    if lcdm_fit_results and lcdm_fit_results.get('success'):
        p_lcdm = list(lcdm_fit_results['fitted_cosmological_params'].values())
        mu_model_lcdm = lcdm_plugin.distance_modulus_model(z_plot_smooth, *p_lcdm)
        res_lcdm = mu_obs_data - lcdm_plugin.distance_modulus_model(z_data, *p_lcdm)
        chi2_lcdm = f"{lcdm_fit_results.get('chi2_min', np.nan):.2f}"
        axs[0].plot(z_plot_smooth, mu_model_lcdm, color='red', ls='-', label=fr'$\Lambda$CDM ($\chi^2$={chi2_lcdm})', lw=2.5)
        axs[1].errorbar(z_data, res_lcdm, yerr=diag_errors_plot, fmt='.', color='red', alpha=0.5, label=r'$\Lambda$CDM Res.', elinewidth=1, capsize=2, ms=4)
        z_lcdm_avg, res_lcdm_avg = get_binned_average(z_data, res_lcdm)
        axs[1].plot(z_lcdm_avg, res_lcdm_avg, color='darkred', ls='-', lw=2, zorder=10, label=r'Avg. $\Lambda$CDM Res.')

    alt_name_raw = getattr(alt_model_plugin, 'MODEL_NAME', 'AltModel')
    alt_name_latex = alt_name_raw.replace('_', r'\_')
    if alt_model_fit_results and alt_model_fit_results.get('success'):
        p_alt = list(alt_model_fit_results['fitted_cosmological_params'].values())
        mu_model_alt = alt_model_plugin.distance_modulus_model(z_plot_smooth, *p_alt)
        res_alt = mu_obs_data - alt_model_plugin.distance_modulus_model(z_data, *p_alt)
        chi2_alt = f"{alt_model_fit_results.get('chi2_min', np.nan):.2f}"
        axs[0].plot(z_plot_smooth, mu_model_alt, color='blue', ls='--', label=fr"{alt_name_latex} ($\chi^2$={chi2_alt})", lw=2.5)
        axs[1].errorbar(z_data, res_alt, yerr=diag_errors_plot, fmt='.', mfc='none', mec='blue', ecolor='lightblue', alpha=0.5, label=fr'{alt_name_latex} Res.', elinewidth=1, capsize=2, ms=4)
        z_alt_avg, res_alt_avg = get_binned_average(z_data, res_alt)
        axs[1].plot(z_alt_avg, res_alt_avg, color='darkblue', ls='--', lw=2, zorder=11, label=f'Avg. {alt_name_latex} Res.')

    axs[0].set_ylabel(r'Distance Modulus ($\mu$)', fontsize=font_sizes['label']); axs[0].legend(fontsize=font_sizes['legend'], loc='lower right'); axs[0].set_title(f'Hubble Diagram: {dataset_name}', fontsize=font_sizes['title']); axs[0].minorticks_on(); axs[0].tick_params(axis='both', which='major', labelsize=font_sizes['ticks'])
    axs[1].axhline(0, color='black', ls='--', lw=1); axs[1].set_xlabel('Redshift (z)', fontsize=font_sizes['label']); axs[1].set_ylabel(r'$\mu_{obs} - \mu_{model}$', fontsize=font_sizes['label']); axs[1].legend(fontsize=font_sizes['legend'], loc='lower right'); axs[1].minorticks_on(); axs[1].tick_params(axis='both', which='major', labelsize=font_sizes['ticks'])

    bbox_lcdm = dict(boxstyle='round,pad=0.5', fc='#FFEEEE', ec='darkred', alpha=0.8)
    bbox_alt = dict(boxstyle='round,pad=0.5', fc='#EEF2FF', ec='darkblue', alpha=0.8)
    # --- LAYOUT MODIFICATION: Move info boxes to the right to accommodate wider plot ---
    fig.text(0.77, 0.90, format_model_summary_text(lcdm_plugin, is_sne_summary=True, fit_results=lcdm_fit_results), fontsize=font_sizes['infobox'], va='top', ha='left', wrap=True, bbox=bbox_lcdm)
    fig.text(0.77, 0.52, format_model_summary_text(alt_model_plugin, is_sne_summary=True, fit_results=alt_model_fit_results), fontsize=font_sizes['infobox'], va='top', ha='left', wrap=True, bbox=bbox_alt)

    filename = os.path.join(plot_dir, f"{base_filename}_{dataset_name.replace(' ', '_')}_{get_timestamp()}.png")
    try:
        plt.savefig(filename, dpi=300); logger.info(f"Hubble diagram saved to {filename}")
    except Exception as e:
        logger.error(f"Error saving Hubble diagram: {e}")
    finally:
        plt.close(fig)

def plot_bao_observables(bao_data_df, lcdm_full_results, alt_model_full_results, lcdm_plugin, alt_model_plugin, plot_dir=".", base_filename="plot_bao"):
    """Generates and saves a plot of BAO observables vs. redshift."""
    _ensure_dir_exists(plot_dir)
    logger = get_logger()
    dataset_name = bao_data_df.attrs.get('dataset_name_attr', 'BAO_data')
    logger.info(f"Generating BAO Plot for {dataset_name}...")

    font_sizes = {'title': 22, 'label': 18, 'legend': 14, 'infobox': 12, 'ticks': 12}
    
    min_z, max_z = bao_data_df['redshift'].min(), bao_data_df['redshift'].max()
    z_plot_smooth = np.geomspace(max(min_z * 0.8, 0.01), max_z * 1.2, 100)

    fig, ax = plt.subplots(figsize=(17, 10))
    # --- LAYOUT MODIFICATION: Widen plot by changing the 'right' parameter ---
    plt.subplots_adjust(left=0.08, bottom=0.1, right=0.75, top=0.90)
    try: plt.style.use('seaborn-v0_8-darkgrid')
    except Exception: logger.warning("Seaborn-v0_8-darkgrid style not found, using default.")

    obs_types = bao_data_df['observable_type'].unique()
    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 0.8, len(obs_types)))
    for i, obs_type in enumerate(obs_types):
        subset = bao_data_df[bao_data_df['observable_type'] == obs_type]
        label = f"Data: {obs_type.replace('_', '/')}"
        ax.errorbar(subset['redshift'], subset['value'], yerr=subset['error'], fmt='o', label=label, capsize=3, color=colors[i], ms=8, zorder=5)

    def plot_model_bao(model_plugin, results, color, line_styles, label_prefix):
        if not results.get('sne_fit_results', {}).get('success'):
            logger.warning(f"Skipping BAO plot for {label_prefix} as SNe fit failed."); return
        p = list(results['sne_fit_results']['fitted_cosmological_params'].values())
        rs = results['rs_Mpc']
        if not np.isfinite(rs):
            logger.warning(f"Skipping BAO plot for {label_prefix} due to invalid r_s value."); return

        dm = model_plugin.get_comoving_distance_Mpc(z_plot_smooth, *p)
        hz = model_plugin.get_Hz_per_Mpc(z_plot_smooth, *p)
        dh = np.where(hz > 0, model_plugin.FIXED_PARAMS["C_LIGHT_KM_S"] / hz, np.nan)
        if hasattr(model_plugin, 'get_DV_Mpc'): dv = model_plugin.get_DV_Mpc(z_plot_smooth, *p)
        else:
            da = model_plugin.get_angular_diameter_distance_Mpc(z_plot_smooth, *p)
            term = (1+z_plot_smooth)**2 * da**2 * model_plugin.FIXED_PARAMS["C_LIGHT_KM_S"] * z_plot_smooth / hz
            dv = np.power(term, 1/3, where=term>=0, out=np.full_like(z_plot_smooth, np.nan))

        def robust_plot(z, y, **kwargs):
            valid_indices = np.isfinite(z) & np.isfinite(y)
            if np.any(valid_indices): ax.plot(z[valid_indices], y[valid_indices], **kwargs)
            else: logger.warning(f"No valid data points to plot for {kwargs.get('label')}")

        if 'DM_over_rs' in obs_types: robust_plot(z_plot_smooth, dm/rs, color=color, ls=line_styles[0], lw=2.5, label=fr'{label_prefix} ($D_M/r_s$)')
        if 'DH_over_rs' in obs_types: robust_plot(z_plot_smooth, dh/rs, color=color, ls=line_styles[1], lw=2.5, label=fr'{label_prefix} ($D_H/r_s$)')
        if 'DV_over_rs' in obs_types: robust_plot(z_plot_smooth, dv/rs, color=color, ls=line_styles[2], lw=2.5, label=fr'{label_prefix} ($D_V/r_s$)')

    line_styles = ['-', '--', ':']
    plot_model_bao(lcdm_plugin, lcdm_full_results, 'red', line_styles, r'$\Lambda$CDM')
    alt_name_raw = getattr(alt_model_plugin, 'MODEL_NAME', 'AltModel')
    alt_name_latex = alt_name_raw.replace('_', r'\_')
    plot_model_bao(alt_model_plugin, alt_model_full_results, 'blue', line_styles, alt_name_latex)

    ax.set_xlabel('Redshift (z)', fontsize=font_sizes['label']); ax.set_ylabel(r'$D_X/r_s$', fontsize=font_sizes['label']); ax.set_title(f'BAO Observables vs. Redshift: {dataset_name}', fontsize=font_sizes['title']); ax.legend(fontsize=font_sizes['legend'], loc='best'); ax.minorticks_on(); ax.tick_params(axis='both', which='major', labelsize=font_sizes['ticks'])
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    bbox_lcdm = dict(boxstyle='round,pad=0.5', fc='#FFEEEE', ec='darkred', alpha=0.8)
    bbox_alt = dict(boxstyle='round,pad=0.5', fc='#EEF2FF', ec='darkblue', alpha=0.8)
    # --- LAYOUT MODIFICATION: Move info boxes to the right to accommodate wider plot ---
    fig.text(0.77, 0.90, format_model_summary_text(lcdm_plugin, is_sne_summary=False, fit_results=lcdm_full_results.get('sne_fit_results',{}), **lcdm_full_results), fontsize=font_sizes['infobox'], va='top', ha='left', wrap=True, bbox=bbox_lcdm)
    fig.text(0.77, 0.55, format_model_summary_text(alt_model_plugin, is_sne_summary=False, fit_results=alt_model_full_results.get('sne_fit_results',{}), **alt_model_full_results), fontsize=font_sizes['infobox'], va='top', ha='left', wrap=True, bbox=bbox_alt)

    filename = os.path.join(plot_dir, f"{base_filename}_{dataset_name.replace(' ', '_')}_{get_timestamp()}.png")
    try:
        plt.savefig(filename, dpi=300); logger.info(f"BAO plot saved to {filename}")
    except Exception as e:
        logger.error(f"Error saving BAO plot: {e}")
    finally:
        plt.close(fig)

def save_sne_fit_results_csv(all_fit_results, sne_data_df, csv_dir=".", base_filename="sne_fit_summary"):
    """Saves a summary of the SNe Ia fitting results to a CSV file."""
    _ensure_dir_exists(csv_dir)
    logger = get_logger(); summary_data = []
    for result in all_fit_results:
        if not result or not result.get('success'): continue
        row = {'Model': result.get('model_name'), 'FitStyle': result.get('fit_style_used'), 'Chi2_min': result.get('chi2_min'), 'DoF': result.get('dof'), 'ReducedChi2': result.get('reduced_chi2'), 'FitSuccess': result.get('success'), 'OptimizerMessage': result.get('message')}
        if result.get('fitted_cosmological_params'): row.update(result['fitted_cosmological_params'])
        if result.get('fitted_nuisance_params'): row.update(result['fitted_nuisance_params'])
        summary_data.append(row)
    
    if not summary_data:
        logger.warning("No successful SNe fits to save to CSV."); return

    summary_df = pd.DataFrame(summary_data)
    filename = os.path.join(csv_dir, f"{base_filename}_{sne_data_df.attrs.get('format_key', 'SNe')}_{get_timestamp()}.csv")
    try:
        summary_df.to_csv(filename, index=False, float_format='%.6g'); logger.info(f"SNe fit summary CSV saved to {filename}")
    except Exception as e:
        logger.error(f"Error saving SNe fit summary CSV: {e}")

def save_bao_results_csv(bao_data_df, lcdm_results, alt_model_results, alt_model_name, csv_dir=".", base_filename="bao_results"):
    """Saves a detailed breakdown of the BAO results to a CSV file."""
    _ensure_dir_exists(csv_dir)
    logger = get_logger()
    if bao_data_df is None or bao_data_df.empty:
        logger.warning("BAO data is empty, skipping CSV save."); return
        
    df_out = bao_data_df.copy()
    
    if lcdm_results and lcdm_results.get('pred_df') is not None and not lcdm_results['pred_df'].empty:
        df_out['pred_lcdm'] = lcdm_results['pred_df']['model_prediction']
        df_out['chi2_contrib_lcdm'] = ((df_out['value'] - df_out['pred_lcdm']) / df_out['error'])**2
    
    alt_model_name_safe = alt_model_name.replace(' ', '_').replace('.', '')
    if alt_model_results and alt_model_results.get('pred_df') is not None and not alt_model_results['pred_df'].empty:
        df_out[f'pred_{alt_model_name_safe}'] = alt_model_results['pred_df']['model_prediction']
        df_out[f'chi2_contrib_{alt_model_name_safe}'] = ((df_out['value'] - df_out[f'pred_{alt_model_name_safe}']) / df_out['error'])**2

    filename = os.path.join(csv_dir, f"{base_filename}_{bao_data_df.attrs.get('format_key', 'BAO')}_{alt_model_name_safe}_{get_timestamp()}.csv")
    try:
        df_out.to_csv(filename, index=False, float_format='%.6g'); logger.info(f"BAO results CSV saved to {filename}")
    except Exception as e:
        logger.error(f"Error saving BAO results CSV: {e}")