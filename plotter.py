# copernican_suite/plotter.py
"""
DEV NOTE (v1.4b-final): This version contains the definitive fix for the
persistent bug that prevented BAO model lines from being plotted.

BUG FIX (Definitive): The helper function `plot_model_curves` was crashing
with a `NameError` because it was not passed the matplotlib axes object (`ax`)
to draw on. This error was not being reported to the logs, causing a "silent
failure". The function signature and the calls to it have been corrected to
pass the `ax` object.

ENHANCEMENT: The calls to `plot_model_curves` are now wrapped in a
try...except block to log any future potential failures loudly and prevent
this kind of silent bug from recurring.
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import importlib.util

# --- Internal Helper Functions ---

def _reconstruct_df_from_split(df_dict):
    """
    Internal helper to safely reconstruct a pandas DataFrame from the 'split'
    dictionary format.
    """
    if not isinstance(df_dict, dict) or 'data' not in df_dict:
        return pd.DataFrame()
    try:
        return pd.DataFrame(df_dict['data'], index=df_dict.get('index'), columns=df_dict['columns'])
    except Exception as e:
        logging.getLogger().error(f"Plotter Error: Could not reconstruct DataFrame from dict. Error: {e}")
        return pd.DataFrame()

def _format_model_summary_text(model_key, results, is_sne_summary):
    """
    Formats the detailed text block for the plot's information boxes.
    It pulls all necessary data directly from the main results dictionary.
    """
    logger = logging.getLogger()
    lines = []
    
    try:
        model_info = results['inputs']['models'][model_key]
        model_name_raw = model_info.get('name', 'N/A')
        model_name_latex = model_name_raw.replace('_', r'\_')
        lines.append(fr"**Model: {model_name_latex}**")

        eq_key = 'equations_sn' if is_sne_summary else 'equations_bao'
        equations = model_info.get(eq_key, [])
        if equations:
            lines.append(r"$\mathbf{Mathematical\ Form:}$")
            for eq_line in equations:
                lines.append(f"  {eq_line}")

        lines.append(r"$\mathbf{Cosmological\ Parameters:}$")
        param_names = model_info.get('parameters', [])
        param_latex_names = [p.replace('_', r'\_') for p in param_names]
        
        fit_results = results['results'][model_key]['sne_fit']
        best_fit_params = fit_results.get('best_fit_params', {})
        
        for i, name in enumerate(param_names):
            val = best_fit_params.get(name)
            latex_name = param_latex_names[i]
            lines.append(fr"  ${latex_name}$ = ${val:.4g}$" if val is not None else f"  {latex_name} = N/A")

        if is_sne_summary:
            nuisance_params = {k: v for k, v in best_fit_params.items() if k not in param_names}
            if nuisance_params:
                lines.append(r"$\mathbf{SNe\ Nuisance\ Parameters:}$")
                for name, val in nuisance_params.items():
                    name_latex = {"M_abs": r"M", "alpha": r"\alpha", "beta": r"\beta"}.get(name, name)
                    lines.append(fr"  ${name_latex}$ = ${val:.4g}$")

            lines.append(r"$\mathbf{SNe\ Fit\ Statistics:}$")
            lines.append(fr"  $\chi^2_{{SNe}}$ = {fit_results.get('chi2', np.nan):.2f}")
        else:
            bao_results = results['results'][model_key]['bao_analysis']
            lines.append(r"$\mathbf{BAO\ Test\ Results:}$")
            lines.append(fr"  $r_s$ = {bao_results.get('rs_Mpc', np.nan):.2f} Mpc")
            lines.append(fr"  $\chi^2_{{BAO}}$ = {bao_results.get('chi2', np.nan):.2f}")

    except KeyError as e:
        logger.error(f"Plotter Error: Missing key '{e}' while formatting model summary for {model_key}.")
        return f"Error: Data for {model_key} is incomplete."

    return "\n".join(lines)
    
def _get_smooth_bao_predictions(z_smooth, model_key, results):
    """
    Calculates smooth model prediction curves for BAO observables, with robust logging.
    """
    logger = logging.getLogger()
    model_info = results['inputs']['models'][model_key]
    model_filepath = model_info.get('filepath', 'lcdm_model.py')
    
    try:
        spec = importlib.util.spec_from_file_location(f"temp_model_for_plot_{model_key}", model_filepath)
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)

        num_cosmo_params = len(model_info['parameters'])
        best_fit_params = list(results['results'][model_key]['sne_fit']['best_fit_params'].values())[:num_cosmo_params]
        
        rs_model = results['results'][model_key]['bao_analysis']['rs_Mpc']
        logger.info(f"PLOTTER: Calculating smooth BAO curves for {model_key} with r_s = {rs_model:.2f}")
        if pd.isna(rs_model):
            logger.error(f"PLOTTER: Cannot calculate smooth BAO curves for {model_key} because r_s is NaN.")
            return {}

        c_km_s = model_module.FIXED_PARAMS.get("C_LIGHT_KM_S", 299792.458)
        hz = model_module.get_Hz_per_Mpc(z_smooth, *best_fit_params)

        dh_raw = np.divide(c_km_s, hz, out=np.full_like(hz, np.nan), where=(hz != 0) & np.isfinite(hz))
        
        predictions = {
            'z': z_smooth,
            'dm_over_rs': model_module.get_comoving_distance_Mpc(z_smooth, *best_fit_params) / rs_model,
            'dh_over_rs': dh_raw / rs_model,
            'dv_over_rs': model_module.get_DV_Mpc(z_smooth, *best_fit_params) / rs_model
        }
        
        for key, val in predictions.items():
            if key != 'z' and np.all(np.isnan(val)):
                logger.warning(f"PLOTTER: Calculation for '{key}' for model '{model_key}' resulted in all NaNs.")

        return predictions

    except Exception as e:
        logger.critical(f"PLOTTER: CRITICAL FAILURE in _get_smooth_bao_predictions for {model_key}. Cannot draw lines.", exc_info=True)
        return {}


# --- Main Public Plotting Functions ---

def create_hubble_diagram(results, output_dir='output'):
    """
    Generates and saves the Hubble diagram, replicating the v1.3 style.
    """
    logger = logging.getLogger()
    dataset_name_raw = results['inputs']['datasets']['sne_data']['attributes'].get('dataset_name_attr', 'SNe_data')
    dataset_name = dataset_name_raw.replace('.dat', '').replace('.txt', '')
    logger.info(f"Generating Hubble Diagram for {dataset_name}...")

    lcdm_df = _reconstruct_df_from_split(results['results']['lcdm'].get('sne_detailed_df', {}))
    alt_df = _reconstruct_df_from_split(results['results']['alt_model'].get('sne_detailed_df', {}))

    if lcdm_df.empty or alt_df.empty:
        logger.error("PLOTTER: Cannot generate SNe plot, detailed dataframes are missing or empty.")
        return

    diag_errors = results['inputs']['datasets']['sne_data']['attributes'].get('diag_errors_for_plot')
    if diag_errors is None:
        diag_errors = lcdm_df['e_mu_obs'].values
    
    font_sizes = {'title': 22, 'label': 18, 'legend': 14, 'infobox': 12, 'ticks': 12, 'footer': 8}
    fig, axs = plt.subplots(2, 1, figsize=(17, 12), sharex=True, gridspec_kw={'height_ratios':[3,1.5],'hspace':0.05})
    plt.subplots_adjust(left=0.08, bottom=0.08, right=0.75, top=0.92)
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except Exception:
        logger.warning("Seaborn-v0_8-darkgrid style not found, using default.")

    axs[0].errorbar(lcdm_df['zcmb'], lcdm_df['mu_obs'], yerr=diag_errors, fmt='.', color='darkgray', alpha=0.6, label=f"Data: {dataset_name}", elinewidth=1, capsize=2, ms=5, ecolor='lightgray', zorder=1)

    axs[0].plot(lcdm_df['zcmb'], lcdm_df['mu_model'], 'r-', lw=2.5, label=fr'$\Lambda$CDM ($\chi^2$={results["results"]["lcdm"]["sne_fit"]["chi2"]:.2f})')
    axs[1].errorbar(lcdm_df['zcmb'], lcdm_df['residual'], yerr=diag_errors, fmt='.', color='red', alpha=0.4, elinewidth=1, capsize=2, ms=4, label=r'Avg. $\Lambda$CDM Res.')
    
    alt_model_name_raw = results['inputs']['models']['alt_model']['name']
    alt_model_name_latex = alt_model_name_raw.replace('_', r'\_')
    axs[0].plot(alt_df['zcmb'], alt_df['mu_model'], 'b--', lw=2.5, label=fr'{alt_model_name_latex} ($\chi^2$={results["results"]["alt_model"]["sne_fit"]["chi2"]:.2f})')
    axs[1].errorbar(alt_df['zcmb'], alt_df['residual'], yerr=diag_errors, fmt='.', mfc='none', mec='blue', ecolor='lightblue', alpha=0.4, elinewidth=1, capsize=2, ms=4, label=f'Avg. {alt_model_name_latex} Res.')
    
    bins = np.histogram_bin_edges(lcdm_df['zcmb'], bins=20)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    lcdm_binned_res, _ = np.histogram(lcdm_df['zcmb'], bins=bins, weights=lcdm_df['residual'])
    alt_binned_res, _ = np.histogram(alt_df['zcmb'], bins=bins, weights=alt_df['residual'])
    counts, _ = np.histogram(lcdm_df['zcmb'], bins=bins)
    
    safe_counts = np.where(counts > 0, counts, 1)
    axs[1].plot(bin_centers, lcdm_binned_res / safe_counts, color='darkred', ls='-', lw=2, zorder=10)
    axs[1].plot(bin_centers, alt_binned_res / safe_counts, color='darkblue', ls='--', lw=2, zorder=11)
    
    axs[0].set_ylabel(r'Distance Modulus ($\mu$)', fontsize=font_sizes['label'])
    axs[0].legend(fontsize=font_sizes['legend'], loc='lower right')
    axs[0].set_title(f'Hubble Diagram: $\Lambda$CDM vs. {alt_model_name_raw} on {dataset_name}', fontsize=font_sizes['title'])
    
    axs[1].axhline(0, color='black', ls='--', lw=1)
    axs[1].set_xlabel('Redshift (z)', fontsize=font_sizes['label'])
    axs[1].set_ylabel(r'$\mu_{obs} - \mu_{model}$', fontsize=font_sizes['label'])
    axs[1].legend(fontsize=font_sizes['legend'], loc='lower right')
    
    for ax in axs:
        ax.minorticks_on()
        ax.tick_params(axis='both', which='major', labelsize=font_sizes['ticks'])

    bbox_lcdm = dict(boxstyle='round,pad=0.5', fc='#FFEEEE', ec='darkred', alpha=0.9)
    bbox_alt = dict(boxstyle='round,pad=0.5', fc='#EEF2FF', ec='darkblue', alpha=0.9)
    fig.text(0.77, 0.90, _format_model_summary_text('lcdm', results, is_sne_summary=True), transform=fig.transFigure, fontsize=font_sizes['infobox'], va='top', ha='left', wrap=True, bbox=bbox_lcdm)
    fig.text(0.77, 0.52, _format_model_summary_text('alt_model', results, is_sne_summary=True), transform=fig.transFigure, fontsize=font_sizes['infobox'], va='top', ha='left', wrap=True, bbox=bbox_alt)

    run_id = results['metadata']['run_id']
    engine_name = results['metadata']['engine_name']
    version = results['metadata']['project_version']
    footer_text = f"Made with Copernican Suite v{version} using {engine_name} on {run_id}"
    fig.text(0.5, 0.01, footer_text, ha='center', va='bottom', fontsize=font_sizes['footer'], color='gray')
    
    run_id = results['metadata']['run_id']
    filename = f"hubble-plot_LambdaCDM-vs-{alt_model_name_raw}_{dataset_name}_{run_id}.png"
    filepath = os.path.join(output_dir, filename)
    try:
        plt.savefig(filepath, dpi=300)
        logger.info(f"Hubble diagram saved to {filename}")
    except Exception as e:
        logger.error(f"Failed to save Hubble diagram: {e}")
    finally:
        plt.close(fig)

def plot_model_curves(ax, smooth_data, colors, linestyle, prefix, obs_types_from_data, alpha=1.0):
    """ Helper function to draw the model lines on a given axis. """
    if not smooth_data: return
    obs_to_key_map = {'DV_rs': 'dv_over_rs', 'DM_rs': 'dm_over_rs', 'DH_rs': 'dh_over_rs'}
    obs_to_label_map = {'DV_rs': r'$D_V/r_s$', 'DM_rs': r'$D_M/r_s$', 'DH_rs': r'$D_H/r_s$'}
    color_map = dict(zip(obs_types_from_data, colors))
    z = smooth_data['z']
    
    for obs_type in obs_types_from_data:
        key = obs_to_key_map.get(obs_type)
        if key in smooth_data and not np.all(np.isnan(smooth_data[key])):
            valid = np.isfinite(smooth_data[key]) & np.isfinite(z)
            if np.any(valid):
                label = f'{prefix} ({obs_to_label_map[obs_type]})'
                ax.plot(z[valid], smooth_data[key][valid], color=color_map[obs_type], ls=linestyle, lw=2.5, alpha=alpha, label=label)

def create_bao_plot(results, output_dir='output'):
    """
    Generates and saves the BAO observables plot, with fixed logic and new styling.
    """
    logger = logging.getLogger()
    dataset_name_raw = results['inputs']['datasets']['bao_data']['attributes'].get('dataset_name_attr', 'BAO_data')
    dataset_name = dataset_name_raw.replace('.json', '')
    logger.info(f"Generating BAO Plot for {dataset_name}...")

    bao_df = _reconstruct_df_from_split(results['results']['lcdm']['bao_analysis']['detailed_df'])
    if bao_df.empty:
        logger.error("PLOTTER: Cannot generate BAO plot, detailed dataframe is missing or empty.")
        return

    font_sizes = {'title': 22, 'label': 18, 'legend': 14, 'infobox': 12, 'ticks': 12, 'footer': 8}
    fig, ax = plt.subplots(figsize=(17, 10))
    plt.subplots_adjust(left=0.08, bottom=0.1, right=0.75, top=0.90)
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except Exception:
        logger.warning("Seaborn-v0_8-darkgrid style not found, using default.")
    
    obs_types_from_data = sorted(list(bao_df['observable_type'].unique()))
    
    data_cmap = plt.get_cmap('viridis')
    data_colors = data_cmap(np.linspace(0, 0.8, len(obs_types_from_data)))
    data_color_map = dict(zip(obs_types_from_data, data_colors))
    
    for obs_type in obs_types_from_data:
        subset = bao_df[bao_df['observable_type'] == obs_type]
        label = fr"Data: ${obs_type.replace('_', r'\/')}$"
        ax.errorbar(subset['redshift'], subset['value'], yerr=subset['error'], fmt='o', label=label, capsize=4, color=data_color_map[obs_type], ms=8, zorder=10)

    z_smooth = np.linspace(0, bao_df['redshift'].max() * 1.1, 300)
    smooth_lcdm = _get_smooth_bao_predictions(z_smooth, 'lcdm', results)
    smooth_alt = _get_smooth_bao_predictions(z_smooth, 'alt_model', results)
    
    alt_model_name_raw = results['inputs']['models']['alt_model']['name']
    alt_model_name_latex = alt_model_name_raw.replace('_', r'\_')
    
    # --- Loudly report if smooth data calculation failed ---
    if not smooth_lcdm:
        logger.error("PLOTTER: Failed to generate smooth curve data for LCDM. Its lines will be missing.")
    if not smooth_alt:
        logger.error("PLOTTER: Failed to generate smooth curve data for the alternative model. Its lines will be missing.")
    
    try:
        lcdm_colors = ['#8B0000', '#FF0000', '#FF6347']
        alt_colors = ['#00008B', '#0000FF', '#4169E1']
        
        # BUG FIX: Pass the `ax` object to the helper function.
        plot_model_curves(ax, smooth_lcdm, lcdm_colors, linestyle='-', prefix=r'$\Lambda$CDM', obs_types_from_data=obs_types_from_data)
        plot_model_curves(ax, smooth_alt, alt_colors, linestyle=(0, (8, 5)), prefix=alt_model_name_latex, obs_types_from_data=obs_types_from_data)

    except Exception as e:
        logger.critical("PLOTTER: CRITICAL FAILURE during BAO line plotting.", exc_info=True)

    ax.set_xlabel('Redshift (z)', fontsize=font_sizes['label'])
    ax.set_ylabel(r'$D_X/r_s$', fontsize=font_sizes['label'])
    ax.set_title(f'BAO Observables vs. Redshift: $\Lambda$CDM vs. {alt_model_name_raw} on {dataset_name}', fontsize=font_sizes['title'])
    ax.legend(fontsize=font_sizes['legend'], loc='best', ncol=2)
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', labelsize=font_sizes['ticks'])
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    bbox_lcdm = dict(boxstyle='round,pad=0.5', fc='#FFEEEE', ec='darkred', alpha=0.9)
    bbox_alt = dict(boxstyle='round,pad=0.5', fc='#EEF2FF', ec='darkblue', alpha=0.9)
    fig.text(0.77, 0.90, _format_model_summary_text('lcdm', results, is_sne_summary=False), transform=fig.transFigure, fontsize=font_sizes['infobox'], va='top', ha='left', wrap=True, bbox=bbox_lcdm)
    fig.text(0.77, 0.55, _format_model_summary_text('alt_model', results, is_sne_summary=False), transform=fig.transFigure, fontsize=font_sizes['infobox'], va='top', ha='left', wrap=True, bbox=bbox_alt)
    
    run_id = results['metadata']['run_id']
    engine_name = results['metadata']['engine_name']
    version = results['metadata']['project_version']
    footer_text = f"Made with Copernican Suite v{version} using {engine_name} on {run_id}"
    fig.text(0.5, 0.015, footer_text, ha='center', va='bottom', fontsize=font_sizes['footer'], color='gray')

    filename = f"bao-plot_LambdaCDM-vs-{alt_model_name_raw}_{dataset_name}_{run_id}.png"
    filepath = os.path.join(output_dir, filename)
    try:
        plt.savefig(filepath, dpi=300)
        logger.info(f"BAO plot saved to {filename}")
    except Exception as e:
        logger.error(f"Failed to save BAO plot: {e}")
    finally:
        plt.close(fig)