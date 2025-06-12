# plotter.py
# Handles all plot generation for the Copernican Suite.

"""
DEV NOTE (v1.4g):
codex/fix-bug-in-data_readers.py-and-explain-long-term-solution
Updated footer text to v1.4g and cleaned up BAO plotting code.
The earlier v1.4rc8 fix for BAO lines remains intact.

gggxa2-codex/fix-bug-in-data_readers.py-and-explain-long-term-solution
Updated footer text to v1.4g and cleaned up BAO plotting code.
The earlier v1.4rc8 fix for BAO lines remains intact.

Minor refinements for the stabilized data loaders.
The footer now reports v1.4g. The BAO plotting fix from v1.4rc8 is retained.
1.4g


---
(Previous notes from v1.4rc2 preserved below)
...
"""

import os
import ast
import matplotlib.pyplot as plt
import pandas as pd

# --- Helper Functions ---

def _reconstruct_df_from_split(split_dict):
    """Reconstructs a Pandas DataFrame from the 'split' dictionary format."""
    return pd.DataFrame(split_dict['data'], index=split_dict['index'], columns=split_dict['columns'])

def _setup_plot_style(style_guide):
    """Sets global matplotlib styles from the style guide."""
    plt.style.use(style_guide.get('base_style', 'seaborn-v0_8-colorblind'))
    plt.rcParams['figure.facecolor'] = style_guide.get('figure_facecolor', '#f0f0f0')
    plt.rcParams['axes.facecolor'] = style_guide.get('axes_facecolor', '#ffffff')
    plt.rcParams['grid.color'] = style_guide.get('grid_color', '#cccccc')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']

def _create_footer_text(run_id, m1_name, m2_name):
    """Creates the standard footer text for all plots."""
    return f"Copernican Suite v1.4g | Run ID: {run_id} | Comparison: {m1_name} vs. {m2_name}"

def _create_info_box_text(model_name, model_meta, fit_results):
    """Creates the text content for a model's info box."""
    params_latex = [p['latex'] for p_name, p in model_meta['parameters'].items() if p['role'] == 'cosmological']
    param_str = ', '.join(params_latex)
    text = f"Model: {model_name}\n"
    text += f"Eq: {model_meta['equations']['sne'][0]}\n" # Show first SNe equation
    text += f"Params: {param_str}\n"
    text += "--- Best Fit ---\n"
    for key, val in fit_results['best_fit_params'].items():
        if key in model_meta['parameters']:
             text += f"{model_meta['parameters'][key]['latex']} = {val:.4f}\n"
    text += f"$\\chi^2 / \\mathrm{{dof}} = {fit_results['min_chi2']:.2f} / {fit_results['dof']} = {fit_results['reduced_chi2']:.3f}"
    return text

def _save_plot(fig, filename_prefix, output_dir):
    """Saves the figure to the output directory."""
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        filepath = os.path.join(output_dir, f"{filename_prefix}.jpg")
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Plot saved to '{os.path.basename(filepath)}'")
    except Exception as e:
        print(f"Error saving plot '{filename_prefix}.jpg': {e}")
    finally:
        plt.close(fig)

# --- Plotting Functions ---

def create_hubble_plot(results_json, style_guide):
    """Generates and saves the Hubble Diagram with residuals."""
    print("Generating Hubble Diagram...")
    _setup_plot_style(style_guide)

    # --- Data Extraction ---
    results_meta = results_json['metadata']
    m1_name = results_meta['model1_name']
    m2_name = results_meta['model2_name']
    df = _reconstruct_df_from_split(results_json['sne_analysis']['detailed_df'])

    # --- Plot Setup ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05})
    fig.suptitle(style_guide.get('title', 'Hubble Diagram'),
                 fontsize=style_guide.get('title_fontsize', 16), weight='bold')

    # --- Main Hubble Plot (ax1) ---
    dp_style = style_guide.get('data_points', {})
    ax1.errorbar(df['z'], df['mu'], yerr=df['mu_err'], fmt='o', **dp_style)

    smooth1 = results_json['sne_analysis']['model1_smooth_curve']
    smooth2 = results_json['sne_analysis']['model2_smooth_curve']
    m1_style = style_guide.get('model_lines', {}).get('lcdm', {})
    m2_style = style_guide.get('model_lines', {}).get('alt_model', {})
    ax1.plot(smooth1['z_smooth'], smooth1['mu_model_smooth'], label=m1_name, **m1_style)
    ax1.plot(smooth2['z_smooth'], smooth2['mu_model_smooth'], label=m2_name, **m2_style)

    ax1.set_ylabel(style_guide.get('ylabel', 'Distance Modulus (μ)'),
                   fontsize=style_guide.get('label_fontsize', 12))
    ax1.legend(loc='upper left', fontsize=style_guide.get('legend_fontsize', 10))
    ax1.grid(True, linestyle='--', color=style_guide.get('grid_color', '#cccccc'), alpha=0.7)

    # --- Info Boxes ---
    m1_box_style = ast.literal_eval(style_guide['info_boxes'].get('lcdm_style', '{}'))
    m2_box_style = ast.literal_eval(style_guide['info_boxes'].get('alt_model_style', '{}'))
    m1_text = _create_info_box_text(m1_name, results_meta['model1_metadata'], results_json['sne_analysis']['model1_fit_results'])
    m2_text = _create_info_box_text(m2_name, results_meta['model2_metadata'], results_json['sne_analysis']['model2_fit_results'])

    ax1.text(0.03, 0.97, m1_text, transform=ax1.transAxes, fontsize=style_guide.get('info_box_fontsize', 9),
             verticalalignment='top', bbox=m1_box_style)
    ax1.text(0.97, 0.03, m2_text, transform=ax1.transAxes, fontsize=style_guide.get('info_box_fontsize', 9),
             verticalalignment='bottom', horizontalalignment='right', bbox=m2_box_style)

    # --- Residuals Plot (ax2) ---
    res_style = style_guide.get('residuals', {})
    ax2.errorbar(df['z'], df['model1_residual'], yerr=df['mu_err'], fmt='o',
                 markersize=3, color=m1_style.get('color', '#D55E00'), alpha=0.5, label=m1_name)
    ax2.errorbar(df['z'], df['model2_residual'], yerr=df['mu_err'], fmt='o',
                 markersize=3, color=m2_style.get('color', '#0072B2'), alpha=0.5, label=m2_name)

    ax2.axhline(0, color=res_style.get('hline_color', 'black'),
                linestyle=res_style.get('hline_style', ':'),
                linewidth=res_style.get('hline_width', 1))
    ax2.set_xlabel(style_guide.get('xlabel', 'Redshift (z)'),
                   fontsize=style_guide.get('label_fontsize', 12))
    ax2.set_ylabel(res_style.get('ylabel', 'Δμ (Data-Model)'), fontsize=style_guide.get('label_fontsize', 12))
    ax2.grid(True, linestyle='--', color=style_guide.get('grid_color', '#cccccc'), alpha=0.7)

    # --- Final Touches ---
    footer_text = _create_footer_text(results_meta['run_id'], m1_name, m2_name)
    fig.text(0.5, 0.01, footer_text, ha='center', va='bottom',
             fontsize=style_guide.get('footer_fontsize', 8), color='#666666')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    output_dir = os.path.join(os.getcwd(), 'output')
    dataset_name = os.path.basename(results_json['sne_analysis']['filepath']).split('.')[0]
    filename = f"hubble-plot_{m1_name}-vs-{m2_name}_{dataset_name}_{results_meta['run_id']}"
    _save_plot(fig, filename, output_dir)

def create_bao_plot(results_json, style_guide):
    """Generates and saves the BAO plot."""
    print("Generating BAO Plot...")
    _setup_plot_style(style_guide)

    # --- Data Extraction ---
    results_meta = results_json['metadata']
    m1_name = results_meta['model1_name']
    m2_name = results_meta['model2_name']
    df = _reconstruct_df_from_split(results_json['bao_analysis']['detailed_df'])
    smooth_curves_data = results_json['bao_analysis']['smooth_curves']
    unique_observables = sorted(df['observable_type'].unique())

    # --- Plot Setup ---
    fig, ax = plt.subplots(figsize=(10, 6)) # Wider to accommodate info boxes
    ax.set_title(style_guide.get('title', 'Baryon Acoustic Oscillations'),
                 fontsize=style_guide.get('title_fontsize', 16), weight='bold')

    # --- Plot Data and Model Lines for Each Observable ---
    dp_style = style_guide.get('data_points', {})
    dp_colors = dp_style.get('colors', ['#009E73', '#F0E442', '#56B4E9'])
    m1_line_style = style_guide.get('model_lines', {}).get('lcdm', {})
    m2_line_style = style_guide.get('model_lines', {}).get('alt_model', {})
    m1_colors = m1_line_style.get('colors', ['#8B0000', '#FF0000', '#FA8072'])
    m2_colors = m2_line_style.get('colors', ['#00008B', '#0000FF', '#4169E1'])

    for i, obs_type in enumerate(unique_observables):
        group = df[df['observable_type'] == obs_type]
        color = dp_colors[i % len(dp_colors)]
        codex/fix-bug-in-data_readers.py-and-explain-long-term-solution

        # Use 'z' for the x-axis instead of the raw 'redshift' column (v1.4g).
        
        gggxa2-codex/fix-bug-in-data_readers.py-and-explain-long-term-solution


        
        # 1.4g
        # FIX (v1.4g): Use 'z' for the x-axis, not 'redshift'.
        x_data, y_data, y_err_data = group['z'], group['value'], group['error']

        # Plot data points
        ax.errorbar(x_data, y_data, yerr=y_err_data,
                    fmt='o', mfc=color, mec='black', label=obs_type,
                    ecolor=color, **dp_style)

        # Plot smooth model curves
        if obs_type in smooth_curves_data:
            curves = smooth_curves_data[obs_type]
            m1_c = m1_colors[i % len(m1_colors)]
            m2_c = m2_colors[i % len(m2_colors)]
            ax.plot(
                curves['z_smooth'],
                curves['model1_y_smooth'],
                color=m1_c,
                linestyle=ast.literal_eval(m1_line_style.get('linestyle', "'-'")),
                linewidth=m1_line_style.get('linewidth', 2),
            )
            ax.plot(
                curves['z_smooth'],
                curves['model2_y_smooth'],
                color=m2_c,
                linestyle=ast.literal_eval(m2_line_style.get('linestyle', "'-'")),
                linewidth=m2_line_style.get('linewidth', 2),
            )
        else:
            print(f"Warning: No smooth curve data found for BAO observable '{obs_type}'.")

    # Add dummy lines for legend
    ax.plot([], [], label=m1_name, **m1_line_style, color=m1_colors[0])
    ax.plot([], [], label=m2_name, **m2_line_style, color=m2_colors[0])

    # --- Axes, Grid, and Legend ---
    ax.set_xlabel(style_guide.get('xlabel', 'Redshift (z)'), fontsize=style_guide.get('label_fontsize', 12))
    ax.set_ylabel('BAO Observable Value', fontsize=style_guide.get('label_fontsize', 12))
    ax.grid(True, linestyle='--', color=style_guide.get('grid_color', '#cccccc'), alpha=0.7)
    ax.legend(loc='best', fontsize=style_guide.get('legend_fontsize', 10))

    # --- Info Boxes Outside Plot ---
    m1_box_style = ast.literal_eval(style_guide['info_boxes'].get('lcdm_style', '{}'))
    m2_box_style = ast.literal_eval(style_guide['info_boxes'].get('alt_model_style', '{}'))
    m1_text = _create_info_box_text(m1_name, results_meta['model1_metadata'], results_json['sne_analysis']['model1_fit_results'])
    m2_text = _create_info_box_text(m2_name, results_meta['model2_metadata'], results_json['sne_analysis']['model2_fit_results'])

    # Position boxes on the right side of the figure
    fig.text(0.99, 0.95, m1_text, transform=fig.transFigure, fontsize=style_guide.get('info_box_fontsize', 9),
             verticalalignment='top', horizontalalignment='right', bbox=m1_box_style)
    fig.text(0.99, 0.65, m2_text, transform=fig.transFigure, fontsize=style_guide.get('info_box_fontsize', 9),
             verticalalignment='top', horizontalalignment='right', bbox=m2_box_style)

    # --- Final Touches ---
    footer_text = _create_footer_text(results_meta['run_id'], m1_name, m2_name)
    fig.text(0.5, 0.01, footer_text, ha='center', va='bottom',
             fontsize=style_guide.get('footer_fontsize', 8), color='#666666')

    # Adjust layout to make room for info boxes and footer
    plt.tight_layout(rect=[0, 0.03, 0.82, 0.95]) # Make right margin smaller

    output_dir = os.path.join(os.getcwd(), 'output')
    filename = f"bao-plot_{m1_name}-vs-{m2_name}_{results_meta['run_id']}"
    _save_plot(fig, filename, output_dir)
