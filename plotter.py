# plotter.py
# Handles all plot generation for the Copernican Suite.

"""
DEV NOTE (v1.4rc): This module has been significantly updated to fix the
BAO plotting bug and implement several stylistic improvements.

1.  REPAIRED BAO PLOTTING: The `create_bao_plot` function has been repaired.
    It now contains the logic to read the 'smooth_curves' data from the
    results dictionary and draw the best-fit model lines using `ax.plot()`.

2.  STYLE GUIDE ADHERENCE: The new plotting logic explicitly uses the
    `style_guide` parameter to set colors and linestyles, ensuring that all
    plots conform to the project's visual standards defined in `doc.json`.

3.  FOOTER FONT SIZE: The font size of the footer text on all plots has
    been increased from the small default to a more legible 8pt.

4.  ROBUSTNESS: The new BAO line plotting logic is enclosed in a
    try/except block to gracefully handle any errors and log them.
"""


import logging
import os
import matplotlib.pyplot as plt
import numpy as np

# --- Helper Functions ---

def _setup_plot_style(style_guide):
    """Sets global matplotlib styles from the style guide."""
    plt.style.use(style_guide.get('base_style', 'seaborn-v0_8-colorblind'))
    plt.rcParams['figure.facecolor'] = style_guide.get('figure_facecolor', '#f0f0f0')
    plt.rcParams['axes.facecolor'] = style_guide.get('axes_facecolor', '#ffffff')
    plt.rcParams['grid.color'] = style_guide.get('grid_color', '#cccccc')
    plt.rcParams['grid.linestyle'] = style_guide.get('grid_linestyle', '--')

def _create_footer_text(run_id, model1_name, model2_name):
    """Creates the standard footer text for all plots."""
    return f"Copernican Suite v1.4rc | Run ID: {run_id} | Models: {model1_name} vs {model2_name}"

def _save_plot(fig, results_meta, plot_type, dataset_name):
    """Saves the figure to the output directory with a standard filename."""
    run_id = results_meta['run_id']
    m1_name = results_meta['model1_name']
    m2_name = results_meta['model2_name']
    
    filename = f"{plot_type}-plot_{m1_name}-vs-{m2_name}_{dataset_name}_{run_id}.png"
    output_path = os.path.join('output', filename)
    
    try:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logging.info(f"Successfully saved plot to {output_path}")
    except Exception as e:
        logging.error(f"Failed to save plot {output_path}: {e}", exc_info=True)

# --- Hubble Diagram (SNe Ia) Plot ---

def create_hubble_plot(results, style_guide):
    """Generates and saves the Hubble Diagram and a residuals plot."""
    _setup_plot_style(style_guide)
    results_meta = results['metadata']
    sne_data = results['sne_analysis']
    df = pd.DataFrame(sne_data['detailed_df'])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True, 
                                   gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05})
    fig.patch.set_facecolor(plt.rcParams['figure.facecolor'])

    # --- Top Panel: Hubble Diagram ---
    ax1.set_title(style_guide.get('hubble_title', "Hubble Diagram"), fontsize=16)
    ax1.set_ylabel(style_guide.get('hubble_ylabel', "Distance Modulus ($\\mu$)"))
    ax1.grid(True)

    # Plot data points
    ax1.errorbar(df['z'], df['mu_data'], yerr=df['mu_err'], fmt='o', 
                 label="SNe Ia Data", color=style_guide.get('data_color', 'black'), 
                 ms=4, elinewidth=1, capsize=3, alpha=0.7)

    # Plot model curves
    m1_style = style_guide['model_lines']['lcdm']
    m2_style = style_guide['model_lines']['alt_model']
    m1_name = results_meta['model1_name']
    m2_name = results_meta['model2_name']
    
    ax1.plot(sne_data['model1_smooth_curve']['z_smooth'], sne_data['model1_smooth_curve']['mu_model_smooth'], 
             label=f"{m1_name} (Best Fit)", **m1_style)
    ax1.plot(sne_data['model2_smooth_curve']['z_smooth'], sne_data['model2_smooth_curve']['mu_model_smooth'],
             label=f"{m2_name} (Best Fit)", **m2_style)
    
    ax1.legend(loc='lower right')

    # --- Bottom Panel: Residuals ---
    ax2.set_xlabel(style_guide.get('xlabel_z', "Redshift (z)"), fontsize=12)
    ax2.set_ylabel(style_guide.get('residual_ylabel', "$\\Delta\\mu$ (Data-Model)"))
    ax2.grid(True)
    ax2.axhline(0, **style_guide['model_lines']['lcdm'])
    
    ax2.errorbar(df['z'], df['model2_residual'], yerr=df['mu_err'], fmt='o',
                 label=f"Residuals vs {m2_name}", color=m2_style['color'],
                 ms=4, elinewidth=1, capsize=3, alpha=0.7)
    
    ax2.legend(loc='lower right')

    # --- Final Touches ---
    footer_text = _create_footer_text(results_meta['run_id'], m1_name, m2_name)
    # MODIFIED (v1.4rc): Increased footer font size for legibility
    fig.text(0.5, 0.015, footer_text, ha='center', va='bottom', 
             fontsize=style_guide.get('footer_fontsize', 8), color='#666666')

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    dataset_name = results['sne_analysis']['detailed_df']['name']
    _save_plot(fig, results_meta, 'hubble', dataset_name)
    plt.close(fig)

# --- BAO Plot ---

def create_bao_plot(results, style_guide):
    """Generates and saves the BAO measurement plots."""
    _setup_plot_style(style_guide)
    results_meta = results['metadata']
    bao_data = results['bao_analysis']
    df = pd.DataFrame(bao_data['detailed_df'])
    
    # One plot per observable type
    for obs_type, group_df in df.groupby('observable_type'):
        fig, ax = plt.subplots(figsize=(10, 7))
        fig.patch.set_facecolor(plt.rcParams['figure.facecolor'])
        
        # Get plot titles and labels from the style guide
        plot_details = style_guide.get(obs_type, {})
        ax.set_title(plot_details.get('title', f"BAO Measurements: {obs_type}"), fontsize=16)
        ax.set_xlabel(style_guide.get('xlabel_z', "Redshift (z)"), fontsize=12)
        ax.set_ylabel(plot_details.get('ylabel', obs_type.replace('_', '/')), fontsize=12)
        ax.grid(True)
        
        # Plot data points from all surveys for this observable
        for survey, survey_df in group_df.groupby('survey'):
            ax.errorbar(survey_df['z'], survey_df['y_data'], yerr=survey_df['y_err'],
                        fmt='o', label=survey, ms=6, elinewidth=1.5, capsize=4, alpha=0.8)

        # --- NEW PLOTTING LOGIC (v1.4rc) ---
        # Draw the smooth model curves
        try:
            smooth_curves = bao_data.get('smooth_curves', {}).get(obs_type)
            if smooth_curves:
                m1_style = style_guide['model_lines']['lcdm']
                m2_style = style_guide['model_lines']['alt_model']
                m1_name = results_meta['model1_name']
                m2_name = results_meta['model2_name']

                # Plot Model 1 (LCDM)
                ax.plot(smooth_curves['z_smooth'], smooth_curves['model1_y_smooth'],
                        label=f"{m1_name} (Best Fit)", **m1_style)
                
                # Plot Model 2 (Alternative)
                ax.plot(smooth_curves['z_smooth'], smooth_curves['model2_y_smooth'],
                        label=f"{m2_name} (Best Fit)", **m2_style)
                logging.info(f"Successfully plotted smooth model curves for '{obs_type}'.")
            else:
                logging.warning(f"No smooth curve data found for BAO observable '{obs_type}'. Model lines will not be plotted.")
        except Exception as e:
            logging.error(f"Failed to plot BAO model lines for '{obs_type}': {e}", exc_info=True)
            
        # The legend should include both data points and the new model lines
        ax.legend(loc='best')
        
        # --- Final Touches ---
        footer_text = _create_footer_text(results_meta['run_id'], results_meta['model1_name'], results_meta['model2_name'])
        # MODIFIED (v1.4rc): Increased footer font size for legibility
        fig.text(0.5, 0.01, footer_text, ha='center', va='bottom', 
                 fontsize=style_guide.get('footer_fontsize', 8), color='#666666')
        
        plt.tight_layout(rect=[0, 0.03, 1, 1])
        
        dataset_name = f"BAO_{obs_type.replace('_over_', '-')}"
        _save_plot(fig, results_meta, 'bao', dataset_name)
        plt.close(fig)

# --- Main Entry Point for the Module ---
# This is a passive module. Functions are called by output_manager.py