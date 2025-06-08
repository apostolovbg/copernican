# copernican_suite/copernican.py
"""
Copernican Suite - Main Orchestrator.
"""

import importlib.util
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Import sibling modules
import data_loaders
import cosmo_engine
import output_manager
import lcdm_model

def get_user_input_filepath(prompt_message, base_dir, must_exist=True):
    """Prompts the user for a filepath and validates it."""
    while True:
        filename = input(f"{prompt_message} (or 'c' to cancel): ").strip()
        if filename.lower() == 'c': return None
        filepath = os.path.join(base_dir, filename)
        if not must_exist or os.path.isfile(filepath): return filepath
        else: print(f"Error: File not found at '{filepath}'. Please try again.")

def load_alternative_model_plugin(model_filepath):
    """Dynamically loads an alternative cosmological model plugin."""
    logger = output_manager.get_logger()
    if not model_filepath.endswith(".py"): model_filepath += ".py"
    if not os.path.isfile(model_filepath):
        logger.error(f"Alternative model plugin file '{model_filepath}' not found.")
        return None
    try:
        module_name = os.path.splitext(os.path.basename(model_filepath))[0]
        spec = importlib.util.spec_from_file_location(module_name, model_filepath)
        alt_model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(alt_model_module)
        required_attrs = ['MODEL_NAME', 'PARAMETER_NAMES', 'INITIAL_GUESSES', 'PARAMETER_BOUNDS', 'distance_modulus_model']
        if not all(hasattr(alt_model_module, attr) for attr in required_attrs):
            logger.error(f"Model plugin '{os.path.basename(model_filepath)}' missing required attributes.")
            return None
        logger.info(f"Successfully loaded alternative model: {alt_model_module.MODEL_NAME}")
        return alt_model_module
    except Exception as e:
        logger.error(f"Error loading model plugin '{os.path.basename(model_filepath)}': {e}", exc_info=True)
        return None

def main_workflow():
    """Main workflow for the Copernican Suite."""
    try: SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError: SCRIPT_DIR = os.getcwd()

    # --- MODIFICATION: Define and create the output subdirectory ---
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- MODIFICATION: Pass the new output directory to the logger ---
    log_file = output_manager.setup_logging(run_name="CopernicanSuite_Run", log_dir=OUTPUT_DIR)
    logger = output_manager.get_logger()
    logger.info("=== Copernican Suite Initialized ===")
    logger.info(f"Running from base directory: {SCRIPT_DIR}")
    logger.info(f"All outputs will be saved to: {OUTPUT_DIR}")


    # --- 1. Configuration ---
    logger.info("\n--- Stage 1: Configuration ---")
    alt_model_filepath = get_user_input_filepath("Enter alternative model plugin filename (e.g., usmf2.py)", base_dir=SCRIPT_DIR)
    if not alt_model_filepath: return logger.info("Model selection canceled. Exiting.")
    alt_model_plugin = load_alternative_model_plugin(alt_model_filepath)
    if not alt_model_plugin: return logger.error("Failed to load alternative model. Exiting.")

    sne_data_filepath = get_user_input_filepath("Enter SNe Ia data filepath", base_dir=SCRIPT_DIR)
    if not sne_data_filepath: return logger.info("SNe Ia data selection canceled. Exiting.")
    
    sne_format_key = data_loaders._select_parser(data_loaders.SNE_PARSERS, "SNe")
    if not sne_format_key: return logger.info("SNe data format selection canceled. Exiting.")

    sne_loader_kwargs = {}
    if sne_format_key == "pantheon_plus_mu_cov_h2":
        cov_path = get_user_input_filepath("Enter SNe covariance matrix filename", base_dir=SCRIPT_DIR)
        if not cov_path: return logger.info("SNe covariance file selection canceled. Exiting.")
        sne_loader_kwargs['cov_filepath'] = cov_path
    
    sne_data_df = data_loaders.load_sne_data(sne_data_filepath, format_key=sne_format_key, **sne_loader_kwargs)
    if sne_data_df is None: return logger.error("Failed to load SNe Ia data. Exiting.")

    bao_data_filepath = get_user_input_filepath("Enter BAO data filename (e.g., bao1.json)", base_dir=SCRIPT_DIR)
    if not bao_data_filepath: return logger.info("BAO data selection canceled. Exiting.")
    bao_format_key = data_loaders._select_parser(data_loaders.BAO_PARSERS, "BAO")
    if not bao_format_key: return logger.info("BAO data format selection canceled. Exiting.")
    bao_data_df = data_loaders.load_bao_data(bao_data_filepath, format_key=bao_format_key)
    if bao_data_df is None: return logger.error("Failed to load BAO data. Exiting.")

    # --- 2. SNe Ia Fitting ---
    logger.info("\n--- Stage 2: Supernovae Ia Fitting ---")
    lcdm_sne_fit_results = cosmo_engine.fit_sne_parameters(sne_data_df, lcdm_model)
    alt_model_sne_fit_results = cosmo_engine.fit_sne_parameters(sne_data_df, alt_model_plugin)
    
    # --- 3. BAO Analysis ---
    logger.info("\n--- Stage 3: BAO Analysis ---")
    
    def run_bao_analysis(model_plugin, sne_fit_results):
        """Helper to run BAO analysis for a given model."""
        if not (sne_fit_results and sne_fit_results.get('success')):
            logger.warning(f"{model_plugin.MODEL_NAME} SNe fit failed; skipping BAO analysis.")
            return {'sne_fit_results': sne_fit_results, 'pred_df': None, 'rs_Mpc': np.nan, 'chi2_bao': np.inf}

        fitted_cosmo_p = list(sne_fit_results['fitted_cosmological_params'].values())
        pred_df, rs_Mpc = cosmo_engine.calculate_bao_observables(bao_data_df, model_plugin, fitted_cosmo_p)
        
        chi2_bao = np.inf
        if pred_df is not None and np.isfinite(rs_Mpc):
            chi2_bao = cosmo_engine.chi_squared_bao(bao_data_df, model_plugin, fitted_cosmo_p, rs_Mpc)
            logger.info(f"{model_plugin.MODEL_NAME} BAO: r_s = {rs_Mpc:.2f} Mpc, Chi2_BAO = {chi2_bao:.2f}")
        else:
            logger.warning(f"{model_plugin.MODEL_NAME} BAO calculation failed or produced invalid r_s.")
            
        return {'sne_fit_results': sne_fit_results, 'pred_df': pred_df, 'rs_Mpc': rs_Mpc, 'chi2_bao': chi2_bao}

    lcdm_full_results = run_bao_analysis(lcdm_model, lcdm_sne_fit_results)
    alt_model_full_results = run_bao_analysis(alt_model_plugin, alt_model_sne_fit_results)

    # --- 4. Output Generation ---
    logger.info("\n--- Stage 4: Generating Outputs ---")
    
    # --- MODIFICATION: Pass OUTPUT_DIR to all output functions ---
    output_manager.plot_hubble_diagram(sne_data_df, lcdm_sne_fit_results, alt_model_sne_fit_results, lcdm_model, alt_model_plugin, plot_dir=OUTPUT_DIR)
    
    if bao_data_df is not None:
        output_manager.plot_bao_observables(bao_data_df, lcdm_full_results, alt_model_full_results, lcdm_model, alt_model_plugin, plot_dir=OUTPUT_DIR)

    output_manager.save_sne_fit_results_csv([lcdm_sne_fit_results, alt_model_sne_fit_results], sne_data_df, csv_dir=OUTPUT_DIR)
    
    if bao_data_df is not None:
        output_manager.save_bao_results_csv(bao_data_df, lcdm_full_results, alt_model_full_results, alt_model_name=alt_model_plugin.MODEL_NAME, csv_dir=OUTPUT_DIR)

    logger.info("\n=== Copernican Suite Finished ===")
    logger.info(f"Main log file: {log_file}")
    logger.info(f"Check the directory '{OUTPUT_DIR}' for output plots and CSV files.")

if __name__ == "__main__":
    try:
        main_workflow()
    except Exception as e:
        # Use a fallback logger if the main one failed to initialize
        logger = output_manager.get_logger()
        if logger.hasHandlers():
            logger.critical("Unhandled exception in main_workflow!", exc_info=True)
        else:
            print("CRITICAL UNHANDLED EXCEPTION IN MAIN WORKFLOW:")
            import traceback
            traceback.print_exc()
    finally:
        # This block ensures plots are displayed in interactive environments
        # but only if figures were actually generated and not closed.
        # With the fixes, this block should ideally not be triggered.
        if plt.get_fignums():
            print("\nDisplaying plot(s). Close plot window(s) to exit script fully.")
            try:
                plt.show(block=True)
            except Exception as e_show:
                print(f"Error during final plt.show(): {e_show}")