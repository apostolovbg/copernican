# output_manager.py
# Dispatches final results to appropriate output generation modules.

"""
DEV NOTE (v1.4rc): Fixed a critical bug where this module was calling a
non-existent function (`create_csv_outputs`) in the csv_writer. The dispatch
logic has been updated to call the correct, specific writer functions
(e.g., `create_sne_csv`, `create_bao_csv`) based on the contents of the
results dictionary. This aligns the dispatcher with the v1.4 architecture.
"""

import logging
import os
import json

# Import the output modules that this manager will dispatch to
import csv_writer
import plotter

def _load_style_guide():
    """Loads the plotting style guide from doc.json."""
    try:
        with open('doc.json', 'r') as f:
            doc = json.load(f)
        return doc.get('plottingStyleGuide', {})
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.warning(f"Could not load or parse plotting style guide from doc.json: {e}. Using defaults.")
        return {}

def generate_outputs(results_json):
    """
    Main entry point for the output manager.
    It inspects the results and calls the correct output modules.
    
    Args:
        results_json (dict): The final results from the cosmo_engine.
    """
    if not results_json:
        logging.error("Output manager received empty results. No outputs will be generated.")
        return

    logging.info("\n--- Stage 4: Generating Outputs ---")
    
    # Load style guide once for all plotting operations
    style_guide = _load_style_guide()

    # --- Dispatch to CSV Writer ---
    # MODIFIED (v1.4rc): Call the correct, specific functions in csv_writer
    try:
        logging.info("Dispatching tasks to csv_writer...")
        csv_writer.create_fit_summary_csv(results_json, style_guide)
        csv_writer.create_sne_csv(results_json, style_guide)
        csv_writer.create_bao_csv(results_json, style_guide)
    except Exception as e:
        logging.error(f"An error occurred during CSV generation: {e}", exc_info=True)

    # --- Dispatch to Plotter ---
    try:
        logging.info("Dispatching tasks to plotter...")
        # Check for SNe data and create Hubble plot
        if 'sne_analysis' in results_json and results_json['sne_analysis'].get('detailed_df'):
            plotter.create_hubble_plot(results_json, style_guide.get('hubble_plot', {}))
        
        # Check for BAO data and create BAO plot(s)
        if 'bao_analysis' in results_json and results_json['bao_analysis'].get('detailed_df'):
            plotter.create_bao_plot(results_json, style_guide.get('bao_plot', {}))
            
    except Exception as e:
        logging.error(f"An error occurred during plot generation: {e}", exc_info=True)
        
    logging.info("All output generation tasks complete.")