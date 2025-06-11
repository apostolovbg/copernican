# copernican_suite/output_manager.py
"""
DEV NOTE (v1.4b): This module has been heavily refactored into a "Dispatcher"
as part of the v1.4b architectural update. Its responsibility is no longer
to generate outputs itself, but to manage and delegate those tasks to the new,
specialized modules: `plotter.py` and `csv_writer.py`.

- The single public function `generate_outputs` remains, but its internal
  logic is now much simpler.
- It parses the Results JSON and calls the main functions from the new modules.
- All internal plotting and CSV-writing functions have been removed, as that
  logic now resides in the specialist modules.
"""

import json
import os
import logging
import plotter
import csv_writer

# --- Main Public Entry Point ---

def generate_outputs(results_json_string, output_dir='output'):
    """
    The single, public entry point for the output manager. It acts as a
    dispatcher, parsing the results JSON and delegating the creation of all
    plots and data files to specialized modules.

    Args:
        results_json_string (str): The complete analysis results from the engine.
        output_dir (str): The directory where all output files will be saved.
    """
    logger = logging.getLogger()
    logger.info("\n--- Stage 4: Output Manager Dispatching ---")

    try:
        results = json.loads(results_json_string)
        if results.get('status') == 'error':
            logger.critical(f"Output generation aborted. Engine reported an error: {results.get('message')}")
            return
    except (json.JSONDecodeError, TypeError) as e:
        logger.critical(f"Output Manager Error: Failed to parse Results JSON. It might be empty or invalid. Error: {e}", exc_info=True)
        return

    # Ensure the output directory exists before delegating tasks.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")

    # --- Delegate tasks to specialist modules ---

    # Call the CSV writer to handle all data file outputs.
    # The writer module has its own internal checks for SNe/BAO data.
    logger.info("Dispatching task to csv_writer...")
    csv_writer.create_csv_outputs(results, output_dir)

    # Conditionally call the plotter for SNe results if they exist.
    if 'sne_detailed_df' in results.get('results', {}).get('lcdm', {}):
        logger.info("Dispatching task to plotter for Hubble Diagram...")
        plotter.create_hubble_diagram(results, output_dir)
    else:
        logger.info("Skipping Hubble Diagram: SNe results not found.")

    # Conditionally call the plotter for BAO results if they exist.
    if 'bao_analysis' in results.get('results', {}).get('lcdm', {}):
        logger.info("Dispatching task to plotter for BAO Plot...")
        plotter.create_bao_plot(results, output_dir)
    else:
        logger.info("Skipping BAO Plot: BAO results not found.")

    logger.info("--- Output Manager tasks complete ---")