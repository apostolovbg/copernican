# copernican.py
"""
DEV NOTE (v1.4b): The version is updated to reflect the new plotting and
output architecture of the suite. No logical changes were required in this
file because the v1.4a architecture correctly abstracts all output generation
behind the `output_manager.py` dispatcher.

This file's responsibility remains focused on user interaction and high-level
workflow orchestration. The only functional change in this version is the
addition of a user-friendly "How-To" message displayed at startup to improve
usability. Comments have also been enhanced for clarity.
"""

import os
import sys
import logging
import time
from datetime import datetime
import importlib.util

# --- Dependency Check at Startup ---
def check_dependencies():
    """Checks for required libraries and prints install commands if missing."""
    required_libs = {'numpy': 'numpy', 'scipy': 'scipy', 'matplotlib': 'matplotlib', 'pandas': 'pandas'}
    missing_libs = []
    for lib, import_name in required_libs.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_libs.append(lib)
    if missing_libs:
        print("\n--- Missing Dependencies ---")
        print("The following required Python libraries are not installed:")
        for lib in missing_libs:
            print(f"  - {lib}")
        print("\nPlease install them by running the following command:")
        print(f"  pip install {' '.join(missing_libs)}")
        print("--------------------------\n")
        sys.exit(1)
    return True

# --- Main Program Execution ---
if __name__ == "__main__":
    check_dependencies()

    # Now that dependencies are confirmed, import the rest of the suite
    import data_loaders
    import input_aggregator
    import output_manager

    # --- UI and Helper Functions ---
    def display_splash_screen():
        """Clears the screen and displays a title splash."""
        os.system('cls' if os.name == 'nt' else 'clear')
        print("="*60)
        print("")
        print("         C  O  P  E  R  N  I  C  A  N    S  U  I  T  E")
        print("")
        print("\n                 A Modular Cosmology Framework")
        print("                            v1.4b") # Updated version
        print("\n" + "="*60)
        print("        ✨ 🔭 🌠 A tool for exploring the cosmos 🪐 ✨")
        print("="*60)
        time.sleep(1.5)

    def display_how_to_message():
        """Displays a brief guide on how to use the program."""
        print("\n--- How to Use ---")
        print("1. Select the computational engine to use for the analysis.")
        print("2. Provide the file path to an alternative model's .py plugin.")
        print("   > Type 'test' to run a quick self-comparison of the LCDM model.")
        print("3. Provide file paths for your SNe and (optionally) BAO data.")
        print("4. Select the correct data parser for each file when prompted.")
        print("\nNOTE: You can type 'c' and press Enter at any prompt to cancel and exit.")
        print("-" * 60)
        
    def setup_logging(base_dir, run_id):
        """Initializes logging to file and console."""
        log_dir = os.path.join(base_dir, 'output')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_filename = os.path.join(log_dir, f'copernican-run_{run_id}.txt')

        # Clear any previous logging handlers to prevent duplicate output
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler(sys.stdout)
            ]
        )
        return logging.getLogger()

    def select_engine(base_dir):
        """Scans for engine files and prompts the user for a selection."""
        print("\n--- 🪐 Select a Computational Engine ---")
        try:
            engine_files = sorted([f for f in os.listdir(base_dir) if f.startswith('cosmo_engine_') and f.endswith('.py')])
            if not engine_files:
                print("Error: No 'cosmo_engine_*.py' files found in the directory.")
                return None

            for i, fname in enumerate(engine_files):
                print(f"  {i+1}. {fname}")

            while True:
                choice = input("Enter the number of the engine to use (or 'c' to cancel): ").strip().lower()
                if choice == 'c': return None
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(engine_files):
                        return engine_files[idx]
                    else:
                        print("Invalid number. Please try again.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
        except Exception as e:
            print(f"Error finding engines: {e}")
            return None

    def get_user_input(prompt, must_exist=True):
        """Generic function to get a valid file path or keyword from the user."""
        while True:
            user_response = input(prompt).strip()
            if user_response.lower() == 'c': return None

            # Handle special 'test' keyword for the model prompt
            if 'model' in prompt.lower() and user_response.lower() == 'test':
                return 'test'

            # Standard file existence check
            if not must_exist or os.path.isfile(user_response):
                return user_response
            else:
                print(f"Error: File not found at '{user_response}'. Please try again.")

    # --- Main Application Workflow ---
    display_splash_screen()
    display_how_to_message() # NEW (v1.4b): Add user guidance

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    while True:
        # Generate a new Run ID for each analysis loop
        RUN_ID = datetime.now().strftime('%Y%m%d_%H%M%S')
        logger = setup_logging(BASE_DIR, RUN_ID)

        logger.info("="*30)
        logger.info(f"    Copernican Suite v{ '1.4b' } Initialized") # Updated version
        logger.info("="*30)
        logger.info(f"Running from base directory: {BASE_DIR}")
        logger.info(f"All outputs will be saved to: {os.path.join(BASE_DIR, 'output')}")
        
        # --- Stage 1: Configuration ---
        # Gather all user inputs for the analysis.
        print("\n--- 🛰️  New Analysis Configuration ---")

        # Step 1.1: Select the computational engine
        engine_name = select_engine(BASE_DIR)
        if engine_name is None: break
        logger.info(f"User selected engine: {engine_name}")

        # Step 1.2: Get the alternative model to test
        alt_model_input = get_user_input("Enter path to alternative model .py file (or 'test'): ")
        if alt_model_input is None: break
        
        if alt_model_input.lower() == 'test':
            alt_model_path = 'lcdm_model.py'
            logger.info("Test mode selected. Running LCDM vs LCDM.")
        else:
            alt_model_path = alt_model_input
        
        # Step 1.3: Get the SNe Ia data and parser info
        print("\n--- 🌠 Type Ia Supernovae Data ---")
        sne_path = get_user_input("Enter path to SNe Ia data file: ")
        if sne_path is None: break

        sne_format_key = data_loaders._select_parser(data_loaders.SNE_PARSERS, "SNe")
        if sne_format_key is None: break

        sne_extra_args = {}
        extra_args_func = data_loaders.SNE_PARSERS[sne_format_key].get('extra_args_func')
        if extra_args_func:
            sne_extra_args = extra_args_func(BASE_DIR)
            if sne_extra_args is None: break # User cancelled

        sne_data_info = {'path': sne_path, 'format_key': sne_format_key, 'extra_args': sne_extra_args}

        # Step 1.4: Get the BAO data and parser info (optional)
        print("\n--- 🌌 Baryon Acoustic Oscillation Data ---")
        bao_path = get_user_input("Enter path to BAO data file (or press Enter to skip): ", must_exist=False)
        bao_data_info = {}
        if bao_path:
            bao_format_key = data_loaders._select_parser(data_loaders.BAO_PARSERS, "BAO")
            if bao_format_key:
                bao_data_info = {'path': bao_path, 'format_key': bao_format_key}
        
        # --- Stage 2: Job Aggregation ---
        # Consolidate all inputs into a single "Job JSON" for the engine.
        logger.info("\n--- Stage 2: Aggregating Job Data ---")
        job_json = input_aggregator.build_job_json(RUN_ID, engine_name, alt_model_path, sne_data_info, bao_data_info)

        if not job_json:
            logger.critical("Failed to build the Job JSON. Aborting this run.")
            continue

        # --- Stage 3: Engine Execution ---
        # Dynamically load and run the selected engine as a black box.
        logger.info(f"\n--- Stage 3: Executing Job with Engine: {engine_name} ---")
        try:
            engine_path = os.path.join(BASE_DIR, engine_name)
            spec = importlib.util.spec_from_file_location("engine", engine_path)
            engine_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(engine_module)

            results_json = engine_module.execute_job(job_json)

            if not results_json:
                 raise RuntimeError("Engine returned an empty result.")

        except Exception as e:
            logger.critical(f"A critical error occurred while executing the engine: {e}", exc_info=True)
            continue

        # --- Stage 4: Output Generation ---
        # Pass the final "Results JSON" to the output manager dispatcher.
        output_manager.generate_outputs(results_json)

        # --- Loop or Exit ---
        run_again = input("\nAnalysis complete. Run another? (y/n): ").strip().lower()
        if run_again != 'y':
            break

    logger.info("Copernican Suite session finished.")