# copernican.py
"""
DEV NOTE (v1.4a): This is the main orchestrator script, significantly refactored
for the v1.4a "black box" architecture. Its responsibilities are now focused
on user interaction and high-level workflow control.

- It no longer directly interacts with the specifics of data loading, model
  fitting, or output generation.
- New Feature: It now scans for available `cosmo_engine_*.py` files and
  prompts the user to select a computational engine for the analysis.
- The main loop has been streamlined to follow the new linear pipeline:
    1. Get user inputs (engine, model, data files).
    2. Call `input_aggregator.build_job_json()` to create the analysis job.
    3. Dynamically load the chosen engine and call `execute_job(job_json)`.
    4. Pass the returned `results_json` to `output_manager.generate_outputs()`.
- A user-friendly splash screen and initial guidance have been added.

BUG FIX (v1.4a): The special 'test' keyword functionality, which allows for a
quick self-test of LCDM vs LCDM, was restored. The code now explicitly checks
for this keyword after user input and sets the model path accordingly.
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
        print("")
        print("         C  O  P  E  R  N  I  C  A  N    S  U  I  T  E")
        print("")
        print("\n                 A Modular Cosmology Framework")
        print("                            v1.4a")
        print("\n" + "="*60)
        print("        ✨ 🔭 🌠 A tool for exploring the cosmos 🪐 ✨")
        print("="*60)
        time.sleep(2.5)

    def setup_logging(base_dir, run_id):
        """Initializes logging to file and console."""
        log_dir = os.path.join(base_dir, 'output')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_filename = os.path.join(log_dir, f'copernican-run_{run_id}.txt')

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
            # For the model prompt, we allow 'test' which doesn't exist as a file.
            is_model_prompt = 'model' in prompt.lower()
            
            user_response = input(prompt).strip()
            if user_response.lower() == 'c': return None

            # Handle special 'test' keyword for the model prompt
            if is_model_prompt and user_response.lower() == 'test':
                return 'test'

            # Standard file existence check
            if not must_exist or os.path.isfile(user_response):
                return user_response
            else:
                print(f"Error: File not found at '{user_response}'. Please try again.")

    # --- Main Application Workflow ---
    display_splash_screen()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    while True:
        # Generate a new Run ID for each loop
        RUN_ID = datetime.now().strftime('%Y%m%d_%H%M%S')
        logger = setup_logging(BASE_DIR, RUN_ID)

        logger.info("="*30)
        logger.info("    Copernican Suite v1.4a Initialized")
        logger.info("="*30)
        logger.info(f"Running from base directory: {BASE_DIR}")
        logger.info(f"All outputs will be saved to: {os.path.join(BASE_DIR, 'output')}")
        
        # --- Stage 1: Configuration ---
        print("\n--- 🛰️  New Analysis Configuration ---")
        print("Enter file paths for your analysis or 'test' for a quick self-test.")
        print("You can type 'c' at any prompt to cancel and exit.")

        engine_name = select_engine(BASE_DIR)
        if engine_name is None: break
        logger.info(f"User selected engine: {engine_name}")

        alt_model_input = get_user_input("Enter path to alternative model .py file (or 'test'): ")
        if alt_model_input is None: break
        
        # **BUG FIX**: Handle the 'test' keyword to run LCDM against itself
        if alt_model_input.lower() == 'test':
            alt_model_path = 'lcdm_model.py'
            logger.info("Test mode selected. Running LCDM vs LCDM.")
        else:
            alt_model_path = alt_model_input
        
        print("\n--- 🌠 Type Ia Supernovae Data ---")
        sne_path = get_user_input("Enter path to SNe Ia data file: ")
        if sne_path is None: break

        sne_format_key = data_loaders._select_parser(data_loaders.SNE_PARSERS, "SNe")
        if sne_format_key is None: break

        sne_extra_args = {}
        extra_args_func = data_loaders.SNE_PARSERS[sne_format_key].get('extra_args_func')
        if extra_args_func:
            sne_extra_args = extra_args_func(BASE_DIR)
            if sne_extra_args is None: break

        sne_data_info = {'path': sne_path, 'format_key': sne_format_key, 'extra_args': sne_extra_args}

        print("\n--- 🌌 Baryon Acoustic Oscillation Data ---")
        bao_path = get_user_input("Enter path to BAO data file (or press Enter to skip): ", must_exist=False)
        bao_data_info = {}
        if bao_path:
            bao_format_key = data_loaders._select_parser(data_loaders.BAO_PARSERS, "BAO")
            if bao_format_key:
                bao_data_info = {'path': bao_path, 'format_key': bao_format_key}
        
        # --- Stage 2: Job Aggregation ---
        logger.info("\n--- Stage 2: Aggregating Job Data ---")
        job_json = input_aggregator.build_job_json(RUN_ID, engine_name, alt_model_path, sne_data_info, bao_data_info)

        if not job_json:
            logger.critical("Failed to build the Job JSON. Aborting this run.")
            continue

        # --- Stage 3: Engine Execution ---
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
        output_manager.generate_outputs(results_json)

        run_again = input("\nAnalysis complete. Run another? (y/n): ").strip().lower()
        if run_again != 'y':
            break

    logger.info("Copernican Suite session finished.")