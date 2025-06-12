# copernican.py
"""
DEV NOTE (v1.4rc):
1.  LOGGING OVERHAUL: The logging system has been completely replaced with a
    'Tee' class that provides a verbatim mirror of the console session to the
    log file. This removes the 'logging' module in favor of a simpler, more
    robust system that captures all output, including user interaction and
    crash tracebacks, exactly as they appear on screen. This fulfills the
    final v1.4rc logging requirement.

2.  EXCEPTION HANDLING: Added a custom exception hook to ensure that fatal
    tracebacks are correctly routed through the Tee logger to be saved in the
    log file.

3.  STABILITY FIX: (from previous version) The main loop now handles graceful
    exits on job failure or engine errors.
"""
import os
import sys
import io
from datetime import datetime
import importlib.util
import traceback

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

# --- Verbatim Console-to-File Logger ---
class Tee:
    """
    A file-like object that duplicates all writes to multiple streams.
    Used to mirror stdout to both the original console and a log file.
    """
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # Flush immediately to ensure real-time logging
    def flush(self):
        for f in self.files:
            f.flush()

# --- Main Program Execution ---
if __name__ == "__main__":
    check_dependencies()

    # Now that dependencies are confirmed, import the rest of the suite
    import data_loaders
    import input_aggregator
    import output_manager

    # --- UI & Orchestration ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    VERSION = "v1.4rc"

    def print_banner():
        """Prints the application's startup banner."""
        print("\n============================================================")
        print(" \n        C O P E R N I C A N   S U I T E \n")
        print("         A Modular Cosmology Framework ")
        print(f"                   {VERSION} \n")
        print("============================================================")
        print("        ✨ 🔭 🌠 A tool for exploring the cosmos 🪐 ✨ ")
        print("============================================================\n")
        print("--- How to Use ---")
        print("1. Select the computational engine to use for the analysis.")
        print("2. Provide the file path to an alternative model's .py plugin.")
        print("   > Type 'test' to run a quick self-comparison of the LCDM model.")
        print("3. Provide file paths for your SNe and (optionally) BAO data.")
        print("4. Select the correct data parser for each file when prompted.")
        print("\nNOTE: You can type 'c' and press Enter at any prompt to cancel and exit.")
        print("------------------------------------------------------------")

    def get_user_input(prompt, validation_func, *validation_args):
        """Generic user input handler with validation."""
        while True:
            user_input = input(prompt).strip()
            if user_input.lower() == 'c':
                return None
            if validation_func(user_input, *validation_args):
                return user_input

    # --- Main Application Loop ---
    original_stdout = sys.stdout  # Save the original stdout stream once at the start.
    
    while True:
        # For each new run, we generate a new run_id and log file.
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = os.path.join(OUTPUT_DIR, f"copernican-run_{run_id}.txt")
        
        try:
            with open(log_filename, 'w', encoding='utf-8') as log_file:
                # Set up the Tee to mirror console to file for the duration of the run.
                sys.stdout = Tee(original_stdout, log_file)
                
                # Custom exception hook to ensure tracebacks are written to the Tee logger
                def log_excepthook(exc_type, exc_value, exc_traceback):
                    """Prints the exception to the Tee logger."""
                    print("\n--- CRITICAL ERROR ---", file=sys.stdout)
                    traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stdout)
                    print("----------------------", file=sys.stdout)
                
                sys.excepthook = log_excepthook

                # --- START OF LOGGED SESSION ---
                print_banner()
                
                print(f"\n--- 🛰️  New Analysis Configuration (Run ID: {run_id}) ---\n")

                # --- Stage 1: User Input ---
                print("--- 🪐 Select a Computational Engine ---")
                engine_files = [f for f in os.listdir(BASE_DIR) if f.startswith('cosmo_engine_') and f.endswith('.py')]
                for i, fname in enumerate(engine_files): print(f"  {i+1}. {fname}")
                
                def validate_engine(choice, options):
                    try:
                        idx = int(choice) - 1
                        if 0 <= idx < len(options): return True
                    except ValueError: pass
                    print("Invalid selection. Please enter a number from the list.")
                    return False
                    
                engine_choice = get_user_input("Enter the number of the engine to use (or 'c' to cancel): ", validate_engine, engine_files)
                if engine_choice is None: break
                engine_name = engine_files[int(engine_choice)-1]
                
                def validate_model_path(path, base_dir):
                    if path.lower() == 'test': return True
                    full_path = path if os.path.isabs(path) else os.path.join(base_dir, path)
                    if os.path.isfile(full_path): return True
                    print(f"Error: File not found at '{full_path}'. Please try again.")
                    return False

                alt_model_path = get_user_input("Enter path to alternative model .py file (or 'test'): ", validate_model_path, BASE_DIR)
                if alt_model_path is None: break
                
                sne_data_info = data_loaders.collect_data_info("SNe Ia", data_loaders.SNE_PARSERS, BASE_DIR)
                if sne_data_info is None: break 
                
                bao_data_info = data_loaders.collect_data_info("BAO", data_loaders.BAO_PARSERS, BASE_DIR, is_optional=True)
                if bao_data_info is None: break
                if not bao_data_info:
                    bao_data_info = {}

                # --- Stage 2: Job Aggregation ---
                print("\n--- Stage 2: Aggregating Job Data ---")
                job_json = input_aggregator.build_job_json(run_id, engine_name, alt_model_path, sne_data_info, bao_data_info)

                if not job_json:
                    raise RuntimeError("Failed to build the Job JSON. See messages above for details.")

                # --- Stage 3: Engine Execution ---
                print(f"\n--- Stage 3: Executing Job with Engine: {engine_name} ---")
                engine_path = os.path.join(BASE_DIR, engine_name)
                spec = importlib.util.spec_from_file_location("engine", engine_path)
                engine_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(engine_module)
                results_json = engine_module.execute_job(job_json)

                if not results_json:
                    raise RuntimeError("Engine returned an empty result.")

                # --- Stage 4: Output Generation ---
                output_manager.generate_outputs(results_json)
                print("\nAnalysis complete.")

        except (RuntimeError, KeyboardInterrupt, Exception) as e:
            # This block catches errors and user cancellations (Ctrl+C)
            # The excepthook handles printing the traceback if it's an unhandled Exception
            if isinstance(e, RuntimeError):
                 print(f"\nRun Aborted: {e}")
            elif isinstance(e, KeyboardInterrupt):
                 print("\n\nRun cancelled by user.")
            else:
                 # The excepthook already printed the detailed traceback
                 print(f"\nRun failed due to an unexpected error.")
        
        finally:
            # Always restore stdout and the default excepthook
            sys.stdout = original_stdout
            sys.excepthook = sys.__excepthook__

        # --- Loop or Exit ---
        run_again = input("Run another analysis? (y/n): ").strip().lower()
        if run_again != 'y':
            break

    print("\nThank you for using the Copernican Suite. Goodbye! ✨")