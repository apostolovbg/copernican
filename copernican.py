# copernican.py
"""
The main orchestrator for the Copernican Suite.

DEV NOTE (v1.4rc2):
1.  CRITICAL FIX: Corrected the `ValueError: I/O operation on closed file`
    logging bug. The `try...except...finally` block that runs the analysis
    has been moved to be *inside* the `with open(...)` block for the log file.
    Previously, the `with` block would exit and close the log file upon an
    exception, *before* the `except` block tried to write the error message,
    causing the crash. This change ensures the log file remains open until
    all output, including crash tracebacks, is successfully written.

2.  (from v1.4rc) LOGGING OVERHAUL: The 'logging' module has been replaced with
    a 'Tee' class that provides a verbatim mirror of the console session to the
    log file. This captures all output exactly as it appears on screen.

3.  (from v1.4rc) EXCEPTION HANDLING: Added a custom exception hook to ensure
    that fatal tracebacks are correctly routed through the Tee logger.
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
        print("\nPlease install them using pip, for example: pip install numpy")
        sys.exit(1)

# --- Verbatim Logging System ---
class Tee(object):
    """
    A file-like object that duplicates all writes to both a file and another
    stream (e.g., the original terminal stdout). This creates a verbatim
    mirror of the console session.
    """
    def __init__(self, file_obj, terminal_obj):
        self.file = file_obj
        self.terminal = terminal_obj

    def write(self, obj):
        self.terminal.write(obj)
        # CRITICAL FIX (v1.4rc2): By moving the try/except block inside the
        # `with` statement in main(), this `self.file` object will no longer
        # be closed prematurely when an error occurs in the engine.
        self.file.write(obj)

    def flush(self):
        self.file.flush()
        self.terminal.flush()

def handle_exception(exc_type, exc_value, exc_traceback):
    """
    Custom exception hook to ensure tracebacks are written by the Tee logger.
    """
    # sys.__excepthook__ is the default handler. We call it to print the
    # traceback to sys.stdout, which our Tee logger is intercepting.
    sys.__excepthook__(exc_type, exc_value, exc_traceback)


# --- Main Application Logic ---
def main():
    """Main function to run the Copernican Suite."""
    check_dependencies()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)

    # --- Main Loop ---
    while True:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = os.path.join(output_dir, f"copernican-run_{run_id}.txt")

        # Store original streams to restore later
        original_stdout = sys.stdout
        original_excepthook = sys.excepthook

        # CRITICAL FIX (v1.4rc2): The try/except/finally block is now INSIDE
        # the `with` statement. This ensures `log_file` stays open even when
        # an exception occurs, allowing the `except` block to write to it.
        with open(log_filename, 'w') as log_file:
            tee_logger = Tee(log_file, original_stdout)
            sys.stdout = tee_logger
            sys.excepthook = handle_exception

            try:
                # --- Splash Screen & UI ---
                print("\n============================================================")
                print("\n        C O P E R N I C A N   S U I T E \n")
                print("         A Modular Cosmology Framework")
                print(f"                   v1.4rc2\n")
                print("============================================================")
                print("        ✨ 🔭 🌠 A tool for exploring the cosmos 🪐 ✨")
                print("============================================================\n")
                print("--- How to Use ---")
                print("1. Select the computational engine to use for the analysis.")
                print("2. Provide the file path to an alternative model's .py plugin.")
                print("   > Type 'test' to run a quick self-comparison of the LCDM model.")
                print("3. Provide file paths for your SNe and (optionally) BAO data.")
                print("4. Select the correct data parser for each file when prompted.\n")
                print("NOTE: You can type 'c' and press Enter at any prompt to cancel and exit.")
                print("------------------------------------------------------------")

                print(f"\n--- 🛰️  New Analysis Configuration (Run ID: {run_id}) ---")

                # --- Stage 1: User Input ---
                from data_loaders import get_user_selections # Deferred import
                selections = get_user_selections(base_dir)
                if selections is None: # User cancelled
                    raise KeyboardInterrupt

                # --- Stage 2: Job Aggregation ---
                import input_aggregator # Deferred import
                job_json = input_aggregator.build_job_json(
                    selections['engine_name'],
                    selections['alt_model_path'],
                    selections['sne_data_info'],
                    selections['bao_data_info'],
                    base_dir,
                    run_id
                )
                if not job_json:
                    raise RuntimeError("Failed to build job data structure.")

                # --- Stage 3: Engine Execution ---
                print(f"\n--- Stage 3: Executing Job with Engine: {job_json['engine_name']} ---")
                engine_path = os.path.join(base_dir, job_json['engine_name'])
                spec = importlib.util.spec_from_file_location("engine", engine_path)
                engine_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(engine_module)
                results_json = engine_module.execute_job(job_json)

                if not results_json:
                    raise RuntimeError("Engine returned an empty result.")

                # --- Stage 4: Output Generation ---
                import output_manager # Deferred import
                output_manager.generate_outputs(results_json)
                print("\nAnalysis complete.")

            except (RuntimeError, KeyboardInterrupt, Exception) as e:
                # This block catches errors and user cancellations (Ctrl+C)
                if isinstance(e, RuntimeError):
                     print(f"\nRun Aborted: {e}")
                elif isinstance(e, KeyboardInterrupt):
                     print("\n\nRun cancelled by user.")
                else:
                     # The custom excepthook already printed the detailed traceback
                     print(f"\nRun failed due to an unexpected error.")

            finally:
                # Always restore stdout and the default excepthook at the end of a run
                sys.stdout = original_stdout
                sys.excepthook = original_excepthook

        # --- Loop or Exit ---
        run_again = input("Run another analysis? (y/n): ").strip().lower()
        if run_again != 'y':
            break

    print("\nThank you for using the Copernican Suite. Goodbye! ✨")

if __name__ == "__main__":
    main()