# copernican_suite/copernican.py
"""
Copernican Suite - Main Orchestrator.
"""
# DEV NOTE (v1.5f): Added placeholders for future data types and bumped version.
# DEV NOTE (v1.5f hotfix): Fixed dependency scanner to ignore relative imports.
# Automatic dependency installer still triggers when packages are missing.
# Plugin validation now occurs on the generated module.
# Previous notes retained below for context.
# DEV NOTE (v1.4.1): Added splash screen, per-run logging with timestamps, and
# migrated the base model import to the new `lcdm.py` plugin file.
# DEV NOTE (v1.4): Refactored into a pluggable architecture. Models, parsers,
# plugins now reside in the `models` package. The summary CSV call removed in
# v1.3 remains omitted.

import importlib.util
import importlib
import os
import sys
import platform
import json
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import shutil
import subprocess
import glob
import time
from scripts import model_parser, model_coder, engine_interface

COPERNICAN_VERSION = "1.5f"

def show_splash_screen():
    """Displays the startup banner once at launch."""
    banner = [
        " " * 70,
        "\n",
        "C O P E R N I C A N   S U I T E".center(70),
        "\n",
        "=" * 70,
        "\n",
        "A tool for rapid development, prototyping and testing of\n".center(70),
        "alternative cosmological frameworks against observational data\n".center(70),
        "-" * 70,
        f"build {COPERNICAN_VERSION}".center(70),
        "=" * 70,
        "\n",
    ]
    for line in banner:
        print(line)
    time.sleep(3)
    print("Follow the prompts to configure a run. Results are saved in the 'output' directory.\n\n")

# --- System Dependency and Sanity Checker ---

def _gather_required_packages():
    """Scan project files for imported modules."""
    pkg_names = set()
    search_dirs = ['.', 'scripts', 'engines', 'parsers']
    for base in search_dirs:
        for root, _, files in os.walk(base):
            for fname in files:
                if fname.endswith('.py'):
                    with open(os.path.join(root, fname), 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if line.startswith('import '):
                                parts = line.split()
                                if len(parts) >= 2:
                                    mod = parts[1].split('.')[0]
                                    if mod and not mod.startswith('.'):
                                        pkg_names.add(mod)
                            elif line.startswith('from '):
                                parts = line.split()
                                if len(parts) >= 2:
                                    mod = parts[1].split('.')[0]
                                    if mod and not mod.startswith('.'):
                                        pkg_names.add(mod)
    ignore = {
        # Standard library modules or local packages that should not trigger
        # the dependency installer
        'os', 'sys', 'time', 'json', 'logging', 'subprocess', 'importlib',
        'multiprocessing', 'glob', 'shutil', 'platform', 'inspect', 'types',
        'pathlib', 'builtins', 'traceback', 'typing',
        # Local modules within this repository
        'data_loaders', 'output_manager', 'csv_writer', 'plotter', 'logger',
        'utils'
    }
    return {pkg for pkg in pkg_names if not pkg.startswith(('scripts', 'engines', 'parsers')) and pkg not in ignore}


def _run_dep_installer(missing_pkgs):
    """Launch the dependency installer in a new terminal window."""
    installer = os.path.join(os.path.dirname(__file__), 'scripts', 'dep_install.py')
    cmd_base = [sys.executable, installer] + list(missing_pkgs)
    system = platform.system()
    if system == 'Windows':
        subprocess.Popen(['cmd', '/c', 'start', 'cmd', '/k'] + cmd_base)
    elif system == 'Darwin':
        subprocess.Popen(['osascript', '-e', f'tell application "Terminal" to do script "{" ".join(cmd_base)}"'])
    else:
        term = shutil.which('x-terminal-emulator') or shutil.which('xterm') or shutil.which('gnome-terminal') or shutil.which('konsole')
        if term:
            subprocess.Popen([term, '-e', ' '.join(cmd_base)])
        else:
            subprocess.Popen(cmd_base)
    print('Missing dependencies installed in separate terminal. Restarting when done.')
    sys.exit(0)


def check_dependencies():
    """Check for required packages and launch installer if any are missing."""
    print("--- Running System Dependency Check ---")
    required = _gather_required_packages()
    missing = [pkg for pkg in required if importlib.util.find_spec(pkg) is None]
    if missing:
        print(f"Missing packages detected: {', '.join(missing)}")
        _run_dep_installer(missing)
    else:
        print("✅ System Dependency Check Passed. Continuing...\n")


# Import sibling modules after the dependency check
import data_loaders
import output_manager

lcdm = None

def get_user_input_filepath(prompt_message, base_dir, must_exist=True):
    """Prompts the user for a filepath and validates it."""
    while True:
        filename = input(f"{prompt_message} (or 'c' to cancel): ").strip()
        if filename.lower() == 'c': return None
        # Allow special keywords like 'test' to pass through without existing as a file
        if not must_exist and not os.path.isabs(filename):
            return filename
        filepath = os.path.join(base_dir, filename)
        if os.path.isfile(filepath): return filepath
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
        if not engine_interface.validate_plugin(alt_model_module):
            logger.error(f"Model plugin '{os.path.basename(model_filepath)}' failed validation.")
            return None
        logger.info(f"Successfully loaded alternative model: {alt_model_module.MODEL_NAME}")
        return alt_model_module
    except Exception as e:
        logger.error(f"Error loading model plugin '{os.path.basename(model_filepath)}': {e}", exc_info=True)
        return None

def select_from_list(options, prompt):
    """Utility to allow user selection from a list."""
    if not options:
        return None
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    while True:
        choice = input(f"{prompt} (number or 'c' to cancel): ").strip()
        if choice.lower() == 'c':
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        print("Invalid selection. Try again.")

def parse_model_header(md_path):
    """Read minimal YAML front matter for plugin lookup."""
    data = {}
    try:
        with open(md_path, 'r') as f:
            lines = f.readlines()
        if lines and lines[0].strip() == '---':
            for line in lines[1:]:
                if line.strip() == '---':
                    break
                if ':' in line:
                    k, v = line.split(':', 1)
                    data[k.strip()] = v.strip().strip('"').strip("'")
    except Exception:
        pass
    return data

def cleanup_cache(base_dir):
    """Finds and removes all __pycache__ directories."""
    logger = output_manager.get_logger()
    logger.info("--- Cleaning up cache files ---")
    for root, dirs, files in os.walk(base_dir):
        if "__pycache__" in dirs:
            pycache_path = os.path.join(root, "__pycache__")
            try:
                shutil.rmtree(pycache_path)
                logger.info(f"Removed cache directory: {pycache_path}")
            except OSError as e:
                logger.error(f"Error removing cache directory {pycache_path}: {e}")
    cache_dir = os.path.join(base_dir, 'models', 'cache')
    if os.path.isdir(cache_dir):
        for fname in os.listdir(cache_dir):
            if fname.startswith('cache_') and fname.endswith('.json'):
                path = os.path.join(cache_dir, fname)
                try:
                    os.remove(path)
                    logger.info(f"Removed cache file: {path}")
                except OSError as e:
                    logger.error(f"Error removing cache file {path}: {e}")

def main_workflow():
    """Main workflow for the Copernican Suite."""
    check_dependencies()

    try: SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError: SCRIPT_DIR = os.getcwd()

    OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    show_splash_screen()

    # Load the baseline LCDM model from JSON and validate it
    def _load_lcdm_from_json():
        models_dir = os.path.join(SCRIPT_DIR, 'models')
        json_path = os.path.join(models_dir, 'cosmo_model_lcdm.json')
        cache_dir = os.path.join(models_dir, 'cache')
        cache_path = model_parser.parse_model_json(json_path, cache_dir)
        func_dict, parsed = model_coder.generate_callables(cache_path)
        return engine_interface.build_plugin(parsed, func_dict)

    global lcdm
    lcdm = _load_lcdm_from_json()
    engine_interface.validate_plugin(lcdm)

    while True:
        log_file = output_manager.setup_logging(log_dir=OUTPUT_DIR)
        logger = output_manager.get_logger()
        start_ts = time.strftime("%y%m%d_%H%M%S")
        logger.info(
            f"Copernican {COPERNICAN_VERSION} has initialized! Current timestamp is {start_ts}. Log file: {log_file}"
        )
        logger.info("Using standard CPU (SciPy) computational backend with multiprocessing.")
        logger.info(f"Running from base directory: {SCRIPT_DIR}")
        logger.info(f"All outputs will be saved to: {OUTPUT_DIR}")

        logger.info("\n--- Stage 1: Configuration ---")

        models_dir = os.path.join(SCRIPT_DIR, 'models')
        model_files = sorted([
            f for f in os.listdir(models_dir)
            if f.startswith('cosmo_model_') and (f.endswith('.md') or f.endswith('.json'))
        ])
        selected_model = select_from_list(model_files + ['test'], 'Select cosmological model')
        if not selected_model:
            break

        if selected_model == 'test':
            alt_model_plugin = lcdm
            logger.info("--- RUNNING IN TEST MODE: Comparing LCDM against itself. ---")
        else:
            if selected_model.endswith('.json'):
                json_path = os.path.join(models_dir, selected_model)
                cache_dir = os.path.join(models_dir, 'cache')
                try:
                    cache_path = model_parser.parse_model_json(json_path, cache_dir)
                except Exception as e:
                    logger.error(str(e))
                    continue
                try:
                    func_dict, parsed = model_coder.generate_callables(cache_path)
                    alt_model_plugin = engine_interface.build_plugin(parsed, func_dict)
                    logger.info(f"Loaded JSON model: {parsed.get('model_name')}")
                except Exception as e:
                    logger.error(f"Error generating model from JSON: {e}", exc_info=True)
                    continue
            else:
                md_path = os.path.join(models_dir, selected_model)
                meta = parse_model_header(md_path)
                plugin_name = meta.get('model_plugin')
                if not plugin_name:
                    logger.error(f"Model file {selected_model} missing 'model_plugin' entry.")
                    continue
                alt_model_filepath = os.path.join(SCRIPT_DIR, plugin_name)
                if not os.path.isfile(alt_model_filepath):
                    alt_model_filepath = os.path.join(models_dir, plugin_name)
                alt_model_plugin = load_alternative_model_plugin(alt_model_filepath)

        if not alt_model_plugin:
            continue

        engines_dir = os.path.join(SCRIPT_DIR, 'engines')
        engine_files = sorted([f for f in os.listdir(engines_dir) if f.startswith('cosmo_engine_') and f.endswith('.py')])
        engine_choice = select_from_list(engine_files, 'Select computation engine')
        if not engine_choice:
            break
        engine_module = importlib.import_module(f"engines.{engine_choice[:-3]}")
        cosmo_engine_selected = engine_module

        sne_data_dir = os.path.join(SCRIPT_DIR, 'data', 'sne')
        sne_files = sorted(os.listdir(sne_data_dir))
        sne_choice = select_from_list(sne_files, 'Select SNe Ia data file')
        if not sne_choice:
            break
        sne_data_filepath = os.path.join(sne_data_dir, sne_choice)
        
        sne_format_key = data_loaders._select_parser(data_loaders.SNE_PARSERS, "SNe")
        if not sne_format_key: break

        sne_loader_kwargs = {}
        parser_info = data_loaders.SNE_PARSERS.get(sne_format_key)
        if parser_info and parser_info.get('extra_args_func'):
            logger.info(f"Parser '{sne_format_key}' requires additional arguments.")
            extra_args = parser_info['extra_args_func'](SCRIPT_DIR)
            if extra_args is None: break
            sne_loader_kwargs.update(extra_args)
        
        sne_data_df = data_loaders.load_sne_data(sne_data_filepath, format_key=sne_format_key, **sne_loader_kwargs)
        if sne_data_df is None: continue

        bao_data_dir = os.path.join(SCRIPT_DIR, 'data', 'bao')
        bao_files = sorted(os.listdir(bao_data_dir))
        bao_choice = select_from_list(bao_files, 'Select BAO data file')
        if not bao_choice:
            break
        bao_data_filepath = os.path.join(bao_data_dir, bao_choice)
        bao_format_key = data_loaders._select_parser(data_loaders.BAO_PARSERS, "BAO")
        if not bao_format_key: break
        bao_data_df = data_loaders.load_bao_data(bao_data_filepath, format_key=bao_format_key)
        if bao_data_df is None: continue

        logger.info("\n--- Stage 2: Supernovae Ia Fitting ---")
        lcdm_sne_fit_results = cosmo_engine_selected.fit_sne_parameters(sne_data_df, lcdm)
        alt_model_sne_fit_results = cosmo_engine_selected.fit_sne_parameters(sne_data_df, alt_model_plugin)
        
        logger.info("\n--- Stage 3: BAO Analysis ---")
        
        min_z, max_z = bao_data_df['redshift'].min(), bao_data_df['redshift'].max()
        z_plot_smooth = np.geomspace(max(min_z * 0.8, 0.01), max_z * 1.2, 100)
        
        def run_bao_analysis(model_plugin, sne_fit_results, z_smooth_arr):
            """Helper to run BAO analysis for a given model."""
            if not (sne_fit_results and sne_fit_results.get('success')):
                logger.warning(f"{model_plugin.MODEL_NAME} SNe fit failed; skipping BAO analysis.")
                return {'sne_fit_results': sne_fit_results, 'pred_df': None, 'rs_Mpc': np.nan, 'chi2_bao': np.inf, 'smooth_predictions': None}

            fitted_cosmo_p = list(sne_fit_results['fitted_cosmological_params'].values())
            pred_df, rs_Mpc, smooth_preds = cosmo_engine_selected.calculate_bao_observables(bao_data_df, model_plugin, fitted_cosmo_p, z_smooth=z_smooth_arr)
            
            chi2_bao = np.inf
            if pred_df is not None and np.isfinite(rs_Mpc):
                chi2_bao = cosmo_engine_selected.chi_squared_bao(bao_data_df, model_plugin, fitted_cosmo_p, rs_Mpc)
                logger.info(f"{model_plugin.MODEL_NAME} BAO: r_s = {rs_Mpc:.2f} Mpc, Chi2_BAO = {chi2_bao:.2f}")
            else:
                logger.warning(f"{model_plugin.MODEL_NAME} BAO calculation failed or produced invalid r_s.")
                
            return {'sne_fit_results': sne_fit_results, 'pred_df': pred_df, 'rs_Mpc': rs_Mpc, 'chi2_bao': chi2_bao, 'smooth_predictions': smooth_preds}

        lcdm_full_results = run_bao_analysis(lcdm, lcdm_sne_fit_results, z_plot_smooth)
        alt_model_full_results = run_bao_analysis(alt_model_plugin, alt_model_sne_fit_results, z_plot_smooth)

        logger.info("\n--- Stage 4: Generating Outputs ---")
        output_manager.plot_hubble_diagram(sne_data_df, lcdm_sne_fit_results, alt_model_sne_fit_results, lcdm, alt_model_plugin, plot_dir=OUTPUT_DIR)
        if bao_data_df is not None:
            output_manager.plot_bao_observables(bao_data_df, lcdm_full_results, alt_model_full_results, lcdm, alt_model_plugin, plot_dir=OUTPUT_DIR)
        
        # The call to the redundant summary CSV has been removed.
        # output_manager.save_sne_fit_results_csv(...)
        
        # Save the detailed point-by-point SNe results CSV
        output_manager.save_sne_results_detailed_csv(sne_data_df, lcdm_sne_fit_results, alt_model_sne_fit_results, lcdm, alt_model_plugin, csv_dir=OUTPUT_DIR)
        
        if bao_data_df is not None:
            output_manager.save_bao_results_csv(bao_data_df, lcdm_full_results, alt_model_full_results, alt_model_name=alt_model_plugin.MODEL_NAME, csv_dir=OUTPUT_DIR)

        print("\n" + "="*50)
        print("Evaluation complete. All files saved to the 'output' directory.")
        print("="*50 + "\n")

        end_ts = time.strftime("%y%m%d_%H%M%S")
        logger.info(f"Run completed at {end_ts}.")

        while True:
            another_run = input("Would you like to run another evaluation? (yes/no): ").strip().lower()
            if another_run in ["yes", "y", "1"]:
                break 
            elif another_run in ["no", "n", "2"]:
                cleanup_cache(SCRIPT_DIR)
                logger.info("Exiting Copernican Suite. Goodbye!")
                return
            else:
                print("Invalid input. Please enter 'yes' or 'no'.")
        
        cleanup_cache(SCRIPT_DIR)

if __name__ == "__main__":
    # This is essential for multiprocessing to work correctly on all platforms
    mp.freeze_support()
    try:
        main_workflow()
    except Exception as e:
        logger = output_manager.get_logger()
        if logger.hasHandlers():
            logger.critical("Unhandled exception in main_workflow!", exc_info=True)
        else:
            print("CRITICAL UNHANDLED EXCEPTION IN MAIN WORKFLOW:")
            import traceback
            traceback.print_exc()
    finally:
        # Ensure that any generated plot windows are displayed at the very end
        if plt.get_fignums():
            print("\nDisplaying plot(s). Close plot window(s) to exit script fully.")
            try:
                plt.show(block=True)
            except Exception as e_show:
                print(f"Error during final plt.show(): {e_show}")
