# copernican_suite/copernican.py
"""
Copernican Suite - Main Orchestrator.
"""


import importlib.util
import importlib
from importlib.metadata import version as package_version, PackageNotFoundError
import ast
import os
import sys
import platform
import shutil
import time
import datetime
import argparse
from pathlib import Path

from copernican_lib import console_output as console

# Verify interpreter version early so users see clear feedback
MIN_PYTHON = (3, 12)


def exit_clean(code: int = 0) -> None:
    """Exit the program after printing a newline."""
    console.write("")
    sys.exit(code)


if sys.version_info < MIN_PYTHON:
    console.write(
        f"ERROR: Copernican Suite requires Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or later.",
        error=True,
    )
    exit_clean(1)

# Delay heavy third-party imports until after the dependency check.
# Doing so keeps startup quick and lets ``check_dependencies`` provide a
# clean error message before the interpreter tries to import missing
# modules.
np = None
plt = None
mp = None

model_parser = None
model_coder = None
engine_interface = None
plotter = None
csv_writer = None
log_mod = None
logger = None
data_loaders = None

# Use a fixed version string to avoid confusion when the package metadata is
# outdated. Automatic releases are not yet enabled. Version 3.1.0 adds
# unified exponent syntax across model YAML files and drops all
# legacy JSON dataset support in favour of YAML-only inputs.
COPERNICAN_VERSION = "3.3.7"
CURRENT_LOG_FILE = None


def _delete_log_file(path: str) -> None:
    """Remove the given log file if it exists."""
    if path and os.path.isfile(path):
        try:
            os.remove(path)
            console.write(f"Removed log file {path}")
        except OSError:
            pass


def _get_cpu_info() -> tuple[str, str]:
    """Return CPU model and current clock speed."""
    cpu = platform.processor() or platform.uname().processor or "Unknown CPU"
    freq = None
    try:
        import psutil  # type: ignore

        freq_info = psutil.cpu_freq()
        if freq_info:
            freq = freq_info.current / 1000.0
    except Exception:
        pass
    if freq is None and platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", "r") as fh:
                for line in fh:
                    if line.startswith("model name") and cpu == "Unknown CPU":
                        cpu = line.split(":", 1)[1].strip()
                    if line.startswith("cpu MHz") and freq is None:
                        freq = float(line.split(":", 1)[1]) / 1000.0
        except Exception:
            pass
    freq_str = f"{freq:.2f} GHz" if freq else "Unknown GHz"
    return cpu, freq_str


# The high-level workflow is broken into small helper functions below. Each
# helper is documented in plain language so non-programmers can follow the
# logic of the program.


def run_startup_tests():
    """Discover and execute functional tests within the ``tests`` package."""
    # This routine is invoked when the ``--run-tests`` flag is supplied.
    # It uses Python's built-in unittest discovery to execute all test modules
    # under the ``tests`` folder. The boolean return value determines whether
    # the suite ran successfully.
    import unittest

    try:
        suite = unittest.defaultTestLoader.discover("tests")
    except Exception as exc:
        console.write(f"Error discovering startup tests: {exc}")
        return False
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    return result.wasSuccessful()


def parse_args():
    """Parse command line flags provided by the user."""
    parser = argparse.ArgumentParser(description="Copernican Suite")
    # ``--run-tests`` triggers the functional test suite and then exits.
    parser.add_argument(
        "--run-tests", action="store_true", help="execute functional tests and exit"
    )
    return parser.parse_args()


def show_splash_screen():
    """Displays the startup banner once at launch."""
    banner = [
        "=" * 70,
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
        console.write(line)
    time.sleep(1)
    console.write(
        "Follow the prompts to configure a run. Results are saved in the 'output' directory.\n\n"
    )


# --- System Dependency and Sanity Checker ---


def _gather_required_packages():
    """Return external packages imported across project modules."""
    # Rather than rely on ``pip freeze`` or manual lists this function
    # walks through the source tree and parses each ``import`` statement
    # with :mod:`ast`.  This keeps the dependency check accurate even
    # when new optional modules are added.
    pkg_names = set()
    search_dirs = ["copernican_lib", "engines", "tests", "."]
    ignore_dirs = {
        "venv",
        ".venv",
        "env",
        "build",
        "dist",
        "__pycache__",
        "copernican_suite.egg-info",
    }
    for base in search_dirs:
        if not os.path.isdir(base):
            continue
        for root, dirs, files in os.walk(base):
            dirs[:] = [
                d
                for d in dirs
                if d not in ignore_dirs
                and not d.startswith(".")
                and "site-packages" not in d
            ]
            for fname in files:
                if not fname.endswith(".py"):
                    continue
                path = os.path.join(root, fname)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        tree = ast.parse(f.read(), filename=path)
                except SyntaxError:
                    continue
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            pkg_names.add(alias.name.split(".")[0])
                    elif isinstance(node, ast.ImportFrom):
                        if node.level == 0 and node.module:
                            pkg_names.add(node.module.split(".")[0])
    ignore = {
        # Standard library modules or local packages that should not trigger
        # the dependency installer
        "os",
        "sys",
        "time",
        "logging",
        "subprocess",
        "importlib",
        "multiprocessing",
        "glob",
        "shutil",
        "platform",
        "inspect",
        "types",
        "pathlib",
        "builtins",
        "traceback",
        "typing",
        # Local modules within this repository (under ``copernican_lib``)
        "data_loaders",
        "csv_writer",
        "plotter",
        "logger",
        "utils",
    }
    return {
        pkg
        for pkg in pkg_names
        if pkg not in ignore and not pkg.startswith(("copernican_lib", "engines"))
    }


def _print_install_instructions(missing: list[str]) -> None:
    """Show platform-specific commands to install missing packages."""
    # The messages here intentionally rely on plain ``pip`` so users on
    # any platform can copy & paste them directly.  Additional hints for
    # Homebrew, APT or pipx are printed when those tools are available
    # to guide less experienced users.
    pkgs = " ".join(sorted(set(missing)))
    pip_cmd = f"{Path(sys.executable).name} -m pip install {pkgs}"
    console.write("Please install them with:")
    console.write(f"  {pip_cmd}")

    sys_name = platform.system()
    if sys_name == "Darwin" and shutil.which("brew"):
        console.write(f"  brew install python && {pip_cmd}")
    elif sys_name == "Linux":
        if shutil.which("apt"):
            console.write(f"  sudo apt install python3-pip && {pip_cmd}")
        elif shutil.which("apt-get"):
            console.write(f"  sudo apt-get install python3-pip && {pip_cmd}")
    if shutil.which("pipx"):
        console.write(f"  pipx install {pkgs}")


def check_dependencies():
    """Ensure all required packages are installed and activate venv if needed."""
    console.write("--- Running System Dependency Check ---")
    required = sorted(_gather_required_packages())
    missing = []
    for pkg in required:
        try:
            if importlib.util.find_spec(pkg) is None:
                missing.append(pkg)
        except ValueError:
            # Python 3.13 may raise ValueError when __main__.__spec__ is None.
            # Fallback to a simple import attempt in that case.
            try:
                importlib.import_module(pkg)
            except Exception:
                missing.append(pkg)

    if missing:
        console.write(f"Missing packages detected: {', '.join(missing)}")
        _print_install_instructions(missing)
        exit_clean(1)
    else:
        console.write("✅ System Dependency Check Passed. Continuing...\n")


# Modules that rely on optional packages will be imported in ``main_workflow``

lcdm = None


def get_user_input_filepath(prompt_message, base_dir, must_exist=True):
    """Prompt for a file path relative to ``base_dir`` and ensure it exists."""

    # The loop continues until a valid path is provided or the user cancels.
    # This prevents accidental typos from immediately aborting the workflow.
    while True:
        filename = console.ask(f"{prompt_message} (or 'c' to cancel): ").strip()
        if filename.lower() == "c":
            return None
        filepath = os.path.join(base_dir, filename)
        if os.path.isfile(filepath):
            return filepath
        else:
            # Inform the user and loop again so they can correct the path.
            console.write(f"Error: File not found at '{filepath}'. Please try again.")


def load_alternative_model_plugin(model_filepath):
    """Dynamically loads an alternative cosmological model plugin."""
    logger = log_mod.get_logger()
    if not model_filepath.endswith(".py"):
        model_filepath += ".py"
    if not os.path.isfile(model_filepath):
        logger.error(f"Alternative model plugin file '{model_filepath}' not found.")
        return None
    try:
        module_name = os.path.splitext(os.path.basename(model_filepath))[0]
        spec = importlib.util.spec_from_file_location(module_name, model_filepath)
        alt_model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(alt_model_module)
        if not engine_interface.validate_plugin(alt_model_module):
            logger.error(
                f"Model plugin '{os.path.basename(model_filepath)}' failed validation."
            )
            return None
        logger.info(
            f"Successfully loaded alternative model: {alt_model_module.MODEL_NAME}"
        )
        return alt_model_module
    except Exception as e:
        logger.error(
            f"Error loading model plugin '{os.path.basename(model_filepath)}': {e}",
            exc_info=True,
        )
        return None


def select_from_list(options, prompt):
    """Display ``options`` and return the item chosen by the user."""

    # The caller supplies a short prompt ("Select model").  This helper prints
    # each option with a number so the user can respond with just an integer.
    # Returning ``None`` signals that the user cancelled the operation.
    if not options:
        return None
    header = prompt.replace("Select ", "").strip()
    if not header.endswith("s"):
        header += "s"
    console.write(f"\nAvailable {header}:")
    for i, opt in enumerate(options, 1):
        console.write(f"  {i}. {opt}")
    console.write("Write the number of your preferred choice or 'c' to cancel:")
    while True:
        choice = console.ask("> ").strip()
        if choice.lower() == "c":
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        console.write("Invalid selection. Try again.")


def parse_model_header(md_path):
    """Read minimal YAML front matter for plugin lookup."""
    # Only the YAML block at the start of the Markdown file is needed in
    # order to locate the generated Python module.  This keeps startup
    # snappy and avoids parsing the entire document.
    data = {}
    try:
        with open(md_path, "r") as f:
            lines = f.readlines()
        if lines and lines[0].strip() == "---":
            for line in lines[1:]:
                if line.strip() == "---":
                    break
                if ":" in line:
                    k, v = line.split(":", 1)
                    data[k.strip()] = v.strip().strip('"').strip("'")
    except Exception:
        pass
    return data


def cleanup_cache(base_dir):
    """Remove temporary files left behind by previous runs."""

    # Python leaves ``__pycache__`` folders behind when modules are imported.
    # Removing them ensures that stale bytecode doesn't interfere with
    # subsequent executions, especially when models are re-generated.
    logger = log_mod.get_logger()
    logger.info("--- Cleaning up cache files ---")
    for root, dirs, files in os.walk(base_dir):
        if "__pycache__" in dirs:
            pycache_path = os.path.join(root, "__pycache__")
            try:
                shutil.rmtree(pycache_path)
                logger.info(f"Removed cache directory: {pycache_path}")
            except OSError as e:
                logger.error(f"Error removing cache directory {pycache_path}: {e}")
    cache_dir = os.path.join(base_dir, "models", "cache")
    if os.path.isdir(cache_dir):
        for fname in os.listdir(cache_dir):
            if fname.startswith("cache_") and fname.endswith(".yml"):
                path = os.path.join(cache_dir, fname)
                try:
                    os.remove(path)
                    logger.info(f"Removed cache file: {path}")
                except OSError as e:
                    logger.error(f"Error removing cache file {path}: {e}")


def main_workflow():
    """Main workflow for the Copernican Suite."""
    # This routine coordinates the entire user interaction:
    #  * parse command-line flags
    #  * verify Python dependencies
    #  * load the reference ΛCDM model
    #  * repeatedly ask the user for models, data sources and engines
    #  * produce plots and CSV files with the results
    args = parse_args()
    check_dependencies()
    if args.run_tests:
        success = run_startup_tests()
        exit_clean(0 if success else 1)

    # Import optional third-party packages after confirming they are installed
    global np, plt, mp, model_parser, model_coder, engine_interface, data_loaders, plotter, csv_writer, log_mod, logger
    import numpy as np
    import matplotlib.pyplot as plt
    import multiprocessing as mp
    from copernican_lib import model_parser, model_coder, engine_interface
    from copernican_lib import (
        data_loaders,
        plotter,
        csv_writer,
        logger as log_mod,
        utils,
    )

    try:
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        SCRIPT_DIR = os.getcwd()

    OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    show_splash_screen()

    # Load the baseline LCDM model from YAML and validate it
    def _load_lcdm_model():
        models_dir = os.path.join(SCRIPT_DIR, "models")
        yaml_path = os.path.join(models_dir, "cosmo_model_lcdm.yml")
        cache_dir = os.path.join(models_dir, "cache")
        cache_path = model_parser.parse_model(yaml_path, cache_dir)
        func_dict, parsed = model_coder.generate_callables(cache_path)
        plugin = engine_interface.build_plugin(parsed, func_dict)
        plugin.MODEL_FILENAME = os.path.basename(yaml_path)
        return plugin

    global lcdm
    lcdm = _load_lcdm_model()
    engine_interface.validate_plugin(lcdm)

    while True:
        global CURRENT_LOG_FILE
        log_file = log_mod.setup_logging(log_dir=OUTPUT_DIR, base_dir=SCRIPT_DIR)
        CURRENT_LOG_FILE = log_file
        logger = log_mod.get_logger()
        utils.set_random_seed(0)
        start_ts = time.strftime("%y%m%d_%H%M%S")
        run_start_dt = datetime.datetime.now()
        run_start_pc = time.perf_counter()
        logger.info(
            f"Copernican {COPERNICAN_VERSION} has initialized! Current timestamp is {start_ts}. Log file: {log_file}"
        )
        logger.info(
            "Using standard CPU (SciPy) computational backend with multiprocessing."
        )
        logger.info(f"Running from base directory: {SCRIPT_DIR}")
        logger.info(f"All outputs will be saved to: {OUTPUT_DIR}")

        logger.info("\n--- Stage 1: Configuration ---\n")

        models_dir = os.path.join(SCRIPT_DIR, "models")
        model_files = sorted(
            [
                f
                for f in os.listdir(models_dir)
                if f.startswith("cosmo_model_") and f.endswith(".yml")
            ]
        )
        selected_model = select_from_list(model_files, "Select cosmological model")
        if not selected_model:
            _delete_log_file(log_file)
            cleanup_cache(SCRIPT_DIR)
            console.write("")
            return
        yaml_path = os.path.join(models_dir, selected_model)
        cache_dir = os.path.join(models_dir, "cache")
        try:
            cache_path = model_parser.parse_model(yaml_path, cache_dir)
        except Exception as e:
            logger.error(str(e))
            continue
        try:
            func_dict, parsed = model_coder.generate_callables(cache_path)
            alt_model_plugin = engine_interface.build_plugin(parsed, func_dict)
            alt_model_plugin.MODEL_FILENAME = os.path.basename(yaml_path)
            logger.info(f"Loaded YAML model: {parsed.get('model_name')}")
        except Exception as e:
            logger.error(f"Error generating model from YAML: {e}", exc_info=True)
            continue

        if not alt_model_plugin:
            continue

        engines_dir = os.path.join(SCRIPT_DIR, "engines")
        engine_files = sorted(
            [
                f
                for f in os.listdir(engines_dir)
                if f.startswith("cosmo_engine_") and f.endswith(".py")
            ]
        )
        engine_choice = select_from_list(engine_files, "Select computation engine")
        if not engine_choice:
            _delete_log_file(log_file)
            cleanup_cache(SCRIPT_DIR)
            console.write("")
            return
        engine_module = importlib.import_module(f"engines.{engine_choice[:-3]}")
        cosmo_engine_selected = engine_module

        sne_data_df = data_loaders.load_sne_data()
        if sne_data_df is None:
            continue

        bao_data_df = data_loaders.load_bao_data()
        if bao_data_df is None:
            continue

        cmb_data_df = data_loaders.load_cmb_data()
        if cmb_data_df is None:
            continue

        lcdm_time = 0.0
        alt_time = 0.0
        if hasattr(cosmo_engine_selected, "fit_combined_parameters"):
            logger.info("\n--- Stage 2: Combined Fit (SNe + BAO + CMB) ---\n")
            t0 = time.perf_counter()
            lcdm_sne_fit_results = cosmo_engine_selected.fit_combined_parameters(
                sne_data_df, bao_data_df, cmb_data_df, lcdm
            )
            lcdm_time += time.perf_counter() - t0
            t0 = time.perf_counter()
            alt_model_sne_fit_results = cosmo_engine_selected.fit_combined_parameters(
                sne_data_df, bao_data_df, cmb_data_df, alt_model_plugin
            )
            alt_time += time.perf_counter() - t0
        else:
            logger.info("\n--- Stage 2: Supernovae Ia Fitting ---\n")
            t0 = time.perf_counter()
            lcdm_sne_fit_results = cosmo_engine_selected.fit_sne_parameters(
                sne_data_df, lcdm
            )
            lcdm_time += time.perf_counter() - t0
            t0 = time.perf_counter()
            alt_model_sne_fit_results = cosmo_engine_selected.fit_sne_parameters(
                sne_data_df, alt_model_plugin
            )
            alt_time += time.perf_counter() - t0

        logger.info("\n--- Stage 3: BAO Analysis ---\n")

        min_z, max_z = bao_data_df["redshift"].min(), bao_data_df["redshift"].max()
        z_plot_smooth = np.geomspace(max(min_z * 0.8, 0.01), max_z * 1.2, 100)

        def run_bao_analysis(model_plugin, sne_fit_results, z_smooth_arr):
            """Helper to run BAO analysis for a given model."""
            if not (sne_fit_results and sne_fit_results.get("success")):
                logger.warning(
                    f"{model_plugin.MODEL_NAME} fit failed; skipping BAO analysis."
                )
                return {
                    "sne_fit_results": sne_fit_results,
                    "pred_df": None,
                    "rs_Mpc": np.nan,
                    "chi2_bao": np.inf,
                    "smooth_predictions": None,
                }

            fitted_cosmo_p = list(
                sne_fit_results["fitted_cosmological_params"].values()
            )
            pred_df, rs_Mpc, smooth_preds = (
                cosmo_engine_selected.calculate_bao_observables(
                    bao_data_df, model_plugin, fitted_cosmo_p, z_smooth=z_smooth_arr
                )
            )

            chi2_bao = np.inf
            if pred_df is not None and np.isfinite(rs_Mpc):
                chi2_bao = cosmo_engine_selected.chi_squared_bao(
                    bao_data_df, model_plugin, fitted_cosmo_p, rs_Mpc
                )
                logger.info(
                    f"{model_plugin.MODEL_NAME} BAO: r_s = {rs_Mpc:.2f} Mpc, Chi2_BAO = {chi2_bao:.2f}"
                )
            else:
                logger.warning(
                    f"{model_plugin.MODEL_NAME} BAO calculation failed or produced invalid r_s."
                )

            return {
                "sne_fit_results": sne_fit_results,
                "pred_df": pred_df,
                "rs_Mpc": rs_Mpc,
                "chi2_bao": chi2_bao,
                "smooth_predictions": smooth_preds,
            }

        def run_cmb_analysis(cmb_df, model_plugin, cosmo_params, cmb_extras=None):
            """Run CMB analysis for a given model."""
            # Skip the CMB step entirely when the model declares it is invalid
            # for such data. This prevents misleading chi-squared calculations
            # and ensures plugin validation does not fail for missing CMB
            # functions.
            if getattr(model_plugin, "valid_for_cmb", True) is False:
                logger.info(
                    f"{model_plugin.MODEL_NAME} does not support CMB; skipping analysis."
                )
                return {"chi2_cmb": np.inf, "theory_spectrum": None}

            if cmb_df is None or cmb_df.empty:
                return {"chi2_cmb": np.inf, "theory_spectrum": None}

            # Convert the fitted cosmological parameters to CAMB's expected
            # dictionary format using the helper provided by the model plugin.
            camb_params = model_plugin.get_camb_params(cosmo_params)
            # Append any additional CMB parameters recovered from a combined
            # fit so that the theoretical spectrum reflects the actual
            # optimisation result instead of falling back to defaults.
            if cmb_extras:
                camb_params.update(cmb_extras)

            components = ["TT"]
            if "Dl_te_obs" in cmb_df.columns:
                components.append("TE")
            if "Dl_ee_obs" in cmb_df.columns:
                components.append("EE")

            theory = cosmo_engine_selected.compute_cmb_spectrum(
                camb_params,
                cmb_df["ell"].values,
                spectra=tuple(components),
            )
            chi2_val = cosmo_engine_selected.chi_squared_cmb(
                cosmo_params,
                cmb_df,
                model_plugin,
                cmb_extras,
            )
            logger.info(f"{model_plugin.MODEL_NAME} CMB chi2 = {chi2_val:.2f}")
            return {"chi2_cmb": chi2_val, "theory_spectrum": theory}

        t0 = time.perf_counter()
        lcdm_full_results = run_bao_analysis(lcdm, lcdm_sne_fit_results, z_plot_smooth)
        lcdm_time += time.perf_counter() - t0
        t0 = time.perf_counter()
        alt_model_full_results = run_bao_analysis(
            alt_model_plugin, alt_model_sne_fit_results, z_plot_smooth
        )
        alt_time += time.perf_counter() - t0

        logger.info("\n--- Stage 4: CMB Analysis ---\n")

        t0 = time.perf_counter()
        lcdm_cmb = run_cmb_analysis(
            cmb_data_df,
            lcdm,
            list(lcdm_sne_fit_results["fitted_cosmological_params"].values()),
            lcdm_sne_fit_results.get("fitted_cmb_params"),
        )
        lcdm_time += time.perf_counter() - t0
        t0 = time.perf_counter()
        alt_cmb = run_cmb_analysis(
            cmb_data_df,
            alt_model_plugin,
            list(alt_model_sne_fit_results["fitted_cosmological_params"].values()),
            alt_model_sne_fit_results.get("fitted_cmb_params"),
        )
        alt_time += time.perf_counter() - t0

        logger.info("\n--- Stage 5: Generating Outputs ---\n")
        logger.info(f"{lcdm.MODEL_NAME} CMB chi2 = {lcdm_cmb['chi2_cmb']:.2f}")
        logger.info(
            f"{alt_model_plugin.MODEL_NAME} CMB chi2 = {alt_cmb['chi2_cmb']:.2f}"
        )

        run_end_dt = datetime.datetime.now()
        end_ts = run_end_dt.strftime("%Y%m%d_%H%M%S")
        new_log = os.path.join(OUTPUT_DIR, f"copernican-run_{end_ts}.txt")
        if log_file != new_log:
            try:
                os.rename(log_file, new_log)
                CURRENT_LOG_FILE = new_log
                logger.info(f"Log file renamed to {os.path.basename(new_log)}")
                log_file = new_log
            except OSError as e_ren:
                logger.error(f"Failed renaming log file: {e_ren}")

        plotter.plot_hubble_diagram(
            sne_data_df,
            lcdm_sne_fit_results,
            alt_model_sne_fit_results,
            lcdm,
            alt_model_plugin,
            plot_dir=OUTPUT_DIR,
            timestamp=end_ts,
        )
        if bao_data_df is not None:
            plotter.plot_bao_observables(
                bao_data_df,
                lcdm_full_results,
                alt_model_full_results,
                lcdm,
                alt_model_plugin,
                sne_data_df,
                plot_dir=OUTPUT_DIR,
                timestamp=end_ts,
            )
        if cmb_data_df is not None:
            plotter.plot_cmb_spectrum(
                cmb_data_df,
                lcdm_cmb,
                alt_cmb,
                lcdm_sne_fit_results,
                alt_model_sne_fit_results,
                lcdm,
                alt_model_plugin,
                plot_dir=OUTPUT_DIR,
                timestamp=end_ts,
            )

        console.write("\n--- Theory Abstracts ---\n")
        console.write(f"ΛCDM Abstract:\n{lcdm.MODEL_ABSTRACT}\n")
        console.write(
            f"{alt_model_plugin.MODEL_NAME} Abstract:\n{alt_model_plugin.MODEL_ABSTRACT}\n"
        )

        def _print_fit(label, sne_res, bao_res, cmb_res, plugin):
            console.write(f"--- {label} Fit Report ---\n")
            if sne_res:
                from copernican_lib import latex_utils

                p_names = getattr(plugin, "PARAMETER_NAMES", [])
                p_latex = getattr(plugin, "PARAMETER_LATEX_NAMES", [])
                for name, latex_name in zip(p_names, p_latex):
                    val = sne_res.get("fitted_cosmological_params", {}).get(name)
                    if val is not None:
                        disp = latex_utils.latex_to_unicode(latex_name)
                        console.write(f"  {disp} = {val:.5g}")
            chi2_sne = sne_res.get("chi2_sne", sne_res.get("chi2_min", float("nan")))
            chi2_total = sne_res.get("chi2_total", float("nan"))
            console.write(f"  χ²_Total = {chi2_total:.2f}")
            console.write(f"  χ²_SNe = {chi2_sne:.2f}")
            if bao_res:
                console.write(f"  χ²_BAO = {bao_res.get('chi2_bao', float('nan')):.2f}")
            if cmb_res:
                console.write(f"  χ²_CMB = {cmb_res.get('chi2_cmb', float('nan')):.2f}")
            console.write("")

        _print_fit("ΛCDM", lcdm_sne_fit_results, lcdm_full_results, lcdm_cmb, lcdm)
        _print_fit(
            alt_model_plugin.MODEL_NAME,
            alt_model_sne_fit_results,
            alt_model_full_results,
            alt_cmb,
            alt_model_plugin,
        )

        # The call to the redundant summary CSV has been removed.
        # csv_writer.save_sne_fit_results_csv(...)

        # Save the detailed point-by-point SNe results CSV
        csv_writer.save_sne_results_detailed_csv(
            sne_data_df,
            lcdm_sne_fit_results,
            alt_model_sne_fit_results,
            lcdm,
            alt_model_plugin,
            csv_dir=OUTPUT_DIR,
            timestamp=end_ts,
        )

        if bao_data_df is not None:
            csv_writer.save_bao_results_csv(
                bao_data_df,
                lcdm_full_results,
                alt_model_full_results,
                alt_model_name=alt_model_plugin.MODEL_NAME,
                csv_dir=OUTPUT_DIR,
                timestamp=end_ts,
            )
        if cmb_data_df is not None:
            csv_writer.save_cmb_results_csv(
                cmb_data_df,
                lcdm_cmb,
                alt_cmb,
                alt_model_name=alt_model_plugin.MODEL_NAME,
                csv_dir=OUTPUT_DIR,
                timestamp=end_ts,
            )

        console.write("\n" + "=" * 50)
        console.write("Evaluation complete. All files saved to the 'output' directory.")
        console.write("=" * 50 + "\n")

        total_time = time.perf_counter() - run_start_pc
        cpu_model, cpu_freq = _get_cpu_info()
        os_info = platform.platform()

        logger.info(f"Run completed at {end_ts}.")

        console.write(f"Run started on {run_start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
        console.write(f"Run ended on {run_end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
        console.write(
            f"Run took {lcdm_time:.2f}s for LCDM and {alt_time:.2f}s for {alt_model_plugin.MODEL_NAME}, "
            f"or {total_time:.2f}s in total, on a system with a {cpu_model} {cpu_freq}, "
            f"under {os_info}"
        )

        while True:
            another_run = (
                console.ask("Would you like to run another evaluation? (yes/no): ")
                .strip()
                .lower()
            )
            if another_run in ["yes", "y", "1"]:
                break
            elif another_run in ["no", "n", "2"]:
                cleanup_cache(SCRIPT_DIR)
                logger.info("Exiting Copernican Suite. Goodbye!")
                console.write("")
                return
            else:
                console.write("Invalid input. Please enter 'yes' or 'no'.")

        cleanup_cache(SCRIPT_DIR)


if __name__ == "__main__":
    # Multiprocessing start method must be 'spawn' so that each child process
    # inherits a pristine interpreter state. This avoids subtle issues when
    # worker processes import project modules that expect to run only once.
    import multiprocessing as _mp

    _mp.freeze_support()
    try:
        _mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # The start method was already set (e.g. by another library). Using
        # 'force=True' above normally prevents this, but wrap in try/except for
        # absolute safety.
        pass
    try:
        main_workflow()
    except Exception:
        logger_obj = log_mod.get_logger() if log_mod else None
        if logger_obj and logger_obj.hasHandlers():
            logger_obj.critical("Unhandled exception in main_workflow!", exc_info=True)
        else:
            console.write("CRITICAL UNHANDLED EXCEPTION IN MAIN WORKFLOW:")
            import traceback

            traceback.print_exc()
    finally:
        # Ensure that any generated plot windows are displayed at the very end
        if plt is not None and hasattr(plt, "get_fignums") and plt.get_fignums():
            console.write(
                "\nDisplaying plot(s). Close plot window(s) to exit script fully."
            )
            try:
                plt.show(block=True)
            except Exception as e_show:
                console.write(f"Error during final plt.show(): {e_show}")
        console.write("")
