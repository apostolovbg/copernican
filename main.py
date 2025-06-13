# main.py
"""
DEV NOTE (v2.0): Consolidated command-line interface for the Copernican Suite.
This script checks dependencies, shows a splash screen, discovers plugins,
and runs selected engines with CosmoDSL models.
"""

import os
import sys
import importlib.util
import argparse
from engines.cosmo_engine_new1 import discover_engines as _discover_engines

VERSION = "2.0"

# --- Dependency Check -------------------------------------------------------

def check_dependencies():
    required = ["numpy", "scipy", "matplotlib", "pandas"]
    missing = []
    for lib in required:
        try:
            __import__(lib)
        except ImportError:
            missing.append(lib)
    if missing:
        print("Missing required libraries: " + ", ".join(missing))
        sys.exit(1)

# --- Splash Screen ---------------------------------------------------------

def splash_screen():
    print("\n===========================================================")
    print("        C O P E R N I C A N   S U I T E")
    print(f"                v{VERSION}\n")
    print("    A Modular Cosmology Framework built with CosmoDSL")
    print("===========================================================\n")

# --- Discovery Helpers ------------------------------------------------------

def discover_engines(path):
    return _discover_engines(path)


def discover_models(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.md')]


def discover_data(path):
    return [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]


def list_cov_matrices(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.cov')]

# --- Simple Prompt Utilities -----------------------------------------------

def prompt_choice(msg, options):
    """Ask the user to pick a single option by number."""
    while True:
        print(msg)
        for i, opt in enumerate(options, 1):
            print(f"  {i}. {os.path.basename(opt)}")
        inp = input("Choice: ").strip()
        if inp.isdigit() and 1 <= int(inp) <= len(options):
            return options[int(inp) - 1]
        print("Invalid choice. Please try again.\n")


def prompt_multichoice(msg, options):
    """Prompt the user to select multiple options separated by commas."""
    selected = []
    while not selected:
        print(msg)
        for i, opt in enumerate(options, 1):
            print(f"  {i}. {os.path.basename(opt)}")
        inp = input("Comma separated numbers: ")
        indices = []
        for c in inp.split(','):
            c = c.strip()
            if c.isdigit() and 1 <= int(c) <= len(options):
                indices.append(int(c) - 1)
        if indices:
            selected = [options[i] for i in indices]
        else:
            print("No valid selections made. Please try again.\n")
    return selected

# --- CosmoDSL Loader --------------------------------------------------------

def load_model(path):
    sections = {}
    current = None
    with open(path, 'r') as f:
        for line in f:
            if line.startswith(':::'):
                current = line.strip()[3:]
                sections[current] = []
            else:
                if current:
                    sections[current].append(line.rstrip())
    for key, val in sections.items():
        sections[key] = '\n'.join(val).strip()
    params = {}
    for line in sections.get('parameters', '').splitlines():
        if '=' in line:
            k, v = line.split('#')[0].split('=', 1)
            params[k.strip()] = float(v)
    sections['parameters'] = params
    return sections


def load_engine(path):
    spec = importlib.util.spec_from_file_location('engine_mod', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# --- Main Workflow ---------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="Copernican Suite")
    parser.add_argument('--version', action='store_true', help='Print version and exit')
    args = parser.parse_args(argv)

    if args.version:
        print(f"Copernican Suite v{VERSION}")
        return

    check_dependencies()
    splash_screen()

    engines = discover_engines('engines')
    models = discover_models('models')
    data = discover_data('data')
    covars = list_cov_matrices('data')

    engine_path = prompt_choice('Choose engine:', engines)
    model_path = prompt_choice("Choose model (or 'test'→lcdm):", models + ['test'])
    data_files = prompt_multichoice('Choose data files:', data)
    covar_file = prompt_choice("Choose covariance matrix (or 'none'):", covars + ['none'])

    engine_module = load_engine(engine_path)
    model_ast = load_model(model_path if model_path != 'test' else 'models/cosmo_model_lcdm.md')
    engine = engine_module.Engine(model_ast)
    print(engine.run(model_ast, data_files, covar_file))

if __name__ == '__main__':
    main(sys.argv[1:])
