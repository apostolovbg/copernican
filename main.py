# main.py
"""
DEV NOTE: Introduces the new command-line orchestrator using CosmoDSL and
plugin-based engines. This script dynamically discovers available engines,
models, and data files.
"""

import os
import importlib.util
from engines.cosmo_engine_new1 import discover_engines as _discover_engines

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
    print(msg)
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {os.path.basename(opt)}")
    idx = int(input("Choice: ")) - 1
    return options[idx]


def prompt_multichoice(msg, options):
    print(msg)
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {os.path.basename(opt)}")
    choices = input("Comma separated numbers: ")
    indices = [int(c.strip()) - 1 for c in choices.split(',') if c.strip().isdigit()]
    return [options[i] for i in indices]

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
    # Parameters block -> dict of values
    params = {}
    for line in sections.get('parameters', '').splitlines():
        if '=' in line:
            k, v = line.split('#')[0].split('=',1)
            params[k.strip()] = float(v)
    sections['parameters'] = params
    return sections


def load_engine(path):
    spec = importlib.util.spec_from_file_location('engine_mod', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# --- Main Workflow ---------------------------------------------------------

def main():
    engines = discover_engines('engines')
    models = discover_models('models')
    data = discover_data('data')
    covars = list_cov_matrices('data')

    engine_path = prompt_choice('Choose engine:', engines)
    model_path = prompt_choice("Choose model (or 'test'\u2192lcdm):", models + ['test'])
    data_files = prompt_multichoice('Choose data files:', data)
    covar_file = prompt_choice("Choose covariance matrix (or 'none'):", covars + ['none'])

    engine_module = load_engine(engine_path)
    model_ast = load_model(model_path if model_path != 'test' else 'models/cosmo_model_lcdm.md')
    engine = engine_module.Engine(model_ast)
    print(engine.run(model_ast, data_files, covar_file))

if __name__ == '__main__':
    main()
