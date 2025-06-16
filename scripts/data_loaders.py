# copernican_suite/data_loaders.py
# DEV NOTE (v1.5f): Added registries and loader functions for CMB, gravitational
# wave, and standard siren data types as preparation for future datasets.
# DEV NOTE (v1.5f hotfix): File moved to ``scripts/`` package; import paths for parsers updated.
# DEV NOTE (v1.5f hotfix 9): Corrected parser discovery path to top-level ``parsers`` directory.
"""
Modular data loading for various cosmological datasets (SNe, BAO, etc.).
"""
import pandas as pd
import numpy as np
import json
import os
import logging
import importlib

# --- Parser Registry ---
SNE_PARSERS = {}
BAO_PARSERS = {}
CMB_PARSERS = {}
GW_PARSERS = {}
SIREN_PARSERS = {}

# --- Helper function for user input, localized to this module ---
def _get_user_input_filepath(prompt_message, base_dir, must_exist=True):
    """Prompts the user for a filepath and validates it."""
    while True:
        # Construct the full prompt to be clear
        full_prompt = f"  > This data format requires an additional file.\n  > {prompt_message} (or 'c' to cancel): "
        filename = input(full_prompt).strip()
        if filename.lower() == 'c': return None
        # We assume the file is in the same base directory as the main script
        filepath = os.path.join(base_dir, filename)
        if not must_exist or os.path.isfile(filepath): return filepath
        else: print(f"Error: File not found at '{filepath}'. Please try again.")

# --- Decorators to register parsers ---
def register_sne_parser(name, description="", extra_args_func=None):
    """
    Decorator to register a SNe data parsing function.
    Can optionally include a function to gather extra arguments from the user.
    """
    def decorator(func):
        SNE_PARSERS[name] = {
            'function': func,
            'description': description,
            'extra_args_func': extra_args_func
        }
        return func
    return decorator

def register_bao_parser(name, description=""):
    """Decorator to register a BAO data parsing function."""
    def decorator(func):
        BAO_PARSERS[name] = {'function': func, 'description': description}
        return func
    return decorator

def register_cmb_parser(name, description=""):
    """Decorator to register a CMB data parsing function."""
    def decorator(func):
        CMB_PARSERS[name] = {'function': func, 'description': description}
        return func
    return decorator

def register_gw_parser(name, description=""):
    """Decorator to register a gravitational wave data parsing function."""
    def decorator(func):
        GW_PARSERS[name] = {'function': func, 'description': description}
        return func
    return decorator

def register_siren_parser(name, description=""):
    """Decorator to register a standard siren data parsing function."""
    def decorator(func):
        SIREN_PARSERS[name] = {'function': func, 'description': description}
        return func
    return decorator

# --- Dynamic Discovery of Parser Modules ---
def _discover_parsers():
    """Imports all parser modules under the project's ``parsers`` directory."""
    # ``data_loaders.py`` lives in ``scripts/`` so we move one level up to find
    # the root ``parsers`` folder next to ``scripts``.
    base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'parsers')
    for sub in ('sne', 'bao', 'cmb', 'gw', 'sirens'):
        subdir = os.path.join(base_dir, sub)
        if not os.path.isdir(subdir):
            continue
        for fname in os.listdir(subdir):
            if fname.startswith('cosmo_parser_') and fname.endswith('.py'):
                module_name = f"parsers.{sub}.{fname[:-3]}"
                try:
                    importlib.import_module(module_name)
                except Exception as e:
                    logging.getLogger().error(f"Failed loading parser module {module_name}: {e}")

# Discover parsers at import time
_discover_parsers()

# --- Helper to list and select parsers ---
def _select_parser(parser_registry, data_type_name):
    """Displays available parsers and prompts user for selection."""
    logger = logging.getLogger()
    if not parser_registry:
        logger.error(f"No parsers registered for {data_type_name} data.")
        return None

    logger.info(f"\nAvailable {data_type_name} data format parsers:")
    options = list(parser_registry.keys())
    for i, key in enumerate(options):
        desc = parser_registry[key]['description']
        # Print to console directly for user interaction
        print(f"  {i+1}. {key} ({desc})")

    while True:
        try:
            choice = input(f"Select a {data_type_name} parser (number) or 'c' to cancel: ")
            if choice.lower() == 'c':
                return None
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(options):
                return options[choice_idx]
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number or 'c'.")

# --- Main Loading Functions ---
def load_sne_data(filepath, format_key=None, **kwargs):
    """Loads SNe data using a registered parser."""
    logger = logging.getLogger()
    if not os.path.isfile(filepath):
        logger.error(f"SNe data file not found at {filepath}")
        return None

    if format_key is None:
        format_key = _select_parser(SNE_PARSERS, "SNe")
        if format_key is None:
            logger.info("SNe data loading canceled by user.")
            return None
    
    if format_key not in SNE_PARSERS:
        logger.error(f"No SNe parser registered for format_key '{format_key}'")
        return None
        
    try:
        logger.info(f"Attempting to load SNe data from '{os.path.basename(filepath)}' using format: {format_key}")
        parser_func = SNE_PARSERS[format_key]['function']
        data_df = parser_func(filepath, **kwargs) 
        if data_df is not None and not data_df.empty:
            data_df.attrs['filepath'] = filepath
            data_df.attrs['format_key'] = format_key
            if 'dataset_name_attr' not in data_df.attrs: 
                data_df.attrs['dataset_name_attr'] = f"SNe_{format_key.replace(' ', '_')}"
            logger.info(f"Successfully loaded {len(data_df)} SNe data points.")
        elif data_df is None:
             logger.error(f"SNe parser '{format_key}' returned None for {filepath}.")
        else: 
             logger.error(f"SNe parser '{format_key}' returned an empty DataFrame for {filepath}.")
        return data_df
    except Exception as e:
        logger.critical(f"CRITICAL Error during SNe data parsing ({filepath}, {format_key}): {e}", exc_info=True)
        return None

def load_bao_data(filepath, format_key=None, **kwargs):
    """Loads BAO data using a registered parser."""
    logger = logging.getLogger()
    if not os.path.isfile(filepath):
        logger.error(f"BAO data file not found at {filepath}")
        return None

    if format_key is None:
        format_key = _select_parser(BAO_PARSERS, "BAO")
        if format_key is None:
            logger.info("BAO data loading canceled by user.")
            return None

    if format_key not in BAO_PARSERS:
        logger.error(f"No BAO parser registered for format_key '{format_key}'")
        return None
        
    try:
        logger.info(f"Attempting to load BAO data from '{os.path.basename(filepath)}' using format: {format_key}")
        parser_func = BAO_PARSERS[format_key]['function']
        data_df = parser_func(filepath, **kwargs)
        if data_df is not None and not data_df.empty:
            data_df.attrs['filepath'] = filepath
            data_df.attrs['format_key'] = format_key
            if 'dataset_name_attr' not in data_df.attrs: 
                data_df.attrs['dataset_name_attr'] = f"BAO_{format_key.replace(' ', '_')}"
            logger.info(f"Successfully loaded {len(data_df)} BAO data points.")
        elif data_df is None:
            logger.error(f"BAO parser '{format_key}' returned None for {filepath}.")
        else: 
            logger.error(f"BAO parser '{format_key}' returned an empty DataFrame for {filepath}.")
        return data_df
    except Exception as e:
        logger.critical(f"CRITICAL Error during BAO data parsing ({filepath}, {format_key}): {e}", exc_info=True)
        return None

def load_cmb_data(filepath, format_key=None, **kwargs):
    """Loads CMB data using a registered parser."""
    logger = logging.getLogger()
    if not os.path.isfile(filepath):
        logger.error(f"CMB data file not found at {filepath}")
        return None

    if format_key is None:
        format_key = _select_parser(CMB_PARSERS, "CMB")
        if format_key is None:
            logger.info("CMB data loading canceled by user.")
            return None

    if format_key not in CMB_PARSERS:
        logger.error(f"No CMB parser registered for format_key '{format_key}'")
        return None

    try:
        logger.info(f"Attempting to load CMB data from '{os.path.basename(filepath)}' using format: {format_key}")
        parser_func = CMB_PARSERS[format_key]['function']
        data_df = parser_func(filepath, **kwargs)
        if data_df is not None and not data_df.empty:
            data_df.attrs['filepath'] = filepath
            data_df.attrs['format_key'] = format_key
            if 'dataset_name_attr' not in data_df.attrs:
                data_df.attrs['dataset_name_attr'] = f"CMB_{format_key.replace(' ', '_')}"
            logger.info(f"Successfully loaded {len(data_df)} CMB data points.")
        elif data_df is None:
            logger.error(f"CMB parser '{format_key}' returned None for {filepath}.")
        else:
            logger.error(f"CMB parser '{format_key}' returned an empty DataFrame for {filepath}.")
        return data_df
    except Exception as e:
        logger.critical(f"CRITICAL Error during CMB data parsing ({filepath}, {format_key}): {e}", exc_info=True)
        return None

def load_gw_data(filepath, format_key=None, **kwargs):
    """Loads gravitational wave data using a registered parser."""
    logger = logging.getLogger()
    if not os.path.isfile(filepath):
        logger.error(f"Gravitational wave data file not found at {filepath}")
        return None

    if format_key is None:
        format_key = _select_parser(GW_PARSERS, "GW")
        if format_key is None:
            logger.info("Gravitational wave data loading canceled by user.")
            return None

    if format_key not in GW_PARSERS:
        logger.error(f"No gravitational wave parser registered for format_key '{format_key}'")
        return None

    try:
        logger.info(f"Attempting to load GW data from '{os.path.basename(filepath)}' using format: {format_key}")
        parser_func = GW_PARSERS[format_key]['function']
        data_df = parser_func(filepath, **kwargs)
        if data_df is not None and not data_df.empty:
            data_df.attrs['filepath'] = filepath
            data_df.attrs['format_key'] = format_key
            if 'dataset_name_attr' not in data_df.attrs:
                data_df.attrs['dataset_name_attr'] = f"GW_{format_key.replace(' ', '_')}"
            logger.info(f"Successfully loaded {len(data_df)} GW data points.")
        elif data_df is None:
            logger.error(f"GW parser '{format_key}' returned None for {filepath}.")
        else:
            logger.error(f"GW parser '{format_key}' returned an empty DataFrame for {filepath}.")
        return data_df
    except Exception as e:
        logger.critical(f"CRITICAL Error during GW data parsing ({filepath}, {format_key}): {e}", exc_info=True)
        return None

def load_siren_data(filepath, format_key=None, **kwargs):
    """Loads standard siren data using a registered parser."""
    logger = logging.getLogger()
    if not os.path.isfile(filepath):
        logger.error(f"Standard siren data file not found at {filepath}")
        return None

    if format_key is None:
        format_key = _select_parser(SIREN_PARSERS, "standard siren")
        if format_key is None:
            logger.info("Standard siren data loading canceled by user.")
            return None

    if format_key not in SIREN_PARSERS:
        logger.error(f"No standard siren parser registered for format_key '{format_key}'")
        return None

    try:
        logger.info(f"Attempting to load siren data from '{os.path.basename(filepath)}' using format: {format_key}")
        parser_func = SIREN_PARSERS[format_key]['function']
        data_df = parser_func(filepath, **kwargs)
        if data_df is not None and not data_df.empty:
            data_df.attrs['filepath'] = filepath
            data_df.attrs['format_key'] = format_key
            if 'dataset_name_attr' not in data_df.attrs:
                data_df.attrs['dataset_name_attr'] = f"SIREN_{format_key.replace(' ', '_')}"
            logger.info(f"Successfully loaded {len(data_df)} standard siren data points.")
        elif data_df is None:
            logger.error(f"Standard siren parser '{format_key}' returned None for {filepath}.")
        else:
            logger.error(f"Standard siren parser '{format_key}' returned an empty DataFrame for {filepath}.")
        return data_df
    except Exception as e:
        logger.critical(f"CRITICAL Error during standard siren data parsing ({filepath}, {format_key}): {e}", exc_info=True)
        return None
