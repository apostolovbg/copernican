# copernican_suite/data_loaders.py
# DEV NOTE (v1.6a): Reworked parser system with explicit registration and
# absolute dynamic imports.
"""Modular data loading for various cosmological datasets."""

import pandas as pd
import numpy as np
import json
import os
import logging
import importlib.util

# --- Global Parser Registry -------------------------------------------------

registry = {}  # {(data_type, source): [{'name': str, 'parser': BaseParser}]}


def register_parser(data_type, source, name, parser):
    """Register a parser instance for a given data type and source."""
    key = (data_type.lower(), source.lower())
    registry.setdefault(key, []).append({'name': name, 'parser': parser})


class BaseParser:
    """Base class for all data parsers."""
    DATA_TYPE = None
    SOURCE_NAME = None
    PARSER_NAME = None
    FILE_EXTENSIONS = []

    def can_parse(self, filepath):
        if not self.FILE_EXTENSIONS:
            return True
        filepath = filepath.lower()
        return any(filepath.endswith(ext.lower()) for ext in self.FILE_EXTENSIONS)

    def get_extra_args(self, base_dir):
        return {}

    def parse(self, filepath, **kwargs):
        raise NotImplementedError

# --- Helper function for user input ---
def _get_user_input_filepath(prompt_message, base_dir, must_exist=True):
    """Prompt the user for an additional file path."""
    while True:
        full_prompt = f"  > This data format requires an additional file.\n  > {prompt_message} (or 'c' to cancel): "
        filename = input(full_prompt).strip()
        if filename.lower() == 'c':
            return None
        filepath = os.path.join(base_dir, filename)
        if not must_exist or os.path.isfile(filepath):
            return filepath
        print(f"Error: File not found at '{filepath}'. Please try again.")

# --- Dynamic Discovery of Parser Modules ---

def _discover_parsers():
    """Import all parser modules under the ./parsers directory."""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'parsers'))
    logger = logging.getLogger()
    for root, _, files in os.walk(base_dir):
        for fname in files:
            if not fname.startswith('cosmo_parser_') or not fname.endswith('.py'):
                continue
            if fname == 'cosmo_parser_template.py':
                continue
            filepath = os.path.join(root, fname)
            module_name = f"parser_{abs(hash(filepath))}"
            try:
                spec = importlib.util.spec_from_file_location(module_name, filepath)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            except Exception as e:
                logger.error(f"\u274c Failed to import '{fname}': {e}")

_discover_parsers()

# --- Utility functions ------------------------------------------------------

def list_available_sources(data_type):
    """Return sorted list of sources for a data type."""
    return sorted({src for dt, src in registry.keys() if dt == data_type})


def list_parsers(data_type, source):
    """Return list of parser info dicts for a source."""
    return registry.get((data_type, source), [])

# --- Generic loading helpers ---

def load_data(data_type, source, parser_info, filepath):
    """Load a file using the specified parser info dict."""
    logger = logging.getLogger()
    parser = parser_info['parser']
    if not parser.can_parse(filepath):
        logger.error(f"{parser_info['name']} cannot parse {os.path.basename(filepath)}.")
        return None
    extra_kwargs = {}
    if hasattr(parser, 'get_extra_args'):
        extra_kwargs = parser.get_extra_args(os.path.dirname(filepath))
        if extra_kwargs is None:
            logger.info(f"{parser_info['name']} parser canceled by user.")
            return None
    try:
        logger.info(f"Using parser {parser_info['name']} on {os.path.basename(filepath)}")
        data_df = parser.parse(filepath, **extra_kwargs)
        if data_df is not None and not data_df.empty:
            data_df.attrs['filepath'] = filepath
            data_df.attrs['parser_name'] = parser_info['name']
            if 'dataset_name_attr' not in data_df.attrs:
                data_df.attrs['dataset_name_attr'] = f"{data_type.upper()}_{parser_info['name']}"
            logger.info(f"Successfully loaded {len(data_df)} points using {parser_info['name']}.")
            return data_df
        elif data_df is None:
            logger.error(f"{parser_info['name']} returned None for {filepath}.")
        else:
            logger.error(f"{parser_info['name']} produced empty DataFrame for {filepath}.")
    except Exception as e:
        logger.error(f"Failed to parse '{filepath}' using {parser_info['name']}: {e}")
    return None
# --- Backwards Compatibility Helpers ---------------------------------------

def _try_all_parsers(data_type, source, filepath):
    """Attempt parsing with all registered parsers for fallback use."""
    for info in list_parsers(data_type, source):
        df = load_data(data_type, source, info, filepath)
        if df is not None:
            return df
    logging.getLogger().error(f"No parser succeeded for {filepath}.")
    return None


def load_sne_data(filepath):
    base_dir = os.path.dirname(filepath)
    source = os.path.basename(os.path.dirname(filepath))
    return _try_all_parsers('sne', source, filepath)


def load_bao_data(filepath):
    base_dir = os.path.dirname(filepath)
    source = os.path.basename(os.path.dirname(filepath))
    return _try_all_parsers('bao', source, filepath)


def load_cmb_data(filepath):
    base_dir = os.path.dirname(filepath)
    source = os.path.basename(os.path.dirname(filepath))
    return _try_all_parsers('cmb', source, filepath)


def load_gw_data(filepath):
    base_dir = os.path.dirname(filepath)
    source = os.path.basename(os.path.dirname(filepath))
    return _try_all_parsers('gw', source, filepath)


def load_siren_data(filepath):
    base_dir = os.path.dirname(filepath)
    source = os.path.basename(os.path.dirname(filepath))
    return _try_all_parsers('sirens', source, filepath)

