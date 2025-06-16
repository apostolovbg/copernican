# copernican_suite/data_loaders.py
# DEV NOTE (v1.6a): Reworked into dynamic parser plugin registry.
"""Modular data loading for various cosmological datasets."""

import pandas as pd
import numpy as np
import json
import os
import logging
import importlib

# --- Parser Registry using metaclass auto-registration ---
PARSER_REGISTRY = {
    'sne': {},
    'bao': {},
    'cmb': {},
    'gw': {},
    'sirens': {}
}

class ParserMeta(type):
    """Metaclass that registers parser subclasses on import."""
    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        dt = getattr(cls, 'DATA_TYPE', None)
        src = getattr(cls, 'SOURCE_NAME', None)
        if not dt or not src or cls.__name__ == 'BaseParser':
            return
        dt = dt.lower()
        src = src.lower()
        PARSER_REGISTRY.setdefault(dt, {}).setdefault(src, []).append(cls)

class BaseParser(metaclass=ParserMeta):
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
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    for root, _, files in os.walk(base_dir):
        for fname in files:
            if not fname.startswith('cosmo_parser_') or not fname.endswith('.py'):
                continue
            if fname == 'cosmo_parser_template.py':
                continue

            rel = os.path.relpath(os.path.join(root, fname), os.path.dirname(__file__))

            module_name = rel.replace(os.sep, '.')[:-3]
            try:
                importlib.import_module(module_name)
            except Exception as e:
                logging.getLogger().error(f"Failed loading parser module {module_name}: {e}")

_discover_parsers()

# --- Utility functions ---

def list_available_sources(data_type):
    """Return sorted list of sources for a data type."""
    return sorted(PARSER_REGISTRY.get(data_type, {}).keys())

def list_parsers(data_type, source):
    """Return list of parser classes for a source."""
    return PARSER_REGISTRY.get(data_type, {}).get(source, [])

# --- Generic loading helpers ---

def _try_parsers(data_type, source, filepath, base_dir):
    logger = logging.getLogger()
    parsers = list_parsers(data_type, source)
    if not parsers:
        logger.error(f"No valid parsers found for {data_type.upper()} data source '{source}'.")
        return None
    for parser_cls in parsers:
        parser = parser_cls()
        if not parser.can_parse(filepath):
            continue
        extra_kwargs = {}
        if hasattr(parser, 'get_extra_args'):
            extra_kwargs = parser.get_extra_args(base_dir)
            if extra_kwargs is None:
                logger.info(f"{parser.PARSER_NAME} parser canceled by user.")
                return None
        try:
            logger.info(f"Attempting {parser.PARSER_NAME} on {os.path.basename(filepath)}")
            data_df = parser.parse(filepath, **extra_kwargs)
            if data_df is not None and not data_df.empty:
                data_df.attrs['filepath'] = filepath
                data_df.attrs['parser_name'] = parser.PARSER_NAME
                if 'dataset_name_attr' not in data_df.attrs:
                    data_df.attrs['dataset_name_attr'] = f"{data_type.upper()}_{parser.PARSER_NAME}"
                logger.info(f"Successfully loaded {len(data_df)} points using {parser.PARSER_NAME}.")
                return data_df
            elif data_df is None:
                logger.error(f"{parser.PARSER_NAME} returned None for {filepath}.")
            else:
                logger.error(f"{parser.PARSER_NAME} produced empty DataFrame for {filepath}.")
        except Exception as e:
            logger.error(f"Failed to parse '{filepath}' using {parser.PARSER_NAME}: {e}")
    logger.error(f"No parser succeeded for {filepath}.")
    return None

# --- Public Loading Functions ---

def load_sne_data(filepath):
    base_dir = os.path.dirname(filepath)
    source = os.path.basename(os.path.dirname(filepath))
    return _try_parsers('sne', source, filepath, base_dir)


def load_bao_data(filepath):
    base_dir = os.path.dirname(filepath)
    source = os.path.basename(os.path.dirname(filepath))
    return _try_parsers('bao', source, filepath, base_dir)


def load_cmb_data(filepath):
    base_dir = os.path.dirname(filepath)
    source = os.path.basename(os.path.dirname(filepath))
    return _try_parsers('cmb', source, filepath, base_dir)


def load_gw_data(filepath):
    base_dir = os.path.dirname(filepath)
    source = os.path.basename(os.path.dirname(filepath))
    return _try_parsers('gw', source, filepath, base_dir)


def load_siren_data(filepath):
    base_dir = os.path.dirname(filepath)
    source = os.path.basename(os.path.dirname(filepath))
    return _try_parsers('sirens', source, filepath, base_dir)
