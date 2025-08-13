# copernican_suite/data_loaders.py
"""
Modular data loading for various cosmological datasets (SNe, BAO, etc.).
"""
import importlib
import logging
import os

from . import console_output as console
from .utils import check_dataset_id, load_metadata_from_dir

# Each parser is registered via a decorator so that ``copernican.py`` can list
# available data sources dynamically. The loaders below simply call the
# registered functions after prompting the user.


# --- Parser Registry ---
# Each registry maps a short human readable key to a dictionary
# describing the parser function, its help text and an optional data
# directory.  Parser modules populate these registries via the
# decorators below when they are imported.
SNE_PARSERS: dict = {}
BAO_PARSERS: dict = {}
CMB_PARSERS: dict = {}
GW_PARSERS: dict = {}
SIREN_PARSERS: dict = {}


# --- Decorators to register parsers ---
def register_sne_parser(name=None, description="", data_dir=None):
    """Decorator to register a SNe data parsing function bound to a data
    source.

    The registration no longer requires the human readable dataset name or
    description up front. When ``name`` is ``None`` the base name of
    ``data_dir`` is used as a temporary key and replaced with the metadata
    ``dataset_name`` during discovery.
    """

    def decorator(func):
        """Store ``func`` in the SNe parser registry under a temporary key."""
        key = name or os.path.basename(data_dir or func.__name__)
        SNE_PARSERS[key] = {
            "function": func,
            "description": description,
            "data_dir": data_dir,
        }
        return func

    return decorator


def register_bao_parser(name=None, description="", data_dir=None):
    """Decorator to register a BAO data parsing function bound to a data
    source.

    When ``name`` is ``None`` the dataset directory name is used as a temporary
    key and replaced with the metadata-supplied ``dataset_name`` during
    discovery.
    """

    def decorator(func):
        """Store ``func`` in the BAO parser registry under a temporary key."""
        key = name or os.path.basename(data_dir or func.__name__)
        BAO_PARSERS[key] = {
            "function": func,
            "description": description,
            "data_dir": data_dir,
        }
        return func

    return decorator


def register_cmb_parser(name=None, description="", data_dir=None):
    """Decorator to register a CMB data parsing function bound to a data
    source.

    When ``name`` is omitted the dataset directory name acts as a temporary key
    until discovery replaces it with the metadata ``dataset_name``.
    """

    def decorator(func):
        """Store ``func`` in the CMB parser registry under a temporary key."""
        key = name or os.path.basename(data_dir or func.__name__)
        CMB_PARSERS[key] = {
            "function": func,
            "description": description,
            "data_dir": data_dir,
        }
        return func

    return decorator


def register_gw_parser(name=None, description="", data_dir=None):
    """Decorator to register a gravitational wave parser bound to a data
    source.

    Omitting ``name`` defers human-readable naming to metadata discovery.
    """

    def decorator(func):
        """Store ``func`` in the GW parser registry under a temporary key."""
        key = name or os.path.basename(data_dir or func.__name__)
        GW_PARSERS[key] = {
            "function": func,
            "description": description,
            "data_dir": data_dir,
        }
        return func

    return decorator


def register_siren_parser(name=None, description="", data_dir=None):
    """Decorator to register a standard siren parser bound to a data source.

    When ``name`` is ``None`` the dataset directory name serves as a temporary
    key and is replaced with the metadata ``dataset_name`` during discovery.
    """

    def decorator(func):
        """Store ``func`` in the standard siren parser registry."""
        key = name or os.path.basename(data_dir or func.__name__)
        SIREN_PARSERS[key] = {
            "function": func,
            "description": description,
            "data_dir": data_dir,
        }
        return func

    return decorator


# --- Dynamic Discovery of Parser Modules ---
def _discover_parsers():
    """Import parser modules and populate registries with dataset metadata."""
    # Parser modules register themselves in the dictionaries above when
    # imported.  Automatically scanning the data directory keeps the core
    # code agnostic to the exact set of available sources.  Metadata is
    # read here rather than inside the parser modules so that discovery and
    # user prompts present human readable dataset names without forcing the
    # parsers to access the metadata files themselves.
    base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    registry_map = {
        "sne": SNE_PARSERS,
        "bao": BAO_PARSERS,
        "cmb": CMB_PARSERS,
        "gw": GW_PARSERS,
        "sirens": SIREN_PARSERS,
    }
    for dtype in ("sne", "bao", "cmb", "gw", "sirens"):
        type_dir = os.path.join(base_dir, dtype)
        if not os.path.isdir(type_dir):
            continue
        for source in os.listdir(type_dir):
            # Skip placeholder folders so unfinished datasets do not appear in
            # the interactive menus.
            if source.lower() == "placeholder":
                continue
            src_dir = os.path.join(type_dir, source)
            if not os.path.isdir(src_dir):
                continue
            meta = load_metadata_from_dir(src_dir)
            placeholder_key = os.path.basename(src_dir)
            for fname in os.listdir(src_dir):
                if fname.startswith("cosmo_parser_") and fname.endswith(".py"):
                    module_name = f"data.{dtype}.{source}.{fname[:-3]}"
                    file_path = os.path.join(src_dir, fname)
                    spec = importlib.util.spec_from_file_location(
                        module_name,
                        file_path,
                    )
                    module = importlib.util.module_from_spec(spec)
                    try:
                        spec.loader.exec_module(module)
                    except Exception as e:
                        logging.getLogger().error(
                            f"Failed loading parser module {file_path}: {e}",
                        )
                    registry = registry_map[dtype]
                    if placeholder_key in registry:
                        entry = registry.pop(placeholder_key)
                        dataset_name = meta.get(
                            "dataset_name",
                            placeholder_key,
                        )
                        entry["description"] = meta.get(
                            "description",
                            entry.get("description", ""),
                        )
                        entry["data_dir"] = src_dir
                        registry[dataset_name] = entry
                        dataset_id = meta.get("dataset_id")
                        if dataset_id:
                            registry[dataset_id] = entry
                    # If the parser registered with an explicit name, simply
                    # update the description if metadata provides one.
                    else:
                        dataset_name = meta.get("dataset_name")
                        if dataset_name and dataset_name in registry:
                            registry[dataset_name]["description"] = meta.get(
                                "description",
                                registry[dataset_name].get("description", ""),
                            )
                            dataset_id = meta.get("dataset_id")
                            if dataset_id:
                                registry[dataset_id] = registry[dataset_name]


# Discover parsers at import time so that functions like
# ``load_sne_data`` can simply refer to the registries without
# additional setup.
_discover_parsers()


# --- Helper to list and select parsers ---
def _select_source(parser_registry, data_type_name):
    """Displays available data sources and prompts user for selection."""
    logger = logging.getLogger()
    if not parser_registry:
        logger.error(f"No parsers registered for {data_type_name} data.")
        return None

    logger.info(f"\nAvailable {data_type_name} data sources:")
    options = list(parser_registry.keys())
    for i, key in enumerate(options):
        desc = parser_registry[key]["description"]
        console.write(
            f"  {i+1}. {key} ({desc})" if desc else f"  {i+1}. {key}",
        )

    console.write(
        "Write the number of your preferred choice or 'c' to cancel:",
    )
    while True:
        try:
            choice = console.ask("> ").strip()
            if choice.lower() == "c":
                return None
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(options):
                return options[choice_idx]
            else:
                console.write("Invalid selection. Please try again.")
        except ValueError:
            console.write("Invalid input. Please enter a number or 'c'.")


# --- Verbose dataset info helper ---
def _log_dataset_info(df, data_type, logger):
    """Log summary and covariance usage for ``df``."""
    # Centralised helper so that every loader reports consistent
    # information about the dataset and whether a covariance matrix was
    # actually used.  ``load_*_data`` attaches all metadata via
    # :func:`load_metadata_from_dir` so the log entries rely solely on the
    # DataFrame attributes.
    if df is None or df.empty:
        return
    # Prefer the human-readable dataset name but fall back to ``dataset_id``
    # when only the identifier is available. Loaders attach both fields so
    # that logs remain descriptive while filenames stay concise.
    name = df.attrs.get("dataset_name", df.attrs.get("dataset_id", ""))
    logger.info(
        f"Loaded {data_type} dataset '{name}' with {len(df)} rows.",
    )
    if "covariance_matrix_inv" in df.attrs:
        if df.attrs["covariance_matrix_inv"] is not None:
            logger.info(
                f"{data_type} covariance matrix inverted successfully.",
            )
        else:
            logger.info(
                f"{data_type} parser provided no usable covariance matrix; "
                "using diagonal errors only.",
            )


# --- Main Loading Functions ---
def load_sne_data(source_key=None, **kwargs):
    """Loads SNe data for the chosen source."""
    logger = logging.getLogger()
    if source_key is None:
        source_key = _select_source(SNE_PARSERS, "SNe")
        if source_key is None:
            logger.info("SNe data loading canceled by user.")
            return None

    if source_key not in SNE_PARSERS:
        logger.error(f"No SNe parser registered for source '{source_key}'")
        return None

    entry = SNE_PARSERS[source_key]
    parser_func = entry["function"]
    data_dir = entry["data_dir"]
    try:
        logger.info(f"Attempting to load SNe data from source '{source_key}'")
        data_df = parser_func(data_dir, **kwargs)
        if data_df is not None and not data_df.empty:
            meta = load_metadata_from_dir(data_dir)
            if meta:
                data_df.attrs.update(meta)
            data_df.attrs["source_key"] = source_key
            dataset_name = data_df.attrs.get("dataset_name", source_key)
            data_df.attrs["dataset_name"] = dataset_name
            dataset_id = check_dataset_id(
                data_df.attrs.get(
                    "dataset_id",
                    dataset_name.replace(" ", "_").lower(),
                )
            )
            data_df.attrs["dataset_id"] = dataset_id
            logger.info(
                f"Successfully loaded {len(data_df)} SNe data points.",
            )
            _log_dataset_info(data_df, "SNe", logger)
        elif data_df is None:
            logger.error(
                f"SNe parser '{source_key}' returned None.",
            )
        else:
            logger.error(
                f"SNe parser '{source_key}' returned an empty DataFrame.",
            )
        return data_df
    except Exception as e:
        logger.critical(
            f"CRITICAL Error during SNe data parsing ({source_key}): {e}",
            exc_info=True,
        )
        return None


def load_bao_data(source_key=None, **kwargs):
    """Loads BAO data for the chosen source."""
    logger = logging.getLogger()
    if source_key is None:
        source_key = _select_source(BAO_PARSERS, "BAO")
        if source_key is None:
            logger.info("BAO data loading canceled by user.")
            return None

    if source_key not in BAO_PARSERS:
        logger.error(f"No BAO parser registered for source '{source_key}'")
        return None

    entry = BAO_PARSERS[source_key]
    parser_func = entry["function"]
    data_dir = entry["data_dir"]
    try:
        logger.info(f"Attempting to load BAO data from source '{source_key}'")
        data_df = parser_func(data_dir, **kwargs)
        if data_df is not None and not data_df.empty:
            meta = load_metadata_from_dir(data_dir)
            if meta:
                data_df.attrs.update(meta)
            data_df.attrs["source_key"] = source_key
            dataset_name = data_df.attrs.get("dataset_name", source_key)
            data_df.attrs["dataset_name"] = dataset_name
            dataset_id = check_dataset_id(
                data_df.attrs.get(
                    "dataset_id",
                    dataset_name.replace(" ", "_").lower(),
                )
            )
            data_df.attrs["dataset_id"] = dataset_id
            logger.info(
                f"Successfully loaded {len(data_df)} BAO data points.",
            )
            _log_dataset_info(data_df, "BAO", logger)
        elif data_df is None:
            logger.error(
                f"BAO parser '{source_key}' returned None.",
            )
        else:
            logger.error(
                f"BAO parser '{source_key}' returned an empty DataFrame.",
            )
        return data_df
    except Exception as e:
        logger.critical(
            f"CRITICAL Error during BAO data parsing ({source_key}): {e}",
            exc_info=True,
        )
        return None


def load_cmb_data(source_key=None, **kwargs):
    """Loads CMB data for the chosen source."""
    logger = logging.getLogger()
    if source_key is None:
        source_key = _select_source(CMB_PARSERS, "CMB")
        if source_key is None:
            logger.info("CMB data loading canceled by user.")
            return None

    if source_key not in CMB_PARSERS:
        logger.error(f"No CMB parser registered for source '{source_key}'")
        return None

    entry = CMB_PARSERS[source_key]
    parser_func = entry["function"]
    data_dir = entry["data_dir"]
    try:
        logger.info(f"Attempting to load CMB data from source '{source_key}'")
        data_df = parser_func(data_dir, **kwargs)
        if data_df is not None and not data_df.empty:
            meta = load_metadata_from_dir(data_dir)
            if meta:
                data_df.attrs.update(meta)
            data_df.attrs["source_key"] = source_key
            dataset_name = data_df.attrs.get("dataset_name", source_key)
            data_df.attrs["dataset_name"] = dataset_name
            dataset_id = check_dataset_id(
                data_df.attrs.get(
                    "dataset_id",
                    dataset_name.replace(" ", "_").lower(),
                )
            )
            data_df.attrs["dataset_id"] = dataset_id
            logger.info(
                f"Successfully loaded {len(data_df)} CMB data points.",
            )
            _log_dataset_info(data_df, "CMB", logger)
        elif data_df is None:
            logger.error(
                f"CMB parser '{source_key}' returned None.",
            )
        else:
            logger.error(
                f"CMB parser '{source_key}' returned an empty DataFrame.",
            )
        return data_df
    except Exception as e:
        logger.critical(
            f"CRITICAL Error during CMB data parsing ({source_key}): {e}",
            exc_info=True,
        )
        return None


def load_gw_data(source_key=None, **kwargs):
    """Loads gravitational wave data for the chosen source."""
    logger = logging.getLogger()
    if source_key is None:
        source_key = _select_source(GW_PARSERS, "GW")
        if source_key is None:
            logger.info("Gravitational wave data loading canceled by user.")
            return None

    if source_key not in GW_PARSERS:
        # fmt: off
        msg = (
            "No gravitational wave parser registered for source "
            f"'{source_key}'"
        )
        # fmt: on
        logger.error(msg)
        return None

    entry = GW_PARSERS[source_key]
    parser_func = entry["function"]
    data_dir = entry["data_dir"]
    try:
        logger.info(
            f"Attempting to load GW data from source '{source_key}'",
        )
        data_df = parser_func(data_dir, **kwargs)
        if data_df is not None and not data_df.empty:
            meta = load_metadata_from_dir(data_dir)
            if meta:
                data_df.attrs.update(meta)
            data_df.attrs["source_key"] = source_key
            dataset_name = data_df.attrs.get("dataset_name", source_key)
            data_df.attrs["dataset_name"] = dataset_name
            dataset_id = check_dataset_id(
                data_df.attrs.get(
                    "dataset_id",
                    dataset_name.replace(" ", "_").lower(),
                )
            )
            data_df.attrs["dataset_id"] = dataset_id
            logger.info(
                f"Successfully loaded {len(data_df)} GW data points.",
            )
            _log_dataset_info(data_df, "GW", logger)
        elif data_df is None:
            logger.error(
                f"GW parser '{source_key}' returned None.",
            )
        else:
            logger.error(
                f"GW parser '{source_key}' returned an empty DataFrame.",
            )
        return data_df
    except Exception as e:
        logger.critical(
            f"CRITICAL Error during GW data parsing ({source_key}): {e}",
            exc_info=True,
        )
        return None


def load_siren_data(source_key=None, **kwargs):
    """Loads standard siren data for the chosen source."""
    logger = logging.getLogger()
    if source_key is None:
        source_key = _select_source(SIREN_PARSERS, "standard siren")
        if source_key is None:
            logger.info("Standard siren data loading canceled by user.")
            return None

    if source_key not in SIREN_PARSERS:
        logger.error(
            f"No standard siren parser registered for source '{source_key}'",
        )
        return None

    entry = SIREN_PARSERS[source_key]
    parser_func = entry["function"]
    data_dir = entry["data_dir"]
    try:
        logger.info(
            f"Attempting to load siren data from source '{source_key}'",
        )
        data_df = parser_func(data_dir, **kwargs)
        if data_df is not None and not data_df.empty:
            meta = load_metadata_from_dir(data_dir)
            if meta:
                data_df.attrs.update(meta)
            data_df.attrs["source_key"] = source_key
            dataset_name = data_df.attrs.get("dataset_name", source_key)
            data_df.attrs["dataset_name"] = dataset_name
            dataset_id = check_dataset_id(
                data_df.attrs.get(
                    "dataset_id",
                    dataset_name.replace(" ", "_").lower(),
                )
            )
            data_df.attrs["dataset_id"] = dataset_id
            # fmt: off
            msg = (
                f"Successfully loaded {len(data_df)} standard siren "
                f"data points."
            )
            # fmt: on
            logger.info(msg)
            _log_dataset_info(data_df, "SIREN", logger)
        elif data_df is None:
            logger.error(
                f"Standard siren parser '{source_key}' returned None.",
            )
        else:
            logger.error(
                f"Standard siren parser '{source_key}' returned an empty "
                f"DataFrame.",
            )
        return data_df
    except Exception as e:
        # fmt: off
        err_msg = (
            "CRITICAL Error during standard siren data parsing "
            f"({source_key}): {e}"
        )
        # fmt: on
        logger.critical(err_msg, exc_info=True)
        return None
