# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.

# copernican_suite/data_loaders.py
"""Dataset discovery and loading helpers.

The suite treats observational data as pluggable components.  Parsers live
alongside their raw tables under ``data/<type>/<source>/`` and register
themselves through decorators provided here.  At runtime
``copernican.py`` imports these modules to populate interactive menus and
returns uniformly formatted :class:`pandas.DataFrame` objects with metadata
stored on ``.attrs``.
"""
import hashlib
import importlib
import logging
import os

from . import console_output as console
from .utils import check_dataset_id, load_metadata_from_dir

# Each parser is registered via a decorator so that ``copernican.py`` can list
# available data sources dynamically. The loaders below simply call the
# registered functions after prompting the user.


# --- Parser Registry ---
# Each registry maps a unique ``dataset_id`` to a dictionary describing the
# parser function, human readable ``dataset_name``, ``description`` and an
# optional ``data_dir``.  Parser modules populate these registries via the
# decorators below when they are imported.  ``dataset_id`` values are derived
# from the metadata files located next to the raw tables and are therefore
# mandatory.
SNE_PARSERS: dict = {}
BAO_PARSERS: dict = {}
CMB_PARSERS: dict = {}
GW_PARSERS: dict = {}
SIREN_PARSERS: dict = {}


# --- Decorators to register parsers ---
def register_sne_parser(name=None, description="", data_dir=None):
    """Register a SNe data parsing function bound to a data source.

    ``name`` acts as a temporary key until metadata discovery provides the
    canonical ``dataset_id``. The same placeholder is stored as
    ``dataset_name`` so that direct calls during tests remain functional before
    discovery runs.
    """

    def decorator(func):
        """Store ``func`` in the SNe parser registry under a temporary key."""
        key = name or os.path.basename(data_dir or func.__name__)
        SNE_PARSERS[key] = {
            "function": func,
            "dataset_name": name or key,
            "description": description,
            "data_dir": data_dir,
        }
        return func

    return decorator


def register_bao_parser(name=None, description="", data_dir=None):
    """Register a BAO data parsing function bound to a data source.

    ``name`` again provides a temporary key and placeholder ``dataset_name``
    until discovery replaces the registry key with the metadata supplied
    ``dataset_id``.
    """

    def decorator(func):
        """Store ``func`` in the BAO parser registry under a temporary key."""
        key = name or os.path.basename(data_dir or func.__name__)
        BAO_PARSERS[key] = {
            "function": func,
            "dataset_name": name or key,
            "description": description,
            "data_dir": data_dir,
        }
        return func

    return decorator


def register_cmb_parser(name=None, description="", data_dir=None):
    """Register a CMB data parsing function bound to a data source.

    ``name`` is used as a placeholder ``dataset_id`` and ``dataset_name`` until
    metadata discovery replaces the key with the canonical identifier.
    """

    def decorator(func):
        """Store ``func`` in the CMB parser registry under a temporary key."""
        key = name or os.path.basename(data_dir or func.__name__)
        CMB_PARSERS[key] = {
            "function": func,
            "dataset_name": name or key,
            "description": description,
            "data_dir": data_dir,
        }
        return func

    return decorator


def register_gw_parser(name=None, description="", data_dir=None):
    """Register a gravitational wave parser bound to a data source.

    ``name`` again serves as a provisional registry key and ``dataset_name``
    until metadata discovery provides the definitive ``dataset_id``.
    """

    def decorator(func):
        """Store ``func`` in the GW parser registry under a temporary key."""
        key = name or os.path.basename(data_dir or func.__name__)
        GW_PARSERS[key] = {
            "function": func,
            "dataset_name": name or key,
            "description": description,
            "data_dir": data_dir,
        }
        return func

    return decorator


def register_siren_parser(name=None, description="", data_dir=None):
    """Register a standard siren parser bound to a data source.

    ``name`` acts as the temporary key and placeholder ``dataset_name`` until
    discovery replaces it with the metadata ``dataset_id``.
    """

    def decorator(func):
        """Store ``func`` in the standard siren parser registry."""
        key = name or os.path.basename(data_dir or func.__name__)
        SIREN_PARSERS[key] = {
            "function": func,
            "dataset_name": name or key,
            "description": description,
            "data_dir": data_dir,
        }
        return func

    return decorator


TRUSTED_PARSER_HASHES = {
    # ``relative_path`` -> ``sha256``
    "sne/pantheon/cosmo_parser_pantheon.py": (
        "a15cfd8cec9104e62aebeb03fc72b148d8da76b33e90ede4537eddbe3310d0a6"
    ),
    "sne/jla2014/cosmo_parser_jla2014.py": (
        "27b553519fa4545153c675f82141be2e2ed35a69b91ce5d72b0add794fb25339"
    ),
    "bao/bossdr12/cosmo_parser_bossdr12.py": (
        "4de5b07156d65e4e075810745c6b61cf8b7f10f0e4c575be9d6d16ebbfcf37b8"
    ),
    "bao/compound/cosmo_parser_compound.py": (
        "a203b6f5efe3742c4cb2253d1f96691c7af769af1718f9526a73069afd3cf126"
    ),
    "cmb/planck2018lite/cosmo_parser_cmb_planck2018lite.py": (
        "3017407d77779873a0eb145d9f8f420c0ea83da33431fab70ecfd8b5ee6a23de"
    ),
    "gw/placeholder/cosmo_parser_gw_placeholder.py": (
        "10d0159cdd879a74324c852be92e877308b949ff0375c9e9609da3a95c0fe3e2"
    ),
    "sirens/placeholder/cosmo_parser_sirens_placeholder.py": (
        "816f2624ff8452ae7fd41c138fcc73b5a5272117d931aae342c9eee6246d3f58"
    ),
}


def _file_sha256(path: str) -> str:
    """Return the SHA256 digest for ``path``."""
    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


# --- Dynamic Discovery of Parser Modules ---
def _discover_parsers(base_dir: str | None = None):
    """Import parser modules and populate registries with dataset metadata.

    The scan walks ``data/`` recursively, ignoring ``placeholder`` folders so
    unfinished datasets stay hidden.  Each candidate parser is verified against
    ``TRUSTED_PARSER_HASHES`` before import to guard against tampering.  Only
    trusted modules are executed, keeping the discovery step resilient to
    untrusted files shipped alongside the data tables.
    Metadata is read here to keep the parser implementations small and focused
    solely on table parsing.
    """
    if base_dir is None:
        base_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data"
        )
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
            dataset_id = check_dataset_id(meta.get("dataset_id", ""))
            dataset_name = meta.get("dataset_name")
            if not dataset_id or not dataset_name:
                logging.getLogger().error(
                    "Missing dataset_id or dataset_name in %s",
                    src_dir,
                )
                continue
            placeholder_key = os.path.basename(src_dir)
            for fname in os.listdir(src_dir):
                if fname.startswith("cosmo_parser_") and fname.endswith(".py"):
                    module_name = f"data.{dtype}.{source}.{fname[:-3]}"
                    file_path = os.path.join(src_dir, fname)
                    rel_path = os.path.relpath(file_path, base_dir)
                    # Normalise path separators
                    # for cross-platform hash lookup.
                    rel_path = rel_path.replace("\\", "/")
                    expected_hash = TRUSTED_PARSER_HASHES.get(rel_path)
                    if expected_hash is None:
                        logging.getLogger().warning(
                            "Skipping untrusted parser %s", file_path
                        )
                        continue
                    actual_hash = _file_sha256(file_path)
                    if actual_hash != expected_hash:
                        logging.getLogger().error(
                            "Hash mismatch for parser %s; expected %s but "
                            "got %s",
                            file_path,
                            expected_hash,
                            actual_hash,
                        )
                        continue
                    spec = importlib.util.spec_from_file_location(
                        module_name,
                        file_path,
                    )
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        try:
                            spec.loader.exec_module(module)
                        except Exception as e:
                            logging.getLogger().error(
                                "Failed loading parser module %s: %s",
                                file_path,
                                e,
                            )
                    else:
                        logging.getLogger().error(
                            "Missing loader for parser module %s", file_path
                        )
                    registry = registry_map[dtype]
                    key = None
                    if placeholder_key in registry:
                        key = placeholder_key
                    else:
                        for tmp_key, tmp_entry in registry.items():
                            if tmp_entry.get("data_dir") == src_dir:
                                key = tmp_key
                                break
                    if key is None:
                        continue
                    entry = registry.pop(key)
                    entry["description"] = meta.get(
                        "description",
                        entry.get("description", ""),
                    )
                    entry["data_dir"] = src_dir
                    entry["dataset_name"] = dataset_name
                    registry[dataset_id] = entry


# Discover parsers at import time so that functions like
# ``load_sne_data`` can simply refer to the registries without
# additional setup.
_discover_parsers()


# --- Helper to list and select parsers ---
def _select_source(parser_registry, data_type_name):
    """Display available data sources and return the chosen ``dataset_id``."""
    logger = logging.getLogger()
    if not parser_registry:
        logger.error(f"No parsers registered for {data_type_name} data.")
        return None

    logger.info(f"\nAvailable {data_type_name} data sources:")
    options = list(parser_registry.items())
    for i, (ds_id, entry) in enumerate(options):
        name = entry.get("dataset_name", ds_id)
        desc = entry.get("description", "")
        console.write(
            f"  {i+1}. {name} ({desc})" if desc else f"  {i+1}. {name}",
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
                return options[choice_idx][0]
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
def load_sne_data(dataset_id=None, **kwargs):
    """Load SNe data for the chosen ``dataset_id``."""
    logger = logging.getLogger()
    if dataset_id is None:
        dataset_id = _select_source(SNE_PARSERS, "SNe")
        if dataset_id is None:
            logger.info("SNe data loading canceled by user.")
            return None

    if dataset_id not in SNE_PARSERS:
        logger.error(f"No SNe parser registered for '{dataset_id}'")
        return None

    entry = SNE_PARSERS[dataset_id]
    parser_func = entry["function"]
    data_dir = entry["data_dir"]
    try:
        logger.info(
            f"Attempting to load SNe data from '{entry['dataset_name']}'",
        )
        data_df = parser_func(data_dir, **kwargs)
        if data_df is not None and not data_df.empty:
            meta = load_metadata_from_dir(data_dir)
            if meta:
                entry["dataset_name"] = meta.get(
                    "dataset_name", entry.get("dataset_name", "")
                )
                data_df.attrs.update(meta)
            data_df.attrs["dataset_name"] = entry["dataset_name"]
            data_df.attrs["dataset_id"] = check_dataset_id(dataset_id)
            logger.info(
                f"Successfully loaded {len(data_df)} SNe data points.",
            )
            _log_dataset_info(data_df, "SNe", logger)
        elif data_df is None:
            logger.error(
                f"SNe parser '{dataset_id}' returned None.",
            )
        else:
            logger.error(
                f"SNe parser '{dataset_id}' returned an empty DataFrame.",
            )
        return data_df
    except Exception as e:
        logger.critical(
            f"CRITICAL Error during SNe data parsing ({dataset_id}): {e}",
            exc_info=True,
        )
        return None


def load_bao_data(dataset_id=None, **kwargs):
    """Load BAO data for the chosen ``dataset_id``."""
    logger = logging.getLogger()
    if dataset_id is None:
        dataset_id = _select_source(BAO_PARSERS, "BAO")
        if dataset_id is None:
            logger.info("BAO data loading canceled by user.")
            return None

    if dataset_id not in BAO_PARSERS:
        logger.error(f"No BAO parser registered for '{dataset_id}'")
        return None

    entry = BAO_PARSERS[dataset_id]
    parser_func = entry["function"]
    data_dir = entry["data_dir"]
    try:
        logger.info(
            f"Attempting to load BAO data from '{entry['dataset_name']}'",
        )
        data_df = parser_func(data_dir, **kwargs)
        if data_df is not None and not data_df.empty:
            meta = load_metadata_from_dir(data_dir)
            if meta:
                entry["dataset_name"] = meta.get(
                    "dataset_name", entry.get("dataset_name", "")
                )
                data_df.attrs.update(meta)
            data_df.attrs["dataset_name"] = entry["dataset_name"]
            data_df.attrs["dataset_id"] = check_dataset_id(dataset_id)
            logger.info(
                f"Successfully loaded {len(data_df)} BAO data points.",
            )
            _log_dataset_info(data_df, "BAO", logger)
        elif data_df is None:
            logger.error(
                f"BAO parser '{dataset_id}' returned None.",
            )
        else:
            logger.error(
                f"BAO parser '{dataset_id}' returned an empty DataFrame.",
            )
        return data_df
    except Exception as e:
        logger.critical(
            f"CRITICAL Error during BAO data parsing ({dataset_id}): {e}",
            exc_info=True,
        )
        return None


def load_cmb_data(dataset_id=None, **kwargs):
    """Load CMB data for the chosen ``dataset_id``."""
    logger = logging.getLogger()
    if dataset_id is None:
        dataset_id = _select_source(CMB_PARSERS, "CMB")
        if dataset_id is None:
            logger.info("CMB data loading canceled by user.")
            return None

    if dataset_id not in CMB_PARSERS:
        logger.error(f"No CMB parser registered for '{dataset_id}'")
        return None

    entry = CMB_PARSERS[dataset_id]
    parser_func = entry["function"]
    data_dir = entry["data_dir"]
    try:
        logger.info(
            f"Attempting to load CMB data from '{entry['dataset_name']}'",
        )
        data_df = parser_func(data_dir, **kwargs)
        if data_df is not None and not data_df.empty:
            meta = load_metadata_from_dir(data_dir)
            if meta:
                entry["dataset_name"] = meta.get(
                    "dataset_name", entry.get("dataset_name", "")
                )
                data_df.attrs.update(meta)
            data_df.attrs["dataset_name"] = entry["dataset_name"]
            data_df.attrs["dataset_id"] = check_dataset_id(dataset_id)
            logger.info(
                f"Successfully loaded {len(data_df)} CMB data points.",
            )
            _log_dataset_info(data_df, "CMB", logger)
        elif data_df is None:
            logger.error(
                f"CMB parser '{dataset_id}' returned None.",
            )
        else:
            logger.error(
                f"CMB parser '{dataset_id}' returned an empty DataFrame.",
            )
        return data_df
    except Exception as e:
        logger.critical(
            f"CRITICAL Error during CMB data parsing ({dataset_id}): {e}",
            exc_info=True,
        )
        return None


def load_gw_data(dataset_id=None, **kwargs):
    """Load gravitational wave data for the chosen ``dataset_id``."""
    logger = logging.getLogger()
    if dataset_id is None:
        dataset_id = _select_source(GW_PARSERS, "GW")
        if dataset_id is None:
            logger.info("Gravitational wave data loading canceled by user.")
            return None

    if dataset_id not in GW_PARSERS:
        msg = f"No gravitational wave parser registered for '{dataset_id}'"
        logger.error(msg)
        return None

    entry = GW_PARSERS[dataset_id]
    parser_func = entry["function"]
    data_dir = entry["data_dir"]
    try:
        logger.info(
            f"Attempting to load GW data from '{entry['dataset_name']}'",
        )
        data_df = parser_func(data_dir, **kwargs)
        if data_df is not None and not data_df.empty:
            meta = load_metadata_from_dir(data_dir)
            if meta:
                entry["dataset_name"] = meta.get(
                    "dataset_name", entry.get("dataset_name", "")
                )
                data_df.attrs.update(meta)
            data_df.attrs["dataset_name"] = entry["dataset_name"]
            data_df.attrs["dataset_id"] = check_dataset_id(dataset_id)
            logger.info(
                f"Successfully loaded {len(data_df)} GW data points.",
            )
            _log_dataset_info(data_df, "GW", logger)
        elif data_df is None:
            logger.error(
                f"GW parser '{dataset_id}' returned None.",
            )
        else:
            logger.error(
                f"GW parser '{dataset_id}' returned an empty DataFrame.",
            )
        return data_df
    except Exception as e:
        logger.critical(
            f"CRITICAL Error during GW data parsing ({dataset_id}): {e}",
            exc_info=True,
        )
        return None


def load_siren_data(dataset_id=None, **kwargs):
    """Load standard siren data for the chosen ``dataset_id``."""
    logger = logging.getLogger()
    if dataset_id is None:
        dataset_id = _select_source(SIREN_PARSERS, "standard siren")
        if dataset_id is None:
            logger.info("Standard siren data loading canceled by user.")
            return None

    if dataset_id not in SIREN_PARSERS:
        logger.error(
            f"No standard siren parser registered for '{dataset_id}'",
        )
        return None

    entry = SIREN_PARSERS[dataset_id]
    parser_func = entry["function"]
    data_dir = entry["data_dir"]
    try:
        logger.info(
            f"Attempting to load siren data from '{entry['dataset_name']}'",
        )
        data_df = parser_func(data_dir, **kwargs)
        if data_df is not None and not data_df.empty:
            meta = load_metadata_from_dir(data_dir)
            if meta:
                entry["dataset_name"] = meta.get(
                    "dataset_name", entry.get("dataset_name", "")
                )
                data_df.attrs.update(meta)
            data_df.attrs["dataset_name"] = entry["dataset_name"]
            data_df.attrs["dataset_id"] = check_dataset_id(dataset_id)
            msg = (
                "Successfully loaded "
                f"{len(data_df)} standard siren data points."
            )
            logger.info(msg)
            _log_dataset_info(data_df, "SIREN", logger)
        elif data_df is None:
            logger.error(
                f"Standard siren parser '{dataset_id}' returned None.",
            )
        else:
            logger.error(
                f"Standard siren parser '{dataset_id}' returned an empty "
                f"DataFrame.",
            )
        return data_df
    except Exception as e:
        err_msg = (
            "CRITICAL Error during standard siren data parsing "
            f"({dataset_id}): {e}"
        )
        logger.critical(err_msg, exc_info=True)
        return None
