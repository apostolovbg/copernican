# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.

"""Dataset registry and loader helpers.

The suite treats observational data as pluggable components. Parsers live
alongside their raw tables under ``data/<type>/<source>/`` and register
themselves through decorators provided here. At runtime ``copernican.py``
imports these modules to populate interactive menus and returns uniformly
formatted :class:`pandas.DataFrame` objects with metadata stored on
``.attrs``. The registry terminology below now emphasises the dictionary nature
of each lookup table so callers no longer confuse the registries with
single-function entry points. Each observable category attaches a uniform set
of attributes including reproducibility hashes, dataset versions and explicit
statistical independence statements. The additional metadata is consumed by the
run manifest builder and keeps the suite honest about likelihood assumptions.
"""
import hashlib
import importlib
import logging
import os
from collections.abc import Callable
from typing import Any

import numpy as np

from . import console_output as console
from .utils import check_dataset_id, compute_sha256, load_metadata_from_dir

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
SNE_PARSER_REGISTRY: dict = {}
BAO_PARSER_REGISTRY: dict = {}
CMB_PARSER_REGISTRY: dict = {}
GW_PARSER_REGISTRY: dict = {}

def get_parser_registries() -> dict[str, dict]:
    """Return the live mapping of dataset types to parser registries."""

    return {
        "sne": SNE_PARSER_REGISTRY,
        "bao": BAO_PARSER_REGISTRY,
        "cmb": CMB_PARSER_REGISTRY,
        "gw": GW_PARSER_REGISTRY,
    }

def get_parser_registry(dataset_key: str) -> dict:
    """Return the registry associated with ``dataset_key``.

    The helper resolves the current registry objects each time so test suites
    that replace the module-level dictionaries receive the updated mappings.
    """

    registries = get_parser_registries()
    if dataset_key not in registries:
        raise KeyError(f"Unknown dataset registry '{dataset_key}'")
    return registries[dataset_key]

# The core cosmology pipelines treat the SNe, BAO and CMB likelihoods as
# statistically independent.  Centralising the statements that justify that
# assumption makes it easier to audit and copy the reasoning into manifests and
# documentation.  Each list is intentionally single-element today so future
# work can append additional caveats without altering the consumer code.
OBSERVATION_INDEPENDENCE_NOTES: dict[str, list[str]] = {
    "sne": [
        (
            "Type Ia supernova distance moduli are treated as statistically "
            "independent from BAO and CMB observables once their published "
            "covariance matrices are accounted for."
        )
    ],
    "bao": [
        (
            "BAO distance measurements are assumed independent of SNe and CMB "
            "datasets because overlapping systematics are negligible at the "
            "current precision level."
        )
    ],
    "cmb": [
        (
            "Planck-lite CMB spectra are modelled as independent from SNe and "
            "BAO summaries; cross-covariances are ignored consistently with "
            "published likelihood treatments."
        )
    ],
}

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
        SNE_PARSER_REGISTRY[key] = {
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
        BAO_PARSER_REGISTRY[key] = {
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
        CMB_PARSER_REGISTRY[key] = {
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
        GW_PARSER_REGISTRY[key] = {
            "function": func,
            "dataset_name": name or key,
            "description": description,
            "data_dir": data_dir,
        }
        return func

    return decorator

TRUSTED_PARSER_DIGESTS = {
    # ``relative_path`` -> ``sha256``
    "sne/pantheon/cosmo_parser_pantheon.py": (
        "39caafd7adc483d9a6d0f039fc5a08645b3a92435d96d4d3c5e858c7992a8841"
    ),
    "sne/jla2014/cosmo_parser_jla2014.py": (
        "56a38cd0fa182f291bc08fa7dbbe9aca019f1b9c68ef546f60d5af7ebefd1c46"
    ),
    "bao/bossdr12/cosmo_parser_bossdr12.py": (
        "2780617aa5f84650a6a1e7d1e79a8ab1a420d95bf062c1b50838227bedc83f74"
    ),
    "bao/compound/cosmo_parser_compound.py": (
        "21d0810907e0d18a488c9583097e100ba7c948eb19f4374fc9ad61e1e2a26a7f"
    ),
    "cmb/planck2018lite/cosmo_parser_cmb_planck2018lite.py": (
        "04620b53c3a8d24565eafd7c36ff0c6624bbb97f3f47ccea6e34baf736da6f8c"
    ),
    "gw/placeholder/cosmo_parser_gw_placeholder.py": (
        "0af702546dcc5fac872fa7b68892176ec2400789b18f22e1dce0759093c3ef08"
    ),
    "sne/union3/cosmo_parser_union3.py": (
        "6e9fd184fd5e7b7871a7a8f9ab9e9ab50485d42d40f08608c23c3f297abd7e91"
    ),
}

def _file_sha256(path: str) -> str:
    """Return the SHA256 digest for ``path`` with newline normalisation.

    Git may convert ``\n`` to ``\r\n`` on Windows checkouts.  Normalising
    line endings ensures the same hash across operating systems so trusted
    parser verification behaves consistently on all platforms.
    """

    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            hasher.update(chunk.replace(b"\r\n", b"\n"))
    return hasher.hexdigest()

# --- Dynamic Discovery of Parser Modules ---
def discover_trusted_parsers(base_dir: str | None = None):
    """Import parser modules and populate registries with dataset metadata.

    The scan walks ``data/`` recursively, ignoring ``placeholder`` folders so
    unfinished datasets stay hidden.  Candidate paths are resolved with
    ``os.path.realpath`` and rejected if they are symlinks or if the resolved
    location escapes ``base_dir``.  Each parser is then verified against
    ``TRUSTED_PARSER_DIGESTS`` before import to guard against tampering.  Only
    trusted modules are executed, keeping the discovery step resilient to
    untrusted files shipped alongside the data tables.
    Metadata is read here to keep the parser implementations small and focused
    solely on table parsing.
    """
    if base_dir is None:
        base_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data"
        )
    # Resolve the discovery root to an absolute path so subsequent checks can
    # verify that candidate entries never escape the repository via symlinks
    # or ".." components.
    base_dir = os.path.realpath(base_dir)
    for dtype in ("sne", "bao", "cmb", "gw"):
        type_dir = os.path.join(base_dir, dtype)
        # Skip symlinks or paths that resolve outside the data directory.
        if os.path.islink(type_dir):
            continue
        type_dir = os.path.realpath(type_dir)
        if os.path.commonpath([base_dir, type_dir]) != base_dir:
            continue
        if not os.path.isdir(type_dir):
            continue
        for source in os.listdir(type_dir):
            # Skip placeholder folders so unfinished datasets do not appear in
            # the interactive menus.
            if source.lower() == "placeholder":
                continue
            src_dir = os.path.join(type_dir, source)
            if os.path.islink(src_dir):
                continue
            src_dir = os.path.realpath(src_dir)
            if os.path.commonpath([base_dir, src_dir]) != base_dir:
                continue
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
                    if os.path.islink(file_path):
                        continue
                    file_path = os.path.realpath(file_path)
                    if os.path.commonpath([base_dir, file_path]) != base_dir:
                        continue
                    rel_path = os.path.relpath(file_path, base_dir)
                    # Normalise path separators for cross-platform hash lookup.
                    rel_path = rel_path.replace("\\", "/")
                    expected_hash = TRUSTED_PARSER_DIGESTS.get(rel_path)
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
                    registry = get_parser_registry(dtype)
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
discover_trusted_parsers()

# Bundle the registries and logging messages for observable categories whose
# loaders now share the same control flow.  The shared structure keeps the
# public ``load_*`` helpers concise while preserving informative log output.
DATASET_CONFIG: dict[str, dict[str, Any]] = {
    "sne": {
        "label": "SNe",
        "cancel_message": "SNe data loading canceled by user.",
    },
    "bao": {
        "label": "BAO",
        "cancel_message": "BAO data loading canceled by user.",
    },
    "cmb": {
        "label": "CMB",
        "cancel_message": "CMB data loading canceled by user.",
    },
}

# --- Helper to list and select parsers ---
def prompt_dataset_selection(parser_registry, data_type_name):
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
    cond_number = df.attrs.get("covariance_condition_number")
    if cond_number is not None:
        logger.info(
            "%s covariance condition number: %.3e",
            data_type,
            cond_number,
        )

def _validate_bao_covariance(df, logger):
    """Ensure BAO covariance matrices are symmetric and positive definite."""

    inv_cov = df.attrs.get("covariance_matrix_inv")
    if inv_cov is None:
        logger.warning(
            "BAO dataset is missing an inverse covariance matrix; "
            "falling back to diagonal errors."
        )
        return False
    inv_arr = np.asarray(inv_cov, dtype=float)
    try:
        if inv_arr.ndim != 2 or inv_arr.shape[0] != inv_arr.shape[1]:
            raise ValueError("matrix must be square")
        if not np.allclose(inv_arr, inv_arr.T, atol=1e-10):
            raise ValueError("matrix is not symmetric")
        eigenvalues = np.linalg.eigvalsh(inv_arr)
        if np.any(eigenvalues <= 0.0):
            raise ValueError("matrix must be positive definite")
    except ValueError as exc:
        logger.warning(
            (
                "Invalid BAO covariance inverse (%s); falling back to "
                "diagonal errors."
            ),
            exc,
        )
        return False
    cond_number = float(np.linalg.cond(inv_arr))
    df.attrs["covariance_condition_number"] = cond_number
    logger.info("BAO covariance condition number: %.3e", cond_number)
    return True

def _attach_file_hashes(df, data_dir, logger):
    """Compute SHA256 hashes for files in ``data_dir`` and log them.

    The resulting mapping is stored on ``df.attrs['file_hashes']`` so later
    stages such as the run manifest can embed the exact input digests.  Each
    hash is logged for audit purposes to aid reproducibility.
    """

    file_hashes = {}
    for root, _, files in os.walk(data_dir):
        for fname in sorted(files):
            if fname.endswith(".py"):
                continue
            path = os.path.join(root, fname)
            rel = os.path.relpath(path, data_dir)
            file_hashes[rel] = compute_sha256(path)
    df.attrs["file_hashes"] = file_hashes
    for rel, digest in file_hashes.items():
        logger.info("SHA256 %s: %s", rel, digest)

def _load_dataset(
    dataset_key: str,
    dataset_id: str | None = None,
    **kwargs: Any,
):
    """Return the dataset ``dataset_key`` refers to using the shared loader.

    The helper consolidates all logic shared by the SNe, BAO and CMB loaders.
    Keeping the implementation in a single place guarantees that metadata,
    reproducibility hashes and independence statements remain perfectly
    aligned across likelihood components.  ``dataset_id`` mirrors the human
    selection when the caller overrides the interactive prompt.
    """

    logger = logging.getLogger()
    config = DATASET_CONFIG[dataset_key]
    label = config["label"]
    registry = get_parser_registry(dataset_key)
    cancel_message = config["cancel_message"]

    if dataset_id is None:
        dataset_id = prompt_dataset_selection(registry, label)
        if dataset_id is None:
            logger.info(cancel_message)
            return None

    if dataset_id not in registry:
        logger.error(
            f"No {label} parser registered for '{dataset_id}'",
        )
        return None

    entry = registry[dataset_id]
    parser_func: Callable[..., Any] = entry["function"]
    data_dir = entry["data_dir"]
    try:
        dataset_name = entry.get("dataset_name", dataset_id)
        logger.info(
            "Attempting to load %s data from '%s'",
            label,
            dataset_name,
        )
        data_df = parser_func(data_dir, **kwargs)
        if data_df is not None and not data_df.empty:
            meta = {}
            if data_dir:
                meta = load_metadata_from_dir(data_dir) or {}
            if meta:
                entry["dataset_name"] = meta.get(
                    "dataset_name",
                    entry.get("dataset_name", dataset_id),
                )
                if "version" in meta:
                    entry["dataset_version"] = meta["version"]
                data_df.attrs.update(meta)
            dataset_version = entry.get("dataset_version") or meta.get(
                "version",
                "unknown",
            )
            data_df.attrs["dataset_name"] = entry["dataset_name"]
            data_df.attrs["dataset_id"] = check_dataset_id(dataset_id)
            data_df.attrs["dataset_version"] = dataset_version
            data_df.attrs["data_path"] = data_dir
            data_df.attrs["independence_assumptions"] = list(
                OBSERVATION_INDEPENDENCE_NOTES.get(dataset_key, [])
            )
            if dataset_key == "bao":
                has_cov = _validate_bao_covariance(data_df, logger)
                if not has_cov:
                    data_df.attrs.pop("covariance_matrix_inv", None)
            logger.info(
                "Successfully loaded %s %s data points.",
                len(data_df),
                label,
            )
            if data_dir:
                _attach_file_hashes(data_df, data_dir, logger)
            _log_dataset_info(data_df, label, logger)
        elif data_df is None:
            logger.error(
                f"{label} parser '{dataset_id}' returned None.",
            )
        else:
            logger.error(
                f"{label} parser '{dataset_id}' returned an empty DataFrame.",
            )
        return data_df
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.critical(
            "CRITICAL Error during %s data parsing (%s): %s",
            label,
            dataset_id,
            exc,
            exc_info=True,
        )
        return None

# --- Main Loading Functions ---
def load_sne_data(dataset_id=None, **kwargs):
    """Load SNe data for the chosen ``dataset_id``."""

    return _load_dataset("sne", dataset_id=dataset_id, **kwargs)

def load_bao_data(dataset_id=None, **kwargs):
    """Load BAO data for the chosen ``dataset_id``."""

    return _load_dataset("bao", dataset_id=dataset_id, **kwargs)

def load_cmb_data(dataset_id=None, **kwargs):
    """Load CMB data for the chosen ``dataset_id``."""

    return _load_dataset("cmb", dataset_id=dataset_id, **kwargs)

def load_gw_data(dataset_id=None, **kwargs):
    """Load gravitational-wave standard siren data for ``dataset_id``."""
    logger = logging.getLogger()
    registry = get_parser_registry("gw")
    if dataset_id is None:
        dataset_id = prompt_dataset_selection(registry, "gravitational-wave")
        if dataset_id is None:
            logger.info(
                "Gravitational-wave data loading canceled by user during "
                "placeholder management."
            )
            return None

    if dataset_id not in registry:
        msg = (
            "No gravitational-wave standard siren parser registered for "
            f"'{dataset_id}'"
        )
        logger.error(msg)
        return None

    entry = registry[dataset_id]
    parser_func = entry["function"]
    data_dir = entry["data_dir"]
    try:
        logger.info(
            "Attempting to load gravitational-wave standard siren data from "
            f"'{entry['dataset_name']}'",
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
                "Successfully loaded %s gravitational-wave standard siren "
                "data points.",
                len(data_df),
            )
            _attach_file_hashes(data_df, data_dir, logger)
            _log_dataset_info(
                data_df, "Gravitational-wave standard siren", logger
            )
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
        err_msg = (
            "CRITICAL Error during gravitational-wave standard siren data "
            f"parsing ({dataset_id}): {e}"
        )
        logger.critical(err_msg, exc_info=True)
        return None
