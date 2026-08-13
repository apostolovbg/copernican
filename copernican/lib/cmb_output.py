"""Canonical CMB output names, metadata, and observation-block handling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy

_COMPONENT_PREFIXES = frozenset({"scalar", "tensor", "total", "vector"})
_SPECTRUM_ALIASES = {
    "EPHI": "EP",
    "PHIPHI": "PP",
    "TPHI": "TP",
}
_LENSING_BASE_SPECTRA = frozenset({"EP", "PP", "TP"})
_TEMPERATURE_BASE_SPECTRA = frozenset({"BB", "EE", "TE", "TT"})


@dataclass(frozen=True, slots=True)
class CMBSpectrumMetadata:
    """Describe one named public CMB output without collapsing its surface."""

    requested_name: str
    canonical_name: str
    base_spectrum: str
    component: str | None
    sector: str | None
    observable_family: str
    lensing_state: str
    units: str

    def as_mapping(self) -> dict[str, str | None]:
        """Return a manifest-safe representation of the output metadata."""

        return {
            "requested_name": self.requested_name,
            "canonical_name": self.canonical_name,
            "base_spectrum": self.base_spectrum,
            "component": self.component,
            "sector": self.sector,
            "observable_family": self.observable_family,
            "lensing_state": self.lensing_state,
            "units": self.units,
        }


@dataclass(frozen=True, slots=True)
class CMBObservationBlock:
    """Keep one observed spectrum and its original table rows together."""

    metadata: CMBSpectrumMetadata
    row_indices: numpy.ndarray
    ells: numpy.ndarray
    observed: numpy.ndarray
    observed_column: str

    def __post_init__(self) -> None:
        """Freeze copied observation arrays used by likelihood instances."""

        for field_name in ("row_indices", "ells", "observed"):
            values = numpy.asarray(getattr(self, field_name))
            values.setflags(write=False)
            object.__setattr__(self, field_name, values)


def _canonical_theory_mapping(
    theory: Mapping[str, Any],
) -> dict[str, numpy.ndarray]:
    """Return a canonical theory map while rejecting ambiguous aliases."""

    theory_map: dict[str, numpy.ndarray] = {}
    for name, values in theory.items():
        canonical_name = canonical_cmb_spectrum_name(name)
        if canonical_name in theory_map:
            raise ValueError(
                "CMB theory contains duplicate canonical spectrum "
                f"'{canonical_name}'"
            )
        theory_map[canonical_name] = numpy.asarray(values, dtype=float)
    return theory_map


def split_cmb_spectrum_name(
    spectrum_name: str,
) -> tuple[bool, str | None, str]:
    """Return ``(lensed, component, base_name)`` for one output name."""

    name = str(spectrum_name).strip()
    if not name:
        raise ValueError("CMB spectrum names must not be empty")
    lensed = name.casefold().startswith("lensed_")
    if lensed:
        name = name.split("_", 1)[1]
    component = None
    lower_name = name.casefold()
    for candidate in sorted(_COMPONENT_PREFIXES):
        prefix = f"{candidate}_"
        if lower_name.startswith(prefix):
            component = candidate
            name = name[len(prefix) :]
            break
    base_name = _SPECTRUM_ALIASES.get(name.upper(), name.upper())
    return lensed, component, base_name


def compose_cmb_spectrum_name(
    *,
    lensed: bool,
    component: str | None,
    base_name: str,
) -> str:
    """Return the canonical name for parsed CMB spectrum metadata."""

    name = _SPECTRUM_ALIASES.get(
        str(base_name).upper(), str(base_name).upper()
    )
    if component is not None:
        normalized_component = str(component).casefold()
        if normalized_component not in _COMPONENT_PREFIXES:
            raise ValueError(f"Unknown CMB spectrum component '{component}'")
        name = f"{normalized_component}_{name}"
    return f"lensed_{name}" if lensed else name


def canonical_cmb_spectrum_name(spectrum_name: str) -> str:
    """Return one stable public output name for ``spectrum_name``."""

    lensed, component, base_name = split_cmb_spectrum_name(spectrum_name)
    return compose_cmb_spectrum_name(
        lensed=lensed,
        component=component,
        base_name=base_name,
    )


def describe_cmb_spectrum(spectrum_name: str) -> CMBSpectrumMetadata:
    """Return orthogonal physical and plotting metadata for one spectrum."""

    requested_name = str(spectrum_name)
    canonical_name = canonical_cmb_spectrum_name(requested_name)
    lensed, component, base_name = split_cmb_spectrum_name(canonical_name)
    if canonical_name.casefold().startswith("diagnostic_"):
        observable_family = "diagnostic"
    elif base_name in _LENSING_BASE_SPECTRA:
        observable_family = "lensing"
    else:
        observable_family = "cmb"
    if observable_family == "cmb":
        lensing_state = "lensed" if lensed else "unlensed"
    else:
        lensing_state = "not_applicable"
    sector = component if component in {"scalar", "tensor", "vector"} else None
    if base_name in _TEMPERATURE_BASE_SPECTRA:
        units = "muK^2"
    elif base_name in _LENSING_BASE_SPECTRA:
        units = "dimensionless"
    else:
        units = "declared"
    return CMBSpectrumMetadata(
        requested_name=requested_name,
        canonical_name=canonical_name,
        base_spectrum=base_name,
        component=component,
        sector=sector,
        observable_family=observable_family,
        lensing_state=lensing_state,
        units=units,
    )


def cmb_observation_blocks(cmb_data: Any) -> tuple[CMBObservationBlock, ...]:
    """Return exact spectrum blocks without reordering observation rows."""

    if cmb_data is None or getattr(cmb_data, "empty", True):
        return ()
    row_count = int(len(cmb_data))
    if "spectrum" in cmb_data.columns:
        labels = numpy.asarray(
            [
                canonical_cmb_spectrum_name(value)
                for value in cmb_data["spectrum"].astype(str)
            ],
            dtype=object,
        )
        names = tuple(dict.fromkeys(str(value) for value in labels))
        blocks = []
        for name in names:
            row_indices = numpy.flatnonzero(labels == name)
            blocks.append(
                CMBObservationBlock(
                    metadata=describe_cmb_spectrum(name),
                    row_indices=row_indices,
                    ells=cmb_data.iloc[row_indices]["ell"].to_numpy(
                        dtype=int,
                        copy=True,
                    ),
                    observed=cmb_data.iloc[row_indices]["Dl_obs"].to_numpy(
                        dtype=float,
                        copy=True,
                    ),
                    observed_column="Dl_obs",
                )
            )
        return tuple(blocks)

    column_by_name: dict[str, str] = {}
    if "Dl_obs" in cmb_data.columns:
        column_by_name["TT"] = "Dl_obs"
    for column_name in cmb_data.columns:
        name = str(column_name)
        if not (name.startswith("Dl_") and name.endswith("_obs")):
            continue
        token = name[3:-4]
        if not token:
            continue
        canonical_name = canonical_cmb_spectrum_name(token)
        column_by_name.setdefault(canonical_name, name)
    all_rows = numpy.arange(row_count, dtype=int)
    return tuple(
        CMBObservationBlock(
            metadata=describe_cmb_spectrum(name),
            row_indices=all_rows.copy(),
            ells=cmb_data["ell"].to_numpy(dtype=int, copy=True),
            observed=cmb_data[column_name].to_numpy(dtype=float, copy=True),
            observed_column=column_name,
        )
        for name, column_name in column_by_name.items()
    )


def observed_cmb_spectrum_names(cmb_data: Any) -> tuple[str, ...]:
    """Return exact canonical spectrum names in table declaration order."""

    return tuple(
        block.metadata.canonical_name
        for block in cmb_observation_blocks(cmb_data)
    )


def cmb_theory_values_for_block(
    theory: Mapping[str, Any] | numpy.ndarray,
    block: CMBObservationBlock,
    *,
    total_row_count: int,
) -> numpy.ndarray:
    """Select theory values for one block on full or compact ell surfaces."""

    if isinstance(theory, Mapping):
        theory_map = _canonical_theory_mapping(theory)
    else:
        theory_map = None
    return _theory_values_for_block(
        theory_map if theory_map is not None else theory,
        block,
        total_row_count=total_row_count,
    )


def _theory_values_for_block(
    theory: Mapping[str, Any] | numpy.ndarray,
    block: CMBObservationBlock,
    *,
    total_row_count: int,
) -> numpy.ndarray:
    """Select values from a canonical mapping or a plain TT array."""

    if isinstance(theory, Mapping):
        values = theory.get(block.metadata.canonical_name)
        if values is None:
            raise KeyError(block.metadata.canonical_name)
        values = numpy.asarray(values, dtype=float)
    else:
        if block.metadata.canonical_name != "TT":
            raise KeyError(block.metadata.canonical_name)
        values = numpy.asarray(theory, dtype=float)
    if values.ndim != 1:
        raise ValueError("CMB theory spectra must be one-dimensional")
    if values.size == int(total_row_count):
        return numpy.asarray(values[block.row_indices], dtype=float)
    if values.size == block.row_indices.size:
        return numpy.asarray(values, dtype=float)
    raise ValueError(
        f"CMB theory spectrum '{block.metadata.canonical_name}' has "
        f"{values.size} values for {block.row_indices.size} observations"
    )


def assemble_cmb_theory_vector(
    theory: Mapping[str, Any] | numpy.ndarray,
    blocks: tuple[CMBObservationBlock, ...],
    *,
    total_row_count: int,
) -> numpy.ndarray:
    """Return a row-aligned theory vector for long-form CMB observations."""

    result = numpy.full(int(total_row_count), numpy.nan, dtype=float)
    normalized_theory = (
        _canonical_theory_mapping(theory)
        if isinstance(theory, Mapping)
        else theory
    )
    for block in blocks:
        result[block.row_indices] = _theory_values_for_block(
            normalized_theory,
            block,
            total_row_count=total_row_count,
        )
    if numpy.any(~numpy.isfinite(result)):
        raise ValueError("CMB theory does not cover every observation row")
    return result


__all__ = [
    "CMBObservationBlock",
    "CMBSpectrumMetadata",
    "assemble_cmb_theory_vector",
    "canonical_cmb_spectrum_name",
    "cmb_observation_blocks",
    "cmb_theory_values_for_block",
    "compose_cmb_spectrum_name",
    "describe_cmb_spectrum",
    "observed_cmb_spectrum_names",
    "split_cmb_spectrum_name",
]
