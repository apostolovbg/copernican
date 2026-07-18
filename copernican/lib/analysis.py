# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.

"""Run-analysis helpers that summarise Copernican outputs."""

from __future__ import annotations

import dataclasses
import datetime
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import yaml

from . import plotter, posterior_explorer
from . import run_manifest as run_manifest_module
from . import utils
from .model_selection import (
    ComparisonRequest,
    comparison_from_manifest,
    comparison_slug,
)

_GREEK_REPLACEMENTS = {
    "Λ": "Lambda",
    "λ": "lambda",
    "Ω": "Omega",
    "ω": "omega",
    "Σ": "Sigma",
    "σ": "sigma",
    "Δ": "Delta",
    "δ": "delta",
    "Θ": "Theta",
    "θ": "theta",
    "Φ": "Phi",
    "φ": "phi",
    "Ψ": "Psi",
    "ψ": "psi",
    "Γ": "Gamma",
    "γ": "gamma",
    "Π": "Pi",
    "π": "pi",
}


def _normalize_model_label(label: str) -> str:
    """Convert a model label to a normalized ASCII key for matching."""

    normalized = unicodedata.normalize("NFKD", label)
    for greek, latin in _GREEK_REPLACEMENTS.items():
        normalized = normalized.replace(greek, latin)
    ascii_only = "".join(
        character for character in normalized if ord(character) < 128
    )
    return ascii_only.casefold()


def _parse_float(raw_value: str) -> Optional[float]:
    """Parse ``raw_value`` as float while tolerating unicode minus signs."""

    cleaned = raw_value.replace("−", "-").replace("\u2212", "-").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def _sanitize_metric_name(metric: str) -> str:
    """Transform metric labels such as 'χ²_Total' into ASCII-friendly keys."""

    sanitized = metric.replace("χ²", "chi2").replace("Χ²", "chi2")
    sanitized = sanitized.strip().replace(" ", "_")
    return sanitized.lower()


def _find_latest_file(run_dir: Path, pattern: str) -> Optional[Path]:
    """Return the newest file under `run_dir` that matches `pattern`."""
    matches = sorted(run_dir.glob(pattern))
    if not matches:
        return None
    return matches[-1]


def _load_yaml_or_json(path: Path) -> Any:
    """Read either a YAML or JSON document from `path`."""
    if path.suffix.lower() in {".yml", ".yaml"}:
        with open(path, "r", encoding="utf-8") as file_handle:
            return yaml.safe_load(file_handle)
    with open(path, "r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def load_parameter_summary(
    run_dir: Path,
) -> tuple[Optional[Mapping[str, Any]], Optional[Path]]:
    """Return the latest parameter-summary file and its contents."""

    run_dir = Path(run_dir)
    for suffix in ("yml", "yaml"):
        candidate = _find_latest_file(run_dir, f"parameter-summary_*.{suffix}")
        if candidate:
            return _load_yaml_or_json(candidate), candidate
    candidate = _find_latest_file(run_dir, "parameter-summary_*.json")
    if candidate:
        return _load_yaml_or_json(candidate), candidate
    return None, None


def load_manifest(
    run_dir: Path,
) -> tuple[Optional[Mapping[str, Any]], Optional[Path]]:
    """Return the most recent manifest under ``run_dir``."""

    run_dir = Path(run_dir)
    manifest_file = _find_latest_file(run_dir, "run_manifest_*.yml")
    if manifest_file:
        manifest = run_manifest_module.load_manifest(str(manifest_file))
        return manifest, manifest_file
    return None, None


_LOG_PATTERNS = ("copernican-run_*.txt", "validation_run_*.txt")
_FIT_REPORT_RE = re.compile(r"^--- (?P<label>.+?) Fit Report ---$")
_DATASET_INFO_RE = re.compile(
    r"^Dataset (?P<id>[^ ]+) \((?P<type>[^)]+)\): (?P<name>.+)$"
)
_DATASET_LOADED_RE = re.compile(
    r"^Loaded dataset (?P<id>[^:]+): (?P<count>\d+) entries$"
)
_RHAT_RE = re.compile(
    r"^Rank-normalised R-hat summary: "
    r"min=(?P<min>[\d.+-eE]+) "
    r"median=(?P<median>[\d.+-eE]+) "
    r"max=(?P<max>[\d.+-eE]+)$"
)
_ESS_RE = re.compile(
    r"^Effective sample sizes: "
    r"bulk median=(?P<bulk>[\d.+-eE]+) "
    r"tail median=(?P<tail>[\d.+-eE]+)$"
)
_ACCEPTANCE_RE = re.compile(
    r"^MCMC acceptance for (?P<label>.+): "
    r"mean=(?P<mean>[\d.+-eE]+), "
    r"min=(?P<min>[\d.+-eE]+), "
    r"max=(?P<max>[\d.+-eE]+)$"
)
_BAO_RS_RE = re.compile(
    r"^(?P<label>.+) BAO: r_s = (?P<rs>[\d.+-eE]+) "
    r"Mpc, [^=]+= (?P<chi2>[\d.+-eE]+)$"
)


def _split_log_line(
    line: str,
) -> tuple[Optional[datetime.datetime], Optional[str]]:
    """Split a log line into its timestamp and message components."""
    parts = line.split(" - ", 2)
    if len(parts) < 3:
        return None, line.strip()
    timestamp = _parse_timestamp(parts[0])
    return timestamp, parts[2].strip()


def _parse_timestamp(timestamp: str) -> Optional[datetime.datetime]:
    """Parse a log timestamp string into a `datetime`, or return `None`."""
    try:
        return datetime.datetime.strptime(
            timestamp.strip(), "%Y-%m-%d %H:%M:%S,%f"
        )
    except ValueError:
        return None


def _find_log_file(run_dir: Path) -> Optional[Path]:
    """Locate the most recent run log inside `run_dir`."""
    run_dir = Path(run_dir)
    for pattern in _LOG_PATTERNS:
        candidate = _find_latest_file(run_dir, pattern)
        if candidate:
            return candidate
    txt_candidates = sorted(run_dir.glob("*.txt"))
    if txt_candidates:
        return txt_candidates[-1]
    return None


def parse_log(log_path: Path) -> Mapping[str, Any]:
    """
    Parse the Copernican run log for diagnostics and chi-squared summaries.
    """

    datasets: dict[str, dict[str, str]] = {}
    counts: dict[str, int] = {}
    log_models: dict[str, dict[str, Any]] = {}
    start_time: Optional[datetime.datetime] = None
    end_time: Optional[datetime.datetime] = None
    duration: Optional[float] = None
    acceptance: dict[str, dict[str, float]] = {}
    rhat: dict[str, float] = {}
    ess: dict[str, float] = {}
    last_ts: Optional[datetime.datetime] = None

    with open(log_path, "r", encoding="utf-8") as file_handle:
        previous_message: Optional[str] = None
        current_model_key: Optional[str] = None
        for raw in file_handle:
            stripped = raw.rstrip("\n")
            timestamp, message = _split_log_line(stripped)
            if message is None:
                continue
            if message == previous_message:
                continue
            previous_message = message
            if timestamp:
                last_ts = timestamp
                if start_time is None:
                    start_time = timestamp
            if message == "Evaluation complete.":
                end_time = timestamp
            info_match = _DATASET_INFO_RE.match(message)
            if info_match:
                datasets[info_match.group("id")] = {
                    "name": info_match.group("name"),
                    "type": info_match.group("type"),
                }
            loaded_match = _DATASET_LOADED_RE.match(message)
            if loaded_match:
                counts[loaded_match.group("id")] = int(
                    loaded_match.group("count")
                )
            rhat_match = _RHAT_RE.match(message)
            if rhat_match:
                rhat = {
                    "min": _parse_float(rhat_match.group("min")),
                    "median": _parse_float(rhat_match.group("median")),
                    "max": _parse_float(rhat_match.group("max")),
                }
            ess_match = _ESS_RE.match(message)
            if ess_match:
                ess = {
                    "bulk_median": _parse_float(ess_match.group("bulk")),
                    "tail_median": _parse_float(ess_match.group("tail")),
                }
            acceptance_match = _ACCEPTANCE_RE.match(message)
            if acceptance_match:
                key = _normalize_model_label(acceptance_match.group("label"))
                acceptance[key] = {
                    "mean": _parse_float(acceptance_match.group("mean")),
                    "min": _parse_float(acceptance_match.group("min")),
                    "max": _parse_float(acceptance_match.group("max")),
                }
            fit_match = _FIT_REPORT_RE.match(message)
            if fit_match:
                current_model_key = _normalize_model_label(
                    fit_match.group("label")
                )
                log_models.setdefault(
                    current_model_key,
                    {
                        "label": fit_match.group("label"),
                        "chi2": {},
                        "bao_rs": None,
                    },
                )
                continue
            bao_match = _BAO_RS_RE.match(message)
            if bao_match:
                key = _normalize_model_label(bao_match.group("label"))
                entry = log_models.setdefault(
                    key,
                    {
                        "label": bao_match.group("label"),
                        "chi2": {},
                        "bao_rs": None,
                    },
                )
                entry["bao_rs"] = _parse_float(bao_match.group("rs"))
                if "chi2_bao" not in entry["chi2"]:
                    entry["chi2"]["chi2_bao"] = (
                        _parse_float(bao_match.group("chi2")) or 0.0
                    )
                continue
            if current_model_key and "=" in message:
                content = message.strip()
                if not ("chi" in content.lower() or "χ" in content):
                    continue
                key, metric_value = map(str.strip, content.split("=", 1))
                parsed_metric_value = _parse_float(metric_value)
                log_models[current_model_key]["chi2"][
                    _sanitize_metric_name(key)
                ] = (
                    parsed_metric_value
                    if parsed_metric_value is not None
                    else 0.0
                )

    if end_time is None and last_ts is not None:
        end_time = last_ts
    if start_time and end_time:
        duration = (end_time - start_time).total_seconds()

    return {
        "start_time": start_time,
        "end_time": end_time,
        "duration_seconds": duration,
        "datasets": datasets,
        "dataset_counts": counts,
        "diagnostics": {"rhat": rhat or None, "ess": ess or None},
        "models": log_models,
        "acceptance": acceptance or None,
    }


def _ensure_mapping(
    maybe_mapping: Optional[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Return the supplied mapping or an empty dict when it is None."""
    return maybe_mapping or {}


@dataclass
class ModelSummary:
    """Record summary statistics returned by run analysis for a model."""

    name: str
    parameters: Mapping[str, Any]
    errors_1sigma: Optional[Mapping[str, Any]]
    covariance_matrix: Optional[Mapping[str, Any]]
    sampling: Optional[Mapping[str, Any]]
    chi2: Mapping[str, float]
    acceptance: Optional[Mapping[str, float]]
    bao_rs: Optional[float]


@dataclass
class RunDiagnostics:
    """Hold diagnostics such as R-hat and ESS from a run log."""

    rhat: Optional[Mapping[str, float]]
    ess: Optional[Mapping[str, float]]


@dataclass
class RunAnalysisResult:
    """Represent the evaluated metadata of a Copernican run directory."""

    run_dir: Path
    model_summaries: Mapping[str, ModelSummary]
    manifest: Optional[Mapping[str, Any]]
    manifest_path: Optional[Path]
    parameter_summary_path: Optional[Path]
    datasets: Mapping[str, Mapping[str, Any]]
    dataset_counts: Mapping[str, int]
    diagnostics: RunDiagnostics
    start_time: Optional[datetime.datetime]
    end_time: Optional[datetime.datetime]
    duration_seconds: Optional[float]
    log_path: Optional[Path]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable representation of the analysis result."""
        result = dataclasses.asdict(self)
        result["run_dir"] = str(self.run_dir)
        if self.start_time:
            result["start_time"] = self.start_time.isoformat()
        if self.end_time:
            result["end_time"] = self.end_time.isoformat()
        result["manifest_path"] = (
            str(self.manifest_path) if self.manifest_path else None
        )
        result["parameter_summary_path"] = (
            str(self.parameter_summary_path)
            if self.parameter_summary_path
            else None
        )
        result["log_path"] = str(self.log_path) if self.log_path else None
        result["model_summaries"] = {
            name: dataclasses.asdict(summary)
            for name, summary in self.model_summaries.items()
        }
        return result


def analyze_run(run_dir: Path) -> RunAnalysisResult:
    """Summarise the contents of a Copernican run directory."""

    run_dir = Path(run_dir)
    manifest, manifest_path = load_manifest(run_dir)
    param_summary, summary_path = load_parameter_summary(run_dir)
    log_path = _find_log_file(run_dir)
    log_data = parse_log(log_path) if log_path else {}

    diagnostics = log_data.get("diagnostics", {})
    model_log = log_data.get("models", {})
    acceptance = log_data.get("acceptance") or {}

    summaries: dict[str, ModelSummary] = {}

    for model_name, model_data in _ensure_mapping(param_summary).items():
        norm = _normalize_model_label(model_name)
        log_entry = model_log.get(norm, {})
        chi2_vals = log_entry.get("chi2", {})
        summaries[model_name] = ModelSummary(
            name=model_name,
            parameters=model_data.get("parameters", {}),
            errors_1sigma=model_data.get("errors_1sigma"),
            covariance_matrix=model_data.get("covariance_matrix"),
            sampling=model_data.get("sampling"),
            chi2={
                metric_key: metric_value
                for metric_key, metric_value in chi2_vals.items()
                if isinstance(metric_value, (int, float))
            },
            acceptance=acceptance.get(norm),
            bao_rs=model_log.get(norm, {}).get("bao_rs"),
        )

    if not summaries and model_log:
        for norm_key, log_entry in model_log.items():
            model_label = log_entry.get("label") or norm_key
            summaries[model_label] = ModelSummary(
                name=model_label,
                parameters={},
                errors_1sigma=None,
                covariance_matrix=None,
                sampling=None,
                chi2={
                    metric_key: metric_value
                    for metric_key, metric_value in log_entry.get(
                        "chi2", {}
                    ).items()
                    if isinstance(metric_value, (int, float))
                },
                acceptance=acceptance.get(norm_key),
                bao_rs=log_entry.get("bao_rs"),
            )

    diagnostics_dataclass = RunDiagnostics(
        rhat=diagnostics.get("rhat"),
        ess=diagnostics.get("ess"),
    )

    return RunAnalysisResult(
        run_dir=run_dir,
        model_summaries=summaries,
        manifest=manifest,
        manifest_path=manifest_path,
        parameter_summary_path=summary_path,
        datasets=_ensure_mapping(
            manifest.get("datasets") if manifest else log_data.get("datasets")
        ),
        dataset_counts=log_data.get("dataset_counts", {}),
        diagnostics=diagnostics_dataclass,
        start_time=log_data.get("start_time"),
        end_time=log_data.get("end_time"),
        duration_seconds=log_data.get("duration_seconds"),
        log_path=log_path,
    )


def _normalize_formats(formats: Sequence[str] | str) -> list[str]:
    """Return a lowercase list of format names from the input."""
    if isinstance(formats, str):
        formats = (formats,)
    return [fmt.casefold() for fmt in formats]


def _summary_timestamp(
    result: RunAnalysisResult, *, override: str | None
) -> str:
    """Return the supplied override or a timestamp derived from the run."""
    if override:
        return override
    ts_source = result.start_time
    if ts_source:
        return ts_source.strftime("%Y%m%d_%H%M%S")
    return utils.get_timestamp()


@dataclass
class _PosteriorPlotPlugin:
    """Configuration container describing a posterior plotting plugin."""

    MODEL_NAME: str
    PARAMETER_NAMES: list[str]
    PARAMETER_LATEX_NAMES: list[str]


def _normalize_posterior_kinds(
    kinds: Sequence[str] | str,
) -> list[str]:
    """Normalize requested posterior plot kinds into a clean list."""
    if isinstance(kinds, str):
        kinds = (kinds,)
    normalized = [kind.strip().lower() for kind in kinds if kind.strip()]
    return normalized or ["overview"]


def _resolve_posterior_path(
    run_dir: Path,
    posterior_file: Path | str | None,
) -> Path:
    """Locate a posterior file for plotting, defaulting to the latest."""
    if posterior_file:
        target = Path(posterior_file)
        if not target.is_absolute():
            target = run_dir / target
    else:
        files = posterior_explorer.find_posterior_files(run_dir)
        if not files:
            raise FileNotFoundError(
                f"No posterior files found inside {run_dir}"
            )
        target = files[-1]
    if not target.is_file():
        raise FileNotFoundError(f"Posterior file not found: {target}")
    return target


def _posterior_dataset_id(
    result: RunAnalysisResult,
) -> str:
    """Return the preferred dataset identifier for posterior outputs."""
    if result.datasets:
        return next(iter(result.datasets))
    return "joint"


def _posterior_data_attrs(
    result: RunAnalysisResult, dataset_id: str
) -> dict[str, str]:
    """Produce metadata attributes for the selected dataset."""
    dataset_meta = result.datasets.get(dataset_id, {})
    return {
        "dataset_id": dataset_id,
        "dataset_name": dataset_meta.get("name", dataset_id),
        "description": dataset_meta.get("description", ""),
        "citation": dataset_meta.get("citation", ""),
        "notes": dataset_meta.get("notes", ""),
    }


def _posterior_plugin(
    result: RunAnalysisResult,
    param_names: list[str],
) -> _PosteriorPlotPlugin:
    """Create a plotting plugin summarizing the run's posterior parameters."""
    model_names = list(result.model_summaries.keys())
    model_label = model_names[0] if model_names else "Posterior"
    latex_names = list(param_names)
    return _PosteriorPlotPlugin(
        MODEL_NAME=model_label,
        PARAMETER_NAMES=list(param_names),
        PARAMETER_LATEX_NAMES=latex_names,
    )


def _corner_filename(
    dataset_id: str,
    comparison: ComparisonRequest,
    timestamp: str,
) -> str:
    """Generate the corner plot filename for a selected model pair."""
    return utils.generate_filename(
        "corner-plot",
        dataset_id,
        "png",
        model_name=comparison_slug(comparison),
        timestamp=timestamp,
    )


def _histogram_filename(
    dataset_id: str,
    comparison: ComparisonRequest,
    timestamp: str,
) -> str:
    """Generate the histogram filename for a selected model pair."""
    return utils.generate_filename(
        "parameter-histograms",
        dataset_id,
        "png",
        model_name=comparison_slug(comparison),
        timestamp=timestamp,
    )


def _posterior_comparison(
    result: RunAnalysisResult,
    comparison: ComparisonRequest | None,
) -> ComparisonRequest:
    """Resolve the required control/test pair for posterior diagnostics."""

    if comparison is not None:
        return comparison
    if result.manifest is None:
        raise ValueError(
            "Posterior plotting requires a control/test comparison."
        )
    return comparison_from_manifest(result.manifest)


def plot_posterior(
    run_dir: Path | str,
    output_dir: Path | str | None = None,
    *,
    posterior_file: Path | str | None = None,
    kinds: Sequence[str] | str = ("corner", "histograms", "overview"),
    timestamp: str | None = None,
    result: RunAnalysisResult | None = None,
    overview_path: Path | str | None = None,
    comparison: ComparisonRequest | None = None,
) -> dict[str, Path]:
    """Render cached posterior diagnostics for an existing comparison run.

    The comparison is recovered from the saved manifest unless callers pass
    it explicitly for an in-memory analysis result.
    """

    resolved_run_dir = Path(run_dir)
    resolved_result = result or analyze_run(resolved_run_dir)
    resolved_comparison = _posterior_comparison(
        resolved_result,
        comparison,
    )
    selected_kinds = _normalize_posterior_kinds(kinds)
    allowed = {"corner", "histograms", "overview"}
    invalid = [kind for kind in selected_kinds if kind not in allowed]
    if invalid:
        raise ValueError(f"Unsupported posterior plot kinds: {invalid}")

    out_dir = Path(output_dir) if output_dir else resolved_run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    target = _resolve_posterior_path(resolved_run_dir, posterior_file)
    dataset = posterior_explorer.load_inference_data(target)
    samples, param_names = posterior_explorer.flatten_posterior_arrays(dataset)
    if param_names == []:
        raise RuntimeError("No posterior variables found for plotting.")

    dataset_id = _posterior_dataset_id(resolved_result)
    data_attrs = _posterior_data_attrs(resolved_result, dataset_id)
    plugin = _posterior_plugin(resolved_result, param_names)
    summary_timestamp = _summary_timestamp(resolved_result, override=timestamp)

    saved: dict[str, Path] = {}
    if "corner" in selected_kinds:
        plotter.plot_corner(
            samples,
            plugin,
            data_attrs,
            plot_dir=str(out_dir),
            parameter_names=param_names,
            timestamp=summary_timestamp,
            comparison=resolved_comparison,
        )
        saved["corner"] = out_dir / _corner_filename(
            dataset_id, resolved_comparison, summary_timestamp
        )
    if "histograms" in selected_kinds:
        plotter.plot_parameter_histograms(
            samples,
            plugin,
            data_attrs,
            plot_dir=str(out_dir),
            parameter_names=param_names,
            timestamp=summary_timestamp,
            comparison=resolved_comparison,
        )
        saved["histograms"] = out_dir / _histogram_filename(
            dataset_id, resolved_comparison, summary_timestamp
        )
    if "overview" in selected_kinds:
        figure = posterior_explorer.create_posterior_overview_figure(
            resolved_result, target
        )
        dest = (
            Path(overview_path)
            if overview_path
            else out_dir / f"analysis-posterior_{summary_timestamp}.png"
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(dest)
        try:
            from matplotlib import pyplot as mpl

            mpl.close(figure)
        except ImportError:  # pragma: no cover - Matplotlib always present
            pass
        saved["overview"] = dest

    return saved


def _dump_summary(
    summary_data: Mapping[str, Any], path: Path, fmt: str
) -> None:
    """Serialize summary data to YAML or JSON depending on `fmt`."""
    if fmt in ("yml", "yaml"):
        with open(path, "w", encoding="utf-8") as file_handle:
            yaml.safe_dump(summary_data, file_handle, sort_keys=False)
        return
    if fmt == "json":
        with open(path, "w", encoding="utf-8") as file_handle:
            json.dump(summary_data, file_handle, indent=2)
        return
    raise ValueError(f"Unsupported summary format: {fmt!r}")


def save_run_summary(
    run_dir: Path | str,
    output_dir: Path | str,
    *,
    formats: Sequence[str] | str = (
        "yml",
        "json",
    ),
    timestamp: str | None = None,
    result: RunAnalysisResult | None = None,
) -> dict[str, Path]:
    """Serialize the latest run analysis result to disk.

    Parameters
    ----------
    run_dir:
        Directory that produced the run outputs.  If ``result`` is provided the
        helper skips re-running :func:`analyze_run`.
    output_dir:
        Directory where the summary files will be written.
        Parent directories are created automatically.
    formats:
        Case-insensitive list of file formats to produce. Supported values are
        ``"yml"``/``"yaml"`` and ``"json"``.  Defaults to both YAML and JSON.
    timestamp:
        Optional override for the timestamp embedded in the filename.
    result:
        Optional :class:`RunAnalysisResult` instance; when omitted the run is
        analysed on demand.

    Returns
    -------
    dict
        Mapping from the canonical format name to the written file path.
    """

    resolved_result = result or analyze_run(Path(run_dir))
    summary_payload = resolved_result.to_dict()
    formats_list = _normalize_formats(formats)
    summary_timestamp = _summary_timestamp(resolved_result, override=timestamp)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    base_name = f"analysis-summary_{summary_timestamp}"
    saved: dict[str, Path] = {}
    for fmt in formats_list:
        filename = f"{base_name}.{fmt}"
        target = output_path / filename
        _dump_summary(summary_payload, target, fmt)
        saved[fmt] = target
    return saved


def _run_descriptor(result: RunAnalysisResult) -> dict[str, Any]:
    """Return a compact descriptor describing the analysed run."""

    return {
        "run_dir": str(result.run_dir),
        "manifest_path": (
            str(result.manifest_path) if result.manifest_path else None
        ),
        "parameter_summary_path": (
            str(result.parameter_summary_path)
            if result.parameter_summary_path
            else None
        ),
        "log_path": str(result.log_path) if result.log_path else None,
        "start_time": (
            result.start_time.isoformat() if result.start_time else None
        ),
        "end_time": result.end_time.isoformat() if result.end_time else None,
        "duration_seconds": result.duration_seconds,
        "models": sorted(result.model_summaries.keys()),
        "datasets": sorted(result.datasets.keys()),
    }


def _safe_numeric(candidate_value: Any) -> Optional[float]:
    """Coerce numeric-like inputs into a float when possible."""
    if candidate_value is None:
        return None
    if isinstance(candidate_value, (int, float)):
        return float(candidate_value)
    if isinstance(candidate_value, str):
        return _parse_float(candidate_value)
    return None


def _diff_entry(base_measure: Any, alternative_measure: Any) -> dict[str, Any]:
    """Compare two values and report their delta if numeric."""
    base_num = _safe_numeric(base_measure)
    alt_num = _safe_numeric(alternative_measure)
    delta = None
    if base_num is not None and alt_num is not None:
        delta = alt_num - base_num
    elif alt_num is not None:
        delta = alt_num
    elif base_num is not None:
        delta = -base_num
    return {
        "base": base_measure,
        "alternative": alternative_measure,
        "delta": delta,
    }


def _model_diff(
    base_entry: ModelSummary | None,
    alt_entry: ModelSummary | None,
) -> dict[str, Any]:
    """Produce chi² and parameter deltas between two model summaries."""
    chi2_keys = set()
    if base_entry:
        chi2_keys.update(base_entry.chi2.keys())
    if alt_entry:
        chi2_keys.update(alt_entry.chi2.keys())
    param_keys = set()
    if base_entry:
        param_keys.update(base_entry.parameters.keys())
    if alt_entry:
        param_keys.update(alt_entry.parameters.keys())

    return {
        "chi2": {
            metric: _diff_entry(
                base_entry.chi2.get(metric) if base_entry else None,
                alt_entry.chi2.get(metric) if alt_entry else None,
            )
            for metric in sorted(chi2_keys)
        },
        "parameters": {
            param: _diff_entry(
                base_entry.parameters.get(param) if base_entry else None,
                alt_entry.parameters.get(param) if alt_entry else None,
            )
            for param in sorted(param_keys)
        },
    }


def compare_runs(
    base_result: RunAnalysisResult,
    alternative_result: RunAnalysisResult,
) -> dict[str, Any]:
    """Compare two analysed runs and report deltas."""

    dataset_ids = set(base_result.datasets.keys()) | set(
        alternative_result.datasets.keys()
    )
    model_names = set(base_result.model_summaries.keys()) | set(
        alternative_result.model_summaries.keys()
    )

    return {
        "runs": {
            "base": _run_descriptor(base_result),
            "alternative": _run_descriptor(alternative_result),
        },
        "difference": {
            "duration_seconds": _diff_entry(
                base_result.duration_seconds,
                alternative_result.duration_seconds,
            ),
            "dataset_counts": {
                dataset: _diff_entry(
                    base_result.dataset_counts.get(dataset),
                    alternative_result.dataset_counts.get(dataset),
                )
                for dataset in sorted(dataset_ids)
            },
            "models": {
                model: _model_diff(
                    base_result.model_summaries.get(model),
                    alternative_result.model_summaries.get(model),
                )
                for model in sorted(model_names)
            },
        },
    }


def compare_run_dirs(
    base_dir: Path | str,
    alternative_dir: Path | str,
) -> dict[str, Any]:
    """Analyse two run directories and compare their summaries."""

    base_result = analyze_run(Path(base_dir))
    alt_result = analyze_run(Path(alternative_dir))
    return compare_runs(base_result, alt_result)


def save_comparison_summary(
    base_result: RunAnalysisResult,
    alternative_result: RunAnalysisResult,
    output_dir: Path | str,
    *,
    formats: Sequence[str] | str = ("yml", "json"),
    timestamp: str | None = None,
) -> dict[str, Path]:
    """Serialize a comparison summary to disk."""

    summary = compare_runs(base_result, alternative_result)
    summary_timestamp = timestamp or _summary_timestamp(
        base_result, override=None
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}
    for fmt in _normalize_formats(formats):
        filename = f"analysis-comparison_{summary_timestamp}.{fmt}"
        target = output_path / filename
        _dump_summary(summary, target, fmt)
        saved[fmt] = target
    return saved


def format_run_summary_text(result: RunAnalysisResult) -> str:
    """Render a human-readable summary for terminal or log output."""

    lines: list[str] = []
    lines.append(f"Run directory: {result.run_dir}")
    lines.append(f"Manifest: {result.manifest_path or 'n/a'}")
    lines.append(
        f"Parameter summary: {result.parameter_summary_path or 'n/a'}"
    )
    lines.append(f"Log: {result.log_path or 'n/a'}")
    if result.start_time:
        lines.append(f"Started: {result.start_time.isoformat()}")
    if result.end_time:
        lines.append(f"Finished: {result.end_time.isoformat()}")
    if result.duration_seconds is not None:
        lines.append(f"Duration: {result.duration_seconds:.1f} s")
    lines.append("")

    if result.datasets:
        lines.append("Datasets:")
        for dataset_id, info in result.datasets.items():
            count = result.dataset_counts.get(dataset_id, "n/a")
            lines.append(
                f"  {dataset_id} ({info.get('type', 'unknown')}): "
                f"{info.get('name', 'unknown')} – {count} rows"
            )
        lines.append("")

    diagnostics = result.diagnostics
    if diagnostics.rhat:
        lines.append(
            "R‑hat summary: "
            f"{diagnostics.rhat.get('min', '?')}/"
            f"{diagnostics.rhat.get('median', '?')}/"
            f"{diagnostics.rhat.get('max', '?')}"
        )
    if diagnostics.ess:
        lines.append(
            "Effective sample sizes: "
            f"bulk={diagnostics.ess.get('bulk_median', '?')}, "
            f"tail={diagnostics.ess.get('tail_median', '?')}"
        )
    lines.append("")

    if result.model_summaries:
        for model_name, summary in result.model_summaries.items():
            lines.append(f"Model: {model_name}")
            for chi2_key, chi2_value in sorted(summary.chi2.items()):
                lines.append(f"  {chi2_key}: {chi2_value}")
            if summary.acceptance:
                lines.append(
                    "  Acceptance "
                    f"mean={summary.acceptance.get('mean', '?')}, "
                    f"min={summary.acceptance.get('min', '?')}, "
                    f"max={summary.acceptance.get('max', '?')}"
                )
            if summary.bao_rs is not None:
                lines.append(f"  BAO r_s = {summary.bao_rs:.2f} Mpc")
            lines.append("")
    return "\n".join(lines).strip()


__all__ = [
    "analyze_run",
    "RunAnalysisResult",
    "ModelSummary",
    "RunDiagnostics",
    "compare_runs",
    "compare_run_dirs",
    "save_comparison_summary",
    "format_run_summary_text",
    "plot_posterior",
]
