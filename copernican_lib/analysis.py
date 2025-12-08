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

from . import run_manifest as run_manifest_module
from . import utils

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
    ascii_only = "".join(ch for ch in normalized if ord(ch) < 128)
    return ascii_only.casefold()


def _parse_float(value: str) -> Optional[float]:
    """Parse ``value`` as float while tolerating unicode minus signs."""

    cleaned = value.replace("−", "-").replace("\u2212", "-").strip()
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
    matches = sorted(run_dir.glob(pattern))
    if not matches:
        return None
    return matches[-1]


def _load_yaml_or_json(path: Path) -> Any:
    if path.suffix.lower() in {".yml", ".yaml"}:
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


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
    parts = line.split(" - ", 2)
    if len(parts) < 3:
        return None, line.strip()
    timestamp = _parse_timestamp(parts[0])
    return timestamp, parts[2].strip()


def _parse_timestamp(timestamp: str) -> Optional[datetime.datetime]:
    try:
        return datetime.datetime.strptime(
            timestamp.strip(), "%Y-%m-%d %H:%M:%S,%f"
        )
    except ValueError:
        return None


def _find_log_file(run_dir: Path) -> Optional[Path]:
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

    with open(log_path, "r", encoding="utf-8") as fh:
        previous_message: Optional[str] = None
        current_model_key: Optional[str] = None
        for raw in fh:
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
                key, value = map(str.strip, content.split("=", 1))
                log_models[current_model_key]["chi2"][
                    _sanitize_metric_name(key)
                ] = (_parse_float(value) or 0.0)

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


def _ensure_mapping(value: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    return value or {}


@dataclass
class ModelSummary:
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
    rhat: Optional[Mapping[str, float]]
    ess: Optional[Mapping[str, float]]


@dataclass
class RunAnalysisResult:
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
                key: value
                for key, value in chi2_vals.items()
                if isinstance(value, (int, float))
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
                    key: value
                    for key, value in log_entry.get("chi2", {}).items()
                    if isinstance(value, (int, float))
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
    if isinstance(formats, str):
        formats = (formats,)
    return [fmt.casefold() for fmt in formats]


def _summary_timestamp(
    result: RunAnalysisResult, *, override: str | None
) -> str:
    if override:
        return override
    ts_source = result.start_time
    if ts_source:
        return ts_source.strftime("%Y%m%d_%H%M%S")
    return utils.get_timestamp()


def _dump_summary(data: Mapping[str, Any], path: Path, fmt: str) -> None:
    if fmt in ("yml", "yaml"):
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(data, fh, sort_keys=False)
        return
    if fmt == "json":
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
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
    data = resolved_result.to_dict()
    formats_list = _normalize_formats(formats)
    ts = _summary_timestamp(resolved_result, override=timestamp)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    base_name = f"analysis-summary_{ts}"
    saved: dict[str, Path] = {}
    for fmt in formats_list:
        filename = f"{base_name}.{fmt}"
        target = output_path / filename
        _dump_summary(data, target, fmt)
        saved[fmt] = target
    return saved


__all__ = [
    "analyze_run",
    "RunAnalysisResult",
    "ModelSummary",
    "RunDiagnostics",
]
