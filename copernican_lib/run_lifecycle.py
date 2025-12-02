"""Run lifecycle helpers for managing manifests and program logs."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from copernican_lib import run_manifest, utils

MAX_PROGRAM_LOGS = 30
PROGRAM_LOG_PREFIX = "copernican_log_"
PROGRAM_LOG_SUFFIX = ".txt"


@dataclass
class ManifestDraft:
    """Track a manifest draft while the Run Builder is still configuring."""

    folder: Path
    manifest_path: Path
    creation_timestamp: str


def _draft_folder_name(prefix: str, timestamp: str) -> str:
    """Return the canonical folder name for a manifest timestamp."""

    return f"{prefix}_{timestamp}"


def _manifest_filename(timestamp: str) -> str:
    """Return the filename written inside every draft folder."""

    return f"run_manifest_{timestamp}.yml"


def create_manifest_draft(
    output_root: Path,
    manifest: dict,
    *,
    folder_prefix: str = "copernican-run",
) -> ManifestDraft:
    """Persist the provided manifest in a timestamped draft directory."""

    timestamp = utils.get_timestamp()
    draft_folder = output_root / _draft_folder_name(folder_prefix, timestamp)
    draft_folder.mkdir(parents=True, exist_ok=True)
    manifest_path = draft_folder / _manifest_filename(timestamp)
    run_manifest.save_manifest(
        manifest,
        draft_folder,
        target_path=manifest_path,
    )
    return ManifestDraft(
        folder=draft_folder,
        manifest_path=manifest_path,
        creation_timestamp=timestamp,
    )


def delete_manifest_draft(draft: ManifestDraft) -> None:
    """Delete the draft folder and its manifest when the user cancels."""

    if draft.folder.exists():
        shutil.rmtree(draft.folder)


def import_manifest_to_draft(
    source: Path,
    output_root: Path,
    *,
    folder_prefix: str = "copernican-run",
) -> ManifestDraft:
    """Copy an external manifest into a new draft workspace."""

    timestamp = utils.get_timestamp()
    draft_folder = output_root / _draft_folder_name(folder_prefix, timestamp)
    draft_folder.mkdir(parents=True, exist_ok=True)
    target_manifest = draft_folder / _manifest_filename(timestamp)
    shutil.copy2(source, target_manifest)
    return ManifestDraft(
        folder=draft_folder,
        manifest_path=target_manifest,
        creation_timestamp=timestamp,
    )


def finalize_run_from_draft(
    draft: ManifestDraft,
    *,
    start_timestamp: str,
    folder_prefix: str = "copernican-run",
) -> ManifestDraft:
    """Rename the draft workspace to the run-start timestamp."""

    new_folder = draft.folder.parent / _draft_folder_name(
        folder_prefix, start_timestamp
    )
    draft.folder.rename(new_folder)
    new_manifest = new_folder / _manifest_filename(start_timestamp)
    draft.manifest_path.rename(new_manifest)
    return ManifestDraft(
        folder=new_folder,
        manifest_path=new_manifest,
        creation_timestamp=start_timestamp,
    )


def _collect_program_logs(
    logs_dir: Path,
    *,
    prefix: str = PROGRAM_LOG_PREFIX,
    suffix: str = PROGRAM_LOG_SUFFIX,
) -> list[Path]:
    """Return the program-facing log files sorted by creation time."""

    pattern = f"{prefix}*{suffix}"
    candidates = [path for path in logs_dir.glob(pattern) if path.is_file()]
    return sorted(candidates, key=lambda path: path.stat().st_mtime)


def prune_program_logs(
    logs_dir: Path,
    *,
    max_logs: int = MAX_PROGRAM_LOGS,
    prefix: str = PROGRAM_LOG_PREFIX,
    suffix: str = PROGRAM_LOG_SUFFIX,
) -> None:
    """Keep only the most recent ``max_logs`` program logs."""

    if max_logs <= 0:
        return
    existing = _collect_program_logs(logs_dir, prefix=prefix, suffix=suffix)
    while len(existing) >= max_logs:
        oldest = existing.pop(0)
        try:
            oldest.unlink()
        except FileNotFoundError:
            continue


def prepare_program_log_path(
    logs_dir: Path,
    *,
    prefix: str = PROGRAM_LOG_PREFIX,
    suffix: str = PROGRAM_LOG_SUFFIX,
    max_logs: int = MAX_PROGRAM_LOGS,
) -> Path:
    """Return a path for the next program log file, pruning as needed."""

    logs_dir.mkdir(parents=True, exist_ok=True)
    prune_program_logs(
        logs_dir,
        max_logs=max_logs,
        prefix=prefix,
        suffix=suffix,
    )
    timestamp = utils.get_timestamp()
    return logs_dir / f"{prefix}{timestamp}{suffix}"
