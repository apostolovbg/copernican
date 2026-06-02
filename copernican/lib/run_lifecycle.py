"""Run lifecycle helpers for managing manifests and run workspaces."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from copernican.lib import run_manifest, utils


@dataclass
class ManifestWorkspace:
    """Describe a persisted manifest and its run folder."""

    folder: Path
    manifest_path: Path
    creation_timestamp: str


def _workspace_folder_name(prefix: str, timestamp: str) -> str:
    """Return the canonical folder name for a manifest timestamp."""

    return f"{prefix}_{timestamp}"


def _manifest_filename(timestamp: str) -> str:
    """Return the filename written inside every workspace folder."""

    return f"run_manifest_{timestamp}.yml"


def create_manifest_workspace(
    output_root: Path,
    manifest: dict,
    *,
    folder_prefix: str = "copernican-run",
    folder_name: str | None = None,
    manifest_filename: str | None = None,
) -> ManifestWorkspace:
    """Persist the provided manifest in a workspace."""

    timestamp = utils.get_timestamp()
    if folder_name is None:
        folder_name = _workspace_folder_name(folder_prefix, timestamp)
    workspace_folder = output_root / folder_name
    workspace_folder.mkdir(parents=True, exist_ok=True)
    if manifest_filename is None:
        manifest_filename = _manifest_filename(timestamp)
    manifest_path = workspace_folder / manifest_filename
    run_manifest.save_manifest(
        manifest,
        workspace_folder,
        target_path=manifest_path,
    )
    return ManifestWorkspace(
        folder=workspace_folder,
        manifest_path=manifest_path,
        creation_timestamp=timestamp,
    )


def delete_manifest_workspace(workspace: ManifestWorkspace) -> None:
    """Delete the manifest workspace when the user cancels."""

    if workspace.folder.exists():
        shutil.rmtree(workspace.folder)


def import_manifest_to_workspace(
    source: Path,
    output_root: Path,
    *,
    folder_prefix: str = "copernican-run",
) -> ManifestWorkspace:
    """Copy an external manifest into a new workspace."""

    timestamp = utils.get_timestamp()
    workspace_folder = output_root / _workspace_folder_name(
        folder_prefix, timestamp
    )
    workspace_folder.mkdir(parents=True, exist_ok=True)
    target_manifest = workspace_folder / _manifest_filename(timestamp)
    shutil.copy2(source, target_manifest)
    return ManifestWorkspace(
        folder=workspace_folder,
        manifest_path=target_manifest,
        creation_timestamp=timestamp,
    )


def finalize_run_workspace(
    workspace: ManifestWorkspace,
    *,
    start_timestamp: str,
    folder_prefix: str = "copernican-run",
) -> ManifestWorkspace:
    """Rename the workspace to the run-start timestamp."""

    manifest_name = _manifest_filename(start_timestamp)
    new_manifest = workspace.folder / manifest_name
    workspace.manifest_path.rename(new_manifest)
    new_folder = workspace.folder.parent / _workspace_folder_name(
        folder_prefix, start_timestamp
    )
    if new_folder.exists():
        suffix = 1
        while True:
            candidate = (
                workspace.folder.parent
                / f"{folder_prefix}_{start_timestamp}_{suffix:02d}"
            )
            if not candidate.exists():
                new_folder = candidate
                break
            suffix += 1
    workspace.folder.rename(new_folder)
    final_manifest = new_folder / manifest_name
    return ManifestWorkspace(
        folder=new_folder,
        manifest_path=final_manifest,
        creation_timestamp=start_timestamp,
    )
