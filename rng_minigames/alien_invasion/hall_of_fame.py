"""Hall of fame tracking for Alien Invasion runs."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, List

import yaml

try:  # pragma: no cover - Tk may be unavailable during tests
    import tkinter as tk
    from tkinter import ttk
except Exception:  # pragma: no cover - executed when Tk is missing
    tk = None
    ttk = None


class HallOfFame:
    """Lightweight YAML-backed scoreboard."""

    def __init__(self, storage_dir: Path, *, limit: int = 10) -> None:
        storage_dir.mkdir(parents=True, exist_ok=True)
        self.limit = limit
        self.path = storage_dir / "alien_invasion_hof.yml"
        self.entries: List[dict[str, Any]] = []
        self._load()
        if not self.path.exists():
            self._save()

    def record(self, initials: str, time_left: float) -> None:
        """Persist a new run."""

        if time_left <= 0:
            return
        entry = {
            "initials": initials,
            "time_left": round(time_left, 2),
            "timestamp": int(time.time()),
        }
        self.entries.append(entry)
        self.entries.sort(key=lambda item: item["time_left"], reverse=True)
        self.entries = self.entries[: self.limit]
        self._save()

    def show(self, parent: "tk.Tk") -> None:
        """Display the scoreboard in a modal window."""

        if tk is None or parent is None:
            return
        window = tk.Toplevel(parent)
        window.title("Alien Invasion Hall of Fame")
        window.resizable(False, False)
        if ttk is None:
            text = "\n".join(
                f"{idx + 1}. {entry['initials']} - {entry['time_left']}s remaining"
                for idx, entry in enumerate(self.entries)
            )
            tk.Label(window, text=text or "No runs recorded yet.").pack(
                padx=16, pady=16
            )
            return
        ttk.Label(
            window,
            text="Last ten fastest runs",
            font=("Helvetica", 12, "bold"),
        ).pack(anchor="w", padx=16, pady=(12, 8))
        frame = ttk.Frame(window)
        frame.pack(fill="both", expand=True, padx=16, pady=(0, 12))
        columns = ("Rank", "Initials", "Time left (s)")
        tree = ttk.Treeview(
            frame, columns=columns, show="headings", height=self.limit
        )
        for col in columns:
            tree.heading(col, text=col)
            tree.column(
                col, width=120 if col != "Rank" else 60, anchor="center"
            )
        for idx, entry in enumerate(self.entries):
            tree.insert(
                "",
                "end",
                values=(idx + 1, entry["initials"], entry["time_left"]),
            )
        tree.pack(fill="both", expand=True)

    #
    # Persistence helpers
    #

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = yaml.safe_load(self.path.read_text()) or {}
        except Exception:
            self.entries = []
            self._save()
            return
        entries = data.get("entries", [])
        if isinstance(entries, list):
            converted: List[dict[str, Any]] = []
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                if "time_left" in entry:
                    converted.append(entry)
                elif "duration" in entry:
                    entry["time_left"] = entry["duration"]
                    converted.append(entry)
            self.entries = converted[: self.limit]

    def _save(self) -> None:
        payload = {"entries": self.entries}
        try:
            self.path.write_text(yaml.safe_dump(payload))
        except Exception:
            pass
