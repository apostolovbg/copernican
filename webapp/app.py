# DEV NOTE (v1.5h): Initial Flask interface implementing phases 0-3 of PLAN.md.
"""Flask web server for the Copernican Suite."""

import os
import subprocess
import shutil
import time
import zipfile
from threading import Thread

from flask import Flask, render_template, request, redirect, url_for, send_file, flash

app = Flask(__name__)
app.secret_key = "copernican_secret"

# Root directory of the project used to spawn subprocesses and locate data.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")

# Track the currently running analysis process and session folder.
_current_process = None
_current_session = None


def _create_session_folder() -> str:
    """Return path to a new timestamped output directory."""
    ts = time.strftime("web_%y%m%d_%H%M%S")
    path = os.path.join(OUTPUT_DIR, ts)
    os.makedirs(path, exist_ok=True)
    return path


def _run_analysis_thread(session_path: str):
    """Launch copernican.py in a subprocess and log output to the session."""
    global _current_process
    log_path = os.path.join(session_path, "run.log")
    with open(log_path, "w") as log_f:
        _current_process = subprocess.Popen(
            ["python", "copernican.py"], cwd=ROOT_DIR, stdout=log_f, stderr=subprocess.STDOUT
        )
        _current_process.wait()
    _current_process = None


@app.route("/")
def index():
    """Landing page displaying run controls and tabs."""
    run_active = _current_process is not None
    return render_template("index.html", version="1.5h", run_active=run_active)


@app.route("/toggle_run", methods=["POST"])
def toggle_run():
    """Start a new run or abort the current one."""
    global _current_session
    action = request.form.get("action")
    if action == "start":
        if _current_process:
            flash("Run already in progress.")
            return redirect(url_for("index"))
        if _current_session and os.path.isdir(_current_session):
            return render_template("confirm.html")
        _current_session = _create_session_folder()
        Thread(target=_run_analysis_thread, args=(_current_session,), daemon=True).start()
    elif action == "abort":
        if _current_process and _current_process.poll() is None:
            _current_process.terminate()
            _current_process.wait()
            flash("Run aborted.")
        _current_process = None
    return redirect(url_for("index"))


@app.route("/confirm_discard", methods=["POST"])
def confirm_discard():
    """Handle confirmation before discarding previous results."""
    global _current_session
    choice = request.form.get("choice")
    if choice == "space":
        if _current_session and os.path.isdir(_current_session):
            shutil.rmtree(_current_session)
        _current_session = _create_session_folder()
        Thread(target=_run_analysis_thread, args=(_current_session,), daemon=True).start()
    return redirect(url_for("index"))


@app.route("/download")
def download():
    """Download a ZIP archive of the most recent session results."""
    if not _current_session or not os.path.isdir(_current_session):
        flash("No session results available.")
        return redirect(url_for("index"))
    zip_path = _current_session + ".zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(_current_session):
            for fname in files:
                fpath = os.path.join(root, fname)
                arcname = os.path.relpath(fpath, _current_session)
                zipf.write(fpath, arcname)
    return send_file(zip_path, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
