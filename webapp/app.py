# DEV NOTE (v1.5i hotfix 11): Pantheon+ dataset option added and `_current_process`
# scoping bug fixed in toggle_run.
# DEV NOTE (v1.5i): Web interface now fully functional for Phase 3. Implements
# model upload, engine and dataset selection, and runs copernican via a helper
# script. Session folders contain logs and plots, downloadable as ZIP archives.

import os
import shutil
import subprocess
import time
import zipfile
from threading import Thread

from flask import (
    Flask, render_template, request, redirect, url_for,
    send_file, flash
)

app = Flask(__name__)
app.secret_key = "copernican_secret"

# Version string displayed in the UI
VERSION = "1.5i"

# Root directory of the project used to spawn subprocesses and locate data
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
UPLOAD_DIR = os.path.join(ROOT_DIR, "webapp", "uploads")

# Available dataset sources with fixed parser keys and file paths
SNE_SOURCES = {
    "unistra": {
        "label": "University of Strasbourg (tablef3.dat)",
        "file": os.path.join(ROOT_DIR, "data", "sne", "tablef3.dat"),
        "format": "unistra_fixed_nuisance_h1",
    },
    "pantheon": {
        "label": "Pantheon+ (Pan.dat + Pancm.cov)",
        "file": os.path.join(ROOT_DIR, "data", "sne", "Pan.dat"),
        "cov": os.path.join(ROOT_DIR, "data", "sne", "Pancm.cov"),
        "format": "pantheon_plus_mu_cov_h2",
    },
}

BAO_SOURCES = {
    "basic": {
        "label": "BAO baseline (bao1.json)",
        "file": os.path.join(ROOT_DIR, "data", "bao", "bao1.json"),
        "format": "bao_json_general_v1",
    }
}

# Track the currently running subprocess and session folder
_current_process = None
_current_session = None

# User selections kept between requests
_selected_model = None
_selected_engine = None
_selected_sne = "unistra"
_selected_bao = "basic"


def _create_session_folder() -> str:
    """Return path to a new timestamped output directory."""
    ts = time.strftime("web_%y%m%d_%H%M%S")
    path = os.path.join(OUTPUT_DIR, ts)
    os.makedirs(path, exist_ok=True)
    return path


def _run_analysis_thread(session_path: str, model_json: str, engine: str,
                         sne_key: str, bao_key: str) -> None:
    """Execute analysis via a helper script in a subprocess."""
    global _current_process
    log_path = os.path.join(session_path, "run.log")
    sne_cfg = SNE_SOURCES[sne_key]
    bao_cfg = BAO_SOURCES[bao_key]
    runner = os.path.join(ROOT_DIR, "webapp", "web_runner.py")
    cmd = [
        "python", runner, model_json, engine,
        sne_cfg["file"], bao_cfg["file"],
        sne_cfg["format"], bao_cfg["format"], session_path,
    ]
    if "cov" in sne_cfg:
        cmd.append(sne_cfg["cov"])
    with open(log_path, "w") as log_f:
        _current_process = subprocess.Popen(
            cmd, cwd=ROOT_DIR, stdout=log_f, stderr=subprocess.STDOUT
        )
        _current_process.wait()
    _current_process = None


@app.route("/")
def index():
    """Landing page displaying run controls and tabs."""
    run_active = _current_process is not None
    log_text = ""
    if not run_active and _current_session:
        log_file = os.path.join(_current_session, "run.log")
        if os.path.isfile(log_file):
            with open(log_file, "r") as f:
                log_text = f.read()
    engines = [f for f in os.listdir(os.path.join(ROOT_DIR, "engines"))
               if f.startswith("cosmo_engine_") and f.endswith(".py")]
    hubble_plot = None
    bao_plot = None
    if _current_session:
        for fname in os.listdir(_current_session):
            if not hubble_plot and "hubble" in fname and fname.endswith(".png"):
                hubble_plot = fname
            if not bao_plot and "bao" in fname and fname.endswith(".png"):
                bao_plot = fname
    return render_template(
        "index.html", version=VERSION, run_active=run_active,
        model=os.path.basename(_selected_model) if _selected_model else None,
        engine=_selected_engine, sne=_selected_sne, bao=_selected_bao,
        engines=engines, log_text=log_text,
        hubble_plot=hubble_plot, bao_plot=bao_plot,
    )


@app.route("/upload_model", methods=["POST"])
def upload_model():
    """Handle model JSON upload."""
    global _selected_model
    file = request.files.get("model_json")
    if not file or not file.filename:
        flash("No file selected.")
        return redirect(url_for("index"))
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    save_path = os.path.join(UPLOAD_DIR, os.path.basename(file.filename))
    file.save(save_path)
    _selected_model = save_path
    flash("Model uploaded.")
    return redirect(url_for("index"))


@app.route("/select_engine", methods=["POST"])
def select_engine():
    """Store selected computational engine."""
    global _selected_engine
    choice = request.form.get("engine")
    if choice:
        _selected_engine = choice
        flash(f"Engine set to {choice}.")
    return redirect(url_for("index"))


@app.route("/select_datasets", methods=["POST"])
def select_datasets():
    """Store dataset selections."""
    global _selected_sne, _selected_bao
    sne_choice = request.form.get("sne_source")
    bao_choice = request.form.get("bao_source")
    if sne_choice in SNE_SOURCES:
        _selected_sne = sne_choice
    if bao_choice in BAO_SOURCES:
        _selected_bao = bao_choice
    flash("Datasets updated.")
    return redirect(url_for("index"))


@app.route("/toggle_run", methods=["POST"])
def toggle_run():
    """Start a new run or abort the current one."""
    global _current_session, _current_process
    action = request.form.get("action")
    if action == "start":
        if _current_process:
            flash("Run already in progress.")
            return redirect(url_for("index"))
        if not (_selected_model and _selected_engine):
            flash("Please upload a model and select an engine first.")
            return redirect(url_for("index"))
        if _current_session and os.path.isdir(_current_session):
            return render_template("confirm.html")
        _current_session = _create_session_folder()
        Thread(
            target=_run_analysis_thread,
            args=(
                _current_session,
                _selected_model,
                _selected_engine,
                _selected_sne,
                _selected_bao,
            ),
            daemon=True,
        ).start()
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
        Thread(
            target=_run_analysis_thread,
            args=(
                _current_session,
                _selected_model,
                _selected_engine,
                _selected_sne,
                _selected_bao,
            ),
            daemon=True,
        ).start()
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


@app.route("/session/<path:filename>")
def session_file(filename):
    """Serve files from the current session directory."""
    if not _current_session:
        return "", 404
    fpath = os.path.join(_current_session, filename)
    if os.path.isfile(fpath):
        return send_file(fpath)
    return "", 404


if __name__ == "__main__":
    app.run(debug=True)

