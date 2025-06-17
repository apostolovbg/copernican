"""Flask routes for the Copernican web interface."""
# DEV NOTE (v1.5h): Provides basic landing page and run/abort logic.

import os
import threading
import zipfile
import shutil
from datetime import datetime
from flask import Blueprint, render_template, redirect, url_for, request, send_file, flash

bp = Blueprint('web', __name__)

_run_thread = None
_abort_flag = threading.Event()
_output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')


def init_app(app):
    """Registers blueprint with the given Flask app."""
    app.register_blueprint(bp)
    app.config['SECRET_KEY'] = 'dev'


@bp.route('/')
def index():
    """Landing page showing current status."""
    running = _run_thread is not None and _run_thread.is_alive()
    return render_template('index.html', running=running)


def _analysis_worker():
    from .. import copernican
    timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
    session_dir = os.path.join(_output_dir, f'session_{timestamp}')
    os.makedirs(session_dir, exist_ok=True)
    try:
        copernican.main_workflow(output_dir=session_dir)
    finally:
        global _run_thread
        _run_thread = None
        _abort_flag.clear()


@bp.route('/run', methods=['POST'])
def run_action():
    """Start or abort a run based on current state."""
    global _run_thread
    if _run_thread is None or not _run_thread.is_alive():
        # Start new run
        _run_thread = threading.Thread(target=_analysis_worker, daemon=True)
        _run_thread.start()
    else:
        # Abort current run by setting flag for future use
        _abort_flag.set()
    return redirect(url_for('web.index'))


@bp.route('/download')
def download_results():
    """Download last run outputs as zip."""
    timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
    archive_name = f"copernican_output_{timestamp}.zip"
    archive_path = os.path.join(_output_dir, archive_name)
    with zipfile.ZipFile(archive_path, 'w') as zipf:
        for root, _, files in os.walk(_output_dir):
            for f in files:
                filepath = os.path.join(root, f)
                if filepath.endswith('.zip'):
                    continue
                zipf.write(filepath, arcname=os.path.relpath(filepath, _output_dir))
    return send_file(archive_path, as_attachment=True, download_name=archive_name)
