from flask import Flask, send_from_directory, jsonify, request
import os
import json
import subprocess
import tempfile
import shutil
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)

# Ensure the project root is in ``sys.path`` so the local package imports work
# even when ``serve_web.py`` is launched from the ``web`` directory.
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from copernican_lib import data_loaders

app = Flask(__name__, static_folder=os.path.join(REPO_ROOT, 'web'), static_url_path='/web')

OUTPUT_DIR = os.path.join(REPO_ROOT, 'output')

@app.route('/')
def index():
    return send_from_directory(REPO_ROOT, 'index.html')


@app.route('/api/models')
def models():
    files = [f for f in os.listdir(os.path.join(REPO_ROOT, 'models')) if f.startswith('cosmo_model_') and f.endswith('.json')]
    return jsonify(sorted(files))

@app.route('/api/datasets')
def datasets():
    return jsonify({
        'sne': list(data_loaders.SNE_PARSERS.keys()),
        'bao': list(data_loaders.BAO_PARSERS.keys()),
        'cmb': list(data_loaders.CMB_PARSERS.keys()),
    })

@app.route('/api/run', methods=['POST'])
def run_eval():
    conf = json.loads(request.form.get('config', '{}'))
    upload = request.files.get('file')
    model_path = None
    if upload:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.json', dir=os.path.join(REPO_ROOT, 'models'))
        upload.save(tmp.name)
        model_path = os.path.basename(tmp.name)
    else:
        model_path = conf.get('model')
    if not model_path:
        return 'No model provided', 400

    inputs = []
    models = sorted([f for f in os.listdir(os.path.join(REPO_ROOT, 'models')) if f.startswith('cosmo_model_') and f.endswith('.json')])
    inputs.append(str(models.index(model_path) + 1))

    engines = sorted([f for f in os.listdir(os.path.join(REPO_ROOT, 'engines')) if f.startswith('cosmo_engine_') and f.endswith('.py')])
    inputs.append('1')  # default engine first

    for group, registry in [('sne', data_loaders.SNE_PARSERS), ('bao', data_loaders.BAO_PARSERS), ('cmb', data_loaders.CMB_PARSERS)]:
        name = conf.get(group)
        if name and name in registry:
            index = list(registry.keys()).index(name) + 1
            inputs.append(str(index))
        else:
            inputs.append('c')
            return 'Invalid dataset selection', 400

    inputs.append('no')
    proc = subprocess.Popen(
        ['python', os.path.join(REPO_ROOT, 'copernican.py')],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=REPO_ROOT,
    )
    console_out, _ = proc.communicate('\n'.join(inputs) + '\n')
    proc.wait()

    files = sorted(os.listdir(OUTPUT_DIR))
    plots = [f for f in files if f.lower().endswith('.png')]
    tables = [f for f in files if f.lower().endswith('.csv')]
    logs = [f for f in files if f.lower().endswith('.txt')]
    log_file = logs[0] if logs else ''

    zip_path = shutil.make_archive(os.path.join(BASE_DIR, 'results'), 'zip', OUTPUT_DIR)
    return jsonify({
        'console': console_out,
        'plots': plots,
        'tables': tables,
        'log': log_file,
        'zip': '/api/download/results.zip'
    })


@app.route('/api/file/<path:filename>')
def get_file(filename: str):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=False)


@app.route('/api/download/<path:filename>')
def download(filename: str):
    return send_from_directory(BASE_DIR, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
