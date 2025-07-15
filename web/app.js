// Simple helper to fetch JSON from the Flask backend
async function fetchData(url) {
    const resp = await fetch(url);
    return resp.json();
}

// Populate a <select> element with option tags from an array of strings
function fillSelect(id, items) {
    const sel = document.getElementById(id);
    items.forEach(item => {
        const opt = document.createElement('option');
        opt.value = item;
        opt.textContent = item;
        sel.appendChild(opt);
    });
}

function switchTab(evt) {
    document.querySelectorAll('.tab').forEach(btn => btn.classList.remove('active'));
    evt.target.classList.add('active');
    document.querySelectorAll('.tabContent').forEach(el => el.classList.add('hidden'));
    document.getElementById(evt.target.dataset.tab).classList.remove('hidden');
}

let plots = [];
let tables = [];
let logFile = '';
let plotIdx = 0;
let tableIdx = 0;

function updatePlotViewer() {
    if (plots.length === 0) {
        document.getElementById('plotImg').src = '';
        document.getElementById('plotName').textContent = 'No plots';
        document.getElementById('downloadPlot').href = '#';
        return;
    }
    const file = plots[plotIdx];
    document.getElementById('plotImg').src = `/api/file/${file}`;
    document.getElementById('plotName').textContent = file;
    document.getElementById('downloadPlot').href = `/api/file/${file}`;
}

function updateTableViewer() {
    if (tables.length === 0) {
        document.getElementById('tableText').textContent = 'No tables';
        document.getElementById('tableName').textContent = '';
        document.getElementById('downloadTable').href = '#';
        return;
    }
    const file = tables[tableIdx];
    document.getElementById('tableName').textContent = file;
    document.getElementById('downloadTable').href = `/api/file/${file}`;
    fetch(`/api/file/${file}`).then(r => r.text()).then(t => {
        document.getElementById('tableText').textContent = t;
    });
}

document.addEventListener('DOMContentLoaded', async () => {
    const models = await fetchData('/api/models');
    fillSelect('modelSelect', models);
    const data = await fetchData('/api/datasets');
    fillSelect('sneSelect', data.sne);
    fillSelect('baoSelect', data.bao);
    fillSelect('cmbSelect', data.cmb);

    // Disable all steps except model selection at first
    ['sneSelect', 'baoSelect', 'cmbSelect', 'confirmSneBtn', 'confirmBaoBtn', 'confirmCmbBtn', 'runBtn']
        .forEach(id => document.getElementById(id).disabled = true);

    // Toggle between file upload and dropdown based on the selected radio
    function updateModelInputs() {
        const useUpload = document.getElementById('uploadRadio').checked;
        document.getElementById('modelFile').disabled = !useUpload;
        document.getElementById('modelSelect').disabled = useUpload;
    }

    document.getElementById('uploadRadio').addEventListener('change', updateModelInputs);
    document.getElementById('serverRadio').addEventListener('change', updateModelInputs);
    updateModelInputs();

    document.getElementById('confirmModelBtn').addEventListener('click', () => {
        document.getElementById('uploadRadio').disabled = true;
        document.getElementById('serverRadio').disabled = true;
        document.getElementById('modelFile').disabled = true;
        document.getElementById('modelSelect').disabled = true;
        document.getElementById('confirmModelBtn').disabled = true;
        document.getElementById('sneSelect').disabled = false;
        document.getElementById('confirmSneBtn').disabled = false;
    });

    document.getElementById('confirmSneBtn').addEventListener('click', () => {
        document.getElementById('sneSelect').disabled = true;
        document.getElementById('confirmSneBtn').disabled = true;
        document.getElementById('baoSelect').disabled = false;
        document.getElementById('confirmBaoBtn').disabled = false;
    });

    document.getElementById('confirmBaoBtn').addEventListener('click', () => {
        document.getElementById('baoSelect').disabled = true;
        document.getElementById('confirmBaoBtn').disabled = true;
        document.getElementById('cmbSelect').disabled = false;
        document.getElementById('confirmCmbBtn').disabled = false;
    });

    document.getElementById('confirmCmbBtn').addEventListener('click', () => {
        document.getElementById('cmbSelect').disabled = true;
        document.getElementById('confirmCmbBtn').disabled = true;
        document.getElementById('runBtn').disabled = false;
    });

    document.querySelectorAll('.tab').forEach(btn => btn.addEventListener('click', switchTab));

    document.getElementById('prevPlot').addEventListener('click', () => {
        plotIdx = (plotIdx - 1 + plots.length) % plots.length;
        updatePlotViewer();
    });
    document.getElementById('nextPlot').addEventListener('click', () => {
        plotIdx = (plotIdx + 1) % plots.length;
        updatePlotViewer();
    });
    document.getElementById('prevTable').addEventListener('click', () => {
        tableIdx = (tableIdx - 1 + tables.length) % tables.length;
        updateTableViewer();
    });
    document.getElementById('nextTable').addEventListener('click', () => {
        tableIdx = (tableIdx + 1) % tables.length;
        updateTableViewer();
    });

    // Kick off a run when the user presses the button. Configuration data is
    // sent to the backend as JSON and the optional file upload is included as
    // multipart form data.
    document.getElementById('runBtn').addEventListener('click', async () => {
        const useUpload = document.getElementById('uploadRadio').checked;
        const body = {
            model: useUpload ? null : document.getElementById('modelSelect').value,
            sne: document.getElementById('sneSelect').value,
            bao: document.getElementById('baoSelect').value,
            cmb: document.getElementById('cmbSelect').value,
        };
        const file = useUpload ? document.getElementById('modelFile').files[0] : null;
        const formData = new FormData();
        formData.append('config', JSON.stringify(body));
        if (file) formData.append('file', file);

        const resp = await fetch('/api/run', { method: 'POST', body: formData });
        const result = await resp.json();

        document.getElementById('console').textContent = result.console;
        document.getElementById('downloadZip').href = result.zip;

        plots = result.plots;
        tables = result.tables;
        logFile = result.log;
        plotIdx = 0;
        tableIdx = 0;
        updatePlotViewer();
        updateTableViewer();

        if (logFile) {
            fetch(`/api/file/${logFile}`).then(r => r.text()).then(t => {
                document.getElementById('logText').textContent = t;
                document.getElementById('downloadLog').href = `/api/file/${logFile}`;
            });
        }

        document.getElementById('results').classList.remove('hidden');
    });
});
