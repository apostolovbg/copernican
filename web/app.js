async function fetchData(url) {
    const resp = await fetch(url);
    return resp.json();
}

function renderList(containerId, items, name) {
    const div = document.getElementById(containerId);
    items.forEach((item, idx) => {
        const id = `${name}-${idx}`;
        const lbl = document.createElement('label');
        lbl.innerHTML = `<input type="radio" name="${name}" value="${item}"> ${item}`;
        div.appendChild(lbl);
        div.appendChild(document.createElement('br'));
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
    renderList('modelList', models, 'model');
    const data = await fetchData('/api/datasets');
    renderList('sneList', data.sne, 'sne');
    renderList('baoList', data.bao, 'bao');
    renderList('cmbList', data.cmb, 'cmb');

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

    document.getElementById('runBtn').addEventListener('click', async () => {
        const body = {
            model: document.querySelector('input[name="model"]:checked')?.value,
            sne: document.querySelector('input[name="sne"]:checked')?.value,
            bao: document.querySelector('input[name="bao"]:checked')?.value,
            cmb: document.querySelector('input[name="cmb"]:checked')?.value,
        };
        const file = document.getElementById('modelFile').files[0];
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
