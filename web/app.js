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

document.addEventListener('DOMContentLoaded', async () => {
    const models = await fetchData('/api/models');
    renderList('modelList', models, 'model');
    const data = await fetchData('/api/datasets');
    renderList('sneList', data.sne, 'sne');
    renderList('baoList', data.bao, 'bao');
    renderList('cmbList', data.cmb, 'cmb');

    document.querySelectorAll('.tab').forEach(btn => btn.addEventListener('click', switchTab));

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
        const blob = await resp.blob();
        document.getElementById('downloadZip').href = URL.createObjectURL(blob);
        document.getElementById('results').classList.remove('hidden');
    });
});
