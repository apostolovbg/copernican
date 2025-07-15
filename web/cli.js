const version = '2.1.1';
document.getElementById('version').textContent = version;
let pyodideReadyPromise = loadPyodide({indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.23.4/full/'});

async function runCopernican(file) {
  const pyodide = await pyodideReadyPromise;
  const output = document.getElementById('output');
  output.textContent = 'Initializing Pyodide...\n';
  await pyodide.loadPackage(['numpy']);
  const data = await file.text();
  pyodide.FS.writeFile('model.json', data);
  const script = await (await fetch('copernican_web.py')).text();
  pyodide.runPython(script);
  const result = pyodide.runPython('run("model.json")');
  output.textContent += result + '\n';
}

document.getElementById('runBtn').addEventListener('click', async () => {
  const fileInput = document.getElementById('modelFile');
  if (fileInput.files.length === 0) {
    alert('Please select a cosmo_model_*.json file.');
    return;
  }
  runCopernican(fileInput.files[0]);
});
