import json
import numpy as np

VERSION = '2.1.1'

def run(model_path: str) -> str:
    with open(model_path, 'r') as f:
        model = json.load(f)
    model_name = model.get('model_name', 'Unnamed Model')
    return f"Copernican Suite JS {VERSION} loaded model: {model_name}\nNumpy {np.__version__} available"
