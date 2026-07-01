from pathlib import Path
import json
p = Path(r'd:\UQ_benchmarking\adaptive_sampling\SHIFTWING_surrogate_optimization.ipynb')
with p.open('r', encoding='utf-8') as f:
    data = json.load(f)
for i, cell in enumerate(data['cells']):
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        if 'def predict_mean' in src:
            print('CELL', i)
            print(src)
            break
