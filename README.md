# ARC White-Box Estimation Challenge 2026

Runtime entry point: `estimator.py`.

The authoritative implementation order and acceptance gates are maintained in
`implementation/AGENTS_EXECUTION_READY.md`. Reproducibility state lives under
`manifests/`; raw benchmark evidence lives under `results/raw/`.

Validated local commands use the official starter-kit environment:

```powershell
$env:PYTHONUTF8 = '1'
.\whest-starterkit\.venv\Scripts\python.exe -m pytest -q
.\whest-starterkit\.venv\Scripts\whest.exe validate --estimator estimator.py
```
