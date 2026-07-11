# Rejected Methods

Methods are appended only after paired compute-adjusted evidence is available.

## T06 — Full covariance as a shipped runtime default

- Evidence: 150-network frozen Full validation, canonical paired result in `results/summaries/t06_covariance_vs_scalar.json`.
- Accuracy: favorable versus exact-first scalar (mean adjusted-score delta `-7.09533e-5`; 95% paired bootstrap CI `[-8.23008e-5, -6.04043e-5]`).
- Operational rejection: P95 wall time `17.426 s`, maximum `483.310 s`, and maximum effective-compute ratio `1.12473`; it breaches the 60-second per-call and hard-budget requirements.
- Decision: retain source and unit coverage for future bounded/profiled experiments, but exclude it from the official runtime path until a later candidate satisfies the T10 compute and tail gates.

## T06 — Installed official scalar baseline

- Evidence: 150/150 frozen Full validation calls failed output validation in `results/raw/t06_validation.csv`.
- Decision: do not ship or use as a parent beyond this audit. The project exact-first-layer scalar remains the safe scalar baseline.
