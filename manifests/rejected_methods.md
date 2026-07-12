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

## T08 — Spherical sampling N=49,152, batch_size=512

- Evidence: Mini local run (100 MLPs), first 2 MLPs immediately busted budget.
  - MLP 0: C_m=515,276,365,680 > B=272,000,000,000 (189% effective compute).
  - MLP 1: C_m=457,112,351,665 > B=272,000,000,000 (168% effective compute).
- Root cause: runner IPC + flopscope overhead inflates effective compute to ~2.5× analytical.
  At N=49,152 (wall ~10s), residual charge ≈ 3.1s × 1e11 ≈ 307B extra FLOPs per MLP.
- Decision: rejected as default; reduced to N=17,408 which yields 67.5% effective ratio.

## T08 — Spherical sampling N=17,408, batch_size=256 (batch size 256)

- Evidence: Mini local run (100 MLPs), mean compute ratio 13.4%, adjusted score 1.98e-06.
- Rejection: batch_size=256 incurred higher residual overhead than batch_size=512.
- Decision: superseded by batch_size=512 which reduces overhead; not a correctness failure.

## T09 — Antithetic pairs at N=17,408, batch_size=512

- Evidence: 5 trials each, IID vs antithetic.
  - IID mean standard error: 0.00190842
  - Antithetic mean standard error: 0.00190924  (virtually identical, slightly *worse*)
- Root cause: deep ReLU nonlinearities destroy the U/−U correlation within 2-3 layers.
  After sufficient depth, the antithetic pair is no more correlated than two IID samples.
- Decision: rejected. `antithetic=False` (IID) remains the selection.

## T08 — Orthogonal spherical blocks at N=4,096

- Evidence: paired 10-Mini-MPL study in `results/raw/t08_final_profile_mini10.json`.
- Mean final MSE `9.89934e-6` versus IID `5.14979e-6`; maximum compute ratio `0.1028` exceeds the 10% score-floor target.
- Decision: rejected; IID spherical directions are retained for the final-layer profile.

## T08 — Randomized Latin-hypercube spherical directions at N=4,096

- Evidence: paired ten-Mini profile in `results/raw/t08_final_profile_mini10.json`, with four independent randomizations additionally recorded in `results/raw/t08_rqmc_randomizations.json`.
- Raw final MSE was marginally lower than IID (`5.08114e-6` versus `5.14979e-6`), but mean effective compute ratio rose from `0.09522` to `0.10970`, P95 wall time from `856.99 ms` to `1647.45 ms`, and adjusted score worsened from `5.14979e-7` to `5.57383e-7`.
- Decision: reject as a shipped method.  Preserve only as an experimental implementation and retain IID spherical directions for T09/T10.
