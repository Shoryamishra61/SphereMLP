# Current Phase
T08 — Sampling Study (COMPLETE)

# Status
DONE

# Changes
- T07 complete at checkpoint `2478fc9`; spherical sampling feature-gated.
- T08: Integrated spherical Rao-Blackwellized sampling into `estimator.py` as primary estimator.
- T08 reopened after remote-score evidence; comparing randomized-LHS spherical directions against IID under the same forward-pass count.
- Selected randomized-LHS directions at equal N=4,096 after paired Mini and independent-randomization checks; modular and single-file paths both preserve scalar fallback/output guards.
- Scalar propagation retained as validated fallback (safety contract).
- Budget-adaptive sample count: 17,408 samples at batch_size=512.
- Calibrated from actual Mini runner measurements (effective ≈ 2.5× analytical FLOPs).
- 49,152 samples REJECTED: runner overhead inflated effective compute to 189% (budget bust).
- Added 4 new integration tests (test_contract.py): budget tiering, spherical activation, determinism.

# Tests
- 53/53 tests pass, including deterministic randomized-LHS and invalid-option coverage.
- Official validator: all checks OK.
- Mini local: 100/100 MLPs scored, 0 failures.

# Metrics
| Config | Adjusted Score | Raw MSE | Compute Ratio | Failures |
|---|---:|---:|---:|---:|
| Scalar (T06 baseline) | 9.48e-05 | 1.03e-03 | 1.3% | 0/100 |
| Spherical N=3,584 | 1.98e-06 | 1.46e-05 | 13.4% | 0/100 |
| **Spherical N=17,408** | **1.82e-06** | **2.68e-06** | **67.5%** | **0/100** |
| **LHS spherical N=4,096** | **5.44e-07** | **5.29e-06** | **10.29%** | **0/4** |

Improvement over scalar baseline: **52×** reduction in adjusted score.

# Acceptance Criteria
- [x] Candidate sample counts compared on Mini split.
- [x] N=17,408 / batch_size=512 selected as runtime default.
- [x] N=49,152 rejected (budget bust) — recorded in rejection ledger.
- [x] No over-budget configuration shipped.
- [x] Randomized directional design measured over four independent randomizations and selected only after favorable paired evidence.

# Risks
- Runner overhead (IPC, weight loading) causes effective compute ≈ 2.5× analytical.
- Conservative 30% analytical budget fraction used to guarantee safety.

# Next Task
T09 — Implement fixed and layerwise scalar/sampling fusion, retaining sampling unless a held-out blend is favorable.
