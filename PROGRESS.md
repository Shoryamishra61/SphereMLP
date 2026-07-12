# Current Phase
T10 — Compute Frontier Optimization (IN_PROGRESS)

# Status
DONE

# Changes
- T07 complete at checkpoint `2478fc9`; spherical sampling feature-gated.
- T08: Integrated spherical Rao-Blackwellized sampling into `estimator.py` as primary estimator.
- T08 reopened after remote-score evidence; comparing randomized-LHS spherical directions against IID under the same forward-pass count.
- Tested randomized-LHS directions at equal N=4,096.  The completed paired ten-Mini run rejected it on adjusted score; IID remains selected.  Modular and single-file paths preserve scalar fallback/output guards.
- T09 fixed scalar/spherical fusion implemented and rejected before calibration: on the initial paired Mini gate, every nontrivial blend increased raw final MSE and the optimum was the spherical endpoint.
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
| IID spherical N=4,096 (10 Mini) | **5.15e-07** | **5.15e-06** | **9.52%** | **0/10** |
| LHS spherical N=4,096 (10 Mini, rejected) | 5.57e-07 | 5.08e-06 | 10.97% | 0/10 |

Improvement over scalar baseline: **52×** reduction in adjusted score.

# Acceptance Criteria
- [x] Candidate sample counts compared on Mini split.
- [x] N=17,408 / batch_size=512 selected as runtime default.
- [x] N=49,152 rejected (budget bust) — recorded in rejection ledger.
- [x] No over-budget configuration shipped.
- [x] Randomized directional design measured over independent randomizations and rejected after unfavorable paired compute-adjusted evidence on ten Mini MLPs.

# Risks
- Runner overhead (IPC, weight loading) causes effective compute ≈ 2.5× analytical.
- Conservative 30% analytical budget fraction used to guarantee safety.

# Next Task
T10 — Optimize the compute frontier using only the retained IID spherical estimator.
