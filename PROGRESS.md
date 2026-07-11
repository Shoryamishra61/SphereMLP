# Current Phase
T05 — Implement Full Covariance Propagation

# Status
COMPLETE

# Changes
- Scalar fallback frozen at `1c0f725`.
- Added exact first-layer joint moments, full Gaussian covariance closure, diagnostics, numerical guards, and scalar downgrade.
- Added declared-symmetry handling without runtime eigendecomposition.

# Tests
- Full suite with warnings as errors: 34 passed in 6.00s.
- Ruff: passed; official validator: passed in 23 ms.
- First 10 Mini networks: zero failures/fallbacks.
- Width-12 deep MC: covariance MSE 0.0014062 versus scalar 0.0052655.

# Metrics
- Raw final MSE: 0.00006518351365230046.
- Adjusted score: 0.00002709981734230616.
- All-layer MSE: 0.000042479519834159873.
- Mean/max compute ratio: 0.3984743 / 0.4937434.
- P95/max wall: 9.930 s / 9.930 s; peak traced memory: 96.4 MB.

# Acceptance Criteria
- [x] Covariance tests and independent differentials pass.
- [x] Zero runtime failures and scalar fallback remains available.
- [x] Covariance improves paired validation adjusted score over scalar.
- [x] Full compute/time profile and bivariate-backend error are recorded.
- [x] No unsupported heavy operation or unconditional eigendecomposition is used.

# Risks
- Vectorized 24-node pair integration may be compute/time heavy at width 256 and must be profiled before retention.

# Next Task
T06 — Freeze the Deterministic Baseline Ladder
