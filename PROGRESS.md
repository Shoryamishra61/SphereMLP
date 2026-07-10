# Current Phase
T02 — Build a Trusted Small-Dimension Reference

# Status
COMPLETE

# Changes
- T00/T01 checkpoint retained at `0a623af`.
- Added an intentionally slow development-only NumPy oracle for forward passes, Monte Carlo, scalar closure, and full covariance closure.
- Added independent Gauss-Hermite integration for general bivariate ReLU moments.

# Tests
- `pytest -q tests/test_reference_mc.py`: 6 passed in 12.40s.
- First-layer mean versus 400k MC max error: 0.0003410864.
- Arc-cosine versus order-160 quadrature max error: 0.0007471478.
- General independent-pair identity error: 0.0001875935.
- First-layer covariance versus 500k MC max matrix error: 0.0017945632.
- Full suite: 11 passed; Ruff passed; official validator passed in 56 ms.

# Metrics
- Runtime estimator remains unchanged; T02 reference code is excluded from packaging.

# Acceptance Criteria
- [x] Analytical primitive errors meet recorded tolerances.
- [x] Reference code remains small-dimension, slow, and auditable.
- [x] All stochastic reference tests use fixed seeds.
- [x] Runtime validator remains unchanged and passing.

# Risks
- Gauss-Hermite convergence is algebraic around the ReLU kink; tolerances must be measured rather than guessed.
- NumPy is development-only here; runtime paths continue to use flopscope primitives.

# Next Task
T03 — Implement Mathematical Primitives
