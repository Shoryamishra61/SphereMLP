# Current Phase
T03 — Implement Mathematical Primitives

# Status
COMPLETE

# Changes
- Trusted reference frozen at `0f98d8f`.
- Added metered normal PDF/CDF, univariate ReLU moments, exact zero-mean pair moments, approximate general pair moments, correlation construction, and stable chi mean.
- Added explicit deterministic and equal-variable boundary branches.

# Tests
- Focused moment suite: 7 passed after fixing one missing test import, one invalid expected constant, and the equal-variable limit.
- Full suite: 18 passed in 6.41s.
- Ruff: passed.
- Official validator: passed in 62 ms.
- Univariate maximum error: 0.0 on the differential fixture.
- General bivariate maximum fixture error: 0.0009279188.

# Metrics
- Runtime estimator remains unchanged until the primitive gate passes.

# Acceptance Criteria
- [x] All fixtures return finite values.
- [x] Variances remain non-negative within tolerance.
- [x] Bivariate diagonal, swap symmetry, arc-cosine, and trusted-reference checks pass.
- [x] Correlation and chi-mean primitives pass boundary checks.
- [x] Official validator remains passing.

# Risks
- A general bivariate-normal CDF may require a documented approximation if flopscope has no supported primitive.

# Next Task
T04 — Implement Scalar Propagation Fallback
