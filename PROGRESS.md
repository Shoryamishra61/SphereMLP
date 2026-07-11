# Current Phase
T04 — Implement Scalar Propagation Fallback

# Status
COMPLETE

# Changes
- Moment primitives frozen at `7362bfd`.
- Added exact-first-layer scalar propagation, finite-state downgrade, output guard, budget preflight, and retained emergency result.
- Lazy-loaded runtime modules to satisfy the official subprocess setup cap on this Windows host.

# Tests
- Full suite: 26 passed in 5.91s; Ruff passed.
- Official validator: passed in 45 ms.
- Official Mini local: 100/100, zero failures, zero fallbacks.
- Official Mini subprocess with `--max-threads 1`: 100/100, zero failures.
- Scalar local P95: 526.0004 ms; max: 918.1094 ms; mean FLOPs: 12,427,710.

# Metrics
- Raw final MSE: 0.0009482214922900312.
- Adjusted score: 0.00009482214922900313.
- All-layer MSE: 0.000815381417341996.
- Mean compute ratio: 0.012301507566875909 local; 0.01190949 subprocess.
- Peak traced memory: 2.0216 MB on a 32x256 synthetic profile.

# Acceptance Criteria
- [x] Official validator passes.
- [x] Mini local and subprocess have zero failures.
- [x] Scalar runtime is below one second locally.
- [x] Compute use and score are recorded.
- [x] Valid scalar fallback is always retained before optional work.

# Risks
- Cold Windows subprocess startup is close to the official five-second setup cap; retain lazy imports and one-thread subprocess checks.

# Next Task
T05 — Implement Full Covariance Propagation
