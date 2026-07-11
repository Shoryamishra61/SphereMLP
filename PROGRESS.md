# Current Phase
T07 — Implement Spherical Rao-Blackwellized Sampling

# Status
IN_PROGRESS

# Changes
- T06 complete at checkpoint `5b6681c`; exact-first scalar remains the default runtime path.
- Implementing only the feature-gated spherical sampling primitive and its required differential tests.

# Tests
- Analytic one-layer, homogeneity, deterministic seed, batch invariance, antithetic marginal, and option-validation tests pass.
- Equal-forward fixture and full-shape compute/memory profile are archived.

# Metrics
- Fixture variance ratio (spherical/Gaussian) is `0.54735` at 1,024 forward evaluations; full-shape 64-sample compute ratio is `0.01695`.

# Acceptance Criteria
- [x] Spherical estimator matches analytic one-layer means.
- [x] Positive homogeneity, seed determinism, and batch invariance are tested.
- [x] Equal-forward-pass variance comparison against Gaussian MC is measured.
- [x] Effective compute accounting and memory bound are validated.

# Risks
- Spherical sampling may not reduce variance for all network geometries; no improvement claim may be made before T08 paired evidence.

# Next Task
T08 — Execute the Sampling Study
