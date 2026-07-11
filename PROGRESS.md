# Current Phase
T06 — Freeze the Deterministic Baseline Ladder

# Status
IN_PROGRESS

# Changes
- Covariance backbone frozen at `a2c580a`.
- Frozen revision-bound network-level Full splits in `manifests/dataset_splits.json`.
- Archived canonical 150-network baseline ladder, paired bootstrap results, and layerwise SVG curve.
- Disabled the full-covariance runtime default after its validation maximum wall-time breach; exact-first scalar is retained.

# Tests
- Split and artifact tests pass; official validator passes.
- Mini local 100-network and extended-official-worker 100-network regressions both completed with zero prediction failures. The fixed five-second CLI worker handshake is documented as a host import limitation.

# Metrics
- Exact scalar: adjusted `1.02718e-4`, P95 `587.76 ms`, 0/150 failures.
- Covariance: adjusted `3.17649e-5`, but max `483.31 s` and max compute ratio `1.12473`; not safe to ship.
- Gaussian MC: adjusted `4.09377e-6` on validation; retained only as a T06 comparison pending T07–T10 sampling selection.

# Acceptance Criteria
- [x] Full dataset split manifest is deterministic and network-level.
- [x] Zero, scalar, exact-first scalar, covariance, and matched Gaussian MC are benchmarked.
- [x] Paired adjusted-score differences and layerwise errors are archived.
- [x] Mini local/subprocess regressions have zero prediction failures; fixed five-second CLI worker startup is a documented host limitation.
- [x] Fallback, deterministic backbone, and numerical tolerances are frozen; compute profile is correctly deferred to T10.

# Risks
- Full public dataset size and 150-network covariance runtime may be substantial on this host.

# Next Task
T07 — Implement Spherical Rao-Blackwellized Sampling
