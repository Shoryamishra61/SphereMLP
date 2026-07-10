# Decisions

## T00 — Official contract recovery

- Authority: installed `whestbench==0.12.0rc3`, installed `flopscope==0.8.0rc5`, and starter-kit commit `c99ef4a` override project notes.
- Entrypoint: module `estimator.py`, class `Estimator`, subclassing `whestbench.BaseEstimator`.
- Required call: bound `predict(mlp, budget)`; source declaration is `predict(self, mlp: MLP, budget: int) -> flopscope.numpy.ndarray`.
- Optional lifecycle: `setup(context: SetupContext)` once before prediction and `teardown()` once after all predictions.
- MLP representation: frozen `whestbench.domain.MLP` with `width`, `depth`, ordered `weights`, `seed`, and `name`. Every weight has shape `(width, width)`.
- Weight orientation: official simulation executes sample-row activations as `x @ w`; matrix rows are input neurons and columns are output neurons. Analytical propagation must therefore use `w.T @ mu` and `(w * w).T @ var`. Proven by an asymmetric differential fixture in `tests/test_contract.py`.
- Seeds: use `mlp.seed` for predict-time randomness and `SetupContext.seed` for setup-time randomness. Runtime logic must not use `mlp.name` or public identifiers.
- Budget: `budget` is a Python `int` matching the surrounding `BudgetContext` cap. Default phase-1 value is `272_000_000_000` effective-FLOP units per MLP.
- Accounting: `C = F + 1e11 * R`, where `F` is analytical flopscope FLOPs and `R` is residual wall time in seconds. A valid run uses multiplier `max(0.1, C / B)`; any failure forces multiplier `1.0`. Combined exhaustion is strict `C > B`.
- FLOP context: `predict` runs inside `flopscope.BudgetContext`; an over-budget counted operation raises before execution. Array construction, reshape/transpose/indexing/stack are free; pointwise operations and reductions are element-counted; matrix operations use dimension-dependent analytical costs; RNG sampling uses calibrated per-element costs.
- Output: must be a finite `flopscope.numpy.ndarray` of shape `(mlp.depth, mlp.width)`. T00 additionally enforces floating dtype and non-negativity as project safety invariants.
- Runner: local mode imports and calls the estimator in-process. Subprocess mode serializes width, depth, weights, and seed to a worker and returns prediction plus measured budget statistics. The default per-call wall limit is 60 seconds.
- Validator: current command is `whest validate --estimator estimator.py`; it probes width 4, depth 2, calls setup/predict/teardown, and checks class resolution, shape, and finiteness.
- CLI on this host: invoke with `PYTHONUTF8=1` (and optionally `PYTHONIOENCODING=utf-8`) because Rich output cannot be encoded by the default CP1252 console.
- Packaging: `whest package --estimator <file-or-folder> --output <archive>`. A file ships only that file (renamed to `estimator.py`); a folder ships non-ignored files. The grader installs no third-party packages beyond its provided WhestBench/flopscope environment.
- Existing work: the nested starter kit had a pre-existing modified `pyproject.toml`; T00 does not alter or revert it.
- Minimal T00 adapter: retain a deterministic float32 zero matrix. It is intentionally only a contract-valid checkpoint; scalar propagation begins in dependency order at T04.
- Repository state: the workspace-root `.git` directory is not a valid Git repository, so T00 cannot record a root commit. The two authoritative nested repositories remain intact; no commit was attempted.
- Test portability: the untouched starter suite produced 13 passes and 4 failures. All failures are README command-fence tests whose harness hard-requires `/bin/bash`; this host has no WSL bash. The failures are recorded, not patched, because they do not reflect estimator behavior.

## T01 — Repository and manifest infrastructure

- Initialized the previously empty root `.git` directory as the implementation repository. The supplied `whest-starterkit`, `flopscope`, implementation-source, and PDF assets remain untouched and are ignored by the implementation repository.
- Pinned only the two grader-provided runtime APIs within their proven compatibility ranges: `flopscope>=0.8.0rc5,<0.9.0` and `whestbench>=0.12.0rc3,<0.13.0`. SciPy is not a runtime dependency.
- Defined the full immutable `RuntimeConfig` schema without creating a production default. Numerical and compute values remain unfrozen until their required profiling tasks.
- Configuration identity is SHA-256 over canonical sorted JSON with NaN rejected. Tests prove determinism, immutability, and sensitivity to field changes.
- Created runtime module boundaries and development entry points as import-safe placeholders only. No advanced estimator was implemented ahead of its dependency gate.
- Folder packaging excludes development code, manifests, local references, research assets, and task documentation while retaining `estimator.py` and `whest_solution/`.
- The host had no Git author identity. Checkpoints use repository-local `Codex <codex@local>` metadata; global Git configuration is unchanged.

## T02 — Trusted small-dimension reference

- SciPy is absent, so the trusted bivariate reference uses explicit tensor-product Gauss-Hermite numerical integration with independent standard-normal coordinates. It is development-only and excluded from packaging.
- Reference propagation follows the proven `x @ w` convention, uses float64 NumPy, explicit pair loops, exact univariate Gaussian ReLU moments, and fixed RNG seeds.
- Gauss-Hermite convergence around the ReLU kink is empirically characterized rather than called exact. At order 160 its maximum error against the zero-mean arc-cosine formula on the tested correlations is `7.4715e-4`.
- The installed optimized mean-propagation baseline matches the independent scalar reference to `1e-10` absolute/relative tolerance on a fixed synthetic network.
- First-layer reference covariance is checked against 500,000-sample Monte Carlo; maximum matrix error is `1.7946e-3`.

## T03 — Metered mathematical primitives

- flopscope 0.8.0rc5 provides metered univariate normal PDF/CDF but no bivariate or multivariate CDF.
- The runtime general bivariate CDF is explicitly approximate: 24-point fixed Gauss-Legendre integration of Plackett's identity, implemented entirely with supported flopscope primitives. It must be compute-profiled before full-covariance retention.
- The zero-mean bivariate moment remains the closed-form arc-cosine expression and supports exact `rho=-1,0,1` limits.
- The general pair primitive uses explicit near-deterministic branches, an exact equal-variable diagonal branch, interior correlation clipping, symmetry, and non-negativity guarding.
- Runtime univariate primitives match the independent float64 reference exactly on the tested values; general bivariate maximum error against the independent order-200 Gauss-Hermite fixtures is `9.2792e-4`.
- Chi mean uses scalar `math.lgamma` in log space. Its residual-time cost is negligible but will still enter effective-compute profiling when spherical sampling is evaluated.
