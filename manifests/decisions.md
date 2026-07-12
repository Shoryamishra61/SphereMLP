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

## T04 — Scalar fallback

- The first layer is implemented directly from input-by-output column norms, making its marginal means exact. Later layers use the diagonal Gaussian closure.
- The top-level estimator retains a free float32 zero matrix before attempting scalar propagation. Tiny budgets below a conservative reserve return it without entering a counted operation; this preserves the official validator's budget-100 probe.
- Scalar and guard modules are lazy-imported inside `predict`. This reduced Windows subprocess startup enough to pass the fixed five-second setup cap when combined with `--max-threads 1`.
- Mini local and subprocess both completed 100/100 networks with zero failures and zero scalar fallbacks.
- Official Mini scalar raw final MSE is `9.482214922900312e-4`; adjusted score is `9.482214922900313e-5`; all-layer MSE is `8.15381417341996e-4`.
- The paired zero parent has raw final MSE `0.909290142506361` and adjusted score `0.09092901425063608`. Scalar is retained.
- Local JSON profiling measured mean analytical FLOPs `12,427,710`, P95 estimator wall `526.0004 ms`, max `918.1094 ms`, and zero failures.
- Runner caveat: `whest run --runner subprocess --estimator .` treats the directory as a module file and fails. Runtime testing uses `--estimator estimator.py`; folder mode remains reserved for `whest package`.
- Host caveat: cold worker imports varied around the fixed 5-second cap. Failed setup attempts are environment/runner startup failures before estimator setup; the final warmed one-thread 100-network subprocess run passed.

## T05 — Full-covariance Gaussian backbone

- The first joint layer uses the exact zero-mean arc-cosine moment with `W.T @ W` under the proven input-by-output orientation.
- Later layers use full linear covariance propagation and the T03 approximate nonzero-mean bivariate backend. Covariance is explicitly symmetrized, exact marginal diagonals are restored, and significant negative diagonal or non-finite state raises a safe downgrade.
- No runtime eigendecomposition or PSD projection is used. Eigenspectra appear only in small development tests.
- Flopscope symmetry warnings are suppressed only for intentional diagnostic reductions and quadrature broadcasting whose numerical symmetry is subsequently restored and declared.
- On the first 10 Mini networks, covariance reduces paired raw final MSE from `9.320093100541271e-4` to `6.518351365230046e-5` and adjusted score from `9.320093100541271e-5` to `2.709981734230616e-5`. It is retained for T06.
- Mean analytical FLOPs are `3,744,274,276`; measured mean compute ratio is `0.39847`, P95 estimator wall is `9.930 s`, and peak traced memory is `96.4 MB`.
- A width-12, depth-3 fixed-seed MC fixture shows covariance MSE `0.0014062` versus scalar `0.0052655`, while max absolute covariance error `0.1331` records the expected narrow finite-width closure limitation.

## T06 — Deterministic baseline ladder

- The Full development-validation split is frozen in `manifests/dataset_splits.json`: 1,000 public Full MLP identifiers partitioned by revision-bound SHA-256 into 700 train, 150 validation, 75 calibration, and 75 untouched internal holdout IDs. The split is by network, never by samples or layers.
- Benchmark capture is canonicalized with a predeclared first-observation rule after an interrupted/resumed process overlap created duplicate records. The unmodified append-only capture is retained as `results/raw/t06_validation_interrupted_capture.csv`; the canonical file has exactly 750 rows (150 MLPs × 5 estimators), all within the frozen validation partition.
- Exact-first-layer scalar is the official fallback and the T06 deterministic runtime backbone. It has zero validation failures, mean adjusted score `1.02718e-4`, and P95 wall time `587.76 ms`.
- Full covariance statistically improves over exact scalar (mean paired adjusted-score delta `-7.09533e-5`, 95% bootstrap CI `[-8.23008e-5, -6.04043e-5]`, 97.33% wins) but is rejected as a runtime default: its maximum observed wall time is `483.31 s`, far beyond the 60-second official per-call limit, and its maximum effective-compute ratio is `1.12473`.
- Matched seeded Gaussian MC also improves over both scalar and covariance on this validation split. It is a benchmark comparator only at T06; sampling geometry/profile selection remains gated behind T07–T10.
- Frozen T06 safety values: scalar-only runtime default; no covariance default; scalar reserve remains `32*d*w*w + 128*d*w`; `sigma_epsilon=1e-12`, `correlation_epsilon=1e-7`, `variance_clip_tolerance=1e-8`; safe maximum effective-compute headroom is `0.85` of the official budget; soft runtime target `45 s` and hard limit `60 s`. A new sampling compute profile is deferred to T10 because it depends on the selected spherical sampler.
- Mini runner regression: local run over all 100 Mini MLPs passed with zero failures (`results/raw/t06_mini_local.json`). The official CLI subprocess command cannot meet this host's hardcoded 5-second setup handshake because importing the installed `whestbench.subprocess_worker` alone measured `6.44–8.94 s`, before user code is loaded. The same official `SubprocessRunner`, with a host-only 15-second setup allowance and unchanged 30-second prediction / 60-second wall limits, completed all 100 Mini MLPs with zero failures (`results/raw/t06_mini_subprocess_extended.json`).

## T07 — Spherical Rao-Blackwellized sampler

- The sampler writes a Gaussian input as `X=R U`, draws only `U` from locally seeded normalized Gaussian directions, and multiplies layer outputs by the exact scalar `E[R]` from `chi_mean(width)`. This is valid because the challenge MLP has zero biases and ReLU is positively homogeneous for nonnegative radius.
- The implementation never retains a direction matrix beyond the active batch and keeps only `O(depth * width)` first/second-sum accumulators. It returns all-layer predictions plus directional standard errors.
- A 32-seed, equal-1,024-forward-evaluation width-4 one-layer analytic fixture measured spherical mean MSE `2.68086e-4` versus Gaussian MC `3.30074e-4`; the MSE-across-randomizations variance ratio was `0.54735`. This is a fixture measurement only, not a retained validation-performance claim.
- Full-shape synthetic profile at 64 samples / batch 16: `270,909,504` analytical FLOPs, effective-compute ratio `0.01695`, `409.79 ms`, active traced allocation `1.99 MB`, finite non-negative output. T08 determines actual geometry/profile selection.

## T08 — Submission radial-factor correction

- The Phase 1 submission that scored `0.2143` was packaged before the spherical implementation was corrected. It multiplied `E[R]` inside every ReLU layer. Positive homogeneity applies to the complete zero-bias network, so the correct estimator is `E[R] * E_U[h(U)]`, with the radial factor applied exactly once after all layerwise directional means have been accumulated.
- Direct official Mini differential check after the correction (`daniel-harrison`, `N=17,408`, batch 512) produced final-layer MSE `8.04191e-7`, all-layer MSE `6.60625e-6`, and finite/non-negative output. This establishes the submitted low score as a pre-fix artifact, not expected behavior of the corrected implementation.
- A depth-two identity-ReLU analytic regression now enforces the once-only radial factor. Reapplying it per layer fails this test.

## T08 — Portable submission manifest repair

- WhestBench `0.12.0rc3` on Windows generated tar members with POSIX paths but wrote Windows backslashes in `manifest.json`. The Phase 1 grader reported `MANIFEST_INVALID` because it compared those names literally.
- `tools/repair_submission_manifest.py` normalizes only manifest file names to POSIX separators and verifies each declared member and SHA-256 digest against the repaired tar archive. It does not alter estimator payload bytes.

## T08 — Single-file submission hardening

- The fixed portable folder archive reproduces correctly in an isolated local extraction, yet the remote graded result remained at `0.2153`, indistinguishable from the failed-pre-fix behavior. To remove any remaining remote subpackage-resolution ambiguity, `submission/estimator.py` vendors the audited scalar fallback and corrected spherical method into a single file.
- The single-file archive declares only `estimator.py`; its manifest has no nested paths. Local direct Mini evaluation matches the modular estimator exactly (`8.04191e-7` final MSE on the checked MLP), and the official validator passes.

## T08 — Final-score compute profile

- The official objective scores final-layer MSE, while all-layer MSE is diagnostic only. The public leading result has final MSE `8.16e-7`, compute ratio `0.0994`, and all-layer MSE `0.7537`, demonstrating a score-focused final-layer profile.
- A paired 10-Mini-MPL IID spherical study selected `N=4,096`, batch `512`: mean final MSE `5.14979e-6`, mean compute ratio `0.0952`, maximum ratio `0.0992`, and adjusted score `5.14979e-7`. Higher sampled profiles crossed the 10% multiplier floor and were worse on this slice.
- Output rows before the final layer are now deterministic zeros in the selected profile. This contains no MLP-ID or split-specific behavior and preserves the full forward pass required to estimate the final layer.
- Metered orthogonal blocks (`N=4,096`) were rejected: mean final MSE `9.89934e-6`, adjusted score `9.89934e-7`, and maximum compute ratio `0.1028`.

## T08 — Remote candidate-retention correction

- Submission `315975` charged the selected sampler (`1.72e10` effective compute) but returned an all-zero prediction: all-layer MSE `0.7755` and final MSE `0.6997`. This is a retained-candidate failure, not a sampling-accuracy result.
- The runtime previously used strict `isinstance(value, fnp.ndarray)` validation before retaining a candidate. It is now replaced by coercion to a metered array, explicit shape validation, and deterministic finite/non-negative sanitization. A successful candidate no longer depends on a particular remote wrapper type identity.
- The single-file output assembly also replaced `final[None, :]` with `fnp.reshape(final, (1, width))`. Remote flopscope arrays can differ from the local wrapper in advanced-indexing support; reshape is explicitly metered and avoids a post-computation exception that would otherwise be silently downgraded to zero.

## T08 — Randomized-LHS spherical directions

- A randomized Latin-hypercube design is generated in Gaussian quantile space, with an independent permutation and continuous jitter per input coordinate, then row-normalized.  Each direction is marginally uniform on the sphere; the resulting estimator is therefore unbiased under the same radial identity as IID spherical sampling.  Dependence across rows means its benefit is empirical, not assumed.
- On the paired first four Mini MLPs at 4,096 full forward evaluations, LHS reduced mean raw final MSE from `7.35751e-6` (IID) to `5.28534e-6` and adjusted score from `7.35751e-7` to `5.43816e-7`.  Mean effective-compute ratio was `0.10289`; maximum was `0.10524` in the local profile.
- On one fixed Mini MLP across four independent randomization seeds, LHS reduced mean final MSE from `7.80436e-6` to `4.04783e-6`; MSE-across-randomizations variance fell from `1.04766e-11` to `5.48971e-12`.  The direct metered full-submission check returned finite shape `(32, 256)`, final MSE `1.38559e-6`, and effective ratio `0.08757`.
- Decision: retain LHS as the selected T08 directional design.  It remains covered by scalar fallback and output validation.  The final sample count is a T10 compute-frontier decision, not inferred from this geometry experiment.
