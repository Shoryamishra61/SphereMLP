# ARC White-Box Estimation Challenge 2026

This repository implements a safe, white-box estimator for the ARC WhestBench
challenge. The current shipped entry point is [estimator.py](estimator.py): it
uses exact-first-layer scalar Gaussian propagation and returns a validated
non-negative result for every call. More ambitious methods are deliberately
kept feature-gated until they clear paired accuracy, compute, and reliability
gates.

Current phase: **T08 — sampling study**. The task specification and acceptance
gates are in `implementation/AGENTS_EXECUTION_READY.md`; experiments and
decisions are recorded in `manifests/` and `results/`.

## Method status

| Method | Status | What it does | Decision |
|---|---|---|---|
| Zero matrix | Safety sentinel | Free finite output retained before any optional work. | Kept solely as emergency fallback. |
| Exact-first-layer scalar propagation | **Shipped default** | Computes first-layer pre-activation variances exactly from weight-column norms; propagates later layers with diagonal Gaussian ReLU moments. | Kept: low compute, deterministic, and zero validation failures. |
| Full-covariance Gaussian propagation | Development-only | Tracks full layer covariance, uses exact first zero-mean pair moments, and a fixed-quadrature approximation for general bivariate ReLU moments. | Rejected as default: accurate but a 483.31-second validation tail and an over-budget case violate runtime gates. |
| Gaussian Monte Carlo | Benchmark comparator | Seeded IID normal inputs are passed through the full MLP. | Not shipped: useful reference and strong validation score, but sampling selection is still gated. |
| Spherical Rao–Blackwellized sampling | Feature-gated | Samples normalized Gaussian directions and integrates the Gaussian radius exactly using the chi mean. | T07 implementation complete; T08 must select or reject it on paired validation data. |
| Fusion, K3 correction, routing, learned calibration | Not implemented | Later stages of the specified dependency order. | Blocked until deterministic/sampling/compute gates are complete. |

## Implemented approach

### 1. Official contract adapter

The official MLP convention is row-wise activation propagation:

\[
x_{\ell+1} = \operatorname{ReLU}(x_\ell W_\ell).
\]

Weights are therefore input-by-output matrices; propagation of a mean uses
`W.T @ mean`, while first-layer variances use the squared column norms. The
runtime entry point is `Estimator.predict(mlp, budget)`, and randomized
methods derive their local generator seed only from `mlp.seed`.

### 2. Exact-first-layer scalar Gaussian propagation

For standard-normal input, the first pre-activation marginal variance is

\[
v_{1,j} = \sum_i W_{ij}^2.
\]

The ReLU mean for a Gaussian pre-activation \(Z\sim\mathcal N(\mu,v)\) is

\[
\mathbb E[\max(0,Z)] = \sigma\phi(\mu/\sigma)+\mu\Phi(\mu/\sigma),
\quad \sigma=\sqrt v.
\]

Later scalar layers use the diagonal closure
\(v_{\ell+1}= (W_\ell^2)^\top\operatorname{Var}[x_\ell]\). This is the
current production estimator because it has a small compute footprint and a
validated fallback path.

### 3. Full covariance Gaussian propagation

The development covariance branch computes

\[
\Sigma_z = W^\top\Sigma_x W,
\]

then transforms joint Gaussian pairs through ReLU. The first layer uses the
closed-form zero-mean arc-cosine kernel. Later non-zero-mean pairs use a
24-point Gauss–Legendre integration of Plackett's identity because the
installed `flopscope` provides univariate but not bivariate normal CDFs.

The branch is numerically guarded: symmetry is restored, marginal diagonals
are restored exactly, non-finite states or significant negative variances
raise a safe downgrade, and no eigendecomposition runs in the estimator.

It improved the frozen validation score relative to scalar propagation, but it
is not enabled at runtime because its maximum measured wall time was unsafe.

### 4. Gaussian and spherical sampling

The Gaussian comparator estimates every layer by forwarding seeded IID normal
inputs. The spherical method instead decomposes a standard Gaussian as
\(X=RU\), where \(U\) is uniform on the unit sphere and \(R\) has a chi
distribution. Zero-bias ReLU MLPs are positively homogeneous, so

\[
\mathbb E[h(X)] = \mathbb E[R] \; \mathbb E_U[h(U)].
\]

The spherical implementation batches only directions, forwards the entire MLP
for each active batch, multiplies outputs by exact \(\mathbb E[R]\), and
keeps only layerwise first and second accumulators. It supports IID and
antithetic directions, deterministic seeds, configurable batch/sample counts,
and per-output standard errors.

On a small equal-forward-evaluation analytic fixture, spherical sampling had
mean MSE `2.68086e-4` versus `3.30074e-4` for Gaussian MC. This is evidence for
continuing the study, not a held-out performance claim.

## Reliability and compute safety

Every runtime candidate follows the same retention rule:

1. Start from a free valid zero result.
2. Compute the scalar candidate.
3. Replace the retained result only if shape, dtype, finiteness, and
   non-negativity checks succeed.
4. Optional branches may never replace the retained result until they pass the
   same checks.

The current scalar runtime uses a conservative budget reserve so the official
low-budget validator can still receive a free valid output. Frozen numerical
values are `sigma_epsilon=1e-12`, `correlation_epsilon=1e-7`, and
`variance_clip_tolerance=1e-8`. The safety target is a maximum effective
compute ratio of `0.85`, P95 runtime below 45 seconds, and a hard 60-second
per-call limit.

## Measured baseline results

Frozen Full validation contains 150 public MLPs, split at the network level.

| Estimator | Final MSE | Adjusted score | Mean compute ratio | P95 wall time | Failures |
|---|---:|---:|---:|---:|---:|
| Exact scalar (shipped) | 1.02718e-3 | 1.02718e-4 | 0.01280 | 587.76 ms | 0 |
| Full covariance | 7.89507e-5 | 3.17649e-5 | 0.40455 | 17.43 s | 0 |
| Gaussian MC, 1,536 samples | 4.09377e-5 | 4.09377e-6 | 0.03720 | 447.66 ms | 0 |

The covariance maximum wall time was 483.31 seconds and its maximum compute
ratio was 1.12473; these are why it is excluded from the runtime default.

## Reproducibility

Use the official starter-kit environment on Windows:

```powershell
$env:PYTHONUTF8 = '1'
.\whest-starterkit\.venv\Scripts\python.exe -m pytest -q
.\whest-starterkit\.venv\Scripts\whest.exe validate --estimator estimator.py
```

The current suite has 43 passing tests. Key reproducibility files:

- `manifests/environment.json` — installed package/runtime versions.
- `manifests/dataset_splits.json` — immutable network-level partition.
- `manifests/experiments.csv` — experiment ledger.
- `manifests/decisions.md` — evidence-backed implementation decisions.
- `manifests/rejected_methods.md` — methods excluded from shipping and why.
- `results/raw/` and `results/summaries/` — benchmark captures and paired
  bootstrap comparisons.

## Repository map

```text
estimator.py                 Official runtime entry point
whest_solution/scalar.py     Shipped scalar estimator
whest_solution/covariance.py Development covariance branch
whest_solution/sampling.py   Feature-gated spherical sampler
whest_solution/moments.py    Metered Gaussian/ReLU moment primitives
tests/                       Contract, numerical, safety, and sampling tests
experiments/                 Development-only benchmark/profile utilities
manifests/                   Frozen splits, decisions, and experiment ledger
results/                     Raw captures, summaries, and plots
```
