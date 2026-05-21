# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository implements **Splat Regression Models (SRM)** — a function approximator parameterized as a weighted sum of transformed probability density functions:

```
f(x) = Σ_j V[j] · ρ_{A[j], B[j]}(x)
     where ρ_{A,B}(x) = det(A)^{-1} · ρ(A^{-1}(x − B))
```

Parameters `(V, A, B)` represent weights `[k, p]`, scale/rotation matrices `[k, d, d]`, and centers `[k, d]`. The default mother function `ρ` is the standard multivariate Gaussian. The framework compares SRM against MLPs and KANs on regression and physics-informed learning tasks.

## Stack

- **JAX** for all numerics and autodiff
- **Flax NNX** for MLP/KAN models
- **jaxkan** for KAN models (`jaxkan.KAN`)
- **optax** for optimizers (Adam/SGD)
- GPU configuration via `jax.config.update('jax_platform_name', 'gpu')`

## Running scripts

There is no build system or test runner. Scripts are executed directly:

```bash
# Run the main regression comparison (trains SRM vs KAN vs MLP)
python regression_comparison.py --seed 42 --name experiment1

# Load and plot a saved snapshot
python regression_comparison.py --load logs/regression_*.pkl

# Run physics-informed comparison
python physinf_comparison.py --mode example

# Run CGLS (Complex Ginzburg-Landau) experiment
python physinf_comparison.py --mode cgls

# Run the splat demo (1D and 2D animations, saves .gif/.mp4)
python lib/splat.py

# Test GPU availability
python simple_gpu_test.py
```

Results are saved as `.pkl` snapshots in `logs/` and figures as `.png`/`.pdf`.

## Architecture

### Core model: `lib/splat.py`

- `eval_splat(X, splatnn, rho=None)` — forward pass; `splatnn = (V, A, B)`; `@jax.jit`
- `eval_splat_grad(splatnn, X, Y, variation, ...)` — analytic gradient of the MSE loss w.r.t. `(V, A, B)` using the custom derivation from the paper; does **not** use `jax.grad` (implements the score function estimator)
- `gd_splat_regression(init_splat, train_X, train_Y, ...)` — training loop returning a list of `(V, A, B)` at each step; supports Adam via optax
- `splat_anim_1d` / `splat_anim_2d` — matplotlib animation helpers for notebooks

### Comparison models: `lib/nets.py`

- `gd_net_regression(model, ...)` — trains any `nnx.Module` (MLP or KAN) with the same interface

### Experiment infrastructure: `regression_comparison.py`, `physinf_comparison.py`

- `run_regression_comparison(...)` / `run_pinn_comparison(...)` — orchestrate multi-architecture experiments
- Results are serialized with `pickle` and can be reloaded for plotting without retraining
- `physinf_comparison.py` defines `PDEProblem` subclasses: `PoissonProblem`, `AdvectionDiffusionProblem`, `AllenCahnProblem`, `BurgersProblem`, `ComplexGinzbergLandauProblem`
- PINN loss = boundary loss + `physics_weight` × PDE residual MSE; gradients computed via `jax.grad`/`jax.hessian`

### Known missing code

- `physinf_comparison.py` imports `from v2.lib.splat import ...` and `cgls_solver` — these modules are not present in this directory; `lib/splat.py` is the current version to use
- `lib/test_identification.py` also references `v2.lib.nets`

## Physics-Informed Use (Eikonal)

The Eikonal equation constraint is `|∇u(x)| = 1/c(x)` where `c(x)` is the speed field. The canonical approach in this codebase is:

1. Define a `PDEProblem`-style class with `pde_residual` encoding `|∇u(x)|² − 1/c(x)² = 0`
2. Compute `∇u` via `jax.grad(lambda x_single: eval_splat(x_single[None,:], params)[0,0])`
3. Use `jax.value_and_grad(loss_fn)(curr_splat)` to differentiate the PINN loss through `eval_splat` w.r.t. `(V, A, B)` — this is the **autodiff path**, not `eval_splat_grad` which is specific to least-squares regression
4. Update with `optax.adam` as shown in `train_and_evaluate_splat_pinn`

The `compute_pinn_loss` and `compute_derivatives` functions in `physinf_comparison.py` are the reference implementation for wiring up derivative constraints.

## Data conventions

- Inputs `X`: `[n, d]` float32
- Outputs `Y`: `[n, p]` float32
- Splat parameters: `V: [k, p]`, `A: [k, d, d]`, `B: [k, d]`
- `train_mask=(1,1,1)` controls which of `(V, A, B)` are updated during gradient descent
