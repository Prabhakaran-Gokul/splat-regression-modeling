# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository implements **Splat Regression Models (SRM)** — a function approximator parameterized as a weighted sum of transformed probability density functions:

```
f(x) = Σ_j V[j] · ρ_{A[j], B[j]}(x)
     where ρ_{A,B}(x) = det(A)^{-1} · ρ(A^{-1}(x − B))
```

Parameters `(V, A, B)` represent weights `[k, p]`, scale/rotation matrices `[k, d, d]`, and centers `[k, d]`. The default mother function `ρ` is the standard multivariate Gaussian. The framework compares SRM against MLPs and KANs on regression and physics-informed learning tasks.

## Current focus: Eikonal motion planning with splats (torus / robot arm)

The active line of work applies SRM to **optimal motion planning as an Eikonal value function**
`T(θ)` (time-to-go), whose optimal paths are `−∇T`. Target: a 2-joint arm's configuration space is
the **2-torus**; obstacles are a smooth **slowness field** (high cost near collision). Read
[`theory.md`](theory.md) (why + method derivation, §9), [`investigation.md`](investigation.md)
(running log + **Executive summary** at top), and [`validation_highd.md`](validation_highd.md)
(certifying a learned field without ground truth).

**Files (this line of work):**
- `torus.py` — **main file.** Self-supervised Eikonal on the 2-torus with obstacles. Splats placed
  *intrinsically* (angle space, wrapped log-map `wrap(θ−B)`), metric kept separate via
  `metric_inv(θ)` (identity now, `M(θ)⁻¹` for the arm). Methods (`--method`): `eikonal`
  (`base·(1+g)` + residual²), `ntfields` (`base/τ` + symmetric speed-match loss + causal +
  progressive + optional RRT* anchors), `roadmap` (RRT* soft-min base + Eikonal refinement, optional
  anchors), `supervised` (oracle: fit to FMM), `screened` (Hopf–Cole; explored, not used). FMM
  (`fast_marching_torus`) is the 2D ground truth only.
- `ground_truth.py` — `PlanningProblem` (plane/sphere), analytic + FMM ground truth, `draw_field`.
- `train.py` — supervised splat fit to a known field (plane/sphere milestone).
- `self_supervised.py` — plane-with-obstacles self-supervised Eikonal + feasibility-guarded path
  extraction (`extract_path`, collision-checked descent — the collision *guarantee*).
- `scenes.py` — SDF obstacle primitives (circle/box, unions), smooth slowness, surface samples.
- `diagnose.py` — near-obstacle zoom (τ field, `‖∇T‖/s` ratio, splat centres).
- `run_baselines.sh` — the fair baseline suite (B1..B6). `refine_experiment.py` /
  `thinning_experiment.py` — the RRT*-prior refinement and sparsity-scaling studies.

**Key findings (see investigation.md Executive summary for numbers):** the splat *representation* is
not the bottleneck (oracle ≈ 0.003); pure physics-informed under-determines the level behind
obstacles (~0.31); a **rough mesh-free RRT\* prior that the Eikonal refines** is the win (0.11 at ~300
samples) and *beats trusting the planner's values as anchors* (which is worse than no supervision,
because RRT* costs are suboptimal). Refinement degrades gracefully but not immunely as samples thin
(`~nodes^-0.67`). Retired: dense-RRT* base (cheating), screened-Poisson, antipodal sampling.

**Run examples:**
```bash
python torus.py --method roadmap --base_reg 3 --rrt_iters 350 --roadmap_gamma 0.01   # prior + refine
python torus.py --method ntfields --causal                                            # planner-free (P-NTFields-like)
python torus.py --method supervised                                                   # oracle (representation ceiling)
zsh run_baselines.sh                                                                   # full fair suite → figures/baselines/
```

**Next (planned):** fixed-budget dimension sweep (2→3→4-D, the true scaling test), then the
**anisotropic metric** (`metric_inv → M(θ)⁻¹`) + joint limits toward the 12-DOF arm goal.

## Stack

- **JAX** for all numerics and autodiff
- **Flax NNX** for MLP/KAN models
- **jaxkan** for KAN models (`jaxkan.KAN`)
- **optax** for optimizers (Adam/SGD)
- GPU configuration via `jax.config.update('jax_platform_name', 'gpu')`

## Running scripts

There is no build system or test runner; scripts are executed directly. For the **current
motion-planning line**, see the run examples in "Current focus" above (`python torus.py --method …`,
`zsh run_baselines.sh`).

The original SRM **regression / KAN comparison and physics-informed exploration** scripts
(`regression_comparison.py`, `physinf_comparison.py`, `dubins_*.py`, `sphere_*.py`, `eikonal_*.py`,
etc.) predate this line and have been **moved to `_archive/preexisting/`** — they are kept for
reference but are not part of the active codebase. Restore any by moving it back to the top level.

### Core splat library: `srms/lib/splat.py` (still used)

- `eval_splat(X, splatnn, rho=None)` — forward pass; `splatnn = (V, A, B)`; `@jax.jit`. Imported by
  `train.py`/`self_supervised.py` (the plane/sphere milestones). `torus.py` uses its own periodic
  `eval_splat_torus`; the `srms/methods/backends/srm.py` backend instead builds on
  `srms/lib/manifold_splat.py`'s generic `eval_wrapped_gaussian`.
- `eval_splat_grad`, `gd_splat_regression`, `splat_anim_1d/2d` — analytic-gradient regression and demo
  helpers from the original framework.
- `srms/lib/manifold_splat.py` — generic wrapped-Gaussian evaluator (`eval_wrapped_gaussian`) plus
  sphere and SE(2) log-map/Jacobian implementations; moved here (was `lib/`) as part of the
  ongoing `srms/` reorganization.

## Data conventions

- Inputs `X`: `[n, d]` float32
- Outputs `Y`: `[n, p]` float32
- Splat parameters: `V: [k, p]`, `A: [k, d, d]`, `B: [k, d]`
- `train_mask=(1,1,1)` controls which of `(V, A, B)` are updated during gradient descent
