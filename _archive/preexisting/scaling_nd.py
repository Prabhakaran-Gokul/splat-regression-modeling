#!/usr/bin/env python3
"""
Dimension scaling study: d-D uniform-speed Eikonal (u = ‖x‖).

Three collocation strategies compared across d ∈ {2, 3, 4, 5, 6, 8}:
  static-8k   — 8 000 fixed points, reused every step           (full-A)
  diag-static — 8 000 fixed points, reused every step           (diagonal-A)
  dynamic     — 1 000 fresh points sampled each step            (full-A)
                  → total unique coverage ~ 1000 × 5000 = 5M pts

Metrics recorded per run:
  - Canonical 2-D slice MSE  (x_3=…=x_d=0 vs √(x₁²+x₂²))
  - PDE residual L²  on a fresh hold-out batch of 4 000 points
  - Training wall time

Output: scaling_nd.png
"""

import time

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

jax.config.update("jax_platform_name", "gpu")
jax.config.update("jax_enable_x64", False)

from eikonal_splat import (
    source_sphere_nd,
    _uniform_collocation_nd,
    _canonical_slice_mse,
    _pde_residual_nd,
    init_splat_params,
    init_splat_params_diag,
    train_eikonal_splat,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DIMS        = [2, 3, 4, 5, 6, 8]
K           = 200
N_INT       = 8000     # static strategy
N_BATCH     = 1000     # dynamic strategy — points per step
N_HOLDOUT   = 4000     # fixed hold-out set for final PDE residual eval
NUM_STEPS   = 5000
LR          = 5e-3
EPS         = 0.10
N_SPHERE    = 200
SEED        = 42
SPEED_FN    = lambda x: jnp.ones((x.shape[0], 1))


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def one_run(key, d, use_diag, dynamic):
    domain = [(-1.0, 1.0)] * d
    x_src  = jnp.zeros(d)

    src_pts, src_vals = source_sphere_nd(x_src, EPS, N_SPHERE, c_at_src=1.0)

    # Fixed hold-out batch for consistent final evaluation
    key, sk = jr.split(key)
    holdout_pts = _uniform_collocation_nd(sk, N_HOLDOUT, domain,
                                          exclude_center=[0.] * d, min_dist=EPS)

    key, sk = jr.split(key)
    if use_diag:
        params = init_splat_params_diag(sk, K, d, domain, scale=0.25)
    else:
        params = init_splat_params(sk, K, d, domain, scale=0.25)

    if dynamic:
        # Closure captures domain / d; key advances inside the training loop
        def resample_fn(sk):
            return _uniform_collocation_nd(sk, N_BATCH, domain,
                                           exclude_center=[0.] * d, min_dist=EPS)
        key, train_key = jr.split(key)
        # interior_pts is ignored when resample_fn is provided
        interior_pts = holdout_pts   # placeholder (not used)
    else:
        key, sk = jr.split(key)
        interior_pts = _uniform_collocation_nd(sk, N_INT, domain,
                                               exclude_center=[0.] * d, min_dist=EPS)
        resample_fn = None
        train_key   = None

    t0 = time.time()
    params, _ = train_eikonal_splat(
        params, interior_pts, src_pts, src_vals, SPEED_FN,
        num_steps=NUM_STEPS, lr=LR, physics_weight=1.0,
        log_interval=999_999, diag=use_diag,
        resample_fn=resample_fn, key=train_key,
    )
    elapsed = time.time() - t0

    slice_mse, _, _, _ = _canonical_slice_mse(params, domain, diag=use_diag)
    pde_res = _pde_residual_nd(params, holdout_pts, SPEED_FN, diag=use_diag)

    return slice_mse, pde_res, elapsed


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

# (label, use_diag, dynamic)
strategies = [
    ("static-8k (full-A)",   False, False),
    ("static-8k (diag-A)",   True,  False),
    ("dynamic-1k (full-A)",  False, True),
]

records = {label: [] for label, *_ in strategies}

for d in DIMS:
    print(f"\n── d={d} ──────────────────────────────")
    base_key = jr.PRNGKey(SEED)
    for label, use_diag, dynamic in strategies:
        key, sk = jr.split(base_key)
        slice_mse, pde_res, elapsed = one_run(sk, d, use_diag, dynamic)
        records[label].append(
            dict(d=d, slice_mse=slice_mse, pde_res=pde_res, elapsed=elapsed)
        )
        print(f"  {label:28s}  slice_MSE={slice_mse:.3e}"
              f"  PDE={pde_res:.3e}  {elapsed:.1f}s")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

styles = {
    "static-8k (full-A)":  ("o-",  "C0", "Static 8k  (full-A)"),
    "static-8k (diag-A)":  ("s--", "C1", "Static 8k  (diag-A)"),
    "dynamic-1k (full-A)": ("^-",  "C2", "Dynamic 1k/step  (full-A)"),
}

for label, (style, color, legend_label) in styles.items():
    recs       = records[label]
    ds         = [r["d"]         for r in recs]
    slice_mses = [r["slice_mse"] for r in recs]
    pde_ress   = [r["pde_res"]   for r in recs]
    times      = [r["elapsed"]   for r in recs]

    axes[0].semilogy(ds, slice_mses, style, color=color, lw=2, markersize=7,
                     label=legend_label)
    axes[1].semilogy(ds, pde_ress,   style, color=color, lw=2, markersize=7,
                     label=legend_label)
    axes[2].semilogy(ds, times,      style, color=color, lw=2, markersize=7,
                     label=legend_label)

for ax, ylabel, title in [
    (axes[0], "Canonical slice MSE  (vs $\\sqrt{x_1^2+x_2^2}$)", "Accuracy: slice MSE vs dimension"),
    (axes[1], "PDE residual $L^2$  (hold-out)",                  "Accuracy: PDE residual vs dimension"),
    (axes[2], "Wall time  (s)",                                   "Training time vs dimension"),
]:
    ax.set_xlabel("Dimension  $d$", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.set_xticks(DIMS)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

fig.suptitle(
    f"Eikonal PINN dimension scaling  —  k={K}, steps={NUM_STEPS}\n"
    f"Static: n_int={N_INT} fixed pts/step  |  Dynamic: {N_BATCH} fresh pts/step"
    f"  (~{N_BATCH*NUM_STEPS//1000}k unique pts total)",
    fontsize=12,
)
plt.tight_layout()
plt.savefig("scaling_nd.png", dpi=150, bbox_inches="tight")
print("\nSaved scaling_nd.png")
plt.show()
