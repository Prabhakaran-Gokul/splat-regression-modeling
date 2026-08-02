#!/usr/bin/env python3
"""
Ablation study: 3-D uniform-speed Eikonal (u = ‖x‖).

Sweeps each of {k, n_int, num_steps} independently while holding the other
two fixed at the baseline, then plots mean-slice MSE vs each hyperparameter.

Baseline: k=200, n_int=2000, steps=5000
"""

import time

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

jax.config.update("jax_platform_name", "gpu")
jax.config.update("jax_enable_x64", False)

from eikonal_splat import (
    source_sphere_3d,
    _uniform_collocation_3d,
    _eval_slices_3d,
    init_splat_params,
    train_eikonal_splat,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DOMAIN     = [(-1.0, 1.0)] * 3
EPS        = 0.10
N_SPHERE   = 100
SPEED_FN   = lambda x: jnp.ones((x.shape[0], 1))
LR         = 5e-3
SEED       = 42

# Baseline values used as fixed axes during each sweep
BASE_K     = 200
BASE_NINT  = 2000
BASE_STEPS = 5000

# Sweep grids
K_VALUES     = [50, 100, 200, 300, 500, 800]
NINT_VALUES  = [250, 500, 1000, 2000, 4000, 8000]
STEPS_VALUES = [500, 1000, 2000, 5000, 10000, 15000]


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def mean_slice_mse(params):
    """Mean MSE over the three midplane slices vs the analytical u=‖x‖."""
    slices = _eval_slices_3d(params, DOMAIN, Ng=60)
    mses = []
    for sname in ("xy", "xz", "yz"):
        u_pred    = slices[sname][0]
        _, Gh, Gv = slices[sname]
        u_true    = jnp.sqrt(Gh ** 2 + Gv ** 2)
        mses.append(float(jnp.mean((u_pred - u_true) ** 2)))
    return sum(mses) / 3


def one_run(key, k, n_int, num_steps):
    """Train one configuration and return (mean_slice_mse, wall_time_s)."""
    x_src = jnp.zeros(3)
    src_pts, src_vals = source_sphere_3d(x_src, EPS, N_SPHERE, c_at_src=1.0)

    key, sk = jr.split(key)
    int_pts = _uniform_collocation_3d(sk, n_int, DOMAIN,
                                      exclude_center=[0., 0., 0.],
                                      min_dist=EPS)

    key, sk = jr.split(key)
    params = init_splat_params(sk, k, 3, DOMAIN, scale=0.25)

    t0 = time.time()
    params, _ = train_eikonal_splat(
        params, int_pts, src_pts, src_vals, SPEED_FN,
        num_steps=num_steps, lr=LR, physics_weight=1.0,
        log_interval=999_999,
    )
    elapsed = time.time() - t0

    mse = mean_slice_mse(params)
    return mse, elapsed


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------

def sweep(label, values, k_fn, nint_fn, steps_fn):
    """Run a single sweep, printing progress."""
    print(f"\n{'='*60}")
    print(f"Sweep: {label}")
    print(f"{'='*60}")
    mses, times = [], []
    base_key = jr.PRNGKey(SEED)
    for v in values:
        k_, ni_, st_ = k_fn(v), nint_fn(v), steps_fn(v)
        key, sk = jr.split(base_key)
        mse, t = one_run(sk, k_, ni_, st_)
        mses.append(mse)
        times.append(t)
        print(f"  {label}={v:6d}  MSE={mse:.4e}  ({t:.1f}s)")
    return mses, times


k_mses, k_times = sweep(
    "k", K_VALUES,
    k_fn=lambda v: v,
    nint_fn=lambda _: BASE_NINT,
    steps_fn=lambda _: BASE_STEPS,
)

nint_mses, nint_times = sweep(
    "n_int", NINT_VALUES,
    k_fn=lambda _: BASE_K,
    nint_fn=lambda v: v,
    steps_fn=lambda _: BASE_STEPS,
)

steps_mses, steps_times = sweep(
    "steps", STEPS_VALUES,
    k_fn=lambda _: BASE_K,
    nint_fn=lambda _: BASE_NINT,
    steps_fn=lambda v: v,
)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

sweep_data = [
    (axes[0], K_VALUES,     k_mses,     k_times,
     "Number of splats  $k$",    f"n_int={BASE_NINT}, steps={BASE_STEPS}"),
    (axes[1], NINT_VALUES,  nint_mses,  nint_times,
     "Collocation points  $n_{int}$", f"k={BASE_K}, steps={BASE_STEPS}"),
    (axes[2], STEPS_VALUES, steps_mses, steps_times,
     "Training steps",             f"k={BASE_K}, n_int={BASE_NINT}"),
]

for ax, xs, mses, times, xlabel, subtitle in sweep_data:
    ax.loglog(xs, mses, "o-", lw=2, markersize=7)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Mean slice MSE  (vs analytical $\\|x\\|$)", fontsize=10)
    ax.set_title(subtitle, fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    # annotate wall-clock times
    for x, mse, t in zip(xs, mses, times):
        ax.annotate(f"{t:.0f}s", (x, mse),
                    textcoords="offset points", xytext=(4, 4), fontsize=7, color="grey")

fig.suptitle(
    "3-D Eikonal PINN ablation  —  effect of k, n_int, steps on slice MSE\n"
    f"(baseline: k={BASE_K}, n_int={BASE_NINT}, steps={BASE_STEPS},  seed={SEED})",
    fontsize=12,
)
plt.tight_layout()
plt.savefig("ablation_3d.png", dpi=150, bbox_inches="tight")
print("\nSaved ablation_3d.png")
plt.show()
