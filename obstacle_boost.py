#!/usr/bin/env python3
"""
High-accuracy SRM for the 2D obstacle-field Eikonal.

Baseline (from commit 4b5977e):
  k=200, n_int=2000 (static), steps=3000  →  MSE vs FMM ≈ 2.34e-3

Boost strategy:
  • k=800   (4× more splats, 16,800 parameters in 2D)
  • Dynamic collocation: 4000 free-space pts resampled every step
  • 15 000 steps  (5× longer)
  • FMM reference on N=400 grid (4× denser than baseline)

Results cached to logs/obstacle_boost_results.pkl.
Output figure: figures/obstacle_boost.png
MLflow experiment: eikonal-srm  (run name: eikonal-srm-obstacle-boost)
"""

import os
import pickle
import time

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.interpolate import RegularGridInterpolator

jax.config.update("jax_platform_name", "gpu")
jax.config.update("jax_enable_x64", False)

from eikonal_splat import (
    _fmm_reference_2d,
    _eval_grid_2d,
    _pde_residual_nd,
    source_ring_2d,
    init_splat_params,
    train_eikonal_splat,
)

# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------

MLFLOW_URI        = "sqlite:////home/tassos/.local/share/mlflow/runs.db"
MLFLOW_EXPERIMENT = "eikonal-srm"
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED      = 42
CACHE_FILE = "logs/obstacle_boost_results.pkl"
os.makedirs("logs",    exist_ok=True)
os.makedirs("figures", exist_ok=True)

# Obstacle geometry (same as baseline)
DOMAIN   = [(-1.0, 1.0), (-1.0, 1.0)]
X_SRC    = jnp.array([-0.7, 0.0])
X_SRC_NP = np.array([-0.7, 0.0])
EPS      = 0.08
N_RING   = 128      # dense BC ring for better anchoring (baseline: 32)
OBS_SPEED = 0.05
OBS = [
    dict(x=(-0.1, 0.55), y=( 0.25,  0.85)),   # upper bar
    dict(x=(-0.1, 0.55), y=(-0.85, -0.25)),    # lower bar
]

# Boost hyperparameters
K         = 800
N_BATCH   = 4000   # per step, dynamically resampled
NUM_STEPS = 30000  # 10× baseline; loss still declining at 15k
LR        = 5e-3
FMM_N     = 200    # matches baseline reference resolution

# Baseline reference (from commit 4b5977e)
BASELINE = dict(k=200, n_int=2000, num_steps=3000, grid_mse=2.34e-3)


# ---------------------------------------------------------------------------
# Obstacle helpers
# ---------------------------------------------------------------------------

def _in_obs_jax(x):
    """Boolean mask: which rows of x are inside an obstacle? (JAX-traceable)"""
    mask = jnp.zeros(x.shape[0], dtype=bool)
    for o in OBS:
        mask = mask | (
            (x[:, 0] >= o['x'][0]) & (x[:, 0] <= o['x'][1]) &
            (x[:, 1] >= o['y'][0]) & (x[:, 1] <= o['y'][1])
        )
    return mask


def _in_obs_np(x):
    """Same mask for NumPy arrays."""
    mask = np.zeros(len(x), dtype=bool)
    for o in OBS:
        mask = mask | (
            (x[:, 0] >= o['x'][0]) & (x[:, 0] <= o['x'][1]) &
            (x[:, 1] >= o['y'][0]) & (x[:, 1] <= o['y'][1])
        )
    return mask


def speed_fn_jax(x):
    """JAX-traceable speed: c=1 in free space, c=0.05 inside obstacles."""
    return jnp.where(_in_obs_jax(x)[:, None], OBS_SPEED, 1.0)


def speed_fn_np(x):
    """NumPy speed for FMM."""
    return np.where(_in_obs_np(np.asarray(x, dtype=np.float32)), OBS_SPEED, 1.0)


# ---------------------------------------------------------------------------
# Dynamic collocation sampler
# ---------------------------------------------------------------------------

lo = jnp.array([b[0] for b in DOMAIN])
hi = jnp.array([b[1] for b in DOMAIN])


def make_resample_fn(n_batch, x_src, eps, oversample=8):
    """
    Return a resample_fn(key) → [n_batch, 2] of free-space collocation points.

    Samples n_batch × oversample candidates, keeps those outside obstacles
    and outside the source exclusion radius.  Falls back to whatever is
    available if fewer than n_batch pass the filter.
    """
    def resample_fn(key):
        cands   = jr.uniform(key, (n_batch * oversample, 2)) * (hi - lo) + lo
        r_src   = jnp.linalg.norm(cands - x_src, axis=-1)
        is_free = ~_in_obs_jax(cands) & (r_src > eps)
        pts     = cands[is_free][:n_batch]
        return pts
    return resample_fn


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_boost(key):
    print(f"\n{'='*60}")
    print(f"Obstacle-field Eikonal boost")
    print(f"  k={K}  n_batch={N_BATCH}/step  steps={NUM_STEPS}  lr={LR}")
    print(f"{'='*60}")

    src_pts, src_vals = source_ring_2d(X_SRC, EPS, N_RING, c_at_src=1.0)
    print(f"  Source ring: {len(src_pts)} BC pts  (eps={EPS})")

    # Build one holdout set for evaluation (not used in training)
    key, sk = jr.split(key)
    resample_fn = make_resample_fn(N_BATCH, X_SRC, EPS)
    holdout_pts = resample_fn(sk)
    print(f"  Holdout collocation: {len(holdout_pts)} pts")

    key, sk = jr.split(key)
    params   = init_splat_params(sk, K, 2, DOMAIN, scale=0.25)
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"  SRM parameters: {n_params:,}")

    key, train_key = jr.split(key)
    t0 = time.time()
    params, history = train_eikonal_splat(
        params, holdout_pts, src_pts, src_vals, speed_fn_jax,
        num_steps=NUM_STEPS, lr=LR, physics_weight=2.0,
        log_interval=3000,
        resample_fn=resample_fn, key=train_key,
    )
    elapsed = time.time() - t0
    print(f"  Training done in {elapsed:.1f}s")

    # FMM reference
    print(f"  Computing FMM reference (N={FMM_N})…")
    u_fmm, Xf, Yf = _fmm_reference_2d(X_SRC_NP, EPS, speed_fn_np, DOMAIN, N=FMM_N)

    xf = np.linspace(DOMAIN[0][0], DOMAIN[0][1], FMM_N)
    yf = np.linspace(DOMAIN[1][0], DOMAIN[1][1], FMM_N)
    fmm_interp = RegularGridInterpolator(
        (yf, xf), u_fmm, method="linear", bounds_error=False, fill_value=np.nan,
    )

    # Evaluate on a dense grid
    Ng = 200
    u_srm, X1, X2 = _eval_grid_2d(params, DOMAIN, Ng=Ng)
    u_srm  = np.array(u_srm)
    X1_np  = np.array(X1)
    X2_np  = np.array(X2)

    pts_interp = np.stack([X2_np.ravel(), X1_np.ravel()], axis=1)   # (y, x) for interp
    u_ref      = fmm_interp(pts_interp).reshape(Ng, Ng)

    pts_grid   = np.stack([X1_np.ravel(), X2_np.ravel()], axis=1)   # (x, y) for obs
    mask_free  = ~_in_obs_np(pts_grid).reshape(Ng, Ng)

    valid    = mask_free & np.isfinite(u_ref) & np.isfinite(u_srm)
    grid_mse = float(np.mean((u_srm[valid] - u_ref[valid]) ** 2))
    max_err  = float(np.max(np.abs(u_srm[valid] - u_ref[valid])))

    # PDE residual on holdout
    pde_res = _pde_residual_nd(params, holdout_pts, speed_fn_jax)

    print(f"  Grid MSE vs FMM : {grid_mse:.4e}  "
          f"(baseline {BASELINE['grid_mse']:.2e}, "
          f"improvement {BASELINE['grid_mse']/grid_mse:.1f}×)")
    print(f"  Max error       : {max_err:.4e}")
    print(f"  PDE residual L² : {pde_res:.4e}")

    return dict(
        k=K, n_batch=N_BATCH, num_steps=NUM_STEPS, lr=LR,
        params=params, history=history,
        grid_mse=grid_mse, max_err=max_err, pde_res=pde_res,
        elapsed=elapsed, n_params=n_params,
        u_srm=u_srm, u_ref=u_ref,
        X1=X1_np, X2=X2_np, mask_free=mask_free, valid=valid,
        Ng=Ng,
    )


# ---------------------------------------------------------------------------
# Train / load cache
# ---------------------------------------------------------------------------

if os.path.exists(CACHE_FILE):
    print(f"Loading cached results from {CACHE_FILE}")
    with open(CACHE_FILE, "rb") as f:
        r = pickle.load(f)
else:
    key = jr.PRNGKey(SEED)
    r   = train_boost(key)
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(r, f)
    print(f"Saved results to {CACHE_FILE}")


# ---------------------------------------------------------------------------
# MLflow logging
# ---------------------------------------------------------------------------

print("\nLogging to MLflow …")
with mlflow.start_run(run_name="eikonal-srm-obstacle-boost"):
    mlflow.log_params({
        "dim":         2,
        "k":           r["k"],
        "n_batch":     r["n_batch"],
        "num_steps":   r["num_steps"],
        "lr":          r["lr"],
        "n_params":    r["n_params"],
        "n_ring":        N_RING,
        "eps":           EPS,
        "fmm_n":         FMM_N,
        "physics_weight": 2.0,
        "collocation":   "dynamic",
        "A_mode":        "full",
        "run_type":      "boost",
        "obstacle":    "two_bars_gap0.25",
        "x_src":       "(-0.7, 0.0)",
    })
    for step, h in enumerate(r["history"]):
        mlflow.log_metrics({
            "loss/total": h["total"],
            "loss/bc":    h["bc"],
            "loss/pde":   h["pde"],
        }, step=step)
    mlflow.log_metrics({
        "final/grid_mse":        r["grid_mse"],
        "final/max_err":         r["max_err"],
        "final/pde_res":         r["pde_res"],
        "final/wall_time_s":     r["elapsed"],
        "baseline/grid_mse":     BASELINE["grid_mse"],
        "improvement/grid_mse":  BASELINE["grid_mse"] / r["grid_mse"],
    })
    print("  Metrics logged.")

    # ---------------------------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------------------------

    u_srm    = r["u_srm"]
    u_ref    = r["u_ref"]
    X1_np    = r["X1"]
    X2_np    = r["X2"]
    mask_free = r["mask_free"]
    valid    = r["valid"]
    hist     = r["history"]
    Ng       = r["Ng"]

    improvement  = BASELINE["grid_mse"] / r["grid_mse"]
    vmax         = min(float(np.nanpercentile(u_ref[mask_free], 96)), 2.5)
    levels       = np.linspace(float(EPS) + 0.05, vmax, 16)
    extent       = [DOMAIN[0][0], DOMAIN[0][1], DOMAIN[1][0], DOMAIN[1][1]]

    u_ref_disp = np.where(mask_free, np.clip(u_ref, 0, vmax), np.nan)
    u_srm_disp = np.where(mask_free, np.clip(u_srm, 0, vmax), np.nan)
    err_disp   = np.where(valid, np.abs(u_srm - u_ref), np.nan)

    # Speed field on same grid
    pts_grid  = np.stack([X1_np.ravel(), X2_np.ravel()], axis=1)
    spd_grid  = speed_fn_np(pts_grid).reshape(Ng, Ng)

    fig = plt.figure(figsize=(24, 5.5))
    gs  = GridSpec(1, 5, figure=fig, wspace=0.32, width_ratios=[1, 1, 1, 1, 1.3])

    # -- Panel 0: speed field --
    ax0 = fig.add_subplot(gs[0, 0])
    im0 = ax0.imshow(spd_grid, origin="lower", extent=extent,
                     cmap="RdYlGn", vmin=0.0, vmax=1.05)
    ax0.set_title("Speed field $c(x)$\n(dark = obstacle, c=0.05)", fontsize=9)
    ax0.plot(*X_SRC_NP, "b*", markersize=11, label="source")
    ax0.legend(fontsize=8)
    ax0.set_xlabel(r"$x_1$"); ax0.set_ylabel(r"$x_2$")
    fig.colorbar(im0, ax=ax0, label="c(x)")

    # -- Panel 1: FMM reference --
    ax1 = fig.add_subplot(gs[0, 1])
    im1 = ax1.imshow(u_ref_disp, origin="lower", extent=extent,
                     cmap="viridis", vmin=0, vmax=vmax)
    ax1.contour(X1_np, X2_np, u_ref_disp, levels=levels,
                colors="white", linewidths=0.7, alpha=0.85)
    ax1.set_title(f"FMM reference (N={FMM_N})\nground truth", fontsize=9)
    ax1.set_xlabel(r"$x_1$")
    fig.colorbar(im1, ax=ax1, label="travel time")

    # -- Panel 2: SRM prediction --
    ax2 = fig.add_subplot(gs[0, 2])
    im2 = ax2.imshow(u_srm_disp, origin="lower", extent=extent,
                     cmap="viridis", vmin=0, vmax=vmax)
    ax2.contour(X1_np, X2_np, u_srm_disp, levels=levels,
                colors="white", linewidths=0.7, alpha=0.85)
    ax2.set_title(
        f"SRM boost  k={r['k']}  dyn-{r['n_batch']}/step  {r['num_steps']} steps\n"
        f"MSE={r['grid_mse']:.2e}  ({improvement:.1f}× vs baseline {BASELINE['grid_mse']:.2e})",
        fontsize=9,
    )
    ax2.set_xlabel(r"$x_1$")
    fig.colorbar(im2, ax=ax2, label="travel time")

    # -- Panel 3: absolute error --
    ax3 = fig.add_subplot(gs[0, 3])
    im3 = ax3.imshow(err_disp, origin="lower", extent=extent,
                     cmap="hot", vmin=0)
    ax3.set_title(
        f"|SRM − FMM|\nmax={r['max_err']:.2e}   PDE res={r['pde_res']:.2e}",
        fontsize=9,
    )
    ax3.set_xlabel(r"$x_1$")
    fig.colorbar(im3, ax=ax3, label="abs error")

    # -- Panel 4: loss history --
    ax4 = fig.add_subplot(gs[0, 4])
    steps = list(range(len(hist)))
    ax4.semilogy([h["total"] for h in hist], lw=1.5, label="total",  color="C0")
    ax4.semilogy([h["bc"]    for h in hist], lw=1.2, label="BC",     color="C1")
    ax4.semilogy([h["pde"]   for h in hist], lw=1.2, label="PDE",    color="C2")
    ax4.axhline(BASELINE["grid_mse"], ls="--", color="grey", lw=0.9, alpha=0.7,
                label=f"baseline MSE={BASELINE['grid_mse']:.1e}")
    ax4.set_xlabel("step"); ax4.set_ylabel("loss")
    ax4.set_title("Loss history", fontsize=9)
    ax4.legend(fontsize=7); ax4.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        "2D Obstacle-field Eikonal  —  Gaussian SRM, dynamic collocation\n"
        f"k={r['k']} ({r['n_params']:,} params)  dyn-{r['n_batch']}/step  "
        f"{r['num_steps']} steps  "
        f"grid-MSE {r['grid_mse']:.2e}  [baseline: {BASELINE['grid_mse']:.2e}]  "
        f"{improvement:.1f}× improvement  {r['elapsed']:.0f}s",
        fontsize=10, y=1.02,
    )

    out = "figures/obstacle_boost.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved {out}")

    mlflow.log_artifact(out, artifact_path="figures")
    print("Figure logged to MLflow.")

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

print("\n" + "="*72)
print(f"{'config':>32}  {'n_params':>8}  {'grid_mse':>10}  {'improv':>8}  {'time':>7}")
print("-"*72)
print(f"{'baseline k=200 n=2000 s=3000':>32}  "
      f"{'':>8}  {BASELINE['grid_mse']:>10.3e}")
tag = f"boost k={r['k']} n={r['n_batch']} s={r['num_steps']}"
print(f"{tag:>32}  "
      f"{r['n_params']:>8,}  {r['grid_mse']:>10.3e}  "
      f"{BASELINE['grid_mse']/r['grid_mse']:>7.1f}×  {r['elapsed']:>6.0f}s")
print("="*72)
