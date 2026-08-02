#!/usr/bin/env python3
"""
Train SRM for 3-D, 4-D, 5-D uniform-speed Eikonal with dynamic collocation.

Analytical solution: u = ‖x‖  (c=1, source at origin)

Dynamic collocation: N_BATCH fresh random interior points sampled every step
(total unique coverage ≈ N_BATCH × NUM_STEPS per run).

For d=3 all three axis-aligned mid-plane slices are shown.
For d=4, 5 the canonical x₁-x₂ slice (x₃=⋯=x_d=0) is shown.

Results are cached to logs/eikonal_nd_dynamic_results.pkl so re-runs skip
retraining; delete the file to force a fresh run.

Output: figures/eikonal_nd_dynamic.png
"""

import os
import pickle
import time

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import mlflow
from matplotlib.gridspec import GridSpec

jax.config.update("jax_platform_name", "gpu")
jax.config.update("jax_enable_x64", False)

# ---------------------------------------------------------------------------
# MLflow setup
# ---------------------------------------------------------------------------

MLFLOW_URI        = "sqlite:////home/tassos/.local/share/mlflow/runs.db"
MLFLOW_EXPERIMENT = "eikonal-srm"

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT)

from eikonal_splat import (
    source_sphere_nd,
    _uniform_collocation_nd,
    _canonical_slice_mse,
    _pde_residual_nd,
    _eval_slices_3d,
    init_splat_params,
    train_eikonal_splat,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DIMS       = [3, 4, 5]
K          = 200          # splat components
N_BATCH    = 1000         # dynamic: fresh interior points per step
N_HOLDOUT  = 4000         # fixed hold-out set for final PDE residual
NUM_STEPS  = 5000
LR         = 5e-3
EPS        = 0.10         # source-sphere radius
N_SPHERE   = 200          # boundary condition points on source sphere
SEED       = 42
SPEED_FN   = lambda x: jnp.ones((x.shape[0], 1))

CACHE_FILE = "logs/eikonal_nd_dynamic_results.pkl"
os.makedirs("logs",    exist_ok=True)
os.makedirs("figures", exist_ok=True)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one(key, d):
    print(f"\n{'='*60}")
    print(f"d={d}  k={K}  dynamic collocation  n_batch={N_BATCH}/step  "
          f"steps={NUM_STEPS}")
    print(f"{'='*60}")

    domain = [(-1.0, 1.0)] * d
    x_src  = jnp.zeros(d)

    src_pts, src_vals = source_sphere_nd(x_src, EPS, N_SPHERE, c_at_src=1.0)

    # Fixed hold-out set for consistent final evaluation
    key, sk = jr.split(key)
    holdout_pts = _uniform_collocation_nd(sk, N_HOLDOUT, domain,
                                          exclude_center=[0.] * d,
                                          min_dist=EPS)

    # Resample closure for dynamic collocation
    def resample_fn(sk):
        return _uniform_collocation_nd(sk, N_BATCH, domain,
                                       exclude_center=[0.] * d,
                                       min_dist=EPS)

    key, sk = jr.split(key)
    params   = init_splat_params(sk, K, d, domain, scale=0.25)
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"  SRM parameters: {n_params:,}")

    key, train_key = jr.split(key)
    t0 = time.time()
    params, history = train_eikonal_splat(
        params,
        holdout_pts,        # placeholder — ignored when resample_fn is given
        src_pts, src_vals,
        SPEED_FN,
        num_steps=NUM_STEPS,
        lr=LR,
        physics_weight=1.0,
        log_interval=1000,
        resample_fn=resample_fn,
        key=train_key,
    )
    elapsed = time.time() - t0

    # Metrics
    slice_mse, u_slice, Gx, Gy = _canonical_slice_mse(params, domain)
    pde_res = _pde_residual_nd(params, holdout_pts, SPEED_FN)

    print(f"  Canonical slice MSE : {slice_mse:.4e}")
    print(f"  PDE residual L²     : {pde_res:.4e}")
    print(f"  Wall time           : {elapsed:.1f}s")

    result = dict(
        d=d, params=params, history=history,
        slice_mse=slice_mse, pde_res=pde_res, elapsed=elapsed,
        n_params=n_params,
        u_slice=u_slice, Gx=Gx, Gy=Gy,
        domain=domain,
    )

    # For 3-D, cache all three midplane slices
    if d == 3:
        result["slices_3d"] = _eval_slices_3d(params, domain, Ng=60)

    return result


# ---------------------------------------------------------------------------
# Train / load cache
# ---------------------------------------------------------------------------

results = {}

if os.path.exists(CACHE_FILE):
    print(f"Loading cached results from {CACHE_FILE}")
    with open(CACHE_FILE, "rb") as f:
        results = pickle.load(f)
    dims_to_train = [d for d in DIMS if d not in results]
    if dims_to_train:
        print(f"  Cache missing d={dims_to_train} — will train those.")
else:
    dims_to_train = DIMS

base_key = jr.PRNGKey(SEED)
for d in dims_to_train:
    base_key, sk = jr.split(base_key)
    results[d] = train_one(sk, d)

if dims_to_train:
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved results to {CACHE_FILE}")


# ---------------------------------------------------------------------------
# MLflow logging — one run per dimension
# ---------------------------------------------------------------------------

def log_dimension_run(r):
    """Log one dimension's results as a single MLflow run."""
    d    = r["d"]
    hist = r["history"]

    with mlflow.start_run(run_name=f"eikonal-srm-d{d}-dynamic"):
        # ── Shared hyper-parameters ──────────────────────────────────────
        mlflow.log_params({
            "dim":          d,
            "k":            K,
            "n_batch":      N_BATCH,
            "n_holdout":    N_HOLDOUT,
            "num_steps":    NUM_STEPS,
            "lr":           LR,
            "eps":          EPS,
            "n_sphere":     N_SPHERE,
            "seed":         SEED,
            "collocation":  "dynamic",
            "A_mode":       "full",
            "n_params":     r["n_params"],
        })

        # ── Per-step loss curves ─────────────────────────────────────────
        for step, h in enumerate(hist):
            mlflow.log_metrics({
                "loss/total": h["total"],
                "loss/bc":    h["bc"],
                "loss/pde":   h["pde"],
            }, step=step)

        # ── Final summary metrics ────────────────────────────────────────
        mlflow.log_metrics({
            "final/slice_mse":  r["slice_mse"],
            "final/pde_res":    r["pde_res"],
            "final/wall_time_s": r["elapsed"],
            "final/loss_total": hist[-1]["total"],
            "final/loss_bc":    hist[-1]["bc"],
            "final/loss_pde":   hist[-1]["pde"],
        })


print("\nLogging to MLflow …")
for d in DIMS:
    log_dimension_run(results[d])
    print(f"  d={d} logged")
print("Done.\n")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
# Layout:
#   rows 0-2 : d=3  — three midplane slices (truth | SRM | |error|) + loss
#   row  3   : d=4  — canonical x1-x2 slice  (truth | SRM | |error| | loss)
#   row  4   : d=5  — canonical x1-x2 slice  (truth | SRM | |error| | loss)

levels = jnp.linspace(0.10, 1.30, 13)
extent = [-1, 1, -1, 1]
VMIN, VMAX = 0.0, 1.5

n_rows = 5
fig = plt.figure(figsize=(20, n_rows * 4.5))
gs  = GridSpec(n_rows, 4, figure=fig,
               hspace=0.48, wspace=0.30,
               width_ratios=[1, 1, 1, 1.2])

# ── d=3 : three midplane slices ──────────────────────────────────────────────

slice_meta_3d = [
    ("xy", "z = 0  (xy-plane)", r"$x_1$", r"$x_2$"),
    ("xz", "y = 0  (xz-plane)", r"$x_1$", r"$x_3$"),
    ("yz", "x = 0  (yz-plane)", r"$x_2$", r"$x_3$"),
]

r3      = results[3]
slc3    = r3["slices_3d"]
hist3   = r3["history"]

for row, (sname, plane_label, xl, yl) in enumerate(slice_meta_3d):
    u_pred = slc3[sname][0]
    _, Gh, Gv = slc3[sname]
    u_true  = jnp.sqrt(Gh ** 2 + Gv ** 2)
    mse_row = float(jnp.mean((u_pred - u_true) ** 2))

    # --- truth ---
    ax_t = fig.add_subplot(gs[row, 0])
    im = ax_t.imshow(u_true, origin="lower", extent=extent,
                     vmin=VMIN, vmax=VMAX, cmap="viridis", alpha=0.85)
    ax_t.contour(Gh, Gv, u_true, levels=levels,
                 colors="white", linewidths=0.7, alpha=0.85)
    if row == 0:
        title_t = (f"d=3  k={K}  dyn-{N_BATCH}/step  {NUM_STEPS} steps\n"
                   f"mean-slice MSE={r3['slice_mse']:.2e}  "
                   f"PDE={r3['pde_res']:.2e}  {r3['elapsed']:.0f}s\n"
                   f"Analytical  [{plane_label}]")
    else:
        title_t = f"Analytical  [{plane_label}]"
    ax_t.set_title(title_t, fontsize=8)
    ax_t.set_xlabel(xl); ax_t.set_ylabel(yl)
    fig.colorbar(im, ax=ax_t, label="u(x)")

    # --- SRM prediction ---
    ax_p = fig.add_subplot(gs[row, 1])
    im2 = ax_p.imshow(u_pred, origin="lower", extent=extent,
                      vmin=VMIN, vmax=VMAX, cmap="viridis", alpha=0.85)
    ax_p.contour(Gh, Gv, u_pred, levels=levels,
                 colors="white", linewidths=0.7, alpha=0.85)
    ax_p.set_title(f"SRM  [{plane_label}]  MSE={mse_row:.1e}", fontsize=8)
    ax_p.set_xlabel(xl); ax_p.set_ylabel(yl)
    fig.colorbar(im2, ax=ax_p, label="u(x)")

    # --- absolute error ---
    ax_e = fig.add_subplot(gs[row, 2])
    err = jnp.abs(u_pred - u_true)
    im3 = ax_e.imshow(err, origin="lower", extent=extent, cmap="hot", vmin=0)
    ax_e.set_title("|error|", fontsize=8)
    ax_e.set_xlabel(xl); ax_e.set_ylabel(yl)
    fig.colorbar(im3, ax=ax_e)

# Loss curve for d=3 spans rows 0-2 in the rightmost column
ax_loss3 = fig.add_subplot(gs[0:3, 3])
ax_loss3.semilogy([h["total"] for h in hist3], lw=1.5, label="total",  color="C0")
ax_loss3.semilogy([h["bc"]    for h in hist3], lw=1.2, label="BC",     color="C1")
ax_loss3.semilogy([h["pde"]   for h in hist3], lw=1.2, label="PDE",    color="C2")
ax_loss3.set_xlabel("step"); ax_loss3.set_ylabel("loss")
ax_loss3.set_title("d=3  loss history", fontsize=10)
ax_loss3.legend(fontsize=9); ax_loss3.grid(True, which="both", alpha=0.3)


# ── d=4 and d=5 : canonical x1-x2 slice ────────────────────────────────────

for row_offset, d in [(3, 4), (4, 5)]:
    r    = results[d]
    u_s  = r["u_slice"]      # canonical 2D slice
    Gx   = r["Gx"]
    Gy   = r["Gy"]
    u_tr = jnp.sqrt(Gx ** 2 + Gy ** 2)
    hist = r["history"]

    # --- truth ---
    ax_t = fig.add_subplot(gs[row_offset, 0])
    im = ax_t.imshow(u_tr, origin="lower", extent=extent,
                     vmin=VMIN, vmax=VMAX, cmap="viridis", alpha=0.85)
    ax_t.contour(Gx, Gy, u_tr, levels=levels,
                 colors="white", linewidths=0.7, alpha=0.85)
    ax_t.set_title(f"d={d}  Analytical  "
                   rf"($x_3\!=\!\cdots\!=\!x_{d}\!=\!0$)", fontsize=8)
    ax_t.set_xlabel(r"$x_1$"); ax_t.set_ylabel(r"$x_2$")
    fig.colorbar(im, ax=ax_t, label="u(x)")

    # --- SRM prediction ---
    ax_p = fig.add_subplot(gs[row_offset, 1])
    im2 = ax_p.imshow(u_s, origin="lower", extent=extent,
                      vmin=VMIN, vmax=VMAX, cmap="viridis", alpha=0.85)
    ax_p.contour(Gx, Gy, u_s, levels=levels,
                 colors="white", linewidths=0.7, alpha=0.85)
    ax_p.set_title(
        f"d={d}  SRM k={K}  dyn-{N_BATCH}/step  {NUM_STEPS} steps\n"
        f"MSE={r['slice_mse']:.2e}  PDE={r['pde_res']:.2e}  {r['elapsed']:.0f}s",
        fontsize=8)
    ax_p.set_xlabel(r"$x_1$"); ax_p.set_ylabel(r"$x_2$")
    fig.colorbar(im2, ax=ax_p, label="u(x)")

    # --- absolute error ---
    ax_e = fig.add_subplot(gs[row_offset, 2])
    err = jnp.abs(u_s - u_tr)
    im3 = ax_e.imshow(err, origin="lower", extent=extent, cmap="hot", vmin=0)
    ax_e.set_title("|error| on canonical slice", fontsize=8)
    ax_e.set_xlabel(r"$x_1$"); ax_e.set_ylabel(r"$x_2$")
    fig.colorbar(im3, ax=ax_e)

    # --- loss ---
    ax_l = fig.add_subplot(gs[row_offset, 3])
    ax_l.semilogy([h["total"] for h in hist], lw=1.5, label="total",  color="C0")
    ax_l.semilogy([h["bc"]    for h in hist], lw=1.2, label="BC",     color="C1")
    ax_l.semilogy([h["pde"]   for h in hist], lw=1.2, label="PDE",    color="C2")
    ax_l.set_xlabel("step"); ax_l.set_ylabel("loss")
    ax_l.set_title(f"d={d}  loss history", fontsize=9)
    ax_l.legend(fontsize=8); ax_l.grid(True, which="both", alpha=0.3)


# ── Summary suptitle ─────────────────────────────────────────────────────────

lines = [
    f"Eikonal PINN  |∇u|=1  (analytical: u=‖x‖) — Gaussian SRM, dynamic collocation",
    f"k={K},  {N_BATCH} fresh pts/step,  {NUM_STEPS} steps",
]
for d in DIMS:
    r = results[d]
    lines.append(f"  d={d}:  {r['n_params']:,} params  "
                 f"slice-MSE={r['slice_mse']:.2e}  "
                 f"PDE-L²={r['pde_res']:.2e}  "
                 f"{r['elapsed']:.0f}s")
fig.suptitle("\n".join(lines), fontsize=11, y=1.01)

out = "figures/eikonal_nd_dynamic.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved {out}")

# Log the combined figure to MLflow as an artifact on a summary run
with mlflow.start_run(run_name="eikonal-srm-nd-summary"):
    mlflow.log_params({
        "dims":      str(DIMS),
        "k":         K,
        "n_batch":   N_BATCH,
        "num_steps": NUM_STEPS,
        "lr":        LR,
        "seed":      SEED,
        "collocation": "dynamic",
    })
    # Summary metrics for each dimension (flat keys with d tag)
    for d in DIMS:
        r = results[d]
        mlflow.log_metrics({
            f"d{d}/slice_mse":   r["slice_mse"],
            f"d{d}/pde_res":     r["pde_res"],
            f"d{d}/wall_time_s": r["elapsed"],
            f"d{d}/n_params":    r["n_params"],
        })
    mlflow.log_artifact(out, artifact_path="figures")
print("Summary run logged to MLflow.")

# Print compact summary table
print("\n" + "="*65)
print(f"{'d':>4}  {'params':>8}  {'slice-MSE':>12}  {'PDE-L²':>12}  {'time(s)':>8}")
print("-"*65)
for d in DIMS:
    r = results[d]
    print(f"{d:>4}  {r['n_params']:>8,}  "
          f"{r['slice_mse']:>12.3e}  "
          f"{r['pde_res']:>12.3e}  "
          f"{r['elapsed']:>8.1f}")
print("="*65)
