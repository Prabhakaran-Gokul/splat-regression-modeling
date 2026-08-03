#!/usr/bin/env python3
"""
High-accuracy SRM training for d=4 and d=5 uniform-speed Eikonal.

Diagnostics from the baseline (k=200, n_batch=1000, steps=5000):
  d=4  still declining at step 5000 → needs more steps + more capacity
  d=5  plateaued early             → needs a bigger model + denser coverage

Boost configs chosen accordingly:
  d=4  k=800   n_batch=4000  steps=15000
  d=5  k=1000  n_batch=6000  steps=20000

Results are cached to logs/eikonal_hd_boost_results.pkl.
Output figure: figures/eikonal_hd_boost.png
MLflow experiment: eikonal-srm  (runs named eikonal-srm-d{d}-boost)
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

from eikonal_splat import (
    source_sphere_nd,
    _uniform_collocation_nd,
    _canonical_slice_mse,
    _pde_residual_nd,
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
# Shared constants
# ---------------------------------------------------------------------------

EPS      = 0.10
N_SPHERE = 200
SEED     = 42
SPEED_FN = lambda x: jnp.ones((x.shape[0], 1))

CACHE_FILE = "logs/eikonal_hd_boost_results.pkl"
os.makedirs("logs",    exist_ok=True)
os.makedirs("figures", exist_ok=True)

# ---------------------------------------------------------------------------
# Per-dimension boost configs
# ---------------------------------------------------------------------------

CONFIGS = {
    4: dict(k=800,  n_batch=4000, num_steps=15000, lr=5e-3),
    5: dict(k=1000, n_batch=6000, num_steps=20000, lr=5e-3),
}

# Baseline for comparison (from previous run)
BASELINE = {
    4: dict(k=200, n_batch=1000, num_steps=5000, slice_mse=4.991e-2, pde_res=1.659e-2),
    5: dict(k=200, n_batch=1000, num_steps=5000, slice_mse=1.367e-1, pde_res=1.437e-2),
}

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one(key, d, cfg):
    k         = cfg["k"]
    n_batch   = cfg["n_batch"]
    num_steps = cfg["num_steps"]
    lr        = cfg["lr"]

    print(f"\n{'='*60}")
    print(f"d={d}  k={k}  n_batch={n_batch}  steps={num_steps}  lr={lr}")
    print(f"{'='*60}")

    domain = [(-1.0, 1.0)] * d
    x_src  = jnp.zeros(d)

    src_pts, src_vals = source_sphere_nd(x_src, EPS, N_SPHERE, c_at_src=1.0)

    key, sk = jr.split(key)
    holdout_pts = _uniform_collocation_nd(sk, 4000, domain,
                                          exclude_center=[0.] * d,
                                          min_dist=EPS)

    def resample_fn(sk):
        return _uniform_collocation_nd(sk, n_batch, domain,
                                       exclude_center=[0.] * d,
                                       min_dist=EPS)

    key, sk = jr.split(key)
    params   = init_splat_params(sk, k, d, domain, scale=0.25)
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"  SRM parameters: {n_params:,}")

    key, train_key = jr.split(key)
    t0 = time.time()
    params, history = train_eikonal_splat(
        params, holdout_pts, src_pts, src_vals, SPEED_FN,
        num_steps=num_steps, lr=lr, physics_weight=1.0,
        log_interval=2000,
        resample_fn=resample_fn, key=train_key,
    )
    elapsed = time.time() - t0

    slice_mse, u_slice, Gx, Gy = _canonical_slice_mse(params, domain)
    pde_res = _pde_residual_nd(params, holdout_pts, SPEED_FN)

    print(f"  Canonical slice MSE : {slice_mse:.4e}  "
          f"(baseline {BASELINE[d]['slice_mse']:.2e}, "
          f"improvement {BASELINE[d]['slice_mse']/slice_mse:.1f}×)")
    print(f"  PDE residual L²     : {pde_res:.4e}  "
          f"(baseline {BASELINE[d]['pde_res']:.2e})")
    print(f"  Wall time           : {elapsed:.1f}s")

    return dict(
        d=d, k=k, n_batch=n_batch, num_steps=num_steps, lr=lr,
        params=params, history=history,
        slice_mse=slice_mse, pde_res=pde_res, elapsed=elapsed,
        n_params=n_params, u_slice=u_slice, Gx=Gx, Gy=Gy,
        domain=domain,
    )


# ---------------------------------------------------------------------------
# Train / load cache
# ---------------------------------------------------------------------------

results = {}

if os.path.exists(CACHE_FILE):
    print(f"Loading cached results from {CACHE_FILE}")
    with open(CACHE_FILE, "rb") as f:
        results = pickle.load(f)
    dims_to_train = [d for d in CONFIGS if d not in results]
    if dims_to_train:
        print(f"  Missing d={dims_to_train} — will train.")
else:
    dims_to_train = list(CONFIGS.keys())

base_key = jr.PRNGKey(SEED)
for d in dims_to_train:
    base_key, sk = jr.split(base_key)
    results[d] = train_one(sk, d, CONFIGS[d])

if dims_to_train:
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved results to {CACHE_FILE}")


# ---------------------------------------------------------------------------
# MLflow logging
# ---------------------------------------------------------------------------

def log_run(r):
    with mlflow.start_run(run_name=f"eikonal-srm-d{r['d']}-boost"):
        mlflow.log_params({
            "dim":         r["d"],
            "k":           r["k"],
            "n_batch":     r["n_batch"],
            "num_steps":   r["num_steps"],
            "lr":          r["lr"],
            "n_params":    r["n_params"],
            "collocation": "dynamic",
            "A_mode":      "full",
            "run_type":    "boost",
        })
        for step, h in enumerate(r["history"]):
            mlflow.log_metrics({
                "loss/total": h["total"],
                "loss/bc":    h["bc"],
                "loss/pde":   h["pde"],
            }, step=step)
        mlflow.log_metrics({
            "final/slice_mse":       r["slice_mse"],
            "final/pde_res":         r["pde_res"],
            "final/wall_time_s":     r["elapsed"],
            "baseline/slice_mse":    BASELINE[r["d"]]["slice_mse"],
            "baseline/pde_res":      BASELINE[r["d"]]["pde_res"],
            "improvement/slice_mse": BASELINE[r["d"]]["slice_mse"] / r["slice_mse"],
        })


# print("\nLogging to MLflow …")
# for d in CONFIGS:
#     log_run(results[d])
#     print(f"  d={d} logged")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
# Layout: 2 rows (d=4, d=5) × 4 cols (truth | boost | |error| | loss)
# with baseline result inset in the title for quick comparison

levels = jnp.linspace(0.10, 1.30, 13)
extent = [-1, 1, -1, 1]
VMIN, VMAX = 0.0, 1.5

fig = plt.figure(figsize=(14, 9))
gs  = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.28,
               width_ratios=[1, 1])

for row, d in enumerate([4, 5]):
    r    = results[d]
    base = BASELINE[d]
    u_s  = r["u_slice"]
    Gx, Gy = r["Gx"], r["Gy"]
    u_tr = jnp.sqrt(Gx ** 2 + Gy ** 2)
    hist = r["history"]
    improvement = base["slice_mse"] / r["slice_mse"]

    # --- truth ---
    ax_t = fig.add_subplot(gs[row, 0])
    im = ax_t.imshow(u_tr, origin="lower", extent=extent,
                     vmin=VMIN, vmax=VMAX, cmap="viridis", alpha=0.85)
    ax_t.contour(Gx, Gy, u_tr, levels=levels,
                 colors="white", linewidths=0.7, alpha=0.85)
    ax_t.set_title(rf"d={d}  Analytical  ($x_3\!=\!\cdots\!=\!0$)", fontsize=9)
    ax_t.set_xlabel(r"$x_1$"); ax_t.set_ylabel(r"$x_2$")
    fig.colorbar(im, ax=ax_t, label="u(x)")

    # --- boost prediction ---
    ax_p = fig.add_subplot(gs[row, 1])
    im2 = ax_p.imshow(u_s, origin="lower", extent=extent,
                      vmin=VMIN, vmax=VMAX, cmap="viridis", alpha=0.85)
    ax_p.contour(Gx, Gy, u_s, levels=levels,
                 colors="white", linewidths=0.7, alpha=0.85)
    ax_p.set_title(
        f"Boost  k={r['k']}  dyn-{r['n_batch']}/step  {r['num_steps']} steps\n"
        f"MSE={r['slice_mse']:.2e}  PDE={r['pde_res']:.2e}  "
        f"({improvement:.1f}× vs baseline)",
        fontsize=8,
    )
    ax_p.set_xlabel(r"$x_1$"); ax_p.set_ylabel(r"$x_2$")
    fig.colorbar(im2, ax=ax_p, label="u(x)")

    # # --- absolute error ---
    # ax_e = fig.add_subplot(gs[row, 2])
    # err = jnp.abs(u_s - u_tr)
    # im3 = ax_e.imshow(err, origin="lower", extent=extent, cmap="hot", vmin=0)
    # ax_e.set_title(
    #     f"|error|  max={float(jnp.max(err)):.2e}\n"
    #     f"(baseline MSE was {base['slice_mse']:.2e})",
    #     fontsize=8,
    # )
    # ax_e.set_xlabel(r"$x_1$"); ax_e.set_ylabel(r"$x_2$")
    # fig.colorbar(im3, ax=ax_e)

    # # --- loss history with baseline final marked ---
    # ax_l = fig.add_subplot(gs[row, 3])
    # ax_l.semilogy([h["total"] for h in hist], lw=1.5, label="total",  color="C0")
    # ax_l.semilogy([h["bc"]    for h in hist], lw=1.2, label="BC",     color="C1")
    # ax_l.semilogy([h["pde"]   for h in hist], lw=1.2, label="PDE",    color="C2")
    # ax_l.axhline(base["pde_res"], ls="--", color="C2", lw=0.9, alpha=0.7,
    #              label=f"baseline PDE={base['pde_res']:.1e}")
    # ax_l.set_xlabel("step"); ax_l.set_ylabel("loss")
    # ax_l.set_title(f"d={d}  loss history", fontsize=9)
    # ax_l.legend(fontsize=7); ax_l.grid(True, which="both", alpha=0.3)


# Summary suptitle
lines = [
    "High-accuracy Eikonal PINN  |∇u|=1  (u=‖x‖) — Gaussian SRM, dynamic collocation",
]
for d in [4, 5]:
    r, b = results[d], BASELINE[d]
    lines.append(
        f"  d={d}: k={r['k']} ({r['n_params']:,}p)  "
        f"dyn-{r['n_batch']}/step  {r['num_steps']} steps  "
        f"slice-MSE {r['slice_mse']:.2e}  PDE {r['pde_res']:.2e}  "
        f"[baseline: {b['slice_mse']:.2e} / {b['pde_res']:.2e}]  "
        f"{r['elapsed']:.0f}s"
    )
fig.suptitle("\n".join(lines), fontsize=10, y=1.02)

out = "figures/eikonal_hd_boost.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved {out}")

# Log figure to summary run
with mlflow.start_run(run_name="eikonal-srm-hd-boost-summary"):
    mlflow.log_params({"dims": "[4,5]", "run_type": "boost_summary"})
    # for d in [4, 5]:
    #     r, b = results[d], BASELINE[d]
    #     mlflow.log_metrics({
    #         f"d{d}/slice_mse_boost":    r["slice_mse"],
    #         f"d{d}/pde_res_boost":      r["pde_res"],
    #         f"d{d}/slice_mse_baseline": b["slice_mse"],
    #         f"d{d}/improvement_x":      b["slice_mse"] / r["slice_mse"],
    #         f"d{d}/wall_time_s":        r["elapsed"],
    #     })
    mlflow.log_artifact(out, artifact_path="figures")
print("Summary run logged to MLflow.")

# Print comparison table
print("\n" + "="*80)
print(f"{'d':>3}  {'config':>24}  {'n_params':>8}  {'slice_mse':>12}  "
      f"{'pde_res':>10}  {'improv':>8}  {'time':>7}")
print("-"*80)
for d in [4, 5]:
    r, b = results[d], BASELINE[d]
    print(f"{d:>3}  {'baseline k=200 n=1k s=5k':>24}  "
          f"{'':>8}  {b['slice_mse']:>12.3e}  {b['pde_res']:>10.3e}")
    tag = f"boost k={r['k']} n={r['n_batch']} s={r['num_steps']}"
    print(f"{d:>3}  {tag:>24}  "
          f"{r['n_params']:>8,}  {r['slice_mse']:>12.3e}  {r['pde_res']:>10.3e}  "
          f"{b['slice_mse']/r['slice_mse']:>7.1f}×  {r['elapsed']:>6.0f}s")
    print()
print("="*80)
