"""Thinning test: how does error grow as RRT* samples get sparser — linearly or sub-linearly?

For a range of node budgets we report base-alone vs PDE-refined RMS (B5 setting, base_reg=3,
physics-led). If refined RMS grows ~linearly in 1/nodes the physics fails to compensate and the
method won't scale; if it grows sub-linearly (log-like) the Eikonal holds the line as samples thin —
the encouraging signal for higher dimensions with sparser points. Saves a log-log plot + slope fit.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from torus import (
    Config, build_roadmap, fast_marching_torus, init_splat, predict_roadmap,
    solve_roadmap, torus_obstacles, torus_sdf, torus_slowness,
)


def gt_and_grid(cfg, obs):
    axis = np.linspace(-np.pi, np.pi, cfg.resolution, endpoint=False)
    g1, g2 = np.meshgrid(axis, axis)
    thj = jnp.asarray(np.stack([g1.ravel(), g2.ravel()], -1), jnp.float32)
    slow = np.asarray(torus_slowness(thj, obs, cfg.slowness_max, cfg.slow_width)).reshape(cfg.resolution, cfg.resolution)
    gt = fast_marching_torus(1.0 / slow, cfg.start, cfg.resolution).ravel()
    inside = np.asarray(torus_sdf(thj, obs)) < 0
    return thj, gt, inside


def rms(pred, gt, inside):
    e = np.where(inside, np.nan, np.asarray(pred) - gt)
    return float(np.sqrt(np.nanmean(e**2)))


rows = []
for iters in (80, 120, 200, 350, 600):
    cfg = Config(
        method="roadmap", rrt_iters=iters, roadmap_nodes=2000, roadmap_gamma=0.01, roadmap_hop=5,
        base_reg=3.0, num_splats=384, num_collocation=1500, steps=4000, resolution=120, seed=1,
        checkpoint_every=999999, out_dir="/private/tmp/claude-504/-Users-baner121-splat-regression-modeling/bcd05dbc-7428-4c80-836b-9dfb4627440f/scratchpad",
    )
    obs = torus_obstacles(cfg)
    thj, gt, inside = gt_and_grid(cfg, obs)
    nodes, costs = build_roadmap(cfg, obs)
    rm = (nodes, costs, cfg.roadmap_gamma, obs, cfg.slowness_max, cfg.slow_width, cfg.roadmap_hop)
    zero = init_splat(cfg.num_splats, cfg.init_scale, jax.random.PRNGKey(0))
    zero = (jnp.zeros_like(zero[0]), zero[1], zero[2])
    base_rms = rms(predict_roadmap(zero, thj, *rm), gt, inside)
    splat = solve_roadmap(cfg, obs)
    ref_rms = rms(predict_roadmap(splat, thj, *rm), gt, inside)
    n = int(len(nodes))
    rows.append((n, base_rms, ref_rms))
    print(f"nodes={n:4d} | base RMS={base_rms:.3f}  refined={ref_rms:.3f}  (physics gap {base_rms - ref_rms:+.3f})", flush=True)

ns = np.array([r[0] for r in rows], float)
base = np.array([r[1] for r in rows])
ref = np.array([r[2] for r in rows])
slope_ref = float(np.polyfit(np.log(ns), np.log(ref), 1)[0])  # refined RMS ~ nodes^slope
slope_base = float(np.polyfit(np.log(ns), np.log(base), 1)[0])

fig, ax = plt.subplots(figsize=(7, 5))
ax.loglog(ns, base, "o--", label=f"base alone (slope {slope_base:.2f})", color="tab:gray")
ax.loglog(ns, ref, "o-", label=f"PDE-refined (slope {slope_ref:.2f})", color="tab:blue")
ax.set_xlabel("RRT* nodes")
ax.set_ylabel("RMS vs FMM")
ax.set_title("Error vs sample sparsity (2D torus)\nslope ~-1 = linear-in-1/nodes (bad); ~0 = flat (great)")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig("figures/thinning.png", dpi=140)

print("\n===== THINNING SUMMARY =====")
for n, b, r in rows:
    print(f"nodes={n:4d}  base={b:.3f}  refined={r:.3f}")
print(f"refined RMS ~ nodes^({slope_ref:.2f})   base RMS ~ nodes^({slope_base:.2f})")
print(f"[interpret] slope near 0 = error flat as samples thin (scales); near -1 = linear in 1/nodes (fails)")
print("saved figures/thinning.png")
print("THINNING DONE")
