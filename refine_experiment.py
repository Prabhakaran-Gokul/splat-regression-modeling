"""Does the Eikonal PDE *refine* a sparse/noisy RRT* base, or just fit it?

The high-D question: a dense RRT* base is impossible in 12-D, so the method only scales if the
physics can rescue a *sparse* (few-node) base. For each tree sparsity and base-weight we report
base-alone vs PDE-refined **value RMS** (vs FMM) and **Eikonal residual** (physical consistency).
If refined < base — especially for sparse bases — the physics is doing real work, not imitating.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from torus import (
    Config, build_roadmap, fast_marching_torus, init_splat, predict_roadmap,
    roadmap_residual, solve_roadmap, torus_obstacles, torus_sdf, torus_slowness,
)


def gt_and_grid(cfg: Config, obs):
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
for iters in (150, 250, 500):
    for base_reg in (3.0, 30.0):
        cfg = Config(
            method="roadmap", rrt_iters=iters, roadmap_nodes=2000, roadmap_gamma=0.01, roadmap_hop=5,
            base_reg=base_reg, num_splats=384, num_collocation=1500, steps=3000, resolution=120, seed=1,
            checkpoint_every=999999, out_dir="/private/tmp/claude-504/-Users-baner121-splat-regression-modeling/bcd05dbc-7428-4c80-836b-9dfb4627440f/scratchpad",
        )
        obs = torus_obstacles(cfg)
        thj, gt, inside = gt_and_grid(cfg, obs)
        nodes, costs = build_roadmap(cfg, obs)
        rm = (nodes, costs, cfg.roadmap_gamma, obs, cfg.slowness_max, cfg.slow_width, cfg.roadmap_hop)
        free = np.where(~inside)[0]
        free = free[np.random.default_rng(0).choice(len(free), 2000, replace=False)]  # residual sample
        freej = thj[free]
        slowf = torus_slowness(freej, obs, cfg.slowness_max, cfg.slow_width)

        zero = init_splat(cfg.num_splats, cfg.init_scale, jax.random.PRNGKey(0))
        zero = (jnp.zeros_like(zero[0]), zero[1], zero[2])
        base_rms = rms(predict_roadmap(zero, thj, *rm), gt, inside)
        base_res = float(jnp.mean(roadmap_residual(zero, freej, slowf, *rm)))

        splat = solve_roadmap(cfg, obs)
        ref_rms = rms(predict_roadmap(splat, thj, *rm), gt, inside)
        ref_res = float(jnp.mean(roadmap_residual(splat, freej, slowf, *rm)))

        row = f"iters={iters:4d} nodes={len(nodes):4d} base_reg={base_reg:4.0f} | RMS {base_rms:.3f} -> {ref_rms:.3f}  ({'DOWN' if ref_rms < base_rms else 'up'})  | resid {base_res:.3f} -> {ref_res:.3f}"
        print(row, flush=True)
        rows.append(row)

print("\n===== REFINEMENT EXPERIMENT SUMMARY =====")
for r in rows:
    print(r)
print("REFINE DONE")
