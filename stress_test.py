"""Stress test: does the Eikonal PDE genuinely help, or is it just the RRT* prior?

10 d=2 scenes of increasing obstacle complexity (1..10 obstacles), fixed 300 RRT* nodes. For each we
report base-alone (RRT* prior) vs PDE-solved RMS and max error. If the PDE consistently reduces error —
and reduces it MORE on harder scenes (worse base) — physics genuinely benefits. If solved ≈ base
everywhere, the prior is doing all the work. Ground truth is the Godunov-Eikonal FMM.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from torus_nd import (
    Config, fast_marching_nd, grid_points, init_splat, make_obstacles, predict,
    rrt_star, sdf_j, slowness_np, solve,
)

D, N, NODES = 2, 100, 300
rows = []
for i in range(10):
    nobs = 1 + i  # complexity: 1 → 10 obstacles
    cfg = dataclasses.replace(Config(), num_obstacles=nobs, seed=100 + i, steps=3000)
    rng = np.random.default_rng(cfg.seed + D)
    centers, radii, start = make_obstacles(cfg, D, rng)
    cen_j, rad_j = jnp.asarray(centers, jnp.float32), jnp.asarray(radii, jnp.float32)

    pts = grid_points(N, D)
    slow_grid = slowness_np(pts, centers, radii, cfg.slowness_max, cfg.slow_width).reshape((N,) * D)
    start_coord = tuple(int(round((s + np.pi) / (2 * np.pi) * N)) % N for s in np.asarray(start))
    gt = fast_marching_nd(1.0 / slow_grid, start_coord, N, D).ravel()
    inside = np.asarray(sdf_j(jnp.asarray(pts, jnp.float32), cen_j, rad_j)) < 0.0

    nodes, costs = rrt_star(cfg, centers, radii, np.asarray(start), D, rng, NODES)
    rm = (nodes, costs, cfg.roadmap_gamma, cen_j, rad_j, cfg.slowness_max, cfg.slow_width, cfg.roadmap_hop)

    free = np.where(~inside)[0]
    eval_pts, eval_gt = jnp.asarray(pts[free], jnp.float32), gt[free]

    def err(pred):
        e = np.abs(np.asarray(pred) - eval_gt)
        return float(np.sqrt(np.mean(e**2))), float(e.max())

    zero = init_splat(cfg.num_splats, cfg.init_scale, D, jax.random.PRNGKey(0))
    zero = (jnp.zeros_like(zero[0]), zero[1], zero[2])
    b_rms, b_max = err(predict(zero, eval_pts, rm))
    splat = solve(cfg, rm, cen_j, rad_j, start, D)
    s_rms, s_max = err(predict(splat, eval_pts, rm))
    impr = 100.0 * (b_rms - s_rms) / b_rms
    rows.append((nobs, b_rms, s_rms, impr, b_max, s_max))
    print(f"scene {i} obs={nobs}: base RMS={b_rms:.3f} solved={s_rms:.3f} ({impr:+.0f}%) | max {b_max:.2f}->{s_max:.2f}", flush=True)

print("\n===== STRESS TEST (10 scenes, d=2, 300 RRT* nodes) =====")
print(f"{'obs':>3} {'base_RMS':>9} {'solved_RMS':>10} {'RMS_impr':>8} {'base_max':>9} {'solved_max':>10}")
for nobs, br, sr, im, bm, sm in rows:
    print(f"{nobs:>3} {br:>9.3f} {sr:>10.3f} {im:>+7.0f}% {bm:>9.2f} {sm:>10.2f}")
imprs = [im for _, _, _, im, _, _ in rows]
maxred = [100 * (bm - sm) / bm for _, _, _, _, bm, sm in rows]
print(f"mean RMS improvement from PDE: {np.mean(imprs):+.0f}%   mean max-error reduction: {np.mean(maxred):+.0f}%")
print("STRESS DONE")
