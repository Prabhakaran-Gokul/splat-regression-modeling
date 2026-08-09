"""Is the weak PDE in 3-D just under-resolved collocation (fixable) or fundamental?

Same 3-D scene and 600-node RRT* base as the doubling sweep. We vary the number of PDE collocation
points (2048 → 8000) and the base regularisation (3 = follow base, 1 = let physics lead). If more
collocation + a freer PDE drops the solved RMS well below the 2048/base_reg=3 baseline (~0.16), the
weak PDE was a resolution/balance issue we simply under-provisioned; if it barely moves, the physics
is genuinely diluted in high-d and we should say so.
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

D, N, NODES = 3, 46, 600
cfg0 = Config()
rng = np.random.default_rng(cfg0.seed + D)
centers, radii, start = make_obstacles(cfg0, D, rng)
cen_j, rad_j = jnp.asarray(centers, jnp.float32), jnp.asarray(radii, jnp.float32)

pts = grid_points(N, D)
slow_grid = slowness_np(pts, centers, radii, cfg0.slowness_max, cfg0.slow_width).reshape((N,) * D)
start_coord = tuple(int(round((s + np.pi) / (2 * np.pi) * N)) % N for s in np.asarray(start))
gt = fast_marching_nd(1.0 / slow_grid, start_coord, N, D).ravel()
inside = np.asarray(sdf_j(jnp.asarray(pts, jnp.float32), cen_j, rad_j)) < 0.0
nodes, costs = rrt_star(cfg0, centers, radii, np.asarray(start), D, rng, NODES)
rm = (nodes, costs, cfg0.roadmap_gamma, cen_j, rad_j, cfg0.slowness_max, cfg0.slow_width, cfg0.roadmap_hop)

free = np.where(~inside)[0]
eval_idx = free if len(free) <= 20000 else rng.choice(free, 20000, replace=False)
eval_pts, eval_gt = jnp.asarray(pts[eval_idx], jnp.float32), gt[eval_idx]

def err(pred):
    e = np.abs(np.asarray(pred) - eval_gt)
    return float(np.sqrt(np.mean(e**2))), float(e.max())

zero = init_splat(cfg0.num_splats, cfg0.init_scale, D, jax.random.PRNGKey(0))
zero = (jnp.zeros_like(zero[0]), zero[1], zero[2])
b_rms, b_max = err(predict(zero, eval_pts, rm))
print(f"BASE (600 nodes): RMS={b_rms:.3f} max={b_max:.3f}", flush=True)

rows = []
for ncoll, breg in [(2048, 3.0), (8000, 3.0), (8000, 1.0)]:
    cfg = dataclasses.replace(cfg0, num_collocation=ncoll, base_reg=breg, steps=2500)
    splat = solve(cfg, rm, cen_j, rad_j, start, D)
    s_rms, s_max = err(predict(splat, eval_pts, rm))
    rows.append((ncoll, breg, s_rms, s_max))
    print(f"collocation={ncoll} base_reg={breg}: solved RMS={s_rms:.3f} max={s_max:.3f}", flush=True)

print(f"\n===== 3-D DILUTION TEST (600 RRT* nodes; base RMS {b_rms:.3f}, max {b_max:.2f}) =====")
print(f"{'colloc':>7} {'base_reg':>8} {'solved_RMS':>10} {'RMS_impr':>8} {'solved_max':>10}")
for nc, br, sr, sm in rows:
    print(f"{nc:>7} {br:>8.1f} {sr:>10.3f} {100 * (b_rms - sr) / b_rms:>+7.0f}% {sm:>10.2f}")
print("DILUTION DONE")
