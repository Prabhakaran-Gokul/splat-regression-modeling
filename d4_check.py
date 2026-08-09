"""Does the PDE's *absolute* contribution hold at d=4 once base_reg is tuned (not just the % vs a bigger base)?

d=4, 1200-node base, sweep base_reg. Report base vs solved RMS, the absolute reduction, and %.
If the absolute reduction is ~0.03 (like 2-D/3-D), the physics is not diminishing — the lower % is a
denominator effect from the worse high-d base.
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

D, N, NODES = 4, 30, 1200
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
print(f"BASE (1200 nodes, d=4): RMS={b_rms:.3f} max={b_max:.2f}", flush=True)

rows = []
for breg in (3.0, 1.0, 0.3):
    cfg = dataclasses.replace(cfg0, num_collocation=2048, base_reg=breg, num_splats=384, steps=3000)
    splat = solve(cfg, rm, cen_j, rad_j, start, D)
    s_rms, s_max = err(predict(splat, eval_pts, rm))
    rows.append((breg, s_rms, s_max))
    print(f"base_reg={breg}: solved RMS={s_rms:.3f} (abs -{b_rms - s_rms:.3f}, {100 * (b_rms - s_rms) / b_rms:+.0f}%) max={s_max:.2f}", flush=True)

print(f"\n===== d=4 base_reg check (1200 nodes; base RMS {b_rms:.3f}) =====")
print(f"{'base_reg':>8} {'solved_RMS':>10} {'abs_red':>8} {'impr':>6} {'max':>6}")
for br, sr, sm in rows:
    print(f"{br:>8} {sr:>10.3f} {b_rms - sr:>8.3f} {100 * (b_rms - sr) / b_rms:>+5.0f}% {sm:>6.2f}")
print("D4CHECK DONE")
