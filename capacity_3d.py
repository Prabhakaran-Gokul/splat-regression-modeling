"""3-D: does the PDE contribute more with a lower base regulariser and/or more Gaussians?

Fixed collocation (2048) and fixed 3-D scene + 600-node RRT* base. Two sweeps:
  (A) base_reg ∈ {3, 1, 0.3, 0.1} at 384 splats — the prior↔physics tug-of-war (lower = freer physics).
  (B) num_splats ∈ {384, 768, 1500} at base_reg=1 — model capacity, expected to matter more in higher-d.
Training loss is printed live (via solve's progress bar) so we can see whether it converges well in 3-D.
Reports solved RMS + max vs the base-alone RMS.
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
print(f"BASE (600 nodes): RMS={b_rms:.3f} max={b_max:.2f}", flush=True)

rows = []
def run(base_reg, num_splats):
    cfg = dataclasses.replace(cfg0, num_collocation=2048, base_reg=base_reg, num_splats=num_splats, steps=3000)
    splat = solve(cfg, rm, cen_j, rad_j, start, D)
    s_rms, s_max = err(predict(splat, eval_pts, rm))
    rows.append((base_reg, num_splats, s_rms, s_max))
    print(f"base_reg={base_reg:<4} splats={num_splats:<5} -> solved RMS={s_rms:.3f} max={s_max:.2f}", flush=True)

for breg in (3.0, 1.0, 0.3, 0.1):  # (A) regulariser sweep at 384 splats
    run(breg, 384)
for ns in (768, 1500):  # (B) capacity sweep at base_reg=1 (384/reg=1 already covered above)
    run(1.0, ns)

print(f"\n===== 3-D base_reg / capacity sweep (2048 collocation; base RMS {b_rms:.3f}) =====")
print(f"{'base_reg':>8} {'splats':>7} {'solved_RMS':>10} {'RMS_impr':>8} {'solved_max':>10}")
for br, ns, sr, sm in rows:
    print(f"{br:>8} {ns:>7} {sr:>10.3f} {100 * (b_rms - sr) / b_rms:>+7.0f}% {sm:>10.2f}")
print("CAPACITY DONE")
