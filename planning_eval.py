"""What does the PDE buy — measured by PLANNING, not RMS.

Same start, many goals, on one solved field. We follow −∇T by naive gradient descent (NO feasibility
guard) and compare the PDE-solved splat field vs the raw RRT* base vs RRT* itself on:
  - reach rate: does descent reach the start (a correct Eikonal field has no spurious minima)?
  - collision-free rate: does the naive path stay out of obstacles?
  - optimality: path cost / FMM-optimal cost (RRT* costs are suboptimal upper bounds, so a
    physics-refined field can beat them).
2-D so FMM gives the true optimum; reusable at higher d by dropping the FMM-optimal column.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from torus_nd import (
    Config, fast_marching_nd, field, grid_points, init_splat, make_obstacles, roadmap_base,
    rrt_star, sdf_j, slowness_j, slowness_np, solve, wrap,
)

D, N, NODES, NGOALS = 2, 100, 300, 60
cfg = dataclasses.replace(Config(), num_obstacles=5, seed=7, base_reg=1.0, steps=4000)
rng = np.random.default_rng(cfg.seed + D)
centers, radii, start = make_obstacles(cfg, D, rng)
cen_j, rad_j = jnp.asarray(centers, jnp.float32), jnp.asarray(radii, jnp.float32)
start_np = np.asarray(start)

pts = grid_points(N, D)
slow_grid = slowness_np(pts, centers, radii, cfg.slowness_max, cfg.slow_width).reshape((N,) * D)
start_coord = tuple(int(round((s + np.pi) / (2 * np.pi) * N)) % N for s in start_np)
gt = fast_marching_nd(1.0 / slow_grid, start_coord, N, D).ravel().reshape((N,) * D)
nodes, costs = rrt_star(cfg, centers, radii, start_np, D, rng, NODES)
rm = (nodes, costs, cfg.roadmap_gamma, cen_j, rad_j, cfg.slowness_max, cfg.slow_width, cfg.roadmap_hop)
splat = solve(cfg, rm, cen_j, rad_j, start, D)
zero = init_splat(cfg.num_splats, cfg.init_scale, D, jax.random.PRNGKey(0))
zero = (jnp.zeros_like(zero[0]), zero[1], zero[2])

grad_pde = jax.jit(jax.grad(lambda t: field(splat, t, rm)))
grad_base = jax.jit(jax.grad(lambda t: field(zero, t, rm)))


def gt_at(theta):  # bilinear-ish nearest lookup of FMM optimal cost
    idx = tuple(int(round((float(x) + np.pi) / (2 * np.pi) * N)) % N for x in theta)
    return float(gt[idx])


def descend(grad_fn, goal, h=0.03, max_steps=3000):
    theta = np.asarray(goal, float)
    cost, collided = 0.0, False
    for _ in range(max_steps):
        g = np.asarray(grad_fn(jnp.asarray(theta, jnp.float32)))
        n = np.linalg.norm(g)
        if n < 1e-8:
            break  # flat — stuck
        step = -h * g / n
        newt = np.asarray(wrap(jnp.asarray(theta + step)))
        s = float(slowness_np(newt[None], centers, radii, cfg.slowness_max, cfg.slow_width)[0])
        cost += np.linalg.norm(step) * s
        collided |= float(np.asarray(sdf_j(jnp.asarray(newt[None], jnp.float32), cen_j, rad_j))[0]) < 0.0
        theta = newt
        if np.linalg.norm(np.asarray(wrap(jnp.asarray(theta - start_np)))) < 0.12:
            return True, collided, cost
    return False, collided, cost  # never reached start = stuck in a local min / flat region


goals = []
while len(goals) < NGOALS:
    q = rng.uniform(-np.pi, np.pi, D)
    if float(np.asarray(sdf_j(jnp.asarray(q[None], jnp.float32), cen_j, rad_j))[0]) > 0.15 and np.linalg.norm(_ := np.asarray(wrap(jnp.asarray(q - start_np)))) > 1.0:
        goals.append(q)


def evaluate(grad_fn, name):
    reach = coll = 0
    ratios = []
    for g in goals:
        ok, col, cost = descend(grad_fn, g)
        reach += ok
        if ok and not col:
            opt = gt_at(g)
            if opt > 1e-3:
                ratios.append(cost / opt)
        coll += (col and ok)
    r = 100 * reach / NGOALS
    c = 100 * coll / max(1, reach)
    optr = float(np.mean(ratios)) if ratios else float("nan")
    print(f"{name:>14}: reach={r:.0f}%  collided(of reached)={c:.0f}%  path/optimal={optr:.3f} (n={len(ratios)})", flush=True)
    return r, c, optr


# RRT* reference optimality: base value / FMM optimal at each goal (the roadmap's own cost estimate)
rrt_ratios = [float(np.asarray(roadmap_base(jnp.asarray(g, jnp.float32), *rm))) / gt_at(g) for g in goals if gt_at(g) > 1e-3]
print(f"\n===== PLANNING EVAL ({NGOALS} goals, d={D}, naive −∇T descent, no guard) =====", flush=True)
evaluate(grad_pde, "PDE field")
evaluate(grad_base, "base-alone")
print(f"{'RRT* (cost)':>14}: path/optimal={np.mean(rrt_ratios):.3f}  (suboptimal upper bound the PDE refines)")
print("PLANEVAL DONE")
