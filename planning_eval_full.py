"""NTFields-style planning evaluation, 2-D → 4-D: what does the PDE field buy over the raw RRT* base?

For each dimension: a hard scene (6 obstacles), an FMM optimal reference, and a solved PDE field vs the
raw softmin base. From a fixed start we plan to many goals by naive −∇T descent (NO feasibility guard)
and report the metrics NTFields uses: Success Rate (reached, i.e. no local-minimum trap), collision-free
rate, path cost / FMM-optimal, and per-query planning time. RRT*'s own cost estimate is the suboptimal
upper bound. This is the regime (harder scenes, sparser base in higher-d) where the physics should
differentiate from a raw prior — easy scenes plan fine either way.
"""

from __future__ import annotations

import dataclasses
import time

import jax
import jax.numpy as jnp
import numpy as np

from torus_nd import (
    Config, fast_marching_nd, field, grid_points, init_splat, make_obstacles, roadmap_base,
    rrt_star, sdf_j, slowness_np, solve, wrap,
)

# (dim, FMM resolution, RRT* nodes [doubling], base_reg) — 6 obstacles everywhere for hardness
SETUPS = [(2, 100, 300, 1.0), (3, 46, 600, 0.3), (4, 30, 1200, 0.3)]
NGOALS = 50
rows = []

for D, N, NODES, BREG in SETUPS:
    cfg = dataclasses.replace(Config(), num_obstacles=6, obstacle_radius=(0.55, 0.95), seed=11, base_reg=BREG, steps=4000)
    rng = np.random.default_rng(cfg.seed + D)
    centers, radii, start = make_obstacles(cfg, D, rng)
    cen_j, rad_j = jnp.asarray(centers, jnp.float32), jnp.asarray(radii, jnp.float32)
    start_np = np.asarray(start)

    pts = grid_points(N, D)
    slow_grid = slowness_np(pts, centers, radii, cfg.slowness_max, cfg.slow_width).reshape((N,) * D)
    start_coord = tuple(int(round((s + np.pi) / (2 * np.pi) * N)) % N for s in start_np)
    gtgrid = fast_marching_nd(1.0 / slow_grid, start_coord, N, D).ravel().reshape((N,) * D)
    nodes, costs = rrt_star(cfg, centers, radii, start_np, D, rng, NODES)
    rm = (nodes, costs, cfg.roadmap_gamma, cen_j, rad_j, cfg.slowness_max, cfg.slow_width, cfg.roadmap_hop)
    splat = solve(cfg, rm, cen_j, rad_j, start, D)
    zero = init_splat(cfg.num_splats, cfg.init_scale, D, jax.random.PRNGKey(0))
    zero = (jnp.zeros_like(zero[0]), zero[1], zero[2])
    grad_pde = jax.jit(jax.grad(lambda t: field(splat, t, rm)))
    grad_base = jax.jit(jax.grad(lambda t: field(zero, t, rm)))

    def gt_at(theta):
        idx = tuple(int(round((float(x) + np.pi) / (2 * np.pi) * N)) % N for x in theta)
        return float(gtgrid[idx])

    def descend(grad_fn, goal, h=0.03, max_steps=3000):
        theta = np.asarray(goal, float)
        cost, collided = 0.0, False
        for _ in range(max_steps):
            g = np.asarray(grad_fn(jnp.asarray(theta, jnp.float32)))
            n = np.linalg.norm(g)
            if n < 1e-8:
                break
            step = -h * g / n
            newt = np.asarray(wrap(jnp.asarray(theta + step)))
            cost += h * float(slowness_np(newt[None], centers, radii, cfg.slowness_max, cfg.slow_width)[0])
            collided |= float(np.asarray(sdf_j(jnp.asarray(newt[None], jnp.float32), cen_j, rad_j))[0]) < 0.0
            theta = newt
            if np.linalg.norm(np.asarray(wrap(jnp.asarray(theta - start_np)))) < 0.15:
                return True, collided, cost
        return False, collided, cost

    goals = []
    while len(goals) < NGOALS:
        q = rng.uniform(-np.pi, np.pi, D)
        clr = float(np.asarray(sdf_j(jnp.asarray(q[None], jnp.float32), cen_j, rad_j))[0])
        if clr > 0.15 and np.linalg.norm(np.asarray(wrap(jnp.asarray(q - start_np)))) > 1.0:
            goals.append(q)

    def evaluate(grad_fn, name):
        reach = collfree = 0
        ratios = []
        t0 = time.time()
        for g in goals:
            ok, col, cost = descend(grad_fn, g)
            reach += ok
            if ok and not col:
                collfree += 1
                opt = gt_at(g)
                if opt > 1e-3:
                    ratios.append(cost / opt)
        dt = 1000 * (time.time() - t0) / NGOALS
        sr = 100 * reach / NGOALS
        cf = 100 * collfree / NGOALS
        optr = float(np.median(ratios)) if ratios else float("nan")
        rows.append((D, name, sr, cf, optr, dt))
        print(f"d={D} {name:>10}: SR={sr:.0f}%  collision-free={cf:.0f}%  path/opt(med)={optr:.3f}  {dt:.0f} ms/query", flush=True)

    print(f"\n--- d={D}: {NODES} RRT* nodes, 6 obstacles ---", flush=True)
    evaluate(grad_pde, "PDE field")
    evaluate(grad_base, "base-only")
    rrt = [float(np.asarray(roadmap_base(jnp.asarray(g, jnp.float32), *rm))) / gt_at(g) for g in goals if gt_at(g) > 1e-3]
    rows.append((D, "RRT* (cost)", float("nan"), float("nan"), float(np.median(rrt)), float("nan")))
    print(f"d={D} {'RRT*(cost)':>10}: path/opt(med)={np.median(rrt):.3f} (upper bound)", flush=True)

print("\n===== PLANNING EVAL SUMMARY (naive −∇T, no guard) =====")
print(f"{'d':>2} {'method':>11} {'SR%':>5} {'collfree%':>9} {'path/opt':>8} {'ms/query':>9}")
for d, m, sr, cf, o, dt in rows:
    srs = f"{sr:.0f}" if sr == sr else "-"
    cfs = f"{cf:.0f}" if cf == cf else "-"
    dts = f"{dt:.0f}" if dt == dt else "-"
    print(f"{d:>2} {m:>11} {srs:>5} {cfs:>9} {o:>8.3f} {dts:>9}")
print("PLANFULL DONE")
