"""Honest reliability of the ONE adaptive number: run the real ntfields_baseline solve() across seeds.

Not a method comparison — same config (adaptive splats + 30 RRT* bounds, eps off), different seed (=
different scene + init). Reports per-seed RMS/max and the mean±spread, so the headline number is
characterized, not cherry-picked. Uses the identical RMS definition as render() (sqrt(nanmean(e^2)) over
free space).
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np

from ntfields_baseline import Config, predict, raw_fn_for, solve
from torus import fast_marching_torus, torus_obstacles, torus_sdf, torus_slowness, Config as TConfig

SEEDS = [1, 2, 3]
base = Config(field="splat", densify=True, supervision="bounds", epsilon=0.0)
rows = []
for seed in SEEDS:
    cfg = dataclasses.replace(base, seed=seed)
    tcfg = TConfig(start=cfg.start, num_obstacles=cfg.num_obstacles, obstacle_radius=cfg.obstacle_radius,
                   slowness_max=cfg.slowness_max, slow_width=cfg.slow_width, resolution=cfg.resolution, seed=seed)
    obstacles = torus_obstacles(tcfg)
    axis = np.linspace(-np.pi, np.pi, cfg.resolution, endpoint=False)
    g1, g2 = np.meshgrid(axis, axis)
    thetas = jnp.asarray(np.stack([g1.ravel(), g2.ravel()], -1), jnp.float32)
    slow = np.asarray(torus_slowness(thetas, obstacles, cfg.slowness_max, cfg.slow_width)).reshape(cfg.resolution, cfg.resolution)
    gt = fast_marching_torus(1.0 / slow, cfg.start, cfg.resolution).ravel()
    inside = np.asarray(torus_sdf(thetas, obstacles)) < 0.0

    params = solve(cfg, obstacles)
    pred = np.asarray(predict(raw_fn_for(cfg), params, thetas, jnp.asarray(cfg.start), cfg.tau_bias, cfg.tau_min))
    e = np.where(inside, np.nan, pred - gt)
    rms, mx = float(np.sqrt(np.nanmean(e**2))), float(np.nanmax(np.abs(e)))
    k = int(params[0].shape[0])
    rows.append((seed, k, rms, mx))
    print(f"[seed {seed}] splats={k}  RMS={rms:.4f}  max|err|={mx:.4f}", flush=True)

r = np.array([x[2] for x in rows])
print("\n===== adaptive (192->512 splats) + 30 RRT* bounds, eps off =====")
for seed, k, rms, mx in rows:
    print(f"  seed {seed}: RMS={rms:.4f}  max={mx:.3f}  ({k} splats)")
print(f"  MEAN RMS={r.mean():.4f}  STD={r.std():.4f}  (min {r.min():.4f}, max {r.max():.4f})")
print("SWEEP DONE", flush=True)
