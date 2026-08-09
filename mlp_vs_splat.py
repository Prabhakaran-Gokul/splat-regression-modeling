"""MLP vs Splat head-to-head (NeRF-vs-3DGS hypothesis) — identical physics pipeline, matched params.

Same factored field `T=base/τ`, same speed-match Eikonal loss + sparse RRT* bounds, same everything —
only the τ-predictor swapped (MLP ↔ splat), at matched parameter budget (~2.6k). We sweep the amount of
DATA (collocation points) and report RMS, max-error (sharp-feature fidelity — the spectral-bias test),
final loss (convergence), and query time. If the splat degrades *less* as data thins, or nails the sharp
features (lower max) at fewer params — that's the 3DGS-over-NeRF advantage, made concrete. If not, we
say so.
"""

from __future__ import annotations

import dataclasses
import time

import jax
import jax.numpy as jnp
import numpy as np

from ntfields_baseline import Config, predict, raw_fn_for, solve
from torus import Config as TConfig, fast_marching_torus, torus_obstacles, torus_sdf, torus_slowness

base = Config(num_obstacles=6, obstacle_radius=(0.55, 0.95), steps=4000, supervision="bounds",
              widths=(4, 48, 48, 1), num_splats=384, seed=1)  # ~2.6k params each
tcfg = TConfig(start=base.start, num_obstacles=base.num_obstacles, obstacle_radius=base.obstacle_radius,
               slowness_max=base.slowness_max, slow_width=base.slow_width, resolution=120, seed=base.seed)
obstacles = torus_obstacles(tcfg)
axis = np.linspace(-np.pi, np.pi, 120, endpoint=False)
g1, g2 = np.meshgrid(axis, axis)
thetas = jnp.asarray(np.stack([g1.ravel(), g2.ravel()], -1), jnp.float32)
slow = np.asarray(torus_slowness(thetas, obstacles, base.slowness_max, base.slow_width)).reshape(120, 120)
gt = fast_marching_torus(1.0 / slow, base.start, 120).ravel()
inside = np.asarray(torus_sdf(thetas, obstacles)) < 0.0
start = jnp.asarray(base.start)

rows = []
for field in ("mlp", "splat"):
    for ncol in (256, 512, 1024, 2048):
        cfg = dataclasses.replace(base, field=field, num_collocation=ncol)
        params = solve(cfg, obstacles)
        _ = predict(raw_fn_for(cfg), params, thetas[:64], start, cfg.tau_bias, cfg.tau_min)  # warm jit
        t0 = time.time()
        pred = np.asarray(predict(raw_fn_for(cfg), params, thetas, start, cfg.tau_bias, cfg.tau_min))
        qt = 1000 * (time.time() - t0)
        e = np.where(inside, np.nan, pred - gt)
        rms, mx = float(np.sqrt(np.nanmean(e**2))), float(np.nanmax(np.abs(e)))
        rows.append((field, ncol, rms, mx, qt))
        print(f"{field:>5} collocation={ncol:>4}: RMS={rms:.3f}  max={mx:.2f}  query={qt:.0f}ms/grid", flush=True)

print("\n===== MLP vs SPLAT (matched ~2.6k params, 6 obstacles, RRT* bounds) =====")
print(f"{'field':>6} {'colloc':>7} {'RMS':>7} {'max':>6} {'query_ms':>9}")
for f, n, r, m, q in rows:
    print(f"{f:>6} {n:>7} {r:>7.3f} {m:>6.2f} {q:>9.0f}")
print("MVS DONE")
