"""Diagnose WHY adaptive densification diverges — instrument the loss trajectory.

Fast reproduction (short steps, frequent densify, small collocation) with per-step logging of:
  loss, max|residual|, MIN slope over collocation (the 1/q bomb), #splats.
Hypothesis: `q + 1/q` explodes when slope→0 at a flat spot, and densify spawns splats there → feedback.
We log min-slope so we can SEE whether a vanishing slope precedes each loss spike.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ntfields_baseline import (
    Config, residual, splat_init, splat_raw, densify_prune,
)
from torus import torus_obstacles, torus_sdf, torus_slowness, wrap, Config as TConfig

cfg = dataclasses.replace(
    Config(), field="splat", densify=True, supervision="none",
    init_splats=192, max_splats=512, densify_every=60, spawn_per=48,
    num_collocation=512, steps=420, epsilon=0.01, seed=1,
)
tcfg = TConfig(start=cfg.start, num_obstacles=cfg.num_obstacles, obstacle_radius=cfg.obstacle_radius,
               slowness_max=cfg.slowness_max, slow_width=cfg.slow_width, seed=cfg.seed)
obstacles = torus_obstacles(tcfg)
start = jnp.asarray(cfg.start)
start_np = np.asarray(cfg.start)
rng = np.random.default_rng(cfg.seed)

params = splat_init(jax.random.PRNGKey(cfg.seed), cfg.init_splats, cfg.init_scale)
optimizer = optax.adam(cfg.lr)
opt_state = optimizer.init(params)
raw_fn = splat_raw


def slope_of(params, thetas):
    def T(t):
        base = jnp.linalg.norm(wrap(t - start))
        return base / (cfg.tau_min + (1 - cfg.tau_min) * jax.nn.sigmoid(raw_fn(params, t) + cfg.tau_bias))
    grad = jax.vmap(jax.grad(T))(thetas)
    return jnp.sqrt(jnp.clip(jnp.sum(grad**2, axis=1), 1e-12, None))


def loss_fn(params, colloc, slow_full, order, rate, lam):
    slow = 1.0 + lam * (slow_full - 1.0)
    gate = jax.nn.sigmoid(torus_sdf(colloc, obstacles) / 0.2)
    res = residual(raw_fn, params, colloc, slow, start, cfg.tau_bias, cfg.tau_min, cfg.epsilon)
    ro, go = res[order], gate[order]
    up = jnp.cumsum(go * ro) - go * ro
    w = jax.lax.stop_gradient(jnp.exp(-(rate) * up)) * go
    return jnp.sum(w * ro) / (jnp.sum(w) + 1e-9)


@jax.jit
def step(params, opt_state, colloc, slow_full, order, rate, lam):
    loss, grads = jax.value_and_grad(loss_fn)(params, colloc, slow_full, order, rate, lam)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, loss


def resample():
    pool = rng.uniform(-np.pi, np.pi, size=(2 * cfg.num_collocation, 2))
    keep = pool[np.linalg.norm(((pool - start_np + np.pi) % (2 * np.pi)) - np.pi, axis=1) > 0.25][: cfg.num_collocation]
    colloc = jnp.asarray(keep, jnp.float32)
    slow_full = torus_slowness(colloc, obstacles, cfg.slowness_max, cfg.slow_width)
    order = jnp.argsort(jnp.linalg.norm(wrap(colloc - start), axis=1))
    return colloc, slow_full, order


colloc, slow_full, order = resample()
rate0 = cfg.causal_strength / cfg.num_collocation
print(f"{'step':>5} {'#spl':>5} {'loss':>12} {'max|res|':>12} {'min_slope':>10}  note", flush=True)
for i in range(cfg.steps):
    if i % 300 == 0 and i > 0:
        colloc, slow_full, order = resample()
    note = ""
    if cfg.densify and i > 0 and i % cfg.densify_every == 0:
        cur = params
        def rmag(pts):
            sl = torus_slowness(pts, obstacles, cfg.slowness_max, cfg.slow_width)
            return jnp.abs(residual(raw_fn, cur, pts, sl, start, cfg.tau_bias, cfg.tau_min, cfg.epsilon))
        params = densify_prune(cfg, params, rmag, obstacles, rng)
        opt_state = optimizer.init(params)
        note = f"DENSIFY -> {params[0].shape[0]} splats, Adam reset"
    lam = min(1.0, cfg.lambda_init + (1 - cfg.lambda_init) * i / (cfg.anneal_frac * cfg.steps + 1e-9))
    rate = rate0 * (1 - i / cfg.steps)
    params, opt_state, loss = step(params, opt_state, colloc, slow_full, order, jnp.float32(rate), jnp.float32(lam))
    if i % 10 == 0 or note or float(loss) > 5.0:
        sl = slope_of(params, colloc)
        res = residual(raw_fn, params, colloc, 1.0 + lam * (slow_full - 1), start, cfg.tau_bias, cfg.tau_min, cfg.epsilon)
        print(f"{i:>5} {params[0].shape[0]:>5} {float(loss):>12.4f} {float(jnp.max(jnp.abs(res))):>12.2f} "
              f"{float(jnp.min(sl)):>10.5f}  {note}", flush=True)
print("DIAG DONE", flush=True)
