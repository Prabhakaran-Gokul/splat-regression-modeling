"""Test the principled densification fix — preserve Adam moments for survivors + grad clipping.

Keeps the faithful symmetric speed-match residual UNCHANGED (so the MLP baseline stays honest). Only the
densification mechanics change:
  (1) grad clip via optax.clip_by_global_norm  — bounds any single catastrophic update.
  (2) surgical opt-state growth: prune+spawn slices Adam mu/nu by the keep-mask and zero-pads for new
      splats, instead of a full Adam reset that re-kicks every converged splat at full lr.
Logs EVERY 10 steps from 0: loss, max|res|, min/max slope, #splats. Reports final RMS vs FMM.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ntfields_baseline import residual, splat_init, splat_raw, predict
from torus import (
    fast_marching_torus, torus_obstacles, torus_sdf, torus_slowness, wrap, Config as TConfig,
)
from ntfields_baseline import Config

import sys
SMOKE = "--smoke" in sys.argv  # fast validation: few steps, frequent densify
EPS0 = "--eps0" in sys.argv    # drop vanishing-viscosity (splats are already C∞)
SCALE_FLOOR = 0.0 if "--nofloor" in sys.argv else 0.07  # min splat covariance singular value (anti-spike)
cfg = dataclasses.replace(
    Config(), field="splat", densify=True, supervision="bounds",
    init_splats=192, max_splats=512,
    densify_every=60 if SMOKE else 400, spawn_per=48,
    num_collocation=512 if SMOKE else 2048, steps=300 if SMOKE else 4000,
    epsilon=0.0 if EPS0 else 0.01, seed=1, prune_thresh=5e-4, spawn_scale=0.3,
)
GRAD_CLIP = 1.0


def project_scales(params):
    """Floor each splat's covariance singular values so no splat can collapse into a gradient spike."""
    if SCALE_FLOOR <= 0.0:
        return params
    V, A, B = params
    U, S, Vt = jnp.linalg.svd(A)  # A: [k,2,2]
    S = jnp.clip(S, SCALE_FLOOR, None)
    A = jnp.einsum("kij,kj,kjl->kil", U, S, Vt)
    return (V, A, B)

tcfg = TConfig(start=cfg.start, num_obstacles=cfg.num_obstacles, obstacle_radius=cfg.obstacle_radius,
               slowness_max=cfg.slowness_max, slow_width=cfg.slow_width, resolution=120, seed=cfg.seed)
obstacles = torus_obstacles(tcfg)
start = jnp.asarray(cfg.start)
start_np = np.asarray(cfg.start)
rng = np.random.default_rng(cfg.seed)
raw_fn = splat_raw

# FMM ground truth for final RMS
axis = np.linspace(-np.pi, np.pi, 120, endpoint=False)
g1, g2 = np.meshgrid(axis, axis)
thetas = jnp.asarray(np.stack([g1.ravel(), g2.ravel()], -1), jnp.float32)
slow_grid = np.asarray(torus_slowness(thetas, obstacles, cfg.slowness_max, cfg.slow_width)).reshape(120, 120)
gt = fast_marching_torus(1.0 / slow_grid, cfg.start, 120).ravel()
inside = np.asarray(torus_sdf(thetas, obstacles)) < 0.0

# weak RRT* bounds (H-NTFields supervision)
from torus import rrt_star_anchors_shadow
wtcfg = TConfig(start=cfg.start, num_obstacles=cfg.num_obstacles, obstacle_radius=cfg.obstacle_radius,
                slowness_max=cfg.slowness_max, slow_width=cfg.slow_width, num_weak=cfg.num_weak,
                weak_clearance=cfg.weak_clearance, shadow_pref=cfg.shadow_pref, rrt_iters=cfg.rrt_iters,
                rrt_step=cfg.rrt_step, rrt_radius=cfg.rrt_radius, seed=cfg.seed)
weak_x, weak_t = rrt_star_anchors_shadow(wtcfg, obstacles)

params = splat_init(jax.random.PRNGKey(cfg.seed), cfg.init_splats, cfg.init_scale)
optimizer = optax.chain(optax.clip_by_global_norm(GRAD_CLIP), optax.adam(cfg.lr))
opt_state = optimizer.init(params)
rate0 = cfg.causal_strength / cfg.num_collocation


def loss_fn(params, colloc, slow_full, order, rate, lam):
    slow = 1.0 + lam * (slow_full - 1.0)
    gate = jax.nn.sigmoid(torus_sdf(colloc, obstacles) / 0.2)
    res = residual(raw_fn, params, colloc, slow, start, cfg.tau_bias, cfg.tau_min, cfg.epsilon)
    ro, go = res[order], gate[order]
    up = jnp.cumsum(go * ro) - go * ro
    w = jax.lax.stop_gradient(jnp.exp(-rate * up)) * go
    pde = jnp.sum(w * ro) / (jnp.sum(w) + 1e-9)
    pred = predict(raw_fn, params, weak_x, start, cfg.tau_bias, cfg.tau_min)
    return pde + cfg.weak_weight * jnp.mean((pred - weak_t) ** 2)


@jax.jit
def step(params, opt_state, colloc, slow_full, order, rate, lam):
    loss, grads = jax.value_and_grad(loss_fn)(params, colloc, slow_full, order, rate, lam)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    return project_scales(optax.apply_updates(params, updates)), opt_state, loss


def slope_minmax(params, thetas):
    def T(t):
        base = jnp.linalg.norm(wrap(t - start))
        return base / (cfg.tau_min + (1 - cfg.tau_min) * jax.nn.sigmoid(raw_fn(params, t) + cfg.tau_bias))
    grad = jax.vmap(jax.grad(T))(thetas)
    sl = jnp.sqrt(jnp.clip(jnp.sum(grad**2, axis=1), 1e-12, None))
    return float(jnp.min(sl)), float(jnp.max(sl))


def densify_preserve(params, opt_state, res_mag_fn):
    """Prune+spawn AND surgically grow Adam state: keep survivors' moments, zero-init spawns.

    Generic regrow: any leaf (params OR optimizer moments) whose leading dim == old splat count is
    sliced to survivors and zero-padded for spawns; scalars (Adam's step count) pass through. Robust
    to optax.chain nesting — no reliance on the exact state namedtuple layout.
    """
    V = np.asarray(params[0])
    old_k = len(V)
    keep = np.abs(V[:, 0]) > cfg.prune_thresh
    if keep.sum() < 16:
        keep = np.ones(old_k, bool)
    idx = np.where(keep)[0]
    room = cfg.max_splats - len(idx)
    spawn_B = np.zeros((0, 2), np.float32)
    if room > 0:
        pool = rng.uniform(-np.pi, np.pi, (3000, 2))
        clr = np.asarray(torus_sdf(jnp.asarray(pool, jnp.float32), obstacles))
        pool = pool[clr > 0.1]
        r = np.asarray(res_mag_fn(jnp.asarray(pool, jnp.float32)))
        k = int(min(cfg.spawn_per, room, len(pool)))
        spawn_B = pool[np.argsort(r)[-k:]].astype(np.float32)
    ns = len(spawn_B)

    def regrow_leaf(x, spawn_fill=0.0):  # grow along axis-0 if this leaf indexes splats
        a = np.asarray(x)
        if a.ndim >= 1 and a.shape[0] == old_k:
            pad = np.full((ns,) + a.shape[1:], spawn_fill, a.dtype)
            return jnp.asarray(np.concatenate([a[idx], pad], 0))
        return x  # scalar (Adam count) or non-splat leaf

    # params: spawn with near-zero weight, spawn_scale identity cov, at high-residual centres
    newV = np.concatenate([V[idx], np.full((ns, 1), 1e-4, np.float32)])
    newA = np.concatenate([np.asarray(params[1])[idx],
                           np.repeat((cfg.spawn_scale * np.eye(2))[None], ns, 0).astype(np.float32)])
    newB = np.concatenate([np.asarray(params[2])[idx], spawn_B])
    new_params = (jnp.asarray(newV), jnp.asarray(newA), jnp.asarray(newB))
    # optimizer moments: survivors preserved, spawns zero (fresh Adam only for the new splats)
    new_opt_state = jax.tree_util.tree_map(lambda x: regrow_leaf(x), opt_state)
    return new_params, new_opt_state


def resample():
    pool = rng.uniform(-np.pi, np.pi, size=(2 * cfg.num_collocation, 2))
    keep = pool[np.linalg.norm(((pool - start_np + np.pi) % (2 * np.pi)) - np.pi, axis=1) > 0.25][: cfg.num_collocation]
    colloc = jnp.asarray(keep, jnp.float32)
    slow_full = torus_slowness(colloc, obstacles, cfg.slowness_max, cfg.slow_width)
    order = jnp.argsort(jnp.linalg.norm(wrap(colloc - start), axis=1))
    return colloc, slow_full, order


colloc, slow_full, order = resample()
print(f"{'step':>5} {'#spl':>5} {'loss':>11} {'max|res|':>11} {'minSlope':>9} {'maxSlope':>10}  note", flush=True)
for i in range(cfg.steps):
    if i % 300 == 0 and i > 0:
        colloc, slow_full, order = resample()
    note = ""
    if i > 0 and i % cfg.densify_every == 0 and i < 0.8 * cfg.steps:
        cur = params
        def rmag(pts):  # target genuine speed-mismatch (eps=0), NOT the singular viscosity Laplacian
            sl = torus_slowness(pts, obstacles, cfg.slowness_max, cfg.slow_width)
            return jnp.abs(residual(raw_fn, cur, pts, sl, start, cfg.tau_bias, cfg.tau_min, 0.0))
        params, opt_state = densify_preserve(params, opt_state, rmag)
        note = f"DENSIFY->{params[0].shape[0]} (moments preserved)"
    lam = min(1.0, cfg.lambda_init + (1 - cfg.lambda_init) * i / (cfg.anneal_frac * cfg.steps + 1e-9))
    rate = rate0 * (1 - i / cfg.steps)
    params, opt_state, loss = step(params, opt_state, colloc, slow_full, order, jnp.float32(rate), jnp.float32(lam))
    if i % 40 == 0 or note:
        smin, smax = slope_minmax(params, colloc)
        res = residual(raw_fn, params, colloc, 1.0 + lam * (slow_full - 1), start, cfg.tau_bias, cfg.tau_min, cfg.epsilon)
        print(f"{i:>5} {params[0].shape[0]:>5} {float(loss):>11.4f} {float(jnp.max(jnp.abs(res))):>11.2f} "
              f"{smin:>9.4f} {smax:>10.2f}  {note}", flush=True)

pred = np.asarray(predict(raw_fn, params, thetas, start, cfg.tau_bias, cfg.tau_min))
e = np.where(inside, np.nan, pred - gt)
rms, mx = float(np.sqrt(np.nanmean(e**2))), float(np.nanmax(np.abs(e)))
print(f"\nFINAL: splats={params[0].shape[0]}  RMS={rms:.4f}  max|err|={mx:.4f}", flush=True)
print("FIX DONE", flush=True)
