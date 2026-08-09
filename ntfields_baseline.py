"""Faithful single-source NTFields / P-NTFields baseline on the 2-torus — MLP field, PURE PHYSICS.

The physics *constructs* the field (no roadmap, no pre-solution): factored time `T = base/τ` with an MLP
`τ`, trained on the symmetric speed-match Eikonal loss with P-NTFields' progressive speed-annealing and
viscosity `ε·ΔT`. This is the honest baseline our splat should be compared against like-for-like (same
loss, swap MLP↔splat). `--supervision bounds` adds sparse RRT* travel-time anchors → H-NTFields.

Single-source adaptation of the two-point NTFields (we fix the source = start); the mechanism —
MLP + factored field + speed-match physics loss + progressive + viscosity — is faithful. Reuses the
torus scene/FMM from `torus.py`.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from tqdm import trange

from torus import (
    eval_splat_torus, fast_marching_torus, render, rrt_star_anchors_shadow, torus_obstacles,
    torus_sdf, torus_slowness, wrap, Config as TConfig,
)


@dataclasses.dataclass
class Config:
    field: Literal["mlp", "splat"] = "mlp"  # τ-predictor representation — identical pipeline otherwise
    supervision: Literal["none", "bounds"] = "none"  # none = NTFields (pure physics); bounds = H-NTFields
    widths: tuple[int, ...] = (4, 60, 60, 1)  # MLP encoder (~4k params, ≈ splat budget for fairness)
    num_splats: int = 384
    init_scale: float = 0.35
    # adaptive splats (3DGS-style): let the model decide where/how many splats go
    densify: bool = False
    init_splats: int = 192  # start with enough coverage to avoid catastrophic under-constrained regions
    max_splats: int = 512
    densify_every: int = 400
    spawn_per: int = 48  # new splats added per densification (at highest-residual points)
    prune_thresh: float = 5e-4  # prune splats with |weight| below this
    spawn_scale: float = 0.3  # initial scale of a spawned splat (moderate → local but not spiky)
    scale_floor: float = 0.07  # min splat covariance singular value — prevents collapse-to-spike (adaptive only)
    grad_clip: float = 1.0  # global-norm gradient clip during adaptive densification
    start: tuple[float, float] = (-1.5, -1.5)
    num_obstacles: int = 3
    obstacle_radius: tuple[float, float] = (0.5, 0.9)
    slowness_max: float = 10.0
    slow_width: float = 0.15
    epsilon: float = 0.01  # viscosity coefficient ε·ΔT (P-NTFields); 0 disables
    tau_bias: float = 4.0
    tau_min: float = 0.25
    num_collocation: int = 2048
    causal: bool = True
    causal_strength: float = 5.0
    lambda_init: float = 0.15  # progressive speed-annealing s_λ = 1 + λ(s−1)
    anneal_frac: float = 0.4
    weak_weight: float = 0.5
    num_weak: int = 30
    weak_clearance: float = 0.4
    shadow_pref: float = 5.0
    rrt_iters: int = 1200
    rrt_step: float = 0.5
    rrt_radius: float = 0.9
    steps: int = 4000
    lr: float = 2e-3
    resolution: int = 120
    seed: int = 1
    error_clip: float = 0.2
    out_dir: str = "figures"


def mlp_init(key, widths):
    params = []
    for i in range(len(widths) - 1):
        key, k = jax.random.split(key)
        W = jax.random.normal(k, (widths[i], widths[i + 1])) * jnp.sqrt(2.0 / widths[i])
        params.append((W, jnp.zeros(widths[i + 1])))
    return params


def mlp_raw(params, theta):
    x = jnp.concatenate([jnp.cos(theta), jnp.sin(theta)])  # periodic encoding respects the torus
    for W, b in params[:-1]:
        x = jnp.tanh(x @ W + b)
    W, b = params[-1]
    return (x @ W + b)[0]


def splat_init(key, num_splats, init_scale):
    centres = jax.random.uniform(key, (num_splats, 2), minval=-jnp.pi, maxval=jnp.pi)
    scales = jnp.repeat((init_scale * jnp.eye(2))[None], num_splats, axis=0)
    return (jnp.zeros((num_splats, 1)), scales, centres)


def splat_raw(params, theta):
    return eval_splat_torus(theta[None, :], params)[0, 0]


def _param_count(params):
    return int(sum(np.prod(np.asarray(a).shape) for leaf in jax.tree_util.tree_leaves(params) for a in [leaf]))


# raw_fn is chosen once per solve (static), so field/residual/predict are shared MLP↔splat
def field(raw_fn, params, theta, start, tau_bias, tau_min):
    base = jnp.linalg.norm(wrap(theta - start))
    return base / (tau_min + (1.0 - tau_min) * jax.nn.sigmoid(raw_fn(params, theta) + tau_bias))


def residual(raw_fn, params, thetas, slow, start, tau_bias, tau_min, eps):
    def T(t):
        return field(raw_fn, params, t, start, tau_bias, tau_min)

    grad = jax.vmap(jax.grad(T))(thetas)
    slope = jnp.sqrt(jnp.clip(jnp.sum(grad**2, axis=1), 1e-12, None))
    if eps > 0.0:  # viscosity: ‖∇T‖ + ε·ΔT should equal s (P-NTFields vanishing-viscosity)
        lap = jax.vmap(lambda t: jnp.trace(jax.hessian(T)(t)))(thetas)
        slope = slope + eps * lap
    q = jnp.clip(slope, 1e-6, None) / slow
    return q + 1.0 / q - 2.0  # symmetric speed-match (accurate); stability comes from a non-sparse start + gentle spawns


def predict(raw_fn, params, thetas, start, tau_bias, tau_min):
    return jax.vmap(lambda t: field(raw_fn, params, t, start, tau_bias, tau_min))(thetas)


def project_scales(params, floor):
    """Floor each splat's covariance singular values so no splat can collapse into a gradient spike.

    A sum of Gaussians cannot represent the true Eikonal kink, so unconstrained optimization chases it by
    driving a splat's scale → 0 (an infinite-gradient spike that then blows the ‖∇T‖-based residual up).
    Flooring the SVD makes that collapse — and the divergence it caused — impossible.
    """
    if floor <= 0.0:
        return params
    V, A, B = params
    U, S, Vt = jnp.linalg.svd(A)  # A: [k,2,2]
    A = jnp.einsum("kij,kj,kjl->kil", U, jnp.clip(S, floor, None), Vt)
    return (V, A, B)


def densify_prune(cfg, params, opt_state, res_mag_fn, obstacles, rng):
    """3DGS-style prune+spawn that ALSO surgically grows the optimizer state.

    Prune near-zero-weight splats, spawn new ones (near-zero weight) at the highest speed-mismatch free
    points. Any leaf (params or Adam moment) whose leading dim equals the old splat count is sliced to
    survivors and zero-padded for spawns — so survivors keep their Adam moments and only the new splats
    get fresh (zero) moments, instead of a full optimizer reset that re-kicks every converged splat.
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

    def regrow(x):  # grow along axis-0 iff this leaf indexes splats; scalars (Adam count) pass through
        a = np.asarray(x)
        if a.ndim >= 1 and a.shape[0] == old_k:
            return jnp.asarray(np.concatenate([a[idx], np.zeros((ns,) + a.shape[1:], a.dtype)], 0))
        return x

    newV = np.concatenate([V[idx], np.full((ns, 1), 1e-4, np.float32)])  # gentle: spawns start near-zero weight
    newA = np.concatenate([np.asarray(params[1])[idx],
                           np.repeat((cfg.spawn_scale * np.eye(2))[None], ns, 0).astype(np.float32)])
    newB = np.concatenate([np.asarray(params[2])[idx], spawn_B])
    new_params = (jnp.asarray(newV), jnp.asarray(newA), jnp.asarray(newB))
    return new_params, jax.tree_util.tree_map(regrow, opt_state)


def solve(cfg, obstacles):
    if cfg.field == "splat":
        raw_fn = splat_raw
        n0 = cfg.init_splats if cfg.densify else cfg.num_splats
        params = splat_init(jax.random.PRNGKey(cfg.seed), n0, cfg.init_scale)
    else:
        raw_fn = mlp_raw
        params = mlp_init(jax.random.PRNGKey(cfg.seed), cfg.widths)
    print(f"[{cfg.field}] {_param_count(params)} params (densify={cfg.densify})", flush=True)
    start = jnp.asarray(cfg.start)
    start_np = np.asarray(cfg.start)
    rng = np.random.default_rng(cfg.seed)
    adaptive = cfg.densify and cfg.field == "splat"  # scale-floor + grad-clip apply ONLY here (baselines unchanged)
    optimizer = (optax.chain(optax.clip_by_global_norm(cfg.grad_clip), optax.adam(cfg.lr))
                 if adaptive else optax.adam(cfg.lr))
    opt_state = optimizer.init(params)
    rate0 = cfg.causal_strength / cfg.num_collocation

    if cfg.supervision == "bounds":  # H-NTFields: sparse RRT* travel-time anchors
        tcfg = TConfig(start=cfg.start, num_obstacles=cfg.num_obstacles, obstacle_radius=cfg.obstacle_radius,
                       slowness_max=cfg.slowness_max, slow_width=cfg.slow_width, num_weak=cfg.num_weak,
                       weak_clearance=cfg.weak_clearance, shadow_pref=cfg.shadow_pref, rrt_iters=cfg.rrt_iters,
                       rrt_step=cfg.rrt_step, rrt_radius=cfg.rrt_radius, seed=cfg.seed)
        weak_x, weak_t = rrt_star_anchors_shadow(tcfg, obstacles)
    else:
        weak_x, weak_t = jnp.zeros((0, 2)), jnp.zeros((0,))

    def loss_fn(params, colloc, slow_full, order, rate, lam):
        slow = 1.0 + lam * (slow_full - 1.0)
        gate = jax.nn.sigmoid(torus_sdf(colloc, obstacles) / 0.2)
        res = residual(raw_fn, params, colloc, slow, start, cfg.tau_bias, cfg.tau_min, cfg.epsilon)
        if cfg.causal:
            ro, go = res[order], gate[order]
            up = jnp.cumsum(go * ro) - go * ro
            w = jax.lax.stop_gradient(jnp.exp(-rate * up)) * go
            pde = jnp.sum(w * ro) / (jnp.sum(w) + 1e-9)
        else:
            pde = jnp.sum(gate * res) / (jnp.sum(gate) + 1e-9)
        if weak_x.shape[0] > 0:
            pred = predict(raw_fn, params, weak_x, start, cfg.tau_bias, cfg.tau_min)
            pde = pde + cfg.weak_weight * jnp.mean((pred - weak_t) ** 2)
        return pde

    @jax.jit
    def step(params, opt_state, colloc, slow_full, order, rate, lam):
        loss, grads = jax.value_and_grad(loss_fn)(params, colloc, slow_full, order, rate, lam)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        if adaptive:
            new_params = project_scales(new_params, cfg.scale_floor)  # keep every splat above the spike threshold
        return new_params, opt_state, loss

    def resample():
        pool = rng.uniform(-np.pi, np.pi, size=(2 * cfg.num_collocation, 2))
        keep = pool[np.linalg.norm(((pool - start_np + np.pi) % (2 * np.pi)) - np.pi, axis=1) > 0.25][: cfg.num_collocation]
        colloc = jnp.asarray(keep, jnp.float32)
        slow_full = torus_slowness(colloc, obstacles, cfg.slowness_max, cfg.slow_width)
        order = jnp.argsort(jnp.linalg.norm(wrap(colloc - start), axis=1))
        return colloc, slow_full, order

    colloc, slow_full, order = resample()
    prog = trange(cfg.steps, desc=f"{cfg.field} ({cfg.supervision})")
    for i in prog:
        if i % 300 == 0:
            colloc, slow_full, order = resample()
        if adaptive and i > 0 and i % cfg.densify_every == 0 and i < 0.8 * cfg.steps:
            cur = params  # spawn at highest speed-MISMATCH free points (eps=0: not the singular viscosity Laplacian)
            def rmag(pts):
                sl = torus_slowness(pts, obstacles, cfg.slowness_max, cfg.slow_width)
                return jnp.abs(residual(raw_fn, cur, pts, sl, start, cfg.tau_bias, cfg.tau_min, 0.0))
            params, opt_state = densify_prune(cfg, params, opt_state, rmag, obstacles, rng)  # moments preserved
        lam = min(1.0, cfg.lambda_init + (1.0 - cfg.lambda_init) * i / (cfg.anneal_frac * cfg.steps + 1e-9))
        rate = rate0 * (1.0 - i / cfg.steps)
        params, opt_state, loss = step(params, opt_state, colloc, slow_full, order, jnp.float32(rate), jnp.float32(lam))
        if i % 100 == 0:
            k = params[0].shape[0] if cfg.field == "splat" else 0
            prog.set_description(f"{cfg.field}({k}) ({cfg.supervision}) — log10(loss)={float(jnp.log10(loss + 1e-12)):.2f}")
    return params


def raw_fn_for(cfg):
    return splat_raw if cfg.field == "splat" else mlp_raw


def main(cfg: Config) -> None:
    tcfg = TConfig(start=cfg.start, num_obstacles=cfg.num_obstacles, obstacle_radius=cfg.obstacle_radius,
                   slowness_max=cfg.slowness_max, slow_width=cfg.slow_width, resolution=cfg.resolution,
                   seed=cfg.seed, error_clip=cfg.error_clip, out_dir=cfg.out_dir)
    obstacles = torus_obstacles(tcfg)
    axis = np.linspace(-np.pi, np.pi, cfg.resolution, endpoint=False)
    g1, g2 = np.meshgrid(axis, axis)
    thetas = jnp.asarray(np.stack([g1.ravel(), g2.ravel()], -1), jnp.float32)
    shape = (cfg.resolution, cfg.resolution)
    slow = np.asarray(torus_slowness(thetas, obstacles, cfg.slowness_max, cfg.slow_width)).reshape(shape)
    gt = fast_marching_torus(1.0 / slow, cfg.start, cfg.resolution).ravel()
    inside = np.asarray(torus_sdf(thetas, obstacles)) < 0.0

    params = solve(cfg, obstacles)
    pred = np.asarray(predict(raw_fn_for(cfg), params, thetas, jnp.asarray(cfg.start), cfg.tau_bias, cfg.tau_min))
    metrics = render(tcfg, gt, pred, inside, shape, out_name=f"ntfields_{cfg.field}_{cfg.supervision}.png")
    print(f"{cfg.field} ({cfg.supervision}): RMS={metrics['rms']:.4e}  max|err|={metrics['max_abs']:.4e}")


if __name__ == "__main__":
    main(tyro.cli(Config))
