"""Weak-supervision training strategy: RRT* soft-min base × exp(correction) refinement.

The RRT* tree acts as weak supervision for the PDE — a coarse, obstacle-aware
cost-to-come the correction only has to refine locally, rather than route
around obstacles itself.

``T(θ) = roadmap_base(θ) · exp(g(θ))`` where ``roadmap_base`` is a
differentiable, obstacle-aware soft-min cost-to-come over an RRT* tree (mesh-
free, so it scales past grid dimensions) and ``g`` is a small correction
evaluated through whichever backend is passed in
(``srms/methods/backends/{srm,mlp,kan}``; ``exp(0)=1`` at init ⇒ ``T≈base``).
``T(start)=0`` automatically since the source node contributes a zero-cost,
zero-length term. This strategy only calls ``backend.init_params(key, env,
cfg)`` / ``backend.eval_raw(params, X, env)``, so the same objective and
training loop work for any of them.

Training loop is the same "simplest, dumbest" style as ``eikonal.py``: plain
fresh random collocation each step, a symmetric speed-match residual against
the base plus a small regularizer keeping the correction small, no adaptive
sampling/causal weighting/anchors.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import trange

from srms.environments import sampling


def roadmap_base(theta: jnp.ndarray, nodes: jnp.ndarray, costs: jnp.ndarray, gamma: float, env, ksamp: int) -> jnp.ndarray:
    """Obstacle-aware cost-to-come: soft-min_i (cost_i + slowness-weighted ‖hop from node_i to θ‖).

    Uses log_map_ambient (not log_map) since the hop below needs an *ambient* tangent vector it can
    add to `nodes` and then re-project with wrap_point — on a flat chart the two coincide, but not on
    an embedded manifold like the sphere.
    """
    disp = jax.vmap(lambda mu: env.log_map_ambient(mu, theta))(nodes)  # [N, dim]
    ts = jnp.linspace(0.0, 1.0, ksamp)
    seg = nodes[None, :, :] + ts[:, None, None] * disp[None, :, :]  # [K, N, dim] points along each last hop
    mean_slow = env.slowness(env.wrap_point(seg.reshape(-1, env.dim))).reshape(ksamp, -1).mean(0)  # [N]
    wlen = jnp.linalg.norm(disp, axis=1) * mean_slow
    return -gamma * jax.scipy.special.logsumexp(-(costs + wlen) / gamma)


def field_roadmap(
    backend, params, theta: jnp.ndarray, nodes: jnp.ndarray, costs: jnp.ndarray, gamma: float, env, ksamp: int
) -> jnp.ndarray:
    """T(θ) = roadmap_base(θ) · exp(g(θ))."""
    base = roadmap_base(theta, nodes, costs, gamma, env, ksamp)
    correction = backend.eval_raw(params, theta[None, :], env)[0, 0]
    return base * jnp.exp(correction)


def predict(
    backend, params, thetas: jnp.ndarray, nodes: jnp.ndarray, costs: jnp.ndarray, gamma: float, env, ksamp: int
) -> jnp.ndarray:
    """Evaluate the roadmap-based field over a batch of points, returned as [N]."""
    return jax.vmap(lambda t: field_roadmap(backend, params, t, nodes, costs, gamma, env, ksamp))(thetas)


def roadmap_residual(
    backend,
    params,
    thetas: jnp.ndarray,
    slow: jnp.ndarray,
    nodes: jnp.ndarray,
    costs: jnp.ndarray,
    gamma: float,
    env,
    ksamp: int,
) -> jnp.ndarray:
    """Symmetric speed-match penalty ``q + 1/q − 2`` (``q = ‖∇T‖/s``) for the roadmap-based field."""

    def t_single(x):
        return field_roadmap(backend, params, x, nodes, costs, gamma, env, ksamp)

    grad = jax.vmap(jax.grad(t_single))(thetas)
    metric = jax.vmap(env.metric_inv)(thetas)
    grad_sq = jnp.einsum("ni,nij,nj->n", grad, metric, grad)
    q = jnp.sqrt(grad_sq + 1e-12) / slow
    return q + 1.0 / q - 2.0


def solve(env, cfg, backend, checkpoint=None):
    """Fit a small correction on top of a coarse RRT*-roadmap base."""
    params = backend.init_params(jax.random.PRNGKey(cfg.seed), env, cfg)
    nodes, costs = sampling.build_roadmap(
        env, env.start, cfg.rrt_iters, cfg.rrt_step, cfg.rrt_radius, cfg.roadmap_nodes, cfg.seed
    )
    rng = np.random.default_rng(cfg.seed)

    optimizer = optax.adam(cfg.lr)
    opt_state = optimizer.init(params)

    def loss_fn(p, colloc, slow):
        residual = roadmap_residual(backend, p, colloc, slow, nodes, costs, cfg.roadmap_gamma, env, cfg.roadmap_hop)
        correction = backend.eval_raw(p, colloc, env).ravel()
        return jnp.mean(residual**2) + cfg.base_reg * jnp.mean(correction**2)

    @jax.jit
    def step(p, state, colloc, slow):
        loss, grads = jax.value_and_grad(loss_fn)(p, colloc, slow)
        updates, state = optimizer.update(grads, state, p)
        return optax.apply_updates(p, updates), state, loss

    progress = trange(cfg.steps, desc="weak_supervision")
    for i in progress:
        colloc = jnp.asarray(env.sample_domain(rng, cfg.num_collocation), dtype=jnp.float32)
        slow = env.slowness(colloc)
        params, opt_state, loss = step(params, opt_state, colloc, slow)
        if i % 25 == 0:
            progress.set_description(f"weak_supervision — log10(loss) = {float(jnp.log10(loss + 1e-12)):.3f}")
        if checkpoint is not None and i > 0 and i % cfg.checkpoint_every == 0:
            checkpoint(params, i)
    return params
