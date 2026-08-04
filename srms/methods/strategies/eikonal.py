"""Eikonal training strategy: plain physics-informed fit — boundary loss + PDE residual loss.

``T(θ)`` is a free (unfactored) field, evaluated through whichever backend is
passed in (``srms/methods/backends/{srm,mlp,kan}``) — this strategy only calls
``backend.init_params(key, env, cfg)`` / ``backend.eval_raw(params, X, env)``,
so the same objective and training loop work for any of them. The point-source
singularity is handled by fitting ``T ≈ eps·slowness(start)`` on a small sphere
of radius ``eps`` around the source (the local, locally-flat travel time); the
Eikonal PDE ``‖∇T‖ = 1/s`` is enforced on random collocation points excluding
that same ball. This mirrors ``_archive/preexisting/eikonal_splat.py``'s
``eikonal_loss``/``train_eikonal_splat`` (and the dynamic per-step resampling
of ``eikonal_nd_dynamic.py``) as closely as possible: no adaptive sampling,
causal weighting, obstacle-shadow anchors, or factored base — just BC + PDE loss.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import trange


def predict(backend, params, thetas: jnp.ndarray, env) -> jnp.ndarray:
    """Evaluate the raw field T over a batch of points, returned as [N]."""
    return backend.eval_raw(params, thetas, env)[:, 0]


def source_sphere(env, eps: float, n_pts: int, seed: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """``n_pts`` points at geodesic distance ``eps`` from the source, with the known local travel time."""
    rng = np.random.default_rng(seed)
    points = env.boundary_ring_np(rng, eps, n_pts)
    value = eps * float(env.slowness_np(np.asarray(env.start)[None, :])[0])
    return jnp.asarray(points, dtype=jnp.float32), jnp.full(n_pts, value, dtype=jnp.float32)


def sample_collocation(env, rng: np.random.Generator, n: int, exclude_radius: float) -> jnp.ndarray:
    """Uniform random collocation over the domain, excluding a ball around the source."""
    pool = env.sample_domain(rng, n * 10)
    dist = np.linalg.norm(env.displacement_np(np.asarray(env.start), pool), axis=-1)
    return jnp.asarray(pool[dist > exclude_radius][:n], dtype=jnp.float32)


def pde_residual(backend, params, thetas: jnp.ndarray, slow: jnp.ndarray, env) -> jnp.ndarray:
    """Eikonal residual ``‖∇T‖_g − 1/slow`` at each θ, ‖·‖_g via env.metric_inv (identity on a flat
    chart; a tangent-plane projector on an embedded manifold like the sphere)."""

    def u_single(x):
        return backend.eval_raw(params, x[None, :], env)[0, 0]

    grad = jax.vmap(jax.grad(u_single))(thetas)
    metric = jax.vmap(env.metric_inv)(thetas)
    grad_sq = jnp.einsum("ni,nij,nj->n", grad, metric, grad)
    grad_norm = jnp.sqrt(grad_sq + 1e-12)
    return grad_norm - 1.0 / slow


def solve(env, cfg, backend, checkpoint=None, progress_fn=None):
    """Fit a free field to the Eikonal PDE with a small-sphere boundary condition at the source.

    ``progress_fn(step, metrics)``, if given, is called every ``cfg.log_every`` steps with a dict of
    scalar training metrics (``loss``, ``bc``, ``pde``) — e.g. to log to mlflow (see ``run.py``).
    """
    params = backend.init_params(jax.random.PRNGKey(cfg.seed), env, cfg)
    rng = np.random.default_rng(cfg.seed)
    src_pts, src_vals = source_sphere(env, cfg.source_radius, cfg.n_sphere, cfg.seed)

    optimizer = optax.adam(cfg.lr)
    opt_state = optimizer.init(params)

    def loss_fn(p, colloc, slow):
        bc = jnp.mean((predict(backend, p, src_pts, env) - src_vals) ** 2)
        pde = jnp.mean(pde_residual(backend, p, colloc, slow, env) ** 2)
        return bc + cfg.physics_weight * pde, {"bc": bc, "pde": pde}

    @jax.jit
    def step(p, state, colloc, slow):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(p, colloc, slow)
        updates, state = optimizer.update(grads, state, p)
        return optax.apply_updates(p, updates), state, loss, aux

    progress = trange(cfg.steps, desc="eikonal")
    for i in progress:
        colloc = sample_collocation(env, rng, cfg.num_collocation, cfg.source_radius)
        slow = env.slowness(colloc)
        params, opt_state, loss, aux = step(params, opt_state, colloc, slow)
        if i % cfg.log_every == 0:
            progress.set_description(f"eikonal — log10(loss) = {float(jnp.log10(loss + 1e-12)):.3f}")
            if progress_fn is not None:
                progress_fn(i, {"loss": float(loss), "bc": float(aux["bc"]), "pde": float(aux["pde"])})
        if checkpoint is not None and i > 0 and i % cfg.checkpoint_every == 0:
            checkpoint(params, i)
    return params
