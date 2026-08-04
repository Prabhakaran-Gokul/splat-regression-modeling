"""NTFields-style training strategy: ``T(x, y) = ‖x − y‖ / τ(x, y)``.

Straight port of the original NTFields (Ni & Qureshi 2023) factorization: arrival time is the
geodesic distance divided by a network-predicted normalized speed ``τ ∈ (0, 1]`` (``τ=1`` ⇒ free
space, straight-line unit-speed travel time; ``τ→0`` near obstacles inflates ``T``). ``τ`` is
evaluated through whichever backend is passed in (``srms/methods/backends/{srm,mlp,kan}``) via a
sigmoid, ``τ = τ_min + (1−τ_min)·σ(g(θ)+bias)`` — ``bias`` starts ``τ`` near 1 (free space) at
init; ``tau_min=0`` (the default) recovers the *un-floored* paper formulation exactly, ``σ(·)``
alone bounding ``τ`` to ``(0, 1)``. ``tau_min>0`` is available as an ablation knob (torus.py's
``solve_ntfields`` used ``tau_min=0.25`` to floor the ``τ→0 ⇒ T→∞`` runaway) but is off by default
here to test the literal paper form first. ``T(start)=0`` automatically since ``base(start)=0``.

Trained with the same symmetric speed-match residual ``q + 1/q − 2`` as
``weak_supervision.py``'s roadmap field — both strategies are of the form
``T = base(θ) · factor(correction(θ))``; only the base (analytic geodesic here vs. RRT* roadmap
there) and factor (``1/τ`` here vs. ``exp(g)`` there) differ. Optional causal weighting on the
residual term is available (``cfg.causal``, see ``training_aids.py``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import trange

from srms.methods.strategies import training_aids
from srms.methods.strategies.eikonal import sample_collocation


def tau(backend, params, theta: jnp.ndarray, env, tau_bias: float, tau_min: float) -> jnp.ndarray:
    """Normalized speed ``τ(θ) = τ_min + (1−τ_min)·σ(g(θ)+bias) ∈ [τ_min, 1)``."""
    raw = backend.eval_raw(params, theta[None, :], env)[0, 0]
    return tau_min + (1.0 - tau_min) * jax.nn.sigmoid(raw + tau_bias)


def field_ntfields(backend, params, theta: jnp.ndarray, env, tau_bias: float, tau_min: float) -> jnp.ndarray:
    """``T(θ) = geodesic(θ, start) / τ(θ)``."""
    base = env.geodesic(theta, jnp.asarray(env.start, dtype=jnp.float32))
    return base / tau(backend, params, theta, env, tau_bias, tau_min)


def predict(backend, params, thetas: jnp.ndarray, env, tau_bias: float, tau_min: float) -> jnp.ndarray:
    """Evaluate the NTFields-style field over a batch of points, returned as [N]."""
    return jax.vmap(lambda t: field_ntfields(backend, params, t, env, tau_bias, tau_min))(thetas)


def ntfields_residual(
    backend, params, thetas: jnp.ndarray, slow: jnp.ndarray, env, tau_bias: float, tau_min: float
) -> jnp.ndarray:
    """Symmetric speed-match penalty ``q + 1/q − 2`` (``q = ‖∇T‖_{g⁻¹}/s``) for the NTFields field."""

    def t_single(x):
        return field_ntfields(backend, params, x, env, tau_bias, tau_min)

    grad = jax.vmap(jax.grad(t_single))(thetas)
    metric = jax.vmap(env.metric_inv)(thetas)
    grad_sq = jnp.einsum("ni,nij,nj->n", grad, metric, grad)
    q = jnp.sqrt(jnp.clip(grad_sq, 1e-12, None)) / slow
    return q + 1.0 / q - 2.0


def solve(env, cfg, backend, checkpoint=None, progress_fn=None):
    """Fit the NTFields ``T=base/τ`` factorization to the symmetric speed-match residual.

    ``progress_fn(step, metrics)``, if given, is called every ``cfg.log_every`` steps with a dict of
    scalar training metrics (``loss``, ``residual``) — e.g. to log to mlflow (see ``run.py``).
    """
    params = backend.init_params(jax.random.PRNGKey(cfg.seed), env, cfg)
    rng = np.random.default_rng(cfg.seed)

    # Same q + 1/q - 2 singularity as weak_supervision.py's roadmap_residual, here with no obstacle-band
    # gating to damp it (and, at tau_min=0, an unfloored 1/tau that can itself blow up) — clip
    # proactively rather than waiting to rediscover the NaN. Causal weighting (below) helps but, per the
    # sphere+srm NaN seen with tau_min=0, cannot rescue a gradient that has already gone NaN.
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(cfg.lr))
    opt_state = optimizer.init(params)

    def loss_fn(p, colloc, slow, rate):
        residual = ntfields_residual(backend, p, colloc, slow, env, cfg.tau_bias, cfg.tau_min)
        squared = residual**2
        residual_loss = training_aids.causal_loss(env, colloc, squared, rate) if cfg.causal else jnp.mean(squared)
        return residual_loss, {"residual": residual_loss}

    @jax.jit
    def step(p, state, colloc, slow, rate):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(p, colloc, slow, rate)
        updates, state = optimizer.update(grads, state, p)
        return optax.apply_updates(p, updates), state, loss, aux

    progress = trange(cfg.steps, desc="ntfields")
    for i in progress:
        colloc = sample_collocation(env, rng, cfg.num_collocation, cfg.source_radius)
        slow = env.slowness(colloc)
        rate = jnp.float32(training_aids.causal_rate(cfg, i)) if cfg.causal else jnp.float32(0.0)
        params, opt_state, loss, aux = step(params, opt_state, colloc, slow, rate)
        if i % cfg.log_every == 0:
            progress.set_description(f"ntfields — log10(loss) = {float(jnp.log10(loss + 1e-12)):.3f}")
            if progress_fn is not None:
                progress_fn(i, {"loss": float(loss), "residual": float(aux["residual"])})
        if checkpoint is not None and i > 0 and i % cfg.checkpoint_every == 0:
            checkpoint(params, i)
    return params
