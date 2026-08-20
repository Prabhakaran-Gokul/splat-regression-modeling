"""MLP backend: a SIREN-style multi-layer perceptron with periodic (sin) activations.

Alternative to ``srms/methods/backends/srm.py``'s splat mixture — same three entry
points so a training strategy (``srms/methods/strategies``) never needs to know
which one it's using:

- ``init_params(key, env, cfg, p=1) -> params`` — reads ``cfg.mlp_width``,
  ``cfg.mlp_depth``, ``cfg.mlp_omega0``; other backends read their own fields
  off the same flat ``cfg``.
- ``eval_raw(params, X, env) -> [n, p]`` — the raw (unfactored) field value.
- ``post_step(params, cfg, env) -> params`` — identity here (no manifold-embedded
  parameters to reproject; see ``srm.py``'s version for why other backends need it).

Two departures from a vanilla MLP, both aimed at the periodic domains this
repo targets (torus / robot-arm joint angles):

1. **Input is Fourier-encoded, not raw.** Each coordinate ``x_i`` is mapped to
   ``(sin(w·(x_i − lo)), cos(w·(x_i − lo)))`` with ``w = 2π / (hi − lo)`` taken
   from ``env.domain`` — so the network is *exactly* periodic in every input
   coordinate by construction, matching the wrapped splats it's being compared
   against, rather than approximating periodicity from samples confined to
   one period.
2. **Hidden activations are sin, not ReLU/tanh** (SIREN — Sitzmann et al.
   2020), with the paper's ``1/fan_in`` / ``sqrt(6/fan_in)/omega_0`` uniform
   init. This matters specifically for the Eikonal strategies here: the loss
   is built from ``∇T`` (and PDE residuals differentiate through that again),
   and ReLU's derivative is discontinuous / a tanh net's second derivative
   saturates near 0 — sin is smooth to all orders and keeps gradients
   well-scaled across the network via ``omega_0``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

MLPParams = list[tuple[jnp.ndarray, jnp.ndarray]]


def _fourier_features(X: jnp.ndarray, env) -> jnp.ndarray:
    """[n, d] raw coordinates -> [n, 2d] periodic (sin, cos) features using env.domain's period."""
    lo, hi = env.domain
    omega = 2.0 * jnp.pi / (hi - lo)
    phase = omega * (X - lo)
    return jnp.concatenate([jnp.sin(phase), jnp.cos(phase)], axis=-1)


def init_params(key: jax.Array, env, cfg, p: int = 1) -> MLPParams:
    """Initialise a SIREN MLP: input 2*env.dim (Fourier features) -> cfg.mlp_width (x cfg.mlp_depth) -> p."""
    sizes = [2 * env.dim] + [cfg.mlp_width] * cfg.mlp_depth + [p]
    keys = jax.random.split(key, len(sizes) - 1)
    layers: MLPParams = []
    for i, (fan_in, fan_out, k) in enumerate(zip(sizes[:-1], sizes[1:], keys)):
        bound = 1.0 / fan_in if i == 0 else jnp.sqrt(6.0 / fan_in) / cfg.mlp_omega0
        W = jax.random.uniform(k, (fan_in, fan_out), minval=-bound, maxval=bound)
        b = jnp.zeros((fan_out,))
        layers.append((W, b))
    return layers


def eval_raw(params: MLPParams, X: jnp.ndarray, env) -> jnp.ndarray:
    """Evaluate the SIREN MLP at each row of X (raw coordinates on env's chart). Returns [n, p]."""
    h = _fourier_features(X, env)
    *hidden, (W_out, b_out) = params
    for W, b in hidden:
        h = jnp.sin(h @ W + b)
    return h @ W_out + b_out


def post_step(params: MLPParams, cfg, env=None) -> MLPParams:
    """No-op: an MLP has no per-unit scale that can collapse, and no parameter that must lie on the
    manifold, so there is nothing to project or retract.

    Present so the strategies can call ``backend.post_step`` unconditionally (see
    ``srms/methods/backends/srm.py``, where it floors splat covariances and retracts splat centres).
    """
    return params


def adapt(params: MLPParams, opt_state, residual_fn, env, cfg, rng):
    """No-op: the MLP is a fixed-capacity model — there is no meaningful 'add a neuron where the
    error is'. Returns the model unchanged, so a densifying run is well-defined for either backend.
    """
    return params, opt_state, None


def num_params(params: MLPParams) -> int:
    """Total trainable scalars, for matched-budget comparison against the splat mixture."""
    import numpy as _np

    return int(sum(_np.prod(_np.shape(x)) for x in jax.tree_util.tree_leaves(params)))


def decay_mask(params: MLPParams) -> MLPParams:
    """Weight-decay mask for ``optax.adamw``: decay everything.

    An MLP has no position parameter analogous to srm's splat centres ``B`` (see
    ``srm.py``'s ``decay_mask``) — every weight here is an ordinary magnitude/complexity prior
    target, so nothing needs excluding.
    """
    return jax.tree_util.tree_map(lambda _: True, params)
