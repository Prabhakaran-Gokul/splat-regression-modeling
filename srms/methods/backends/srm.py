"""SRM backend: a weighted mixture of wrapped Gaussians ("splats") on any Environment.

Delegates the manifold-specific pieces (log map, Jacobian correction) to the
Environment and the wrapped-Gaussian math to
``srms.lib.manifold_splat.eval_wrapped_gaussian`` — the same generic evaluator
already used for the sphere and SE(2) backends there. This replaces the
original ``torus.py``'s hand-rolled ``eval_splat_torus``, which duplicated
that math with the wrap baked in by name.

Parameters ``(V, A, B)``: ``V: [k, p]`` weights, ``A: [k, d, d]`` scale/rotation,
``B: [k, d]`` centres — same convention as ``srms/lib/splat.py``.

Every backend (this one, and eventually ``mlp``/``kan``) exposes the same two
entry points so a training strategy (``srms/methods/strategies``) never needs
to know which one it's using:

- ``init_params(key, env, cfg) -> params`` — reads whatever ``cfg`` fields it
  needs (here: ``cfg.num_splats``, ``cfg.init_scale``); other backends read
  their own fields off the same flat ``cfg``.
- ``eval_raw(params, X, env) -> [n, p]`` — the raw (unfactored) field value.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from srms.lib.manifold_splat import eval_wrapped_gaussian

SplatParams = tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]


def init_params(key: jax.Array, env, cfg, p: int = 1) -> SplatParams:
    """Initialise (V, A, B) with zero weights and centres uniform over env.domain."""
    lo, hi = env.domain
    centres = jax.random.uniform(key, (cfg.num_splats, env.dim), minval=lo, maxval=hi)
    scales = jnp.repeat((cfg.init_scale * jnp.eye(env.dim))[None], cfg.num_splats, axis=0)
    return jnp.zeros((cfg.num_splats, p)), scales, centres


def eval_raw(params: SplatParams, X: jnp.ndarray, env) -> jnp.ndarray:
    """Evaluate the raw splat mixture g(x) = Σ_j V[j]·N_w(x | B[j], A[j]) at each row of X. Returns [n, p]."""
    V, A, B = params

    def rho_at_x(x: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(
            lambda mu, Ak: eval_wrapped_gaussian(
                x, mu, Ak, log_map_fn=env.log_map, jac_factor_fn=env.jac_factor, dim=env.dim
            )
        )(B, A)

    return jax.vmap(rho_at_x)(X) @ V
