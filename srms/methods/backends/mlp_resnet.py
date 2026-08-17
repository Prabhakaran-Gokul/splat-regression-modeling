"""MLP backend variant: ``mlp.py``'s SIREN with residual (skip) connections between hidden layers.

Same Fourier-encoded periodic input and SIREN sin-activation/init scheme as ``mlp.py`` — this file
changes exactly one thing: every hidden-to-hidden layer computes ``h = h + sin(W h + b)`` instead of
``h = sin(W h + b)``, so gradients have an identity path back to the input regardless of
``cfg.mlp_depth``. Plain deep SIRENs are hard to optimize past a handful of layers (each sin layer
both rotates and saturates its input, so the composition drifts and gradients degrade); residual
connections are the standard fix and let ``cfg.mlp_width``/``cfg.mlp_depth`` scale up without a new
optimization failure mode confounding the capacity question. No new config fields — a bigger network
is just a bigger ``--mlp_width``/``--mlp_depth`` on the existing flags.

Built to test whether ``ntfields`` (pure physics-informed, no roadmap prior) closes its gap to
``weak_supervision`` with more capacity. Per ``investigation.md``'s capacity sweeps on the ``srm``
backend, that gap has repeatedly turned out to be a *missing-information* problem (the Eikonal
residual alone under-determines the level behind an obstacle), not a *missing-capacity* one — more
splats there mildly hurt rather than helped. This backend exists to check that finding transfers to
``mlp`` rather than assume it: if RMS does not improve substantially with width/depth here either,
the fix is supervision (a prior), not architecture.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from srms.methods.backends.mlp import _fourier_features

MLPParams = list[tuple[jnp.ndarray, jnp.ndarray]]


def init_params(key: jax.Array, env, cfg, p: int = 1) -> MLPParams:
    """Initialise a residual SIREN: input 2*env.dim (Fourier features) -> cfg.mlp_width (x cfg.mlp_depth) -> p.

    Same SIREN init as ``mlp.py`` (``1/fan_in`` first layer, ``sqrt(6/fan_in)/omega_0`` after) — the
    residual connections added in ``eval_raw`` change the forward pass, not the init distribution.
    """
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
    """Evaluate the residual SIREN at each row of X (raw coordinates on env's chart). Returns [n, p].

    The first hidden layer (Fourier features -> width) has no skip connection, since its input and
    output dimensions generally differ; every width -> width layer after it is a residual block.
    """
    h = _fourier_features(X, env)
    *hidden, (W_out, b_out) = params
    (W0, b0), *rest = hidden
    h = jnp.sin(h @ W0 + b0)
    for W, b in rest:
        h = h + jnp.sin(h @ W + b)
    return h @ W_out + b_out


def post_step(params: MLPParams, cfg, env=None) -> MLPParams:
    """No-op — see ``mlp.py``'s ``post_step``; identical reasoning applies here."""
    return params


def adapt(params: MLPParams, opt_state, residual_fn, env, cfg, rng):
    """No-op — see ``mlp.py``'s ``adapt``; identical reasoning applies here."""
    return params, opt_state, None


def num_params(params: MLPParams) -> int:
    """Total trainable scalars, for matched-budget comparison against other backends."""
    import numpy as _np

    return int(sum(_np.prod(_np.shape(x)) for x in jax.tree_util.tree_leaves(params)))
