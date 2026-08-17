"""MLP backend ablation: ``mlp.py`` without the Fourier feature encoding, with residual hidden layers.

Isolates whether the periodic (sin/cos) input encoding in ``srms/methods/backends/mlp.py`` is what
produces the torus's wraparound continuity, by feeding the SIREN raw ``env.dim`` coordinates instead
of the ``2*env.dim`` ``(sin, cos)`` pair — see that module's docstring for the rationale being tested
against. Expected result on a periodic chart (e.g. the torus) if Fourier features are the cause: the
field learned here shows a visible seam/discontinuity at the domain wrap (`-π` vs `π`) that ``mlp.py``
does not (confirmed — see the torus seam measurement in the investigation log). On a *non-periodic*
chart (e.g. ``poincare_hyperbolic``, where ``env.domain`` is a chart extent, not a period), this
backend is the architecturally correct one: ``mlp.py`` would force points near opposite sides of the
chart to nearly-identical input features even when they are geodesically far apart, which is wrong
there, not merely untested.

Residual (skip) connections between hidden layers, same construction as
``srms/methods/backends/mlp_resnet.py``: every width -> width layer computes ``h = h + sin(W h + b)``
instead of ``h = sin(W h + b)``, so ``cfg.mlp_width``/``cfg.mlp_depth`` can scale up without a plain
deep SIREN's optimization difficulty confounding whatever question this backend is being used to
answer. The first hidden layer (raw coordinates -> width) has no skip connection, since its input and
output dimensions generally differ.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

MLPParams = list[tuple[jnp.ndarray, jnp.ndarray]]


def init_params(key: jax.Array, env, cfg, p: int = 1) -> MLPParams:
    """Initialise a SIREN MLP: input env.dim (raw coordinates) -> cfg.mlp_width (x cfg.mlp_depth) -> p."""
    sizes = [env.dim] + [cfg.mlp_width] * cfg.mlp_depth + [p]
    keys = jax.random.split(key, len(sizes) - 1)
    layers: MLPParams = []
    for i, (fan_in, fan_out, k) in enumerate(zip(sizes[:-1], sizes[1:], keys)):
        bound = 1.0 / fan_in if i == 0 else jnp.sqrt(6.0 / fan_in) / cfg.mlp_omega0
        W = jax.random.uniform(k, (fan_in, fan_out), minval=-bound, maxval=bound)
        b = jnp.zeros((fan_out,))
        layers.append((W, b))
    return layers


def eval_raw(params: MLPParams, X: jnp.ndarray, env) -> jnp.ndarray:
    """Evaluate the residual SIREN directly on raw coordinates (no Fourier encoding). Returns [n, p].

    The first hidden layer (raw coordinates -> width) has no skip connection, since its input and
    output dimensions generally differ; every width -> width layer after it is a residual block —
    same structure as ``mlp_resnet.py``'s ``eval_raw``.
    """
    h = X
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
