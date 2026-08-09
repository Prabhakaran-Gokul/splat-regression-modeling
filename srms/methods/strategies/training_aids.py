"""Shared training aids ported from torus.py, reusable across strategies.

Currently: causal residual weighting (Wang, Sankaran & Perdikaris 2022's "causal PINN" idea, as used
by torus.py's ``solve``/``solve_ntfields``/``solve_roadmap``) — a source-outward curriculum that
weights each collocation point's residual by how well-fit *nearer* points already are, so training
can't "cheat" by fitting a far wavefront before the near one it causally depends on has converged.

Generic over any ``Environment`` (only needs ``env.geodesic``/``env.start``) and any per-point
residual, so the same helpers apply to ``eikonal.py``'s PDE residual, ``weak_supervision.py``'s and
``ntfields.py``'s symmetric speed-match residual alike.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def causal_rate(cfg, step: int) -> float:
    """Per-step causal decay rate: ``cfg.causal_strength / cfg.num_collocation``, linearly annealed
    to 0 by the end of training if ``cfg.causal_anneal`` (matches torus.py exactly)."""
    rate = cfg.causal_strength / cfg.num_collocation
    if cfg.causal_anneal:
        rate *= 1.0 - step / cfg.steps
    return rate


def causal_loss(env, colloc: jnp.ndarray, squared_residual: jnp.ndarray, rate: jnp.ndarray) -> jnp.ndarray:
    """Causal-weighted mean of a per-point squared residual.

    Orders points source-outward by geodesic distance from ``env.start``, then weights each by
    ``exp(-rate · upstream)`` where ``upstream`` is the cumulative (stop-gradient) residual mass of
    all nearer points in that ordering — so a far point's loss is discounted until nearer points are
    already well-fit.
    """
    dist = env.geodesic(colloc, jnp.asarray(env.start, dtype=jnp.float32))
    order = jnp.argsort(dist)
    ordered = squared_residual[order]
    upstream = jnp.cumsum(ordered) - ordered
    weight = jax.lax.stop_gradient(jnp.exp(-rate * upstream))
    return jnp.mean(weight * ordered)
