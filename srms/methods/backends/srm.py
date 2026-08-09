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
import numpy as np

from srms.lib.manifold_splat import eval_wrapped_gaussian

SplatParams = tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]


def init_params(key: jax.Array, env, cfg, p: int = 1) -> SplatParams:
    """Initialise (V, A, B) with zero weights and centres drawn from env.sample_domain.

    Centres need environment-specific sampling (a box for the torus, a unit sphere for
    SphereEnvironment), not a generic uniform box, so this delegates to the same host-side sampler
    RRT*/collocation already use rather than assuming ``env.domain`` is a box.

    With ``cfg.densify`` the model starts at ``cfg.init_splats`` and grows via ``adapt``; otherwise it
    is a fixed mixture of ``cfg.num_splats``.
    """
    k = cfg.init_splats if getattr(cfg, "densify", False) else cfg.num_splats
    seed = int(jax.random.randint(key, (), 0, 2**31 - 1))
    centres_np = env.sample_domain(np.random.default_rng(seed), k)
    centres = jnp.asarray(centres_np, dtype=jnp.float32)
    scales = jnp.repeat((cfg.init_scale * jnp.eye(env.tangent_dim))[None], k, axis=0)
    return jnp.zeros((k, p)), scales, centres


def eval_raw(params: SplatParams, X: jnp.ndarray, env) -> jnp.ndarray:
    """Evaluate the raw splat mixture g(x) = Σ_j V[j]·N_w(x | B[j], A[j]) at each row of X. Returns [n, p]."""
    V, A, B = params

    def rho_at_x(x: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(
            lambda mu, Ak: eval_wrapped_gaussian(
                x, mu, Ak, log_map_fn=env.log_map, jac_factor_fn=env.jac_factor, dim=env.tangent_dim
            )
        )(B, A)

    return jax.vmap(rho_at_x)(X) @ V


def post_step(params: SplatParams, cfg) -> SplatParams:
    """Floor each splat's covariance singular values, so no splat can collapse into a gradient spike.

    **This is the fix that makes adaptive densification stable.** A sum of Gaussians cannot represent
    the true Eikonal kink at obstacle boundaries and the cut locus, so unconstrained optimization
    chases it by driving a splat's covariance to zero — an effectively infinite ``‖∇T‖`` spike (up to
    ~1e7 was measured) which then blows up any speed-match residual. Flooring the SVD singular values
    at ``cfg.scale_floor`` makes that collapse impossible. Applied after every optimizer step; jittable,
    so it lives inside the strategies' ``step``.

    A no-op when ``cfg.scale_floor <= 0``, and irrelevant to the fixed-count model (which never
    densifies and so never triggers the collapse), but harmless there.
    """
    floor = getattr(cfg, "scale_floor", 0.0)
    if floor <= 0.0:
        return params
    V, A, B = params
    U, S, Vt = jnp.linalg.svd(A)
    return V, jnp.einsum("kij,kj,kjl->kil", U, jnp.clip(S, floor, None), Vt), B


def adapt(params: SplatParams, opt_state, residual_fn, env, cfg, rng: np.random.Generator):
    """3DGS-style prune + spawn, preserving Adam moments for the splats that survive.

    Prunes splats whose weight is below ``cfg.prune_thresh``, then spawns up to ``cfg.spawn_per`` new
    ones at the highest-``residual_fn`` free-space points, capped at ``cfg.max_splats``.

    The optimizer state is grown **surgically**: every leaf whose leading axis indexes splats is
    sliced to the survivors and zero-padded for the spawns. A plain ``optimizer.init()`` reset would
    re-kick every already-converged splat, which is what made earlier densification runs diverge.
    Scalar leaves (Adam's step count) pass through untouched, so this is robust to ``optax.chain``
    nesting without depending on the state's namedtuple layout.

    Returns ``(params, opt_state, num_splats)``.
    """
    V = np.asarray(params[0])
    old_k = len(V)
    keep = np.abs(V[:, 0]) > cfg.prune_thresh
    if keep.sum() < 16:  # never prune the model into nothing
        keep = np.ones(old_k, bool)
    idx = np.where(keep)[0]

    spawn_B = np.zeros((0, env.dim), np.float32)
    room = cfg.max_splats - len(idx)
    if room > 0:
        pool = env.sample_domain(rng, 3000)
        pool = pool[env.sdf_np(pool) > 0.1]  # spawn in free space, not inside obstacles
        if len(pool):
            residual = np.asarray(residual_fn(jnp.asarray(pool, jnp.float32)))
            k = int(min(cfg.spawn_per, room, len(pool)))
            spawn_B = pool[np.argsort(residual)[-k:]].astype(np.float32)
    ns = len(spawn_B)

    def regrow(leaf):
        a = np.asarray(leaf)
        if a.ndim >= 1 and a.shape[0] == old_k:
            return jnp.asarray(np.concatenate([a[idx], np.zeros((ns,) + a.shape[1:], a.dtype)], 0))
        return leaf

    new_v = np.concatenate([V[idx], np.full((ns, V.shape[1]), 1e-4, np.float32)])
    new_a = np.concatenate(
        [
            np.asarray(params[1])[idx],
            np.repeat((cfg.spawn_scale * np.eye(env.tangent_dim))[None], ns, 0).astype(np.float32),
        ]
    )
    new_b = np.concatenate([np.asarray(params[2])[idx], spawn_B])
    new_params = (jnp.asarray(new_v), jnp.asarray(new_a), jnp.asarray(new_b))
    return new_params, jax.tree_util.tree_map(regrow, opt_state), len(new_v)


def num_params(params: SplatParams) -> int:
    """Total trainable scalars — k·(p + d² + d). Reported so SRM/MLP can be compared at matched size."""
    return int(sum(np.prod(np.shape(x)) for x in jax.tree_util.tree_leaves(params)))
