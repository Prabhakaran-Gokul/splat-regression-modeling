"""H-NTFields training strategy: sparse-roadmap weak supervision + PDE regularization.

Port of H-NTFields — *Hierarchical* Neural Time Fields (Ni, Liu & Qureshi 2026, arXiv 2604.13204) —
to this repo's *single-source* setting. H-NTFields extends **TD-NTFields** (Ni, Pan & Qureshi,
ICLR 2025, arXiv 2505.05691), not P-NTFields: it inherits that paper's three PDE losses and
causality weight, and adds weak supervision from a sparse sphere-packing roadmap. The combined
objective (Eq. 9) is

    L = (λ_E·L_E + λ_TD·L_TD + λ_N·L_N + λ_R·L_R) · L_C

with, at one endpoint (this repo's single-source reduction — see *Deviations*):

- ``L_E  = (√(S⋆/S) − 1)²``                                     Eikonal, infinitesimal scale (Eq. 3)
- ``L_TD = [T(θ) − Δt/S⋆(θ) − T(θ + u⋆Δt)]²``,  ``u⋆ = −∇T/‖∇T‖``   Bellman, finite scale (Eq. 4)
- ``L_N  = (1 − S⋆(θ))·‖S⋆(θ)∇T(θ) + ∇S⋆(θ)/‖∇S⋆(θ)‖‖²``        normal alignment near obstacles (Eq. 5)
- ``L_R  = max(0, T − T_ub) + max(0, T_lb − T)``                 roadmap travel-time bounds (Eq. 8)
- ``L_C  = exp(−λ_C·T)``                                         causality curriculum (Eq. 6)

Note ``L_E`` here is the **squared, one-directional** form of TD-NTFields Eq. 3, which is *not*
NTFields' isotropic Eq. 4 (``|1−√q| + |1−1/√q|``, in ``ntfields.py``). The two papers genuinely use
different Eikonal losses; each strategy uses its own.

**Roadmap supervision.** ``srms/environments/sampling.build_sphere_roadmap`` builds the paper's
sphere-packing roadmap; training points are drawn by *perturbing roadmap nodes within their free
spheres* (Sec. IV-A, "start–goal perturbation"), which the paper calls crucial — it is what puts
queries near obstacles for ``L_E``/``L_N`` and spans short and long horizons for ``L_TD``/``L_C``.
For a point θ perturbed off node ``i``, the bounds are the node's graph travel time ± the cost of
the perturbation hop, which is a genuine upper bound (roadmap to node ``i``, then straight to θ) and
lower bound (triangle inequality). The paper adds/subtracts the perturbation *radius*, valid there
because its free-space speed is 1; using the slowness-integrated hop cost keeps the bounds valid
under this repo's varying free-space slowness and reduces to the paper's form when slowness ≡ 1.

**Deviations from the paper** — all of ``ntfields.py``'s (single-source; scene speed model; per-step
resampling; ``srm``/``mlp`` backend), plus:

- *Field parameterization.* TD-NTFields predicts ``T`` as a learned **quasimetric**
  ``D(f(q_s), f(q_g))`` (its Eq. 10) inside a PirateNets architecture. That is inseparable from a
  two-point formulation, so this module keeps the ``T = base/τ`` factorization used by
  ``ntfields.py``/``pntfields.py``. Holding the field form fixed across all three strategies is also
  what makes the SRM-vs-MLP comparison isolate the *loss*, rather than confounding it with
  architecture.
- *``cfg.causal`` ignored.* H-NTFields has its own ``L_C = exp(−λ_C·T)``, which is a different
  object from ``training_aids.causal_loss`` (Wang et al.'s cumulative-upstream-residual weighting).
  ``L_C`` is implemented inline here and the residual weight is stop-gradient, as weights normally
  are; the paper does not specify.
- *``u⋆`` is stop-gradient.* Differentiating the Taylor expansion through the policy direction would
  bring in second derivatives of ``T``; the paper does not state whether it detaches. Detaching is
  the standard reading of "along the optimal policy".
- *Inference.* The paper plans with sampling-based MPC; this repo scores the field against a dense
  fast-marching grid and (elsewhere) descends ``−∇T``.
- *λ_R.* Unspecified in the paper; defaulted to ``λ_E``'s scale. λ_E/λ_TD/λ_N/λ_C/Δt defaults are
  TD-NTFields' published 3D-environment values (1e-2, 1e-3, 1e-3, 0.5, 0.02).
- *Δt is domain-relative.* ``u⋆`` is a unit vector, so ``u⋆Δt`` is a configuration-space
  *displacement*. NTFields normalizes its C-space to ``[-0.5, 0.5]`` per axis (unit span), where
  ``Δt=0.02`` is ~2% of the domain; this torus spans ``2π``, where the same 2% is ``Δt ≈ 0.126``.
  The default keeps the paper's literal number — pass ``--hnt-dt 0.126`` for the domain-matched
  equivalent. The paper notes this choice matters: too small and ``L_TD`` stops contributing, too
  large and the stepped point lands through an obstacle.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import trange

from srms.environments import sampling
from srms.methods.strategies import ntfields
from srms.methods.strategies.ntfields import predict  # noqa: F401  (re-exported: same T = base/τ field)


def sample_perturbed(
    env, nodes: np.ndarray, radii: np.ndarray, times: np.ndarray, rng: np.random.Generator, n: int, ksamp: int = 6
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Perturb random roadmap nodes inside their free spheres; return ``(θ, T_ub, T_lb)``.

    Uniform in the ball (radius scaled by ``u^{1/d}``) so samples are not biased toward the centre.
    The bound slack is the slowness-integrated cost of the perturbation hop (see module docstring).
    """
    dim = env.dim
    idx = rng.integers(0, len(nodes), size=n)
    centres, radius, node_time = nodes[idx], radii[idx], times[idx]

    direction = rng.standard_normal((n, dim))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True) + 1e-12
    delta = direction * (radius * rng.random(n) ** (1.0 / dim))[:, None]
    thetas = env.wrap_point_np(centres + delta)

    ts = np.linspace(0.0, 1.0, ksamp)[:, None, None]
    hop_pts = env.wrap_point_np((centres[None] + ts * delta[None]).reshape(-1, dim))
    mean_slow = env.slowness_np(hop_pts).reshape(ksamp, n).mean(axis=0)
    hop_time = np.linalg.norm(delta, axis=1) * mean_slow

    upper = node_time + hop_time
    lower = np.maximum(node_time - hop_time, 0.0)  # travel time is non-negative
    return (
        jnp.asarray(thetas, dtype=jnp.float32),
        jnp.asarray(upper, dtype=jnp.float32),
        jnp.asarray(lower, dtype=jnp.float32),
    )


def speed_star(env, thetas: jnp.ndarray) -> jnp.ndarray:
    """Ground-truth speed ``S⋆ = 1/slowness`` at each θ."""
    return 1.0 / env.slowness(thetas)


def speed_star_normal(env, thetas: jnp.ndarray) -> jnp.ndarray:
    """Unit obstacle-normal direction ``∇S⋆/‖∇S⋆‖`` at each θ (Eq. 5)."""

    def s_single(x):
        return 1.0 / env.slowness(x[None, :])[0]

    grad = jax.vmap(jax.grad(s_single))(thetas)
    # ‖∇S⋆‖ → 0 in flat free space; the (1 − S⋆) weight kills the term there, but 0·NaN is NaN, so
    # floor the norm rather than let the normalization blow up.
    return grad / jnp.maximum(jnp.linalg.norm(grad, axis=1, keepdims=True), 1e-8)


def loss_terms(backend, params, thetas, upper, lower, env, cfg):
    """The four H-NTFields loss terms plus the causality weight, each as [n]."""
    tau_bias, tau_min = cfg.tau_bias, cfg.tau_min

    def t_single(x):
        return ntfields.field_ntfields(backend, params, x, env, tau_bias, tau_min)

    time = jax.vmap(t_single)(thetas)
    grad = ntfields.time_grad(backend, params, thetas, env, tau_bias, tau_min)
    inv_speed = ntfields.grad_norm(grad, thetas, env)  # = 1/S
    s_star = speed_star(env, thetas)
    slow = 1.0 / s_star

    # L_E — Eikonal, squared one-directional form (Eq. 3): q = S*/S = ‖∇T‖ / slowness.
    q = inv_speed / slow
    l_e = (jnp.sqrt(jnp.clip(q, 1e-12, None)) - 1.0) ** 2

    # L_TD — Bellman over a finite step along the optimal policy u* = −∇T/‖∇T‖ (Eq. 4).
    policy = jax.lax.stop_gradient(-grad / jnp.maximum(inv_speed, 1e-8)[:, None])
    stepped = env.wrap_point(thetas + cfg.hnt_dt * policy)
    l_td = (time - cfg.hnt_dt * slow - jax.vmap(t_single)(stepped)) ** 2

    # L_N — align ∇T with the obstacle normal, active only near obstacles via the (1 − S*) weight (Eq. 5).
    aligned = s_star[:, None] * grad + speed_star_normal(env, thetas)
    l_n = (1.0 - s_star) * jnp.sum(aligned**2, axis=1)

    # L_R — hinge onto the roadmap travel-time corridor (Eq. 8).
    l_r = jnp.maximum(0.0, time - upper) + jnp.maximum(0.0, lower - time)

    # L_C — causality: learn small travel times first (Eq. 6); a weight, so stop-gradient.
    l_c = jax.lax.stop_gradient(jnp.exp(-cfg.hnt_lambda_c * time))
    return l_e, l_td, l_n, l_r, l_c


def solve(env, cfg, backend, checkpoint=None, progress_fn=None):
    """Fit ``T=base/τ`` to H-NTFields' combined roadmap-bounded PDE objective (Eq. 9).

    ``progress_fn(step, metrics)``, if given, is called every ``cfg.log_every`` steps with a dict of
    scalar training metrics (``loss`` and the four terms) — e.g. to log to mlflow (``run.py``).
    """
    if cfg.causal:
        print("[hntfields] cfg.causal ignored — H-NTFields uses its own L_C = exp(-λ_C·T)", flush=True)

    params = backend.init_params(jax.random.PRNGKey(cfg.seed), env, cfg)
    rng = np.random.default_rng(cfg.seed)
    nodes, radii, times = sampling.build_sphere_roadmap(
        env, env.start, cfg.hnt_nodes, cfg.hnt_pool, cfg.hnt_connect_radius, cfg.seed
    )

    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(cfg.lr, weight_decay=0.1))
    opt_state = optimizer.init(params)

    def loss_fn(p, thetas, upper, lower):
        l_e, l_td, l_n, l_r, l_c = loss_terms(backend, p, thetas, upper, lower, env, cfg)
        weighted = (
            cfg.hnt_lambda_e * l_e + cfg.hnt_lambda_td * l_td + cfg.hnt_lambda_n * l_n + cfg.hnt_lambda_r * l_r
        )
        total = jnp.mean(weighted * l_c)
        return total, {
            "eikonal": jnp.mean(l_e),
            "td": jnp.mean(l_td),
            "normal": jnp.mean(l_n),
            "bounds": jnp.mean(l_r),
        }

    @jax.jit
    def step(p, state, thetas, upper, lower):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(p, thetas, upper, lower)
        updates, state = optimizer.update(grads, state, p)
        return optax.apply_updates(p, updates), state, loss, aux

    progress = trange(cfg.steps, desc="hntfields")
    for i in progress:
        thetas, upper, lower = sample_perturbed(env, nodes, radii, times, rng, cfg.num_collocation)
        params, opt_state, loss, aux = step(params, opt_state, thetas, upper, lower)
        if i % cfg.log_every == 0:
            progress.set_description(f"hntfields — log10(loss) = {float(jnp.log10(loss + 1e-12)):.3f}")
            if progress_fn is not None:
                progress_fn(i, {"loss": float(loss), **{k: float(v) for k, v in aux.items()}})
        if checkpoint is not None and i > 0 and i % cfg.checkpoint_every == 0:
            checkpoint(params, i)
    return params
