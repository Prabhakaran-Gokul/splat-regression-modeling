"""P-NTFields training strategy: NTFields + viscosity + progressive speed scheduling.

Port of P-NTFields (Ni & Qureshi, RSS 2023, arXiv 2306.00616) to this repo's *fixed-source*
setting. P-NTFields keeps NTFields' field ``T = base/τ`` and isotropic speed loss verbatim (its
Eq. 2 and Eq. 5 restate them) and adds exactly two mechanisms, both of which target the same
failure mode — the physics-informed optimizer converging to a wrong local minimum "as if the
low-speed obstacles do not exist":

1. **Viscosity (Eq. 6).** The vanishing-viscosity regularization of the Eikonal equation,

       1/S(q_g) = ‖∇_{q_g} T‖ + ε·Δ_{q_g} T,

   turning an ill-posed non-linear PDE into a semi-linear elliptic one with a unique *smooth*
   solution around low-speed obstacles. Following the paper's Eq. 7, the Laplacian is taken of the
   **factored** field ``τ`` rather than of ``T`` ("for computational simplification, which also
   keeps the similar second-order derivative term") — see ``laplacian_tau``.

   Eq. 7 writes the predicted speed as a closed-form chain-rule expansion of ``τ`` and its
   derivatives. That expansion is algebraically identical to ``‖∇T‖`` for ``T = base/τ``: with
   ``∇base`` the unit vector from the source, ``‖∇T‖² = (τ² − 2τ·(d·∇τ) + ‖d‖²‖∇τ‖²)/τ⁴`` where
   ``d`` is the displacement from the source — exactly Eq. 7's radicand. So this module computes
   ``‖∇T‖`` by autodiff (as ``ntfields.py`` does, and picking up ``env.metric_inv`` for free on a
   curved metric) and adds ``ε·Δτ``, rather than transcribing Eq. 7 term by term.

2. **Progressive speed scheduling (Eq. 8).** The ground-truth speed is annealed from a constant
   free-space field toward the true obstacle field,

       S⋆_α(q) = (1 − α) + α·S⋆(q),     α: 0 → 1 (and slightly past),

   so training starts at the trivial solution ``τ ≡ 1`` and follows a continuous interpolation to
   the sharp field. Note this is linear in **speed**; the original ``torus.py`` annealed the
   *slowness* instead (``s_λ = 1 + λ(s−1)``), which is a different path through field space — see
   ``progressive_speed``.

Both are on by default (``cfg.viscosity_eps``, ``cfg.alpha_*``); setting ``viscosity_eps=0`` and
``alpha_init=1.0`` recovers ``ntfields.py`` exactly, which is how the paper's own ablation
("w/o viscosity", "w/o scheduling") is reproduced.

**Deviations from the paper** — all of ``ntfields.py``'s (fixed-source; ``cfg.causal`` ignored;
scene speed model; per-step resampled collocation; ``srm``/``mlp`` backend), plus:

- *α schedule shape.* The paper holds ``α=0.5`` for 1000 epochs, then steps ``+1/4000`` per epoch,
  halving to ``+1/8000`` after epoch 4000, until ``α ≥ 1.05``. Those numbers presuppose their epoch
  budget and dataset, so this module keeps the *shape* — hold, then linear ramp to ``alpha_final`` —
  parameterized as fractions of ``cfg.steps``. Defaults (0.5 / 25% hold / 1.05) match the paper's.
- *No loss-threshold rollback.* Algorithm 1 reloads the previous epoch's parameters and reshuffles
  batches when ``L_i/L_{i−1} > η`` (η=1.5). That guards against a bad pass over a **fixed** dataset;
  with fresh collocation every step there is no epoch to roll back to, so it is **omitted**.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import trange

from srms.methods.strategies import ntfields
from srms.methods.strategies.eikonal import sample_collocation
from srms.methods.strategies.ntfields import isotropic_loss, predict  # noqa: F401  (predict re-exported)


def progressive_speed(slow: jnp.ndarray, alpha: jnp.ndarray) -> jnp.ndarray:
    """P-NTFields Eq. 8: ``S⋆_α = (1 − α) + α·S⋆``, with ``S⋆ = 1/slowness``.

    Returned in *speed* units (α=0 ⇒ constant free-space speed 1; α=1 ⇒ the scene's true ``S⋆``).
    """
    return (1.0 - alpha) + alpha * (1.0 / slow)


def alpha_at(cfg, step: int) -> float:
    """α(t): hold at ``cfg.alpha_init`` for ``cfg.alpha_hold_frac`` of training, then ramp linearly.

    The paper's hold-then-ramp curriculum (Sec. IV-B / Algorithm 1), expressed as fractions of
    ``cfg.steps`` so it transfers across training budgets.
    """
    hold = cfg.alpha_hold_frac * cfg.steps
    if step <= hold:
        return float(cfg.alpha_init)
    frac = (step - hold) / max(cfg.steps - hold, 1.0)
    return float(cfg.alpha_init + (cfg.alpha_final - cfg.alpha_init) * min(frac, 1.0))


def laplacian_tau(backend, params, thetas: jnp.ndarray, env, tau_bias: float, tau_min: float) -> jnp.ndarray:
    """``Δτ`` at each θ — trace of the Hessian of the factored field (P-NTFields Eq. 7).

    Chart-coordinate Laplacian, matching the paper's Euclidean ``Δ``; on the flat torus
    (``metric_inv = I``) this is the Laplace–Beltrami operator. A curved metric would need the
    metric-weighted form, which the paper does not define.
    """

    def tau_single(x):
        return ntfields.tau(backend, params, x, env, tau_bias, tau_min)

    return jax.vmap(lambda x: jnp.trace(jax.hessian(tau_single)(x)))(thetas)


def viscous_speed_ratio(
    backend, params, thetas: jnp.ndarray, slow: jnp.ndarray, env, cfg, alpha: jnp.ndarray
) -> jnp.ndarray:
    """``q = S⋆_α / S`` with the viscosity-corrected ``1/S = ‖∇T‖_{g⁻¹} + ε·Δτ`` (Eq. 6/7 + Eq. 8)."""
    grad = ntfields.time_grad(backend, params, thetas, env, cfg.tau_bias, cfg.tau_min)
    inv_speed = ntfields.grad_norm(grad, thetas, env)
    if cfg.viscosity_eps != 0.0:
        inv_speed = inv_speed + cfg.viscosity_eps * laplacian_tau(
            backend, params, thetas, env, cfg.tau_bias, cfg.tau_min
        )
        # ε·Δτ is signed and can, transiently, drive the predicted 1/S non-positive; the isotropic
        # loss takes √q, so floor it rather than propagate a NaN.
        inv_speed = jnp.clip(inv_speed, 1e-6, None)
    return progressive_speed(slow, alpha) * inv_speed


def solve(env, cfg, backend, checkpoint=None, progress_fn=None):
    """Fit ``T=base/τ`` with viscosity + progressive speed scheduling under NTFields' Eq. 4 loss.

    ``progress_fn(step, metrics)``, if given, is called every ``cfg.log_every`` steps with a dict of
    scalar training metrics (``loss``, ``eikonal``, ``alpha``) — e.g. to log to mlflow (``run.py``).
    """
    if cfg.causal:
        print("[pntfields] paper-faithful: cfg.causal ignored (P-NTFields has no causal weighting)", flush=True)

    params = backend.init_params(jax.random.PRNGKey(cfg.seed), env, cfg)
    rng = np.random.default_rng(cfg.seed)

    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(cfg.lr, weight_decay=0.1))
    opt_state = optimizer.init(params)

    def loss_fn(p, colloc, slow, alpha):
        q = viscous_speed_ratio(backend, p, colloc, slow, env, cfg, alpha)
        eikonal_loss = jnp.mean(isotropic_loss(q))
        return eikonal_loss, {"eikonal": eikonal_loss}

    @jax.jit
    def step(p, state, colloc, slow, alpha):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(p, colloc, slow, alpha)
        updates, state = optimizer.update(grads, state, p)
        return optax.apply_updates(p, updates), state, loss, aux

    progress = trange(cfg.steps, desc="pntfields")
    for i in progress:
        colloc = sample_collocation(env, rng, cfg.num_collocation, cfg.source_radius)
        slow = env.slowness(colloc)
        alpha = jnp.float32(alpha_at(cfg, i))
        params, opt_state, loss, aux = step(params, opt_state, colloc, slow, alpha)
        if i % cfg.log_every == 0:
            progress.set_description(
                f"pntfields — log10(loss) = {float(jnp.log10(loss + 1e-12)):.3f}  α = {float(alpha):.3f}"
            )
            if progress_fn is not None:
                progress_fn(i, {"loss": float(loss), "eikonal": float(aux["eikonal"]), "alpha": float(alpha)})
        if checkpoint is not None and i > 0 and i % cfg.checkpoint_every == 0:
            checkpoint(params, i)
    return params
