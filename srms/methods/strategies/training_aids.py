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
import numpy as np


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


class DensifyController:
    """Let the model decide how many splats it needs, by marginal value.

    This is the mechanism the splat model has and a fixed-width MLP structurally does not: capacity
    can be *added where the residual is* and stopped when it stops paying. An MLP's weight count is
    set at construction, so there is no analogue — ``mlp.adapt`` is necessarily a no-op.

    The rule is one line of arithmetic. At each densify pass, compare the mean training residual since
    the previous pass against the residual before it, and divide the fractional improvement by the
    number of splats that bought it:

        gain_per_splat = (L_prev − L_now) / L_prev / splats_added

    Growth stops when that falls below ``cfg.densify_min_gain`` on **two consecutive passes**. One
    reading is not enough: the window is measured immediately after a densify pass, which is exactly
    when freshly spawned splats perturb the optimisation and the residual transiently rises. Measured
    on the 2-D torus, a single-reading rule stopped at 1194 splats on a *negative* gain (-6.6e-04,
    i.e. the loss had gone up), where the model went on to reach 2676 splats and a better RMS
    (0.2485 vs 0.2527) when allowed to continue — so that stop was reading the disturbance the pass
    itself caused, not saturation. Requiring two in a row keeps the model growing through the
    transient while still stopping when capacity genuinely stops paying.

    Dividing by ``L_prev`` makes the test
    independent of the residual's scale, and dividing by ``splats_added`` makes it independent of the
    spawn schedule — so the same threshold transfers across manifolds, seeds and obstacle sets without
    being re-pinned, which a fixed "improved by less than x%" tolerance does not.

    Two other stops, whichever comes first: the final ``densify_freeze_frac`` of training always runs
    at fixed structure so the model converges against a settled basis, and ``cfg.max_splats`` is a
    runaway backstop rather than a target. ``stop_reason`` records which one fired, so a run reports
    *why* it chose its size instead of leaving it to be inferred.
    """

    def __init__(self, cfg):
        self.every = cfg.densify_every
        self.min_gain = getattr(cfg, "densify_min_gain", 1e-5)
        self.freeze_frac = getattr(cfg, "densify_freeze_frac", 0.1)
        self.freeze_step = (1.0 - self.freeze_frac) * cfg.steps
        self.enabled = getattr(cfg, "densify", False)
        self.history: list[float] = []
        self.stopped = False
        self.stop_reason: str | None = None
        self.stop_step: int | None = None
        self.last_gain: float | None = None
        self._prev_loss: float | None = None
        self._prev_k: int | None = None
        self._below = 0  # consecutive passes whose marginal value was under threshold

    def record(self, loss: float) -> None:
        """Log one step's training residual — the quantity the marginal-value test reads."""
        self.history.append(float(loss))

    def _stop(self, step: int, reason: str) -> None:
        self.stopped, self.stop_reason, self.stop_step = True, reason, step

    def should_densify(self, step: int, num_splats: int) -> bool:
        """True on steps where the model should grow. Call once per training step."""
        if not self.enabled or self.stopped or step == 0 or step % self.every:
            return False
        if step >= self.freeze_step:
            self._stop(step, f"structure frozen for the final {self.freeze_frac:.0%} of training")
            return False
        recent = float(np.mean(self.history[-self.every :]))
        if self._prev_loss is not None and self._prev_k is not None:
            added = num_splats - self._prev_k
            if added > 0 and np.isfinite(recent) and self._prev_loss > 0:
                self.last_gain = (self._prev_loss - recent) / self._prev_loss / added
                if self.last_gain < self.min_gain:
                    self._below += 1
                    if self._below >= 2:
                        self._stop(
                            step,
                            f"marginal value below {self.min_gain:.0e}/splat on two consecutive passes "
                            f"(last {self.last_gain:.2e})",
                        )
                        return False
                else:
                    self._below = 0
        self._prev_loss, self._prev_k = recent, num_splats
        return True

    def summary(self, num_splats: int) -> str:
        """One line: the size the model settled on, and what stopped it growing."""
        why = self.stop_reason or "training ended while still growing (raise steps or max_splats)"
        gain = f", last marginal value {self.last_gain:.2e}/splat" if self.last_gain is not None else ""
        return f"chose {num_splats} splats — densification stopped at step {self.stop_step}: {why}{gain}"
