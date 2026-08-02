"""Fit a Splat Regression Model to a ground-truth geodesic time-to-go field.

This is the first, simplest milestone: an obstacle-free plane or sphere where the
time-to-go from a fixed source is known analytically. We fit the splat field to
that ground truth by least squares (reusing ``lib.splat``) to validate that the
splat representation and the visualisation pipeline are correct before moving to
the self-supervised PDE loss. Output is a single ``[GT | prediction | error]``
comparison figure with a shared error scale.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import tyro
from tqdm import trange

from ground_truth import PlanningProblem, draw_field, make_problem
from lib.splat import eval_splat


@dataclasses.dataclass
class Config:
    """Configuration for a single splat time-to-go fitting run.

    Attributes:
        problem: Domain to fit, ``"plane"`` or ``"sphere"``.
        start: Source position; ``(x, y)`` for the plane, ``(lat_deg, lon_deg)`` for the sphere.
        resolution: Grid resolution per axis for evaluation and rendering.
        num_splats: Number of splats ``k`` in the model.
        num_train: Number of grid points randomly sampled as training targets.
        steps: Number of gradient-descent steps.
        lr: Adam learning rate.
        init_scale: Initial isotropic scale of each splat (diagonal of ``A``).
        seed: PRNG seed for initialisation and training-point sampling.
        error_clip: Symmetric colour limit for the error panel; ``None`` auto-scales.
        out_dir: Directory to write the comparison figure into.
    """

    problem: Literal["plane", "sphere"] = "plane"
    start: tuple[float, float] = (0.0, 0.0)
    resolution: int = 160
    num_splats: int = 256
    num_train: int = 4096
    steps: int = 3000
    lr: float = 5e-3
    init_scale: float = 0.3
    seed: int = 0
    error_clip: float | None = None
    out_dir: str = "figures"


def init_splat(problem: PlanningProblem, num_splats: int, init_scale: float, key: jax.Array) -> tuple:
    """Initialise ``(V, A, B)`` with small weights, isotropic scales, and centres on the domain."""
    key_v, key_b = jax.random.split(key)
    weights = jax.random.normal(key_v, (num_splats, 1)) * 0.1
    scales = jnp.repeat((init_scale * jnp.eye(problem.dim))[None], num_splats, axis=0)
    centre_idx = jax.random.choice(key_b, len(problem), (num_splats,), replace=False)
    centres = problem.points[centre_idx]
    return weights.astype(jnp.float32), scales.astype(jnp.float32), centres.astype(jnp.float32)


def fit_splat(splat: tuple, train_x: jnp.ndarray, train_y: jnp.ndarray, lr: float, steps: int) -> tuple:
    """Fit the splat field to targets by Adam on the mean-squared error, via a jitted step."""
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(splat)

    def loss_fn(params: tuple) -> jnp.ndarray:
        return jnp.mean((eval_splat(train_x, params) - train_y) ** 2)

    @jax.jit
    def step(params: tuple, state: optax.OptState) -> tuple:
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, state = optimizer.update(grads, state, params)
        return optax.apply_updates(params, updates), state, loss

    progress = trange(steps, desc="fitting splat")
    for i in progress:
        splat, opt_state, loss = step(splat, opt_state)
        if i % 25 == 0:
            progress.set_description(f"fitting splat — log10(MSE) = {float(jnp.log10(loss)):.3f}")
    return splat


def render_comparison(
    problem: PlanningProblem,
    prediction: jnp.ndarray,
    path: str,
    error_clip: float | None,
    goal: np.ndarray | None = None,
) -> dict:
    """Save a ``[ground truth | prediction | error]`` figure and return error metrics."""
    gt_image = problem.to_image(problem.ground_truth)
    pred_image = np.array(problem.to_image(prediction))
    pred_image[np.isnan(gt_image)] = np.nan
    error_image = pred_image - gt_image
    value_max = float(np.nanmax(gt_image))
    error_limit = error_clip if error_clip is not None else float(np.nanmax(np.abs(error_image)))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    handles = [
        draw_field(axes[0], problem, gt_image, "viridis", 0.0, value_max),
        draw_field(axes[1], problem, pred_image, "viridis", 0.0, value_max),
        draw_field(axes[2], problem, error_image, "bwr", -error_limit, error_limit, contours=0),
    ]
    for ax, title, handle in zip(axes, ("ground truth", "splat prediction", "error (pred − GT)"), handles):
        if goal is not None:
            ax.plot(goal[0], goal[1], "X", color="magenta", markersize=12, markeredgecolor="white", zorder=6)
        ax.set_title(title)
        fig.colorbar(handle, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"time-to-go fit — {problem.name}")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)

    rms = float(np.sqrt(np.nanmean(error_image**2)))
    return {
        "rms": rms,
        "max_abs": float(np.nanmax(np.abs(error_image))),
        "rel_rms": rms / (float(np.nanstd(gt_image)) + 1e-12),
        "error_limit": error_limit,
    }


def main(cfg: Config) -> None:
    """Build the problem, fit the splat field by least squares, and render the comparison."""
    problem = make_problem(cfg.problem, cfg.resolution, cfg.start)
    splat = init_splat(problem, cfg.num_splats, cfg.init_scale, jax.random.PRNGKey(cfg.seed))

    free = jnp.asarray(problem.free_indices())
    pick = jax.random.choice(
        jax.random.PRNGKey(cfg.seed + 1), free.shape[0], (min(cfg.num_train, free.shape[0]),), replace=False
    )
    sample_idx = free[pick]
    train_x = problem.points[sample_idx]
    train_y = problem.ground_truth[sample_idx]

    splat = fit_splat(splat, train_x, train_y, cfg.lr, cfg.steps)
    prediction = eval_splat(problem.points, splat)

    path = f"{cfg.out_dir}/fit_{problem.name}.png"
    metrics = render_comparison(problem, prediction, path, cfg.error_clip)
    print(f"saved {path}")
    print(
        f"RMS={metrics['rms']:.4e}  max|err|={metrics['max_abs']:.4e}  rel_RMS={metrics['rel_rms']:.4e}  error_limit={metrics['error_limit']:.4e}"
    )


if __name__ == "__main__":
    main(tyro.cli(Config))
