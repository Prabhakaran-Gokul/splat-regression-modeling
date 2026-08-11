"""One SRM formulation, three curvatures: the combined torus / sphere / hyperbolic figure.

Trains the *same* splat model with the *same* strategy and the *same* hyperparameters on T², S² and
H², and renders them as one 3x3 panel (rows = manifold, columns = ground truth | SRM | error). The
only thing that differs between rows is the ``Environment``, which supplies ``log_map``,
``jac_factor`` and ``metric_inv``. ``srms/methods/backends/srm.py`` is not touched, and neither is
the strategy.

Every number the figure shows is scored against that manifold's dense fast-marching field, and each
error panel is annotated with the marcher's *own* discretisation error (measured by
``--calibrate``-style obstacle-free runs, see ``results/manifolds.md``) so a residual at the noise
floor is not read as a modelling error.

Run:
    python manifold_figure.py --method hntfields    # roadmap-supervised + PDE (strongest)
    python manifold_figure.py --method ntfields     # planner-free: PDE only, no roadmap

Do **not** use ``--method eikonal`` for this comparison. Its unfactored free field starts at
``T ≡ 0`` (``srm.init_params`` sets ``V = 0``), which is an exact stationary point of the Eikonal
residual — measured ``|∂/∂params mean(‖∇T‖−s)²| = 0.000e+00`` at initialisation against a PDE loss of
10.98, so only the 32-point boundary ring has any gradient and the field never leaves a small bump at
the source. That is a property of the strategy on every manifold, not of the geometry; see
``results/manifolds.md`` §5.
"""

from __future__ import annotations

import argparse
import pickle

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from srms.methods.backends import BACKENDS
from srms.methods.strategies import eikonal, hntfields, ntfields, pntfields, weak_supervision
from srms.run import Config, _build_env

STRATEGIES = {
    "eikonal": eikonal,
    "weak_supervision": weak_supervision,
    "ntfields": ntfields,
    "pntfields": pntfields,
    "hntfields": hntfields,
}
_NTFIELDS_FAMILY = ("ntfields", "pntfields", "hntfields")

# Fast-marching error floor at resolution 120; no RMS below this is interpretable (results/manifolds.md).
GT_FLOOR_RMS = {"torus": 4.65e-2, "sphere": 2.13e-2, "hyperbolic": 4.04e-2}

ROWS = ("torus", "sphere", "hyperbolic")
CURVATURE = {"torus": "K = 0", "sphere": "K = +1", "hyperbolic": "K = −1"}


def load_one(pkl_path: str):
    """Rebuild one manifold's figure inputs from a saved run, without retraining.

    ``srms/run.py`` pickles ``(splat params, obstacles, cfg)`` per run, so a completed sweep can be
    re-rendered — and, more importantly, the figure is then guaranteed to show the *same* fitted
    models the reported metrics came from, rather than a fresh fit that might differ.
    """
    with open(pkl_path, "rb") as f:
        saved = pickle.load(f)
    cfg = Config(**saved["cfg"])
    env = _build_env(cfg)
    params = jax.tree_util.tree_map(jnp.asarray, saved["splat"])
    return finish(env, cfg, params, cfg.method)


def solve_one(name: str, method: str, steps: int, resolution: int, seed: int, tau_min: float):
    """Train on one manifold and return everything the figure needs."""
    cfg = Config(
        environment=name,
        method=method,
        backend="srm",
        dim=2,
        steps=steps,
        resolution=resolution,
        seed=seed,
        tau_min=tau_min,
        # eikonal.solve never calls adapt, so densify would silently pin it at init_splats.
        densify=method in ("ntfields", "hntfields"),
    )
    env = _build_env(cfg)
    print(f"\n=== {name} ({CURVATURE[name]}) — {method} ===", flush=True)
    params = STRATEGIES[method].solve(env, cfg, BACKENDS["srm"])
    return finish(env, cfg, params, method)


def finish(env, cfg, params, method: str):
    """Score a fitted model against the manifold's dense fast-marching field."""
    backend = BACKENDS["srm"]
    points, shape = env.grid(cfg.resolution)
    gt = env.ground_truth(cfg.resolution)
    inside = np.asarray(env.sdf(points)) < 0.0

    if method in _NTFIELDS_FAMILY:
        prediction = np.asarray(ntfields.predict(backend, params, points, env, cfg.tau_bias, cfg.tau_min))
    else:
        prediction = np.asarray(eikonal.predict(backend, params, points, env))
    return env, gt, prediction, inside, shape, backend.num_params(params)


def draw(results: dict, method: str, out_path: str) -> None:
    """Render the 3x3 panel. Colour scales are per-row: the three manifolds have different diameters
    (T² 4.4, S² 3.1, H² 5.9), so one shared scale would flatten two of the three rows."""
    fig, axes = plt.subplots(3, 3, figsize=(16.5, 15.0))

    for row, name in enumerate(ROWS):
        env, gt, prediction, inside, shape, n_params = results[name]
        gt_img = np.where(inside, np.nan, gt).reshape(shape)
        pred_img = np.where(inside, np.nan, prediction).reshape(shape)
        err_img = pred_img - gt_img
        vmax = float(np.nanmax(gt_img))
        rms = float(np.sqrt(np.nanmean(err_img**2)))
        rel = rms / (np.nanstd(gt_img) + 1e-12)
        # Clip the diverging scale at the 98th percentile so outlier cells do not wash the panel out.
        clip = max(float(np.nanpercentile(np.abs(err_img), 98)), 1e-3)
        extent = env.render_extent
        mesh1, mesh2 = np.meshgrid(
            np.linspace(extent[0], extent[1], shape[1]), np.linspace(extent[2], extent[3], shape[0])
        )
        blocked = inside.reshape(shape).astype(float)

        panels = [
            (env.gt_label, gt_img, "viridis", 0.0, vmax, 14),
            (f"SRM ({n_params} params)", pred_img, "viridis", 0.0, vmax, 14),
            (f"error — RMS {rms:.3f} (rel {rel:.1%}), ±{clip:.2f}", err_img, "bwr", -clip, clip, 0),
        ]
        for col, (title, img, cmap, lo, hi, levels) in enumerate(panels):
            ax = axes[row, col]
            ax.set_facecolor("0.85")
            handle = ax.imshow(img, origin="lower", extent=extent, cmap=cmap, vmin=lo, vmax=hi, aspect="auto")
            if levels:
                ax.contour(mesh1, mesh2, img, levels=levels, colors="white", linewidths=0.6, alpha=0.7)
            ax.contour(mesh1, mesh2, blocked, levels=[0.5], colors="black", linewidths=1.2)
            if hasattr(env, "trunc_radius"):  # make the Poincaré rim a wall, not a rendering artifact
                ax.add_patch(Circle((0, 0), env.trunc_radius, fill=False, ec="black", lw=1.4, ls="--"))
            ax.plot(*env.render_marker_deg(), "*", color="red", markersize=15, markeredgecolor="white")
            ax.set_title(title, fontsize=10)
            ax.set_xlabel(env.axis_labels[0], fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{name}  ({CURVATURE[name]})\n{env.axis_labels[1]}", fontsize=10)
            fig.colorbar(handle, ax=ax, fraction=0.046, pad=0.04)
            if col == 2:
                ax.text(
                    0.02,
                    0.02,
                    f"FMM floor {GT_FLOOR_RMS[name]:.3f}",
                    transform=ax.transAxes,
                    fontsize=8,
                    va="bottom",
                    ha="left",
                    bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.6", "alpha": 0.85},
                )

    fig.suptitle(
        f"One SRM, three curvatures — self-supervised Eikonal time-to-go via '{method}'\n"
        "identical backend, strategy and hyperparameters; only log_map / jac_factor / metric_inv differ",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nsaved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", default="ntfields", choices=sorted(STRATEGIES))  # see docstring re: eikonal
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--resolution", type=int, default=120)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--tau_min",
        type=float,
        default=0.01,
        help="floor on tau in T=base/tau. The paper's 0.0 lets tau collapse (measured "
        "4.3e-22 on S^2 at ~200 splats, giving T=6.4e21 and NaN); 0.01 bounds it.",
    )
    parser.add_argument(
        "--load",
        metavar="DIR",
        help="render from saved runs DIR/<manifold>/splat.pkl instead of retraining, so "
        "the figure shows exactly the models the reported metrics came from",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.load:
        results = {n: load_one(f"{args.load}/taumin_{n}/splat.pkl") for n in ROWS}
    else:
        results = {n: solve_one(n, args.method, args.steps, args.resolution, args.seed, args.tau_min) for n in ROWS}
    out = args.out or f"figures/manifolds/srm_three_manifolds_{args.method}.png"
    draw(results, args.method, out)

    print(f"\n{'manifold':<12} {'curvature':>10} {'params':>8} {'RMS':>10} {'rel_RMS':>9} {'FMM floor':>10}")
    for name in ROWS:
        _, gt, prediction, inside, shape, n_params = results[name]
        gt_img = np.where(inside, np.nan, gt).reshape(shape)
        err = np.where(inside, np.nan, prediction).reshape(shape) - gt_img
        rms = float(np.sqrt(np.nanmean(err**2)))
        print(
            f"{name:<12} {CURVATURE[name]:>10} {n_params:>8} {rms:>10.4f} "
            f"{rms / (np.nanstd(gt_img) + 1e-12):>8.1%} {GT_FLOOR_RMS[name]:>10.4f}"
        )


if __name__ == "__main__":
    main()
