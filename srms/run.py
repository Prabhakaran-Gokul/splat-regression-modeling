"""CLI entry point: solve the Eikonal time-to-go on an Environment with a chosen strategy.

Generalizes the original ``torus.py``'s ``main`` to dispatch through the
``STRATEGIES`` registry instead of an inline solver dict, and to build the
environment (obstacles, slowness field, ground truth) from the ``ENVIRONMENTS``
registry (``srms/environments``) instead of module-level torus functions.
"""

from __future__ import annotations

import dataclasses
import pickle
from typing import Literal

import jax
import mlflow
import numpy as np
import tyro

from srms.environments import ENVIRONMENTS, sampling
from srms.methods.backends import BACKENDS
from srms.methods.strategies import eikonal, ntfields, weak_supervision
from srms.viz import render

STRATEGIES = {"eikonal": eikonal, "weak_supervision": weak_supervision, "ntfields": ntfields}


@dataclasses.dataclass
class Config:
    """Configuration for an Eikonal splat solve with obstacles on a chosen manifold.

    Attributes:
        environment: ``torus`` (flat Tⁿ, e.g. an n-joint revolute arm's configuration space) or
            ``sphere`` (Sⁿ embedded in R^(n+1)).
        dim: intrinsic manifold dimension — the ``n`` in Tⁿ / Sⁿ. Training (mesh-free collocation)
            works at any dim; the dense-grid ground truth (fast marching) and rendering only make
            sense at dim=2 (a dense grid is intractable and unplottable beyond that) and are skipped
            for dim>2 — only the final training loss is reported.
        method: ``eikonal`` (free field, BC ring + PDE residual loss),
            ``weak_supervision`` (RRT* soft-min base × exp(correction) refinement), or
            ``ntfields`` (analytic-geodesic base / network-predicted speed τ).
        backend: the function-approximator backend the strategy trains (see
            ``srms/methods/backends``) — ``srm`` (splat mixture) or ``mlp`` (SIREN-style
            periodic-activation MLP).

        Scene — start: source point (rad joint angles for torus, unit ambient vector for sphere;
            defaults to a dim-appropriate point if left unset); num_obstacles / obstacle_radius:
            count and inclusive (min, max) angular radius of the obstacles; slowness_max /
            slow_width: peak slowness inside obstacles and the ramp width (rad).

        Causal weighting (all strategies) — causal: weight each strategy's residual term
            source-outward, so a far collocation point is only trusted once nearer points are
            already well-fit (Wang et al. 2022; ``training_aids.py``); causal_strength: base decay
            rate scale (divided by num_collocation); causal_anneal: relax the rate to 0 by the end
            of training.

        Eikonal strategy — source_radius: BC-ring geodesic radius / PDE collocation exclusion
            radius around the source; n_sphere: number of BC points on that ring; physics_weight:
            weight on the PDE loss term relative to the boundary-condition loss.

        Weak-supervision strategy (RRT*) — rrt_iters / rrt_step / rrt_radius: RRT* tree size,
            step, and rewiring radius; roadmap_nodes: base node budget; roadmap_gamma: soft-min
            temperature; roadmap_hop: samples along each last hop for its slowness weighting;
            base_reg: pull of the splat correction toward zero (small ⇒ physics leads).

        NTFields strategy — tau_bias: initial sigmoid bias so τ starts near 1 (free space) at
            init; tau_min: floor of τ = τ_min + (1−τ_min)·σ(g+bias) (0 ⇒ the un-floored paper
            formulation; >0 is an ablation against the τ→0 runaway).

        MLP backend — mlp_width: hidden layer width; mlp_depth: number of hidden layers;
            mlp_omega0: SIREN frequency scale for the hidden-layer init (Sitzmann et al.).

        Training / output — num_splats, num_collocation, steps, lr, init_scale, resolution
            (eval + FMM grid, dim=2 only), seed, error_clip (error colour limit),
            checkpoint_every, log_every (training-metric logging cadence, incl. mlflow), out_dir.
    """

    environment: Literal["torus", "sphere"] = "torus"
    dim: int = 2
    method: Literal["eikonal", "weak_supervision", "ntfields"] = "eikonal"
    backend: Literal["srm", "mlp"] = "srm"
    # scene
    start: tuple[float, ...] | None = None
    num_obstacles: int = 3
    obstacle_radius: tuple[float, float] = (0.5, 0.9)
    slowness_max: float = 10.0
    slow_width: float = 0.15
    # causal weighting (all strategies)
    causal: bool = True
    causal_strength: float = 5.0
    causal_anneal: bool = True
    # eikonal strategy
    source_radius: float = 0.25
    n_sphere: int = 32
    physics_weight: float = 1.0
    # weak_supervision strategy (RRT*)
    rrt_iters: int = 350
    rrt_step: float = 0.5
    rrt_radius: float = 0.9
    roadmap_nodes: int = 300
    roadmap_gamma: float = 0.05
    roadmap_hop: int = 5
    base_reg: float = 1.0
    # ntfields strategy
    tau_bias: float = 4.0
    tau_min: float = 0.0
    # mlp backend
    mlp_width: int = 128
    mlp_depth: int = 3
    mlp_omega0: float = 30.0
    # training / output
    num_splats: int = 384
    num_collocation: int = 2048
    steps: int = 4000
    lr: float = 3e-3
    init_scale: float = 0.35
    resolution: int = 120
    seed: int = 1
    error_clip: float = 0.2
    checkpoint_every: int = 1500
    log_every: int = 25
    out_dir: str = "figures"


def _build_env(cfg: Config):
    """Construct the configured Environment, filling in a dim-appropriate default start if unset."""
    if cfg.environment == "torus":
        start = cfg.start if cfg.start is not None else (-1.5,) * cfg.dim
        return ENVIRONMENTS["torus"](
            start=start,
            dim=cfg.dim,
            num_obstacles=cfg.num_obstacles,
            obstacle_radius=cfg.obstacle_radius,
            slowness_max=cfg.slowness_max,
            slow_width=cfg.slow_width,
            seed=cfg.seed,
        )
    start = cfg.start if cfg.start is not None else (0.0,) * cfg.dim + (1.0,)
    return ENVIRONMENTS["sphere"](
        start=start,
        n=cfg.dim,
        num_obstacles=cfg.num_obstacles,
        obstacle_radius=cfg.obstacle_radius,
        slowness_max=cfg.slowness_max,
        slow_width=cfg.slow_width,
        seed=cfg.seed,
    )


def main(cfg: Config) -> None:
    """Solve the Eikonal PDE with obstacles and, at dim=2, score against a dense fast-marching grid."""
    env = _build_env(cfg)
    backend = BACKENDS[cfg.backend]
    dense = cfg.dim == 2  # grid ground truth / rendering only tractable at dim=2

    thetas = shape = gt = inside = None
    if dense:
        thetas, shape = env.grid(cfg.resolution)
        gt = env.ground_truth(cfg.resolution)
        inside = np.asarray(env.sdf(thetas)) < 0.0

    roadmap = None
    if cfg.method == "weak_supervision":
        roadmap = sampling.build_roadmap(
            env, env.start, cfg.rrt_iters, cfg.rrt_step, cfg.rrt_radius, cfg.roadmap_nodes, cfg.seed
        )

    def predict_current(current) -> np.ndarray:
        if cfg.method == "weak_supervision":
            nodes, costs = roadmap
            return np.asarray(
                weak_supervision.predict(backend, current, thetas, nodes, costs, cfg.roadmap_gamma, env, cfg.roadmap_hop)
            )
        if cfg.method == "ntfields":
            return np.asarray(ntfields.predict(backend, current, thetas, env, cfg.tau_bias, cfg.tau_min))
        return np.asarray(eikonal.predict(backend, current, thetas, env))

    checkpoint = None
    if dense:

        def checkpoint(current, stepnum: int) -> None:
            out_name = f"{cfg.environment}_ckpt_{stepnum}.png"
            marks = render(env, cfg, gt, predict_current(current), inside, shape, out_name=out_name)
            print(f"  [ckpt {stepnum}] saved {out_name}  RMS={marks['rms']:.4e}", flush=True)
            mlflow.log_metric("ckpt_rms", marks["rms"], step=stepnum)

    run_name = f"{cfg.environment}-{cfg.method}-{cfg.backend}-d{cfg.dim}"
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags(
            {
                "environment": cfg.environment,
                "method": cfg.method,
                "backend": cfg.backend,
                "dim": str(cfg.dim),
                # same for the srm/mlp pair of a scene -> group by this tag to compare backends head-to-head
                "comparison_group": f"{cfg.environment}-{cfg.method}-d{cfg.dim}",
            }
        )
        mlflow.log_params(dataclasses.asdict(cfg))

        def progress_fn(step: int, metrics: dict) -> None:
            mlflow.log_metrics(metrics, step=step)

        strategy = STRATEGIES[cfg.method]
        splat = strategy.solve(env, cfg, backend, checkpoint, progress_fn=progress_fn)
        with open(f"{cfg.out_dir}/splat.pkl", "wb") as f:  # save params + scene for the near-obstacle diagnostic
            pickle.dump(
                {
                    "splat": jax.tree_util.tree_map(np.asarray, splat),
                    "obstacles": env.obstacles,
                    "cfg": dataclasses.asdict(cfg),
                },
                f,
            )

        if not dense:
            print(
                f"dim={cfg.dim} > 2: no dense-grid ground truth / rendering (intractable past dim=2); training done."
            )
            print(f"saved {cfg.out_dir}/splat.pkl  ({len(env.obstacles)} obstacles)")
            return

        out_name = f"{cfg.environment}_obstacles.png"
        prediction = predict_current(splat)
        metrics = render(env, cfg, gt, prediction, inside, shape, out_name=out_name)
        mlflow.log_metrics({f"final_{k}": v for k, v in metrics.items()})
        mlflow.log_artifact(f"{cfg.out_dir}/{out_name}")
        print(f"saved {cfg.out_dir}/{out_name}  ({len(env.obstacles)} obstacles)")
        print(f"RMS={metrics['rms']:.4e}  max|err|={metrics['max_abs']:.4e}  rel_RMS={metrics['rel_rms']:.4e}")


if __name__ == "__main__":
    main(tyro.cli(Config))
