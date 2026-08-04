"""CLI entry point: solve the Eikonal time-to-go on an Environment with a chosen strategy.

Generalizes the original ``torus.py``'s ``main`` to dispatch through the
``STRATEGIES`` registry instead of an inline solver dict, and to build the
environment (obstacles, slowness field, ground truth) from ``TorusEnvironment``
instead of module-level torus functions.
"""

from __future__ import annotations

import dataclasses
import pickle
from typing import Literal

import numpy as np
import tyro

from srms.environments import sampling
from srms.environments.torus import TorusEnvironment
from srms.methods.backends import BACKENDS
from srms.methods.strategies import eikonal, weak_supervision
from srms.viz import render

STRATEGIES = {"eikonal": eikonal, "weak_supervision": weak_supervision}


@dataclasses.dataclass
class Config:
    """Configuration for a torus Eikonal splat solve with obstacles.

    Attributes:
        method: ``eikonal`` (free field, BC ring + PDE residual loss) or
            ``weak_supervision`` (RRT* soft-min base × exp(correction) refinement).
        backend: the function-approximator backend the strategy trains (see
            ``srms/methods/backends``); currently only ``srm`` is implemented.

        Scene — start: source joint angles (rad); num_obstacles / obstacle_radius: count and
            inclusive (min, max) radius of the circular angle-space obstacles; slowness_max /
            slow_width: peak slowness inside obstacles and the ramp width (rad).

        Eikonal strategy — source_radius: BC-sphere radius / PDE collocation exclusion radius
            around the source; n_sphere: number of BC points on that sphere; physics_weight:
            weight on the PDE loss term relative to the boundary-condition loss.

        Weak-supervision strategy (RRT*) — rrt_iters / rrt_step / rrt_radius: RRT* tree size,
            step, and rewiring radius; roadmap_nodes: base node budget; roadmap_gamma: soft-min
            temperature; roadmap_hop: samples along each last hop for its slowness weighting;
            base_reg: pull of the splat correction toward zero (small ⇒ physics leads).

        Training / output — num_splats, num_collocation, steps, lr, init_scale, resolution
            (eval + FMM grid), seed, error_clip (error colour limit), checkpoint_every, out_dir.
    """

    method: Literal["eikonal", "weak_supervision"] = "eikonal"
    backend: Literal["srm"] = "srm"
    # scene
    start: tuple[float, float] = (-1.5, -1.5)
    num_obstacles: int = 3
    obstacle_radius: tuple[float, float] = (0.5, 0.9)
    slowness_max: float = 10.0
    slow_width: float = 0.15
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
    out_dir: str = "figures"


def main(cfg: Config) -> None:
    """Solve the torus Eikonal with obstacles and score against periodic fast marching."""
    env = TorusEnvironment(
        start=cfg.start,
        num_obstacles=cfg.num_obstacles,
        obstacle_radius=cfg.obstacle_radius,
        slowness_max=cfg.slowness_max,
        slow_width=cfg.slow_width,
        seed=cfg.seed,
    )
    backend = BACKENDS[cfg.backend]
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
        return np.asarray(eikonal.predict(backend, current, thetas, env))

    def checkpoint(current, stepnum: int) -> None:
        marks = render(env, cfg, gt, predict_current(current), inside, shape, out_name=f"torus_ckpt_{stepnum}.png")
        print(f"  [ckpt {stepnum}] saved torus_ckpt_{stepnum}.png  RMS={marks['rms']:.4e}", flush=True)

    strategy = STRATEGIES[cfg.method]
    splat = strategy.solve(env, cfg, backend, checkpoint)
    prediction = predict_current(splat)
    metrics = render(env, cfg, gt, prediction, inside, shape)
    with open(f"{cfg.out_dir}/splat.pkl", "wb") as f:  # save params + scene for the near-obstacle diagnostic
        pickle.dump(
            {"splat": [np.asarray(a) for a in splat], "obstacles": env.obstacles, "cfg": dataclasses.asdict(cfg)}, f
        )
    print(f"saved {cfg.out_dir}/torus_obstacles.png  ({len(env.obstacles)} obstacles)")
    print(f"RMS={metrics['rms']:.4e}  max|err|={metrics['max_abs']:.4e}  rel_RMS={metrics['rel_rms']:.4e}")


if __name__ == "__main__":
    main(tyro.cli(Config))
