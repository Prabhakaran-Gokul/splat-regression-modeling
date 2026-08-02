"""Batched, reproducible obstacle-plane experiments for splat time-to-go fitting.

A single global seed deterministically generates a batch of scenarios. Each
scenario randomises the source, the goal, and one or more circular obstacles on
the Euclidean plane, computes the fast-marching ground truth, fits a splat field
to it, and records error metrics. Every scenario is reproducible from
``(global_seed, index)``, and per-scenario parameters plus metrics are written to
a TSV so problem cases can be located and replayed exactly.
"""

from __future__ import annotations

import dataclasses
import json
import os

import jax
import numpy as np
import tyro

from ground_truth import Circle, make_plane_fmm
from lib.splat import eval_splat
from train import fit_splat, init_splat, render_comparison


@dataclasses.dataclass
class Config:
    """Configuration for a batch of randomised obstacle-plane experiments.

    Attributes:
        global_seed: Master seed; scenario ``i`` derives from ``(global_seed, i)``.
        batch_size: Number of scenarios to generate and fit.
        num_obstacles: Number of circular obstacles per scenario.
        radius_range: Inclusive ``(min, max)`` obstacle radius.
        clearance: Minimum distance of the source and goal from any obstacle.
        half_width: Half-width of the square domain ``[-half_width, half_width]^2``.
        resolution: Fast-marching and evaluation grid resolution per axis.
        num_splats: Number of splats ``k`` in the model.
        num_train: Number of free-space points sampled as training targets.
        steps: Number of gradient-descent steps per scenario.
        lr: Adam learning rate.
        init_scale: Initial isotropic scale of each splat.
        error_clip: Shared symmetric error colour limit across scenarios.
        out_dir: Directory for the per-scenario comparison figures.
        log_path: TSV file collecting per-scenario parameters and metrics.
    """

    global_seed: int = 0
    batch_size: int = 4
    num_obstacles: int = 1
    radius_range: tuple[float, float] = (0.2, 0.4)
    clearance: float = 0.1
    half_width: float = 1.0
    resolution: int = 140
    num_splats: int = 256
    num_train: int = 4096
    steps: int = 1500
    lr: float = 5e-3
    init_scale: float = 0.3
    error_clip: float = 0.15
    out_dir: str = "figures/obstacle_batch"
    log_path: str = "logs/obstacle_batch.tsv"


def sample_obstacles(
    rng: np.random.Generator, num: int, half_width: float, radius_range: tuple[float, float]
) -> tuple[Circle, ...]:
    """Sample circular obstacles fully contained in the domain."""
    obstacles = []
    for _ in range(num):
        radius = float(rng.uniform(*radius_range))
        centre = rng.uniform(-half_width + radius, half_width - radius, size=2)
        obstacles.append((float(centre[0]), float(centre[1]), radius))
    return tuple(obstacles)


def sample_free_point(
    rng: np.random.Generator, obstacles: tuple[Circle, ...], half_width: float, clearance: float
) -> np.ndarray:
    """Reject-sample a point at least ``clearance`` away from every obstacle."""
    while True:
        point = rng.uniform(-half_width, half_width, size=2)
        if all((point[0] - cx) ** 2 + (point[1] - cy) ** 2 > (radius + clearance) ** 2 for cx, cy, radius in obstacles):
            return point


def run_scenario(cfg: Config, index: int) -> dict:
    """Generate scenario ``index``, fit a splat to its ground truth, and return a metric row."""
    rng = np.random.default_rng([cfg.global_seed, index])
    obstacles = sample_obstacles(rng, cfg.num_obstacles, cfg.half_width, cfg.radius_range)
    start = sample_free_point(rng, obstacles, cfg.half_width, cfg.clearance)
    goal = sample_free_point(rng, obstacles, cfg.half_width, cfg.clearance)

    problem = make_plane_fmm(cfg.resolution, (float(start[0]), float(start[1])), obstacles, cfg.half_width)
    key = jax.random.PRNGKey(cfg.global_seed * 10_000 + index)
    splat = init_splat(problem, cfg.num_splats, cfg.init_scale, key)

    free = problem.free_indices()
    pick = rng.choice(free.shape[0], size=min(cfg.num_train, free.shape[0]), replace=False)
    sample_idx = free[pick]
    splat = fit_splat(splat, problem.points[sample_idx], problem.ground_truth[sample_idx], cfg.lr, cfg.steps)
    prediction = eval_splat(problem.points, splat)
    figure_path = f"{cfg.out_dir}/scenario_seed{cfg.global_seed}_{index}.png"
    metrics = render_comparison(problem, prediction, figure_path, cfg.error_clip, goal=goal)
    return {
        "index": index,
        "global_seed": cfg.global_seed,
        "start": [float(start[0]), float(start[1])],
        "goal": [float(goal[0]), float(goal[1])],
        "obstacles": [list(o) for o in obstacles],
        "rms": metrics["rms"],
        "max_abs": metrics["max_abs"],
        "rel_rms": metrics["rel_rms"],
        "figure": figure_path,
    }


def write_log(path: str, rows: list[dict]) -> None:
    """Write per-scenario parameters and metrics as a tab-separated table."""
    columns = ["index", "global_seed", "start", "goal", "obstacles", "rms", "max_abs", "rel_rms", "figure"]

    def format_row(row: dict) -> str:
        return "\t".join(json.dumps(row[c]) if isinstance(row[c], list) else str(row[c]) for c in columns) + "\n"

    with open(path, "w") as handle:
        handle.write("\t".join(columns) + "\n")
        handle.writelines(format_row(row) for row in rows)


def main(cfg: Config) -> None:
    """Run the batch, render each scenario, and summarise the error distribution."""
    os.makedirs(cfg.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(cfg.log_path), exist_ok=True)

    rows = [run_scenario(cfg, index) for index in range(cfg.batch_size)]
    write_log(cfg.log_path, rows)

    rms_values = np.array([row["rms"] for row in rows])
    worst = max(rows, key=lambda row: row["rms"])
    print(f"\nwrote {len(rows)} scenarios to {cfg.log_path}")
    print(f"RMS  mean={rms_values.mean():.4e}  median={np.median(rms_values):.4e}  max={rms_values.max():.4e}")
    print(
        f"worst scenario: index={worst['index']} (seed {worst['global_seed']}) rms={worst['rms']:.4e} -> {worst['figure']}"
    )


if __name__ == "__main__":
    main(tyro.cli(Config))
