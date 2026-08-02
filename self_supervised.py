"""Self-supervised (physics-informed) splat solve of the Eikonal time-to-go field.

No ground-truth targets are used for training. The splat field ``T`` is fit to
satisfy the viscosity-regularised Eikonal ``|∇T|² − ε ΔT = 1`` in free space,
factored as ``T(x) = ‖x − start‖ · (1 + g(x))`` with the splat representing the
deviation ``g`` (initialised to zero) — this bakes in ``T(start) = 0`` and makes
the free-space distance the exact starting point.

Obstacles are encoded as a **smooth high-slowness field** derived from the scene
SDF (``scenes.smooth_slowness``): slowness ~1 in free space rising smoothly to
``slowness_max`` inside each object, i.e. the conformal Riemannian metric
``g = s(x)² I``. The high interior cost is a *global* signal that forces the wave
to route around, and the smooth (not discontinuous) transition keeps the field
representable by a smooth splat, so per-object rim error stays small and does not
compound as objects are added. The residual is taken *relative* (divided by
``s²``) so free-space and high-slowness points are balanced. This one formulation
handles one circle, many objects, and intricate non-convex shapes (SDF unions).

A reflecting Neumann boundary (``∂T/∂n = 0``) was tried first; on its own it under-
determined the nonlocal shadow behind obstacles under gradient descent (the field
stayed near straight-through), so the global slowness signal above is used instead.
Ground truth (fast marching on the same speed field) is used only to score.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import tyro
from tqdm import trange

import scenes
from ground_truth import PlanningProblem, draw_field, make_plane, make_plane_scene
from lib.splat import eval_splat
from scenes import Scene
from train import init_splat, render_comparison


@dataclasses.dataclass
class Config:
    """Configuration for a self-supervised Eikonal splat solve on the plane.

    Attributes:
        start: Source position ``(x, y)`` that time-to-go is measured from.
        num_objects: Number of random obstacles (ignored when ``scene_name`` is set).
        scene_name: Named intricate scene (``"cross"`` / ``"u_trap"``); ``None`` for random/free.
        box_fraction: Fraction of random obstacles that are boxes rather than circles.
        epsilon: Viscosity coefficient of the ``−ε ΔT`` regularisation term.
        num_collocation: Number of PDE collocation points.
        source_radius: Collocation exclusion radius around ``start`` (avoids the 1/r singularity).
        slowness_max: Peak slowness inside obstacles (metric contrast).
        slow_width: SDF transition width of the smooth slowness ramp.
        num_splats: Number of splats ``k``.
        steps: Number of optimisation steps.
        lr: Adam learning rate.
        init_scale: Initial isotropic scale of each splat.
        resolution: Evaluation and fast-marching grid resolution.
        adaptive: Enable residual-adaptive collocation (resample points toward high residual).
        resample_every: Steps between collocation resamples.
        pool_size: Candidate pool size scored by residual at each resample.
        resample_uniform_frac: Fraction of resampled points drawn uniformly (coverage floor).
        residual_power: Exponent on the residual when forming resample probabilities.
        weak_weight: Weight of the weak-supervision anchor term (0 disables it).
        num_weak: Number of sparse weak-supervision anchor points.
        weak_resolution: Grid resolution of the cheap coarse solver providing weak targets.
        weak_clearance: Minimum distance of weak anchors from obstacles (cheap solver is bad at rims).
        plan_goals: Number of demonstration goals to extract paths to.
        path_step: Euler step size for path integration of ``−∇T``.
        guard_margin: SDF clearance the feasibility guard keeps paths away from obstacles.
        seed: Global seed for splat init, scene, and collocation sampling.
        error_clip: Symmetric error colour limit for the figure.
        out_dir: Directory for the comparison figure.
    """

    start: tuple[float, float] = (-0.6, -0.6)
    num_objects: int = 1
    scene_name: str | None = None
    box_fraction: float = 0.4
    epsilon: float = 0.02
    num_collocation: int = 1536
    source_radius: float = 0.12
    slowness_max: float = 12.0
    slow_width: float = 0.04
    num_splats: int = 256
    steps: int = 4000
    lr: float = 3e-3
    init_scale: float = 0.3
    resolution: int = 140
    adaptive: bool = True
    resample_every: int = 400
    pool_size: int = 4096
    resample_uniform_frac: float = 0.4
    residual_power: float = 2.0
    weak_weight: float = 0.0
    num_weak: int = 96
    weak_resolution: int = 60
    weak_clearance: float = 0.15
    plan_goals: int = 4
    path_step: float = 0.008
    guard_margin: float = 0.02
    seed: int = 0
    error_clip: float = 0.15
    out_dir: str = "figures"


def field_value(splat: tuple, x: jnp.ndarray, start: jnp.ndarray) -> jnp.ndarray:
    """Evaluate the factored field ``T(x) = ‖x − start‖ · (1 + g(x))`` at a single point ``x``."""
    return jnp.linalg.norm(x - start) * (1.0 + eval_splat(x[None, :], splat)[0, 0])


def field_grad(splat: tuple, points: jnp.ndarray, start: jnp.ndarray) -> jnp.ndarray:
    """Return ``∇T`` at each point, shape ``[N, 2]``."""
    return jax.vmap(lambda x: jax.grad(field_value, argnums=1)(splat, x, start))(points)


def eikonal_residual(
    splat: tuple, points: jnp.ndarray, slow: jnp.ndarray, epsilon: float, start: jnp.ndarray
) -> jnp.ndarray:
    """Return the relative viscosity-Eikonal residual ``(|∇T|² − ε ΔT)/s² − 1`` at each point."""
    grad = field_grad(splat, points, start)
    laplacian = jax.vmap(lambda x: jnp.trace(jax.hessian(field_value, argnums=1)(splat, x, start)))(points)
    return (jnp.sum(grad**2, axis=1) - epsilon * laplacian) / slow**2 - 1.0


def predict_field(splat: tuple, points: jnp.ndarray, start: jnp.ndarray) -> jnp.ndarray:
    """Evaluate the factored field ``T`` over a batch of points, returned as ``[N, 1]``."""
    return jax.vmap(lambda x: field_value(splat, x, start))(points)[:, None]


def sample_collocation(cfg: Config) -> jnp.ndarray:
    """Sample collocation points over the domain, clear of the source singularity."""
    sampled = np.random.default_rng(cfg.seed).uniform(-1.0, 1.0, size=(2 * cfg.num_collocation, 2))
    keep = np.linalg.norm(sampled - np.asarray(cfg.start), axis=1) > cfg.source_radius
    return jnp.asarray(sampled[keep][: cfg.num_collocation], dtype=jnp.float32)


def slowness_of(points: jnp.ndarray, scene: Scene, cfg: Config) -> jnp.ndarray:
    """Return the slowness at ``points`` (1 on the free plane, the SDF ramp with a scene)."""
    if scene:
        return scenes.smooth_slowness(points, scene, cfg.slowness_max, cfg.slow_width)
    return jnp.ones(points.shape[0])


def weak_anchors(cfg: Config, scene: Scene) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Sample sparse weak-supervision anchors from a cheap coarse solver, biased to large distance.

    The coarse fast-marching field is a stand-in for any weak/cheap absolute-time signal (a
    low-resolution planner, sparse measured travel times). Anchors are drawn with probability
    proportional to distance from the source, where the accumulated viscosity bias is largest.
    Anchors are kept clear of obstacle rims, where the cheap coarse solver is least reliable and
    would otherwise corrupt the field locally.
    """
    if scene:
        coarse = make_plane_scene(
            cfg.weak_resolution, cfg.start, scene, slowness_max=cfg.slowness_max, width=cfg.slow_width
        )
    else:
        coarse = make_plane(cfg.weak_resolution, cfg.start)
    free = coarse.free_indices()
    points = np.asarray(coarse.points)[free]
    times = np.asarray(coarse.ground_truth)[free, 0]
    if scene:
        clearance = np.asarray(scenes.scene_sdf(jnp.asarray(points, dtype=jnp.float32), scene)) > cfg.weak_clearance
        points, times = points[clearance], times[clearance]
    distance = np.linalg.norm(points - np.asarray(cfg.start), axis=1)
    probability = distance / distance.sum()
    chosen = np.random.default_rng(cfg.seed + 7).choice(len(points), size=cfg.num_weak, replace=False, p=probability)
    return jnp.asarray(points[chosen], dtype=jnp.float32), jnp.asarray(times[chosen], dtype=jnp.float32)


def solve(cfg: Config, scene: Scene) -> tuple:
    """Fit the splat deviation with adaptive collocation and optional weak supervision."""
    weights, scales, centres = init_splat(
        make_plane(cfg.resolution, cfg.start), cfg.num_splats, cfg.init_scale, jax.random.PRNGKey(cfg.seed)
    )
    splat = (jnp.zeros_like(weights), scales, centres)
    start = jnp.asarray(cfg.start)
    rng = np.random.default_rng(cfg.seed)
    collocation = sample_collocation(cfg)
    slow = slowness_of(collocation, scene, cfg)
    weak_x, weak_t = weak_anchors(cfg, scene) if cfg.weak_weight > 0 else (jnp.zeros((0, 2)), jnp.zeros((0,)))

    optimizer = optax.adam(cfg.lr)
    opt_state = optimizer.init(splat)

    def loss_fn(params: tuple, points: jnp.ndarray, slowness: jnp.ndarray) -> jnp.ndarray:
        pde = jnp.mean(eikonal_residual(params, points, slowness, cfg.epsilon, start) ** 2)
        if weak_x.shape[0] == 0:
            return pde
        predicted = jax.vmap(lambda x: field_value(params, x, start))(weak_x)
        return pde + cfg.weak_weight * jnp.mean((predicted - weak_t) ** 2)

    @jax.jit
    def step(params: tuple, state: optax.OptState, points: jnp.ndarray, slowness: jnp.ndarray) -> tuple:
        loss, grads = jax.value_and_grad(loss_fn)(params, points, slowness)
        updates, state = optimizer.update(grads, state, params)
        return optax.apply_updates(params, updates), state, loss

    @jax.jit
    def residual_magnitude(params: tuple, points: jnp.ndarray, slowness: jnp.ndarray) -> jnp.ndarray:
        return jnp.abs(eikonal_residual(params, points, slowness, cfg.epsilon, start))

    def resample(params: tuple) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Draw collocation points with probability blended from the residual and a uniform floor."""
        pool = rng.uniform(-1.0, 1.0, size=(cfg.pool_size, 2))
        pool = pool[np.linalg.norm(pool - np.asarray(cfg.start), axis=1) > cfg.source_radius]
        slow_pool = slowness_of(jnp.asarray(pool, dtype=jnp.float32), scene, cfg)
        weight = (
            np.asarray(residual_magnitude(params, jnp.asarray(pool, dtype=jnp.float32), slow_pool))
            ** cfg.residual_power
        )
        residual_p = weight / weight.sum()
        blended = (1.0 - cfg.resample_uniform_frac) * residual_p + cfg.resample_uniform_frac / len(pool)
        chosen = rng.choice(len(pool), size=cfg.num_collocation, replace=False, p=blended / blended.sum())
        return jnp.asarray(pool[chosen], dtype=jnp.float32), slow_pool[chosen]

    progress = trange(cfg.steps, desc="self-supervised eikonal")
    for iteration in progress:
        if cfg.adaptive and iteration > 0 and iteration % cfg.resample_every == 0:
            collocation, slow = resample(splat)
        splat, opt_state, loss = step(splat, opt_state, collocation, slow)
        if iteration % 25 == 0:
            progress.set_description(f"self-supervised eikonal — log10(loss) = {float(jnp.log10(loss)):.3f}")
    return splat


def extract_path(
    splat: tuple,
    start: np.ndarray,
    goal: np.ndarray,
    step: float,
    scene: Scene,
    guard: bool,
    margin: float = 0.02,
    max_steps: int = 6000,
    tol: float = 0.02,
) -> np.ndarray:
    """Integrate ``ẋ = −∇T/|∇T|`` from a goal down to the source, returning the source→goal polyline.

    When ``guard`` is set, a hard feasibility check (the scene SDF) overrides the soft field: any
    step heading into an obstacle has its inward normal component removed (slide along the surface)
    and any penetration is projected back out along ``∇d``. This makes the path collision-free
    regardless of how permeable the soft slowness field is.
    """
    gradient = jax.jit(lambda x: jax.grad(field_value, argnums=1)(splat, x, jnp.asarray(start, dtype=jnp.float32)))
    guarded = guard and bool(scene)
    if guarded:
        sdf_value_grad = jax.jit(jax.value_and_grad(lambda p: scenes.scene_sdf(p, scene)))
    start_np = np.asarray(start, dtype=float)
    position = np.asarray(goal, dtype=float)
    path = [position.copy()]
    for _ in range(max_steps):
        raw = np.asarray(gradient(jnp.asarray(position, dtype=jnp.float32)))
        direction = -raw / (np.linalg.norm(raw) + 1e-9)
        if guarded:
            distance, normal = sdf_value_grad(jnp.asarray(position, dtype=jnp.float32))
            unit_normal = np.asarray(normal) / (np.linalg.norm(np.asarray(normal)) + 1e-9)
            inward = float(np.dot(direction, unit_normal))
            if float(distance) < margin and inward < 0.0:
                tangential = direction - inward * unit_normal
                if np.linalg.norm(tangential) < 0.3:
                    tangential = np.array([-unit_normal[1], unit_normal[0]])
                    if np.dot(tangential, start_np - position) < 0.0:
                        tangential = -tangential
                direction = tangential / (np.linalg.norm(tangential) + 1e-9)
        position = position + step * direction
        if guarded:
            distance, normal = sdf_value_grad(jnp.asarray(position, dtype=jnp.float32))
            if float(distance) < margin:
                unit_normal = np.asarray(normal) / (np.linalg.norm(np.asarray(normal)) + 1e-9)
                position = position + (margin - float(distance)) * unit_normal
        path.append(position.copy())
        if np.linalg.norm(position - start_np) < tol:
            break
    path.append(start_np)
    return np.array(path)[::-1]


def choose_goals(scene: Scene, count: int) -> np.ndarray:
    """Return up to ``count`` demonstration goals in free space around the domain."""
    candidates = np.array([[0.85, 0.85], [0.85, -0.85], [-0.85, 0.85], [0.9, 0.0], [0.0, 0.9], [-0.85, -0.85]])
    if scene:
        free = np.asarray(scenes.scene_sdf(jnp.asarray(candidates, dtype=jnp.float32), scene)) > 0.06
        candidates = candidates[free]
    return candidates[:count]


def build_scene(cfg: Config) -> tuple[Scene, str]:
    """Return the obstacle scene and a short tag from the configuration."""
    if cfg.scene_name is not None:
        return scenes.named_scene(cfg.scene_name), cfg.scene_name
    if cfg.num_objects <= 0:
        return (), "free"
    return scenes.random_scene(
        cfg.seed, cfg.num_objects, box_fraction=cfg.box_fraction, avoid=cfg.start
    ), f"{cfg.num_objects}obj"


def render_paths(
    problem: PlanningProblem,
    prediction: jnp.ndarray,
    start: np.ndarray,
    goals: np.ndarray,
    guarded: list,
    unguarded: list,
    out_path: str,
) -> None:
    """Draw the predicted field with guarded (feasible) and unguarded extracted paths overlaid."""
    image = np.array(problem.to_image(prediction))
    if problem.valid is not None:
        image[(~np.asarray(problem.valid)).reshape(problem.grid_shape)] = np.nan
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    handle = draw_field(ax, problem, image, "viridis", 0.0, float(np.nanmax(image)))
    for path in unguarded:
        ax.plot(path[:, 0], path[:, 1], color="orange", linewidth=1.6, linestyle="--", zorder=6)
    for goal, path in zip(goals, guarded):
        ax.plot(path[:, 0], path[:, 1], color="red", linewidth=2.2, zorder=7)
        ax.plot(goal[0], goal[1], "X", color="magenta", markersize=12, markeredgecolor="white", zorder=8)
    ax.plot([], [], color="red", linewidth=2.2, label="feasibility-guarded")
    ax.plot([], [], color="orange", linewidth=1.6, linestyle="--", label="raw −∇T")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title(f"extracted paths — {problem.name}")
    fig.colorbar(handle, ax=ax, label="time-to-go")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main(cfg: Config) -> None:
    """Solve the Eikonal self-supervised, score it, and extract optimal paths from the field."""
    scene, tag = build_scene(cfg)
    if scene:
        problem = make_plane_scene(
            cfg.resolution, cfg.start, scene, slowness_max=cfg.slowness_max, width=cfg.slow_width
        )
    else:
        problem = make_plane(cfg.resolution, cfg.start)

    splat = solve(cfg, scene)
    prediction = predict_field(splat, problem.points, jnp.asarray(cfg.start))
    comparison_path = f"{cfg.out_dir}/selfsup_{tag}.png"
    metrics = render_comparison(problem, prediction, comparison_path, cfg.error_clip)
    print(f"saved {comparison_path}  (scene: {tag}, {len(scene)} objects)")
    print(f"RMS={metrics['rms']:.4e}  max|err|={metrics['max_abs']:.4e}  rel_RMS={metrics['rel_rms']:.4e}")

    goals = choose_goals(scene, cfg.plan_goals)
    start = np.asarray(cfg.start)
    guarded = [extract_path(splat, start, goal, cfg.path_step, scene, True, cfg.guard_margin) for goal in goals]
    unguarded = [extract_path(splat, start, goal, cfg.path_step, scene, False) for goal in goals]
    render_paths(problem, prediction, start, goals, guarded, unguarded, f"{cfg.out_dir}/path_{tag}.png")
    infeasible = (
        sum(int(np.asarray(scenes.scene_sdf(jnp.asarray(p, jnp.float32), scene)).min() < 0) for p in unguarded)
        if scene
        else 0
    )
    print(
        f"saved {cfg.out_dir}/path_{tag}.png  ({len(goals)} paths; {infeasible} raw paths entered an obstacle, 0 guarded)"
    )


if __name__ == "__main__":
    main(tyro.cli(Config))
