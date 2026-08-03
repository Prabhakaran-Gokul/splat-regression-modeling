"""Self-supervised splat solve of the Eikonal time-to-go on the 2-torus ``T²``.

``T²`` is the configuration space of a 2-joint revolute arm. We use the **intrinsic**
placement (approach A): splats live in angle space and are evaluated at the flat-torus
log map ``wrap(θ − B)``, so the model dimension equals the number of joints (``d = n``)
and periodicity is automatic — no ambient embedding. The metric is kept separate
(``metric_inv``: identity for the flat torus, ``M(θ)⁻¹`` for the arm), entering only the
residual, so a curved metric plugs in without touching the placement.

Design: **known base + splat correction.** The analytic flat-torus geodesic
``‖wrap(θ − start)‖`` is the base (correct global structure, in any dimension); the splat
learns only *local* corrections — here the routing around obstacles. Obstacles are circles
in angle space encoded as a smooth high-slowness field (the conformal metric ``g = s²I``).
Ground truth is a periodic fast-marching solve on the same field.
"""

from __future__ import annotations

import dataclasses
import heapq
import pickle
from typing import Literal

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import tyro
from tqdm import trange

Obstacle = tuple[float, float, float]


@dataclasses.dataclass
class Config:
    """Configuration for a self-supervised torus Eikonal splat solve with obstacles.

    Attributes:
        method: Which solver — ``eikonal`` (base·(1+g) + residual²), ``ntfields`` (base/τ + speed-match
            loss + causal + progressive + optional RRT* anchors), ``roadmap`` (RRT* soft-min base +
            Eikonal refinement + optional anchors), ``supervised`` (oracle: fit to FMM ground truth).

        Scene — start: source joint angles ``(θ1, θ2)`` (rad); num_obstacles / obstacle_radius: count and
            inclusive ``(min, max)`` radius of the circular angle-space obstacles; slowness_max / slow_width:
            peak slowness inside obstacles (metric contrast) and the ramp width (rad).

        Eikonal residual — epsilon: viscosity coefficient of ``−ε ΔT`` (0 disables it); cut_band: cut-locus
            exclusion band used only when ``epsilon>0``; num_collocation / source_radius: number of PDE
            collocation points and the exclusion radius around the source.

        Causal weighting — causal: enable near-to-far residual weighting; causal_strength: ordering strength;
            causal_anneal: relax the causal weight to zero by the end.

        Roadmap / factored field — roadmap_gamma: soft-min temperature of the RRT* base; roadmap_nodes:
            base node budget; roadmap_hop: samples along each last hop for its slowness weighting; base_reg:
            pull of ``T=base·exp(g)`` toward the base (small ⇒ physics leads); tau_bias / tau_min: bias and
            floor of ``τ=τ_min+(1−τ_min)·σ(splat+bias)`` in ``T=base/τ``; lambda_init / anneal_frac:
            progressive obstacle-contrast schedule ``s_λ=1+λ(s−1)``.

        Adaptive sampling — resample_every / pool_size: resample cadence and candidate pool; rad_floor:
            floor ``c`` in RAD density ``|R|/mean|R|+c``; band_boost / band_width: obstacle-rim sampling
            boost; curriculum_frac / curriculum_width / radius_min / radius_max: source-outward frontier.

        Weak supervision (RRT* anchors) — weak_weight: anchor-term weight (0 disables); num_weak: anchor
            count; shadow_pref: bias toward occluded (shadow) nodes; weak_clearance: min obstacle clearance;
            rrt_iters / rrt_step / rrt_radius: RRT* tree size, step, and rewiring radius.

        Training / output — num_splats, steps, lr, init_scale, resolution (eval + FMM grid), seed,
            error_clip (error colour limit), checkpoint_every, out_dir.
    """

    method: Literal["eikonal", "ntfields", "roadmap", "supervised"] = "eikonal"
    # scene
    start: tuple[float, float] = (-1.5, -1.5)
    num_obstacles: int = 3
    obstacle_radius: tuple[float, float] = (0.5, 0.9)
    slowness_max: float = 10.0
    slow_width: float = 0.15
    # Eikonal residual
    epsilon: float = 0.0
    cut_band: float = 0.35
    num_collocation: int = 2048
    source_radius: float = 0.25
    # causal weighting
    causal: bool = True
    causal_strength: float = 5.0
    causal_anneal: bool = True
    # roadmap / factored field
    roadmap_gamma: float = 0.05
    roadmap_nodes: int = 300
    roadmap_hop: int = 5
    base_reg: float = 1.0
    tau_bias: float = 4.0
    tau_min: float = 0.25
    lambda_init: float = 0.15
    anneal_frac: float = 0.4
    # adaptive sampling
    resample_every: int = 300
    pool_size: int = 6000
    rad_floor: float = 1.0
    band_boost: float = 2.0
    band_width: float = 0.2
    curriculum_frac: float = 0.5
    curriculum_width: float = 0.5
    radius_min: float = 0.8
    radius_max: float = 4.5
    # weak supervision (RRT* anchors)
    weak_weight: float = 0.0
    num_weak: int = 30
    shadow_pref: float = 5.0
    weak_clearance: float = 0.4
    rrt_iters: int = 1200
    rrt_step: float = 0.5
    rrt_radius: float = 0.9
    # training / output
    num_splats: int = 384
    steps: int = 4000
    lr: float = 3e-3
    init_scale: float = 0.35
    resolution: int = 120
    seed: int = 1
    error_clip: float = 0.2
    checkpoint_every: int = 1500
    out_dir: str = "figures"


def wrap(angle: jnp.ndarray) -> jnp.ndarray:
    """Wrap angles to ``[−π, π)``."""
    return (angle + jnp.pi) % (2 * jnp.pi) - jnp.pi


def _wrap_np(angle: np.ndarray) -> np.ndarray:
    """NumPy angle wrap to ``[−π, π)``."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def eval_splat_torus(thetas: jnp.ndarray, splat: tuple) -> jnp.ndarray:
    """Evaluate a periodic splat field intrinsically: Gaussians of the WRAPPED displacement."""
    weights, scales, centres = splat
    displacement = wrap(thetas[:, None, :] - centres[None, :, :])
    solved = jnp.linalg.solve(scales[None], displacement[..., None]).squeeze(-1)
    dim = thetas.shape[1]
    density = jnp.exp(-0.5 * jnp.sum(solved**2, axis=-1)) / jnp.power(2 * jnp.pi, dim / 2.0)
    return (density / jnp.linalg.det(scales)) @ weights


def geodesic(theta: jnp.ndarray, start: jnp.ndarray) -> jnp.ndarray:
    """Analytic flat-torus geodesic distance ``‖wrap(θ − start)‖`` (the known base / free GT)."""
    return jnp.linalg.norm(wrap(theta - start), axis=-1)


def metric_inv(theta: jnp.ndarray) -> jnp.ndarray:
    """Inverse metric ``gⁱʲ`` at ``θ``; identity for the flat torus (M6 swaps in ``M(θ)⁻¹``)."""
    return jnp.eye(2)


def torus_obstacles(cfg: Config) -> tuple[Obstacle, ...]:
    """Generate reproducible circular obstacles in angle space, clear of the source."""
    rng = np.random.default_rng(cfg.seed + 5)
    start = np.asarray(cfg.start)
    obstacles: list[Obstacle] = []
    while len(obstacles) < cfg.num_obstacles:
        centre = rng.uniform(-np.pi, np.pi, size=2)
        radius = float(rng.uniform(*cfg.obstacle_radius))
        if np.linalg.norm(_wrap_np(centre - start)) > radius + 0.3:
            obstacles.append((float(centre[0]), float(centre[1]), radius))
    return tuple(obstacles)


def torus_sdf(thetas: jnp.ndarray, obstacles: tuple[Obstacle, ...]) -> jnp.ndarray:
    """Signed distance (wrapped) to the union of obstacle circles on the torus."""
    per = [jnp.linalg.norm(wrap(thetas - jnp.array([c1, c2])), axis=-1) - r for c1, c2, r in obstacles]
    return jnp.min(jnp.stack(per, axis=0), axis=0)


def torus_slowness(
    thetas: jnp.ndarray, obstacles: tuple[Obstacle, ...], slowness_max: float, width: float
) -> jnp.ndarray:
    """Smooth slowness: ~1 in free space, rising to ``slowness_max`` inside obstacles."""
    return 1.0 + (slowness_max - 1.0) * jax.nn.sigmoid(-torus_sdf(thetas, obstacles) / width)


def field_value(splat: tuple, theta: jnp.ndarray, start: jnp.ndarray) -> jnp.ndarray:
    """Factored field ``T(θ) = geodesic(θ, start) · (1 + g(θ))`` — analytic base times a periodic splat."""
    base = jnp.linalg.norm(wrap(theta - start))
    correction = eval_splat_torus(theta[None, :], splat)[0, 0]
    return base * (1.0 + correction)


def eikonal_residual(
    splat: tuple, thetas: jnp.ndarray, slow: jnp.ndarray, epsilon: float, start: jnp.ndarray
) -> jnp.ndarray:
    """Return the relative metric-Eikonal residual ``(gⁱʲ ∂ᵢT ∂ⱼT − ε ΔT)/s² − 1`` at each θ."""
    grad = jax.vmap(lambda t: jax.grad(field_value, argnums=1)(splat, t, start))(thetas)
    quad = jax.vmap(lambda g, t: g @ metric_inv(t) @ g)(grad, thetas)
    if epsilon == 0.0:
        return quad / slow**2 - 1.0
    laplacian = jax.vmap(lambda t: jnp.trace(jax.hessian(field_value, argnums=1)(splat, t, start)))(thetas)
    return (quad - epsilon * laplacian) / slow**2 - 1.0


def predict(splat: tuple, thetas: jnp.ndarray, start: jnp.ndarray) -> jnp.ndarray:
    """Evaluate ``T`` over a batch of torus angles, returned as ``[N]``."""
    return jax.vmap(lambda t: field_value(splat, t, start))(thetas)


def init_splat(num_splats: int, init_scale: float, key: jax.Array) -> tuple:
    """Initialise ``(V, A, B)`` with zero weights and centres in angle space (``d = 2``)."""
    centres = jax.random.uniform(key, (num_splats, 2), minval=-jnp.pi, maxval=jnp.pi)
    scales = jnp.repeat((init_scale * jnp.eye(2))[None], num_splats, axis=0)
    return jnp.zeros((num_splats, 1)), scales, centres


def fast_marching_torus(speed: np.ndarray, start: tuple[float, float], n: int) -> np.ndarray:
    """Periodic fast marching of ``|∇T| = 1/speed`` on the flat-torus grid (wrap-around neighbours)."""
    axis = np.linspace(-np.pi, np.pi, n, endpoint=False)
    step = 2 * np.pi / n
    grid1, grid2 = np.meshgrid(axis, axis)
    slowness = 1.0 / speed
    time = np.full((n, n), np.inf)
    accepted = np.zeros((n, n), dtype=bool)
    heap: list[tuple[float, int, int]] = []
    seed = np.sqrt(_wrap_np(grid1 - start[0]) ** 2 + _wrap_np(grid2 - start[1]) ** 2)
    for j, i in zip(*np.where(seed <= step)):
        time[j, i] = float(seed[j, i])
        heapq.heappush(heap, (float(time[j, i]), int(j), int(i)))

    while heap:
        _, j, i = heapq.heappop(heap)
        if accepted[j, i]:
            continue
        accepted[j, i] = True
        for dj, di in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nj, ni = (j + dj) % n, (i + di) % n
            if accepted[nj, ni]:
                continue
            left, right = time[nj, (ni - 1) % n], time[nj, (ni + 1) % n]
            down, up = time[(nj - 1) % n, ni], time[(nj + 1) % n, ni]
            along_x = min(
                left if accepted[nj, (ni - 1) % n] else np.inf, right if accepted[nj, (ni + 1) % n] else np.inf
            )
            along_y = min(down if accepted[(nj - 1) % n, ni] else np.inf, up if accepted[(nj + 1) % n, ni] else np.inf)
            cost = slowness[nj, ni] * step
            if np.isinf(along_x) and np.isinf(along_y):
                continue
            if np.isinf(along_x):
                candidate = along_y + cost
            elif np.isinf(along_y):
                candidate = along_x + cost
            else:
                lo, hi = min(along_x, along_y), max(along_x, along_y)
                candidate = (
                    lo + cost if hi - lo >= cost else 0.5 * (lo + hi + np.sqrt(max(0.0, 2 * cost**2 - (hi - lo) ** 2)))
                )
            if candidate < time[nj, ni]:
                time[nj, ni] = candidate
                heapq.heappush(heap, (float(candidate), int(nj), int(ni)))
    return time


def _slow_np(points: np.ndarray, obstacles: tuple[Obstacle, ...], slowness_max: float, width: float) -> np.ndarray:
    """NumPy smooth slowness (for the host-side RRT*): 1 in free space, up to ``slowness_max`` in obstacles."""
    sdf = np.min(
        [np.linalg.norm(_wrap_np(points - np.array([c1, c2])), axis=-1) - r for c1, c2, r in obstacles], axis=0
    )
    return 1.0 + (slowness_max - 1.0) / (1.0 + np.exp(sdf / width))


def rrt_star(cfg: Config, obstacles: tuple[Obstacle, ...]) -> tuple[np.ndarray, np.ndarray]:
    """RRT* on the torus with slowness-weighted edges: returns tree nodes and their cost-to-come from the source.

    Mesh-free and dimension-scalable — the intended high-dimensional anchor source (unlike grid FMM).
    Edge cost = wrapped length × mean slowness along the edge, matching the soft-obstacle field.
    """
    rng = np.random.default_rng(cfg.seed + 11)
    start = np.asarray(cfg.start, dtype=float)

    def edge_cost(a: np.ndarray, b: np.ndarray) -> float:
        ts = np.linspace(0.0, 1.0, 6)[:, None]
        pts = _wrap_np(a[None, :] + ts * _wrap_np(b - a)[None, :])
        return float(
            np.linalg.norm(_wrap_np(b - a)) * _slow_np(pts, obstacles, cfg.slowness_max, cfg.slow_width).mean()
        )

    nodes = [start]
    costs = [0.0]
    node_arr = np.array(nodes)
    for _ in range(cfg.rrt_iters):
        q_rand = rng.uniform(-np.pi, np.pi, size=2)
        distances = np.linalg.norm(_wrap_np(node_arr - q_rand), axis=1)
        near = int(distances.argmin())
        direction = _wrap_np(q_rand - node_arr[near])
        length = np.linalg.norm(direction)
        if length < 1e-6:
            continue
        q_new = _wrap_np(node_arr[near] + min(cfg.rrt_step, length) * direction / length)
        neighbours = np.where(np.linalg.norm(_wrap_np(node_arr - q_new), axis=1) < cfg.rrt_radius)[0]
        best_cost = costs[near] + edge_cost(node_arr[near], q_new)
        for i in neighbours:
            best_cost = min(best_cost, costs[i] + edge_cost(node_arr[i], q_new))
        nodes.append(q_new)
        costs.append(best_cost)
        node_arr = np.array(nodes)
        for i in neighbours:
            costs[i] = min(costs[i], best_cost + edge_cost(q_new, node_arr[i]))
    return node_arr, np.array(costs)


def rrt_star_anchors(cfg: Config, obstacles: tuple[Obstacle, ...]) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return ``num_weak`` sparse ``(config, cost-to-come)`` anchors sampled from an RRT* tree."""
    nodes, costs = rrt_star(cfg, obstacles)
    keep = np.asarray(torus_sdf(jnp.asarray(nodes, dtype=jnp.float32), obstacles)) > cfg.weak_clearance
    nodes, costs = nodes[keep], costs[keep]
    chosen = np.random.default_rng(cfg.seed + 12).choice(len(nodes), size=min(cfg.num_weak, len(nodes)), replace=False)
    return jnp.asarray(nodes[chosen], dtype=jnp.float32), jnp.asarray(costs[chosen], dtype=jnp.float32)


def rrt_star_anchors_shadow(cfg: Config, obstacles: tuple[Obstacle, ...]) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Sparse RRT* ``(config, cost-to-come)`` anchors biased into obstacle *shadows* (occluded from source).

    The pointwise Eikonal loss constrains slope, not level; behind obstacles the level is
    under-determined and drifts. These anchors pin that free integration constant exactly where it is
    free — the region whose straight source→node ray is blocked. Mesh-free (RRT*, not FMM), so it
    scales past the grid dimensions.
    """
    nodes, costs = rrt_star(cfg, obstacles)
    clear = np.asarray(torus_sdf(jnp.asarray(nodes, dtype=jnp.float32), obstacles)) > cfg.weak_clearance
    nodes, costs = nodes[clear], costs[clear]
    start = np.asarray(cfg.start, dtype=float)
    disp = _wrap_np(nodes - start)
    ts = np.linspace(0.1, 0.95, 8)
    ray = start[None, None, :] + ts[None, :, None] * disp[:, None, :]  # [N, 8, 2] straight source→node rays
    ray_sdf = np.asarray(torus_sdf(jnp.asarray(ray.reshape(-1, 2), dtype=jnp.float32), obstacles)).reshape(len(nodes), -1)
    occluded = ray_sdf.min(axis=1) < 0.0  # blocked ray ⇒ node sits in a routing shadow
    weight = 1.0 + cfg.shadow_pref * occluded
    weight = weight / weight.sum()
    k = min(cfg.num_weak, len(nodes))
    chosen = np.random.default_rng(cfg.seed + 12).choice(len(nodes), size=k, replace=False, p=weight)
    print(f"[anchors] {k} RRT* anchors — {int(occluded[chosen].sum())} in shadow", flush=True)
    return jnp.asarray(nodes[chosen], dtype=jnp.float32), jnp.asarray(costs[chosen], dtype=jnp.float32)


def solve(cfg: Config, obstacles: tuple[Obstacle, ...], checkpoint=None) -> tuple:
    """Fit the splat correction with RAD residual-importance + obstacle-band + source-outward sampling.

    Sampling density (resampled every ``resample_every`` steps): a RAD term ``|R|/mean|R| + c``
    (Wu et al. 2023) that auto-targets high-residual regions (obstacle rims, cut locus), times an
    obstacle-boundary band, times a source-outward curriculum frontier that expands over training.
    On top, causal residual weighting fits nearer-to-source points first. No weak supervision.
    """
    splat = init_splat(cfg.num_splats, cfg.init_scale, jax.random.PRNGKey(cfg.seed))
    start = jnp.asarray(cfg.start)
    start_np = np.asarray(cfg.start)
    rng = np.random.default_rng(cfg.seed)
    causal_rate = cfg.causal_strength / cfg.num_collocation
    weak_x, weak_t = rrt_star_anchors(cfg, obstacles) if cfg.weak_weight > 0 else (jnp.zeros((0, 2)), jnp.zeros((0,)))

    optimizer = optax.adam(cfg.lr)
    opt_state = optimizer.init(splat)

    def loss_fn(params, colloc, slow, order, rate):
        squared = eikonal_residual(params, colloc, slow, cfg.epsilon, start) ** 2
        if cfg.causal:
            ordered = squared[order]
            upstream = jnp.cumsum(ordered) - ordered
            weight = jax.lax.stop_gradient(jnp.exp(-rate * upstream))
            pde = jnp.mean(weight * ordered)
        else:
            pde = jnp.mean(squared)
        if weak_x.shape[0] == 0:
            return pde
        predicted = jax.vmap(lambda t: field_value(params, t, start))(weak_x)
        return pde + cfg.weak_weight * jnp.mean((predicted - weak_t) ** 2)

    @jax.jit
    def step(params, state, colloc, slow, order, rate):
        loss, grads = jax.value_and_grad(loss_fn)(params, colloc, slow, order, rate)
        updates, state = optimizer.update(grads, state, params)
        return optax.apply_updates(params, updates), state, loss

    @jax.jit
    def residual_mag(params, pts, slow):
        return jnp.abs(eikonal_residual(params, pts, slow, cfg.epsilon, start))

    def resample(params, radius):
        pool = rng.uniform(-np.pi, np.pi, size=(cfg.pool_size, 2))
        wrapped = _wrap_np(pool - start_np)
        keep = np.linalg.norm(wrapped, axis=1) > cfg.source_radius
        if cfg.epsilon > 0.0:
            keep &= np.abs(wrapped).max(axis=1) < np.pi - cfg.cut_band
        pool = pool[keep]
        pool_j = jnp.asarray(pool, dtype=jnp.float32)
        slow_pool = torus_slowness(pool_j, obstacles, cfg.slowness_max, cfg.slow_width)
        residual = np.asarray(residual_mag(params, pool_j, slow_pool))
        pool_max = float(np.abs(np.asarray(jax.vmap(lambda t: field_value(params, t, start))(pool_j))).max())
        print(f"[resample r={radius:.2f}] maxT={pool_max:.2f} (GT max ~5.6)  mean|R|={residual.mean():.3f}", flush=True)
        base = np.asarray(geodesic(pool_j, start))
        sdf = np.asarray(torus_sdf(pool_j, obstacles))
        w_rad = residual / (residual.mean() + 1e-9) + cfg.rad_floor
        w_curr = 1.0 / (1.0 + np.exp((base - radius) / cfg.curriculum_width))
        w_band = 1.0 + cfg.band_boost * np.exp(-((sdf / cfg.band_width) ** 2))
        density = w_rad * w_curr * w_band
        density /= density.sum()
        idx = rng.choice(len(pool), size=min(cfg.num_collocation, len(pool)), replace=False, p=density)
        colloc = jnp.asarray(pool[idx], dtype=jnp.float32)
        return (
            colloc,
            torus_slowness(colloc, obstacles, cfg.slowness_max, cfg.slow_width),
            jnp.argsort(geodesic(colloc, start)),
        )

    collocation, slow, order = resample(splat, cfg.radius_min)
    progress = trange(cfg.steps, desc="torus eikonal")
    for i in progress:
        if i % cfg.resample_every == 0:
            radius = cfg.radius_min + (cfg.radius_max - cfg.radius_min) * min(
                1.0, i / (cfg.curriculum_frac * cfg.steps + 1e-9)
            )
            collocation, slow, order = resample(splat, radius)
        rate = causal_rate * (1.0 - i / cfg.steps) if cfg.causal_anneal else causal_rate
        splat, opt_state, loss = step(splat, opt_state, collocation, slow, order, jnp.float32(rate))
        if i % 25 == 0:
            progress.set_description(f"torus eikonal — log10(loss) = {float(jnp.log10(loss)):.3f}")
        if checkpoint is not None and i > 0 and i % cfg.checkpoint_every == 0:
            checkpoint(splat, i)
    return splat


def field_ntfields(
    splat: tuple, theta: jnp.ndarray, start: jnp.ndarray, tau_bias: float, tau_min: float
) -> jnp.ndarray:
    """NTFields-style factored field ``T(θ) = base(θ) / τ(θ)`` with ``τ = τ_min + (1−τ_min)·σ(splat+bias)``.

    ``τ ∈ [τ_min, 1]`` so ``base ≤ T ≤ base/τ_min``: the free-space geodesic is a hard lower bound
    (the under-scaling failure of ``base·(1+g)`` is unrepresentable) *and* the floor ``τ_min`` bounds
    ``T`` from above, preventing the ``τ→0 ⇒ T→∞`` runaway seen at full obstacle contrast. ``bias``
    starts ``τ`` near 1 (free space) at init; the splat lowers ``τ`` in the shadow. ``T(start)=0``.
    """
    base = jnp.linalg.norm(wrap(theta - start))
    raw = eval_splat_torus(theta[None, :], splat)[0, 0]
    return base / (tau_min + (1.0 - tau_min) * jax.nn.sigmoid(raw + tau_bias))


def ntfields_residual(
    splat: tuple, thetas: jnp.ndarray, slow: jnp.ndarray, start: jnp.ndarray, tau_bias: float, tau_min: float
) -> jnp.ndarray:
    """Per-point symmetric speed-match penalty ``q + 1/q − 2`` with ``q = ‖∇T‖_{g⁻¹} / s``.

    This matches the predicted speed ``Ŝ = 1/‖∇T‖`` to the known speed ``S = 1/s`` in a bounded,
    symmetric, √-smoothed form (NTFields): ``Ŝ/S + S/Ŝ − 2 ≥ 0``, zero iff ``Ŝ = S``. Unlike the
    slowness residual ``(‖∇T‖²/s² − 1)²`` it is not obstacle-dominated, so the level-setting free
    space is not starved. ``metric_inv`` carries the (an)isotropy of the metric.
    """
    grad = jax.vmap(lambda t: jax.grad(field_ntfields, argnums=1)(splat, t, start, tau_bias, tau_min))(thetas)
    quad = jax.vmap(lambda g, t: g @ metric_inv(t) @ g)(grad, thetas)
    q = jnp.sqrt(jnp.clip(quad, 1e-12, None)) / slow
    return q + 1.0 / q - 2.0


def predict_ntfields(
    splat: tuple, thetas: jnp.ndarray, start: jnp.ndarray, tau_bias: float, tau_min: float
) -> jnp.ndarray:
    """Evaluate the NTFields-style field over a batch of torus angles, returned as ``[N]``."""
    return jax.vmap(lambda t: field_ntfields(splat, t, start, tau_bias, tau_min))(thetas)


def solve_ntfields(cfg: Config, obstacles: tuple[Obstacle, ...], checkpoint=None) -> tuple:
    """Solve the torus Eikonal NTFields-style: bounded factorisation ``T=base/τ`` + symmetric speed
    loss + progressive obstacle annealing (``λ: λ₀→1``), no causal/DP/visibility crutches.

    The only training aids are RAD residual-importance resampling (obstacle rims, cut locus) and the
    source-outward curriculum frontier — everything that made the level converge is the formulation
    itself, for a clean comparison against ``solve`` (``base·(1+g)`` + slowness residual²).
    """
    splat = init_splat(cfg.num_splats, cfg.init_scale, jax.random.PRNGKey(cfg.seed))
    start = jnp.asarray(cfg.start)
    start_np = np.asarray(cfg.start)
    rng = np.random.default_rng(cfg.seed)
    weak_x, weak_t = (
        rrt_star_anchors_shadow(cfg, obstacles) if cfg.weak_weight > 0 else (jnp.zeros((0, 2)), jnp.zeros((0,)))
    )

    optimizer = optax.adam(cfg.lr)
    opt_state = optimizer.init(splat)

    causal_rate = cfg.causal_strength / cfg.num_collocation

    def loss_fn(params, colloc, slow_full, order, rate, lam):
        slow = 1.0 + lam * (slow_full - 1.0)  # anneal obstacle contrast: free space (λ→0) → full (λ→1)
        gate = jax.nn.sigmoid(torus_sdf(colloc, obstacles) / cfg.band_width)  # ~0 inside obstacles, ~1 outside
        residual = ntfields_residual(params, colloc, slow, start, cfg.tau_bias, cfg.tau_min)
        if not cfg.causal:
            pde = jnp.sum(gate * residual) / (jnp.sum(gate) + 1e-9)
        else:
            res_o, gate_o = residual[order], gate[order]  # sorted source-outward; upstream = nearer-to-source mass
            upstream = jnp.cumsum(gate_o * res_o) - gate_o * res_o
            weight = jax.lax.stop_gradient(jnp.exp(-rate * upstream)) * gate_o
            pde = jnp.sum(weight * res_o) / (jnp.sum(weight) + 1e-9)
        if weak_x.shape[0] == 0:
            return pde
        pred = jax.vmap(lambda t: field_ntfields(params, t, start, cfg.tau_bias, cfg.tau_min))(weak_x)
        return pde + cfg.weak_weight * jnp.mean((pred - weak_t) ** 2)  # sparse shadow anchors pin the level

    @jax.jit
    def step(params, state, colloc, slow_full, order, rate, lam):
        loss, grads = jax.value_and_grad(loss_fn)(params, colloc, slow_full, order, rate, lam)
        updates, state = optimizer.update(grads, state, params)
        return optax.apply_updates(params, updates), state, loss

    @jax.jit
    def residual_mag(params, pts, slow):
        return ntfields_residual(params, pts, slow, start, cfg.tau_bias, cfg.tau_min)

    def resample(params, radius, lam):
        pool = rng.uniform(-np.pi, np.pi, size=(cfg.pool_size, 2))
        wrapped = _wrap_np(pool - start_np)
        keep = np.linalg.norm(wrapped, axis=1) > cfg.source_radius
        pool, wrapped = pool[keep], wrapped[keep]
        pool_j = jnp.asarray(pool, dtype=jnp.float32)
        slow_full = torus_slowness(pool_j, obstacles, cfg.slowness_max, cfg.slow_width)
        slow_lam = 1.0 + lam * (slow_full - 1.0)
        residual = np.asarray(residual_mag(params, pool_j, slow_lam))
        pool_max = float(np.asarray(predict_ntfields(params, pool_j, start, cfg.tau_bias, cfg.tau_min)).max())
        print(f"[resample r={radius:.2f} λ={lam:.2f}] maxT={pool_max:.2f} (GT max ~5.6)  mean loss={residual.mean():.3f}", flush=True)
        base = np.asarray(geodesic(pool_j, start))
        sdf = np.asarray(torus_sdf(pool_j, obstacles))
        w_rad = residual / (residual.mean() + 1e-9) + cfg.rad_floor
        w_curr = 1.0 / (1.0 + np.exp((base - radius) / cfg.curriculum_width))
        w_band = 1.0 + cfg.band_boost * np.exp(-((sdf / cfg.band_width) ** 2))
        density = w_rad * w_curr * w_band
        density /= density.sum()
        idx = rng.choice(len(pool), size=min(cfg.num_collocation, len(pool)), replace=False, p=density)
        colloc = jnp.asarray(pool[idx], dtype=jnp.float32)
        return (
            colloc,
            torus_slowness(colloc, obstacles, cfg.slowness_max, cfg.slow_width),
            jnp.argsort(geodesic(colloc, start)),
        )

    lam0 = cfg.lambda_init
    collocation, slow_full, order = resample(splat, cfg.radius_min, lam0)
    progress = trange(cfg.steps, desc="torus ntfields")
    for i in progress:
        lam = min(1.0, lam0 + (1.0 - lam0) * i / (cfg.anneal_frac * cfg.steps + 1e-9))
        if i % cfg.resample_every == 0:
            radius = cfg.radius_min + (cfg.radius_max - cfg.radius_min) * min(
                1.0, i / (cfg.curriculum_frac * cfg.steps + 1e-9)
            )
            collocation, slow_full, order = resample(splat, radius, lam)
        rate = causal_rate * (1.0 - i / cfg.steps) if cfg.causal_anneal else causal_rate
        splat, opt_state, loss = step(splat, opt_state, collocation, slow_full, order, jnp.float32(rate), jnp.float32(lam))
        if i % 25 == 0:
            progress.set_description(f"torus ntfields — log10(loss) = {float(jnp.log10(loss + 1e-12)):.3f}")
        if checkpoint is not None and i > 0 and i % cfg.checkpoint_every == 0:
            checkpoint(splat, i)
    return splat


def build_roadmap(cfg: Config, obstacles: tuple[Obstacle, ...]) -> tuple[jnp.ndarray, jnp.ndarray]:
    """RRT* tree subsampled to a fixed node budget — the coarse, obstacle-aware cost-to-come prior.

    Deterministic in ``cfg.seed`` so ``solve`` and rendering rebuild the identical roadmap. Mesh-free
    (RRT*, not FMM) so the base scales past the grid dimensions; the source (cost 0) is always kept.
    """
    nodes, costs = rrt_star(cfg, obstacles)
    if len(nodes) > cfg.roadmap_nodes:
        idx = np.random.default_rng(cfg.seed + 13).choice(len(nodes) - 1, cfg.roadmap_nodes - 1, replace=False) + 1
        idx = np.concatenate([[0], idx])  # keep the source node (cost 0) so T(start)=0
        nodes, costs = nodes[idx], costs[idx]
    return jnp.asarray(nodes, dtype=jnp.float32), jnp.asarray(costs, dtype=jnp.float32)


def roadmap_base(
    theta: jnp.ndarray, nodes: jnp.ndarray, costs: jnp.ndarray, gamma: float,
    obstacles: tuple[Obstacle, ...], smax: float, width: float, ksamp: int,
) -> jnp.ndarray:
    """Obstacle-aware cost-to-come ``soft-min_i (cost_i + slowness-weighted ‖wrap(θ−node_i)‖)``.

    The last hop from the best tree node to θ is weighted by the *mean slowness along it* (``ksamp``
    samples), so hops that cut through an obstacle are penalised — the fix that took the raw base from
    0.52 to 0.065 RMS (straight hops cut corners and underestimated the shadow). Fully differentiable
    in θ and mesh-free (RRT* tree), so it scales past the grid dimensions.
    """
    disp = wrap(theta[None, :] - nodes)  # [N, 2]
    ts = jnp.linspace(0.0, 1.0, ksamp)  # [K]
    seg = nodes[None, :, :] + ts[:, None, None] * disp[None, :, :]  # [K, N, 2] points along each last hop
    mean_slow = torus_slowness(wrap(seg.reshape(-1, 2)), obstacles, smax, width).reshape(ksamp, -1).mean(0)  # [N]
    wlen = jnp.linalg.norm(disp, axis=1) * mean_slow  # slowness-weighted hop length
    return -gamma * jax.scipy.special.logsumexp(-(costs + wlen) / gamma)  # differentiable soft-min


def field_roadmap(
    splat: tuple, theta: jnp.ndarray, nodes: jnp.ndarray, costs: jnp.ndarray, gamma: float,
    obstacles: tuple[Obstacle, ...], smax: float, width: float, ksamp: int,
) -> jnp.ndarray:
    """``T(θ) = roadmap_base(θ) · exp(splat(θ))`` — accurate obstacle-aware base, small splat correction.

    The base already routes (RMS ~0.065), so the splat only smooths/refines locally (``exp(0)=1`` at
    init ⇒ ``T≈base``); it never has to do the non-local shadow work that broke the straight-line prior.
    ``T(start)=0`` since the source node contributes a zero term.
    """
    base = roadmap_base(theta, nodes, costs, gamma, obstacles, smax, width, ksamp)
    return base * jnp.exp(eval_splat_torus(theta[None, :], splat)[0, 0])


def roadmap_residual(
    splat: tuple, thetas: jnp.ndarray, slow: jnp.ndarray, nodes: jnp.ndarray, costs: jnp.ndarray, gamma: float,
    obstacles: tuple[Obstacle, ...], smax: float, width: float, ksamp: int,
) -> jnp.ndarray:
    """Symmetric speed-match penalty ``q + 1/q − 2`` (``q = ‖∇T‖_{g⁻¹}/s``) for the roadmap-based field."""
    grad = jax.vmap(lambda t: jax.grad(field_roadmap, argnums=1)(splat, t, nodes, costs, gamma, obstacles, smax, width, ksamp))(thetas)
    quad = jax.vmap(lambda g, t: g @ metric_inv(t) @ g)(grad, thetas)
    q = jnp.sqrt(jnp.clip(quad, 1e-12, None)) / slow
    return q + 1.0 / q - 2.0


def predict_roadmap(
    splat: tuple, thetas: jnp.ndarray, nodes: jnp.ndarray, costs: jnp.ndarray, gamma: float,
    obstacles: tuple[Obstacle, ...], smax: float, width: float, ksamp: int,
) -> jnp.ndarray:
    """Evaluate the roadmap-based field over a batch of torus angles, returned as ``[N]``."""
    return jax.vmap(lambda t: field_roadmap(splat, t, nodes, costs, gamma, obstacles, smax, width, ksamp))(thetas)


def solve_roadmap(cfg: Config, obstacles: tuple[Obstacle, ...], checkpoint=None) -> tuple:
    """Solve with a coarse RRT*-roadmap base + small splat correction; causal + RAD + progressive, no anchors.

    The obstacle-aware base carries the routing, so the splat does only a small *local* correction — the
    dimension-robust alternative to making the splat (or ever-more anchors) route around obstacles.
    """
    splat = init_splat(cfg.num_splats, cfg.init_scale, jax.random.PRNGKey(cfg.seed))
    nodes, costs = build_roadmap(cfg, obstacles)
    start_np = np.asarray(cfg.start)
    start = jnp.asarray(cfg.start)
    rng = np.random.default_rng(cfg.seed)
    optimizer = optax.adam(cfg.lr)
    opt_state = optimizer.init(splat)
    causal_rate = cfg.causal_strength / cfg.num_collocation

    rm_args = (nodes, costs, cfg.roadmap_gamma, obstacles, cfg.slowness_max, cfg.slow_width, cfg.roadmap_hop)
    weak_x, weak_t = (  # same RRT* samples reused as anchors (base gives shape, anchors pin the level)
        rrt_star_anchors_shadow(cfg, obstacles) if cfg.weak_weight > 0 else (jnp.zeros((0, 2)), jnp.zeros((0,)))
    )

    def loss_fn(params, colloc, slow_full, order, rate, lam):
        slow = 1.0 + lam * (slow_full - 1.0)
        gate = jax.nn.sigmoid(torus_sdf(colloc, obstacles) / cfg.band_width)
        residual = roadmap_residual(params, colloc, slow, *rm_args)
        if not cfg.causal:
            pde = jnp.sum(gate * residual) / (jnp.sum(gate) + 1e-9)
        else:
            res_o, gate_o = residual[order], gate[order]
            upstream = jnp.cumsum(gate_o * res_o) - gate_o * res_o
            weight = jax.lax.stop_gradient(jnp.exp(-rate * upstream)) * gate_o
            pde = jnp.sum(weight * res_o) / (jnp.sum(weight) + 1e-9)
        correction = eval_splat_torus(colloc, params).ravel()  # log-correction g; T = base·exp(g)
        loss = pde + cfg.base_reg * jnp.mean(correction**2)  # keep T≈accurate base; splat only smooths kinks
        if weak_x.shape[0] > 0:
            pred = jax.vmap(lambda t: field_roadmap(params, t, *rm_args))(weak_x)
            loss = loss + cfg.weak_weight * jnp.mean((pred - weak_t) ** 2)  # anchors pin the near-optimal level
        return loss

    @jax.jit
    def step(params, state, colloc, slow_full, order, rate, lam):
        loss, grads = jax.value_and_grad(loss_fn)(params, colloc, slow_full, order, rate, lam)
        updates, state = optimizer.update(grads, state, params)
        return optax.apply_updates(params, updates), state, loss

    @jax.jit
    def residual_mag(params, pts, slow):
        return roadmap_residual(params, pts, slow, *rm_args)

    def resample(params, radius, lam):
        pool = rng.uniform(-np.pi, np.pi, size=(cfg.pool_size, 2))
        wrapped = _wrap_np(pool - start_np)
        keep = np.linalg.norm(wrapped, axis=1) > cfg.source_radius
        pool, wrapped = pool[keep], wrapped[keep]
        pool_j = jnp.asarray(pool, dtype=jnp.float32)
        slow_full = torus_slowness(pool_j, obstacles, cfg.slowness_max, cfg.slow_width)
        slow_lam = 1.0 + lam * (slow_full - 1.0)
        residual = np.asarray(residual_mag(params, pool_j, slow_lam))
        pool_max = float(np.asarray(predict_roadmap(params, pool_j, *rm_args)).max())
        print(f"[resample r={radius:.2f} λ={lam:.2f}] maxT={pool_max:.2f} (GT max ~5.6)  mean loss={residual.mean():.3f}", flush=True)
        base = np.asarray(geodesic(pool_j, start))
        sdf = np.asarray(torus_sdf(pool_j, obstacles))
        w_rad = residual / (residual.mean() + 1e-9) + cfg.rad_floor
        w_curr = 1.0 / (1.0 + np.exp((base - radius) / cfg.curriculum_width))
        w_band = 1.0 + cfg.band_boost * np.exp(-((sdf / cfg.band_width) ** 2))
        density = w_rad * w_curr * w_band
        density /= density.sum()
        idx = rng.choice(len(pool), size=min(cfg.num_collocation, len(pool)), replace=False, p=density)
        colloc = jnp.asarray(pool[idx], dtype=jnp.float32)
        return colloc, torus_slowness(colloc, obstacles, cfg.slowness_max, cfg.slow_width), jnp.argsort(geodesic(colloc, start))

    lam0 = cfg.lambda_init
    collocation, slow_full, order = resample(splat, cfg.radius_min, lam0)
    progress = trange(cfg.steps, desc="torus roadmap")
    for i in progress:
        lam = min(1.0, lam0 + (1.0 - lam0) * i / (cfg.anneal_frac * cfg.steps + 1e-9))
        if i % cfg.resample_every == 0:
            radius = cfg.radius_min + (cfg.radius_max - cfg.radius_min) * min(1.0, i / (cfg.curriculum_frac * cfg.steps + 1e-9))
            collocation, slow_full, order = resample(splat, radius, lam)
        rate = causal_rate * (1.0 - i / cfg.steps) if cfg.causal_anneal else causal_rate
        splat, opt_state, loss = step(splat, opt_state, collocation, slow_full, order, jnp.float32(rate), jnp.float32(lam))
        if i % 25 == 0:
            progress.set_description(f"torus roadmap — log10(loss) = {float(jnp.log10(loss + 1e-12)):.3f}")
        if checkpoint is not None and i > 0 and i % cfg.checkpoint_every == 0:
            checkpoint(splat, i)
    return splat


def solve_supervised(cfg: Config, obstacles: tuple[Obstacle, ...], checkpoint=None) -> tuple:
    """Oracle baseline: regression-fit the ``T=base/τ`` splat directly to the FMM ground truth.

    Not a solver — it uses the true field as dense supervision, so its error is the *representation
    ceiling*: the best any splat of this budget can do. It separates "the splat can't represent the
    field" from "the solver can't find it". Uses FMM, so 2D-only (a yardstick, not a method).
    """
    splat = init_splat(cfg.num_splats, cfg.init_scale, jax.random.PRNGKey(cfg.seed))
    start = jnp.asarray(cfg.start)
    axis = np.linspace(-np.pi, np.pi, cfg.resolution, endpoint=False)
    grid1, grid2 = np.meshgrid(axis, axis)
    thetas_np = np.stack([grid1.ravel(), grid2.ravel()], axis=-1)
    slow = np.asarray(torus_slowness(jnp.asarray(thetas_np, jnp.float32), obstacles, cfg.slowness_max, cfg.slow_width))
    gt = fast_marching_torus(1.0 / slow.reshape(cfg.resolution, cfg.resolution), cfg.start, cfg.resolution).ravel()
    free = np.where(np.asarray(torus_sdf(jnp.asarray(thetas_np, jnp.float32), obstacles)) >= 0.0)[0]
    train_x = jnp.asarray(thetas_np[free], dtype=jnp.float32)
    train_y = jnp.asarray(gt[free], dtype=jnp.float32)

    optimizer = optax.adam(cfg.lr)
    opt_state = optimizer.init(splat)

    def loss_fn(params):
        return jnp.mean((predict_ntfields(params, train_x, start, cfg.tau_bias, cfg.tau_min) - train_y) ** 2)

    @jax.jit
    def step(params, state):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, state = optimizer.update(grads, state, params)
        return optax.apply_updates(params, updates), state, loss

    progress = trange(cfg.steps, desc="torus supervised")
    for i in progress:
        splat, opt_state, loss = step(splat, opt_state)
        if i % 25 == 0:
            progress.set_description(f"torus supervised — log10(MSE) = {float(jnp.log10(loss + 1e-12)):.3f}")
        if checkpoint is not None and i > 0 and i % cfg.checkpoint_every == 0:
            checkpoint(splat, i)
    return splat


def render(
    cfg: Config,
    gt: np.ndarray,
    prediction: np.ndarray,
    inside: np.ndarray,
    shape: tuple[int, int],
    out_name: str = "torus_obstacles.png",
) -> dict:
    """Save a ``[GT | prediction | error]`` figure on the ``(θ1, θ2)`` torus and return metrics."""
    extent = (-180.0, 180.0, -180.0, 180.0)
    marker = (np.degrees(cfg.start[0]), np.degrees(cfg.start[1]))
    gt_img = np.where(inside, np.nan, gt).reshape(shape)
    pred_img = np.where(inside, np.nan, prediction).reshape(shape)
    error_img = pred_img - gt_img
    vmax = float(np.nanmax(gt_img))
    mesh1, mesh2 = np.meshgrid(np.linspace(-180, 180, shape[1]), np.linspace(-180, 180, shape[0]))
    blocked = inside.reshape(shape).astype(float)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    panels = [
        ("ground truth (periodic FMM)", gt_img, "viridis", 0.0, vmax, 14),
        ("splat prediction", pred_img, "viridis", 0.0, vmax, 14),
        ("error (pred − GT)", error_img, "bwr", -cfg.error_clip, cfg.error_clip, 0),
    ]
    for ax, (title, img, cmap, lo, hi, levels) in zip(axes, panels):
        ax.set_facecolor("lightgray")
        handle = ax.imshow(img, origin="lower", extent=extent, cmap=cmap, vmin=lo, vmax=hi, aspect="auto")
        if levels:
            ax.contour(mesh1, mesh2, img, levels=levels, colors="white", linewidths=0.6, alpha=0.7)
        ax.contour(mesh1, mesh2, blocked, levels=[0.5], colors="black", linewidths=1.2)
        ax.plot(*marker, "*", color="red", markersize=14, markeredgecolor="white")
        ax.set_title(title)
        ax.set_xlabel("θ1 (deg)")
        ax.set_ylabel("θ2 (deg)")
        fig.colorbar(handle, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"torus T² — time-to-go ({cfg.num_obstacles} obstacles) — {out_name}")
    fig.tight_layout()
    fig.savefig(f"{cfg.out_dir}/{out_name}", dpi=140)
    plt.close(fig)

    rms = float(np.sqrt(np.nanmean(error_img**2)))
    return {"rms": rms, "max_abs": float(np.nanmax(np.abs(error_img))), "rel_rms": rms / (np.nanstd(gt_img) + 1e-12)}


def main(cfg: Config) -> None:
    """Solve the torus Eikonal with obstacles self-supervised and score against periodic fast marching."""
    obstacles = torus_obstacles(cfg)
    axis = np.linspace(-np.pi, np.pi, cfg.resolution, endpoint=False)
    grid1, grid2 = np.meshgrid(axis, axis)
    thetas_np = np.stack([grid1.ravel(), grid2.ravel()], axis=-1)
    thetas = jnp.asarray(thetas_np, dtype=jnp.float32)
    shape = (cfg.resolution, cfg.resolution)

    slow = np.asarray(torus_slowness(thetas, obstacles, cfg.slowness_max, cfg.slow_width)).reshape(shape)
    gt = fast_marching_torus(1.0 / slow, cfg.start, cfg.resolution).ravel()
    inside = np.asarray(torus_sdf(thetas, obstacles)) < 0.0

    method = cfg.method
    roadmap = build_roadmap(cfg, obstacles) if method == "roadmap" else None

    def predict_current(current: tuple) -> np.ndarray:
        if method in ("ntfields", "supervised"):
            return np.asarray(predict_ntfields(current, thetas, jnp.asarray(cfg.start), cfg.tau_bias, cfg.tau_min))
        if method == "roadmap":
            return np.asarray(predict_roadmap(
                current, thetas, roadmap[0], roadmap[1], cfg.roadmap_gamma,
                obstacles, cfg.slowness_max, cfg.slow_width, cfg.roadmap_hop,
            ))
        return np.asarray(predict(current, thetas, jnp.asarray(cfg.start)))

    def checkpoint(current: tuple, stepnum: int) -> None:
        marks = render(cfg, gt, predict_current(current), inside, shape, out_name=f"torus_ckpt_{stepnum}.png")
        print(f"  [ckpt {stepnum}] saved torus_ckpt_{stepnum}.png  RMS={marks['rms']:.4e}", flush=True)

    solver = {
        "ntfields": solve_ntfields, "roadmap": solve_roadmap,
        "supervised": solve_supervised, "eikonal": solve,
    }[method]
    splat = solver(cfg, obstacles, checkpoint)
    prediction = predict_current(splat)
    metrics = render(cfg, gt, prediction, inside, shape)
    with open(f"{cfg.out_dir}/splat.pkl", "wb") as f:  # save params + scene for the near-obstacle diagnostic
        pickle.dump({"splat": [np.asarray(a) for a in splat], "obstacles": obstacles, "cfg": dataclasses.asdict(cfg)}, f)
    print(f"saved {cfg.out_dir}/torus_obstacles.png  ({len(obstacles)} obstacles)")
    print(f"RMS={metrics['rms']:.4e}  max|err|={metrics['max_abs']:.4e}  rel_RMS={metrics['rel_rms']:.4e}")


if __name__ == "__main__":
    main(tyro.cli(Config))
