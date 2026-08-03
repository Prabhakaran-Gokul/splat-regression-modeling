"""Signed-distance-field obstacle scenes for planning experiments.

Obstacles are represented as signed distance fields (SDFs): ``d(x) < 0`` inside,
``d(x) > 0`` outside, with ``∇d`` the outward normal on the surface. A *scene* is a
list of primitives (circles and rotated boxes) whose union SDF is the pointwise
minimum. This one representation gives everything the solver needs — membership
(``d < 0``), the reflecting-boundary normals (``∇d``), and surface samples — and it
extends from a single circle to many objects and to intricate non-convex shapes
(built as unions of primitives) without changing any downstream code.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

Primitive = tuple  # ("circle", cx, cy, r) or ("box", cx, cy, hx, hy, angle)
Scene = tuple[Primitive, ...]


def _sdf_circle(points: jnp.ndarray, cx: float, cy: float, r: float) -> jnp.ndarray:
    """Signed distance to a circle of radius ``r`` centred at ``(cx, cy)``."""
    return jnp.linalg.norm(points - jnp.array([cx, cy]), axis=-1) - r


def _sdf_box(points: jnp.ndarray, cx: float, cy: float, hx: float, hy: float, angle: float) -> jnp.ndarray:
    """Signed distance to a rotated box with half-extents ``(hx, hy)`` centred at ``(cx, cy)``."""
    cos_a, sin_a = jnp.cos(-angle), jnp.sin(-angle)
    delta = points - jnp.array([cx, cy])
    local_x = cos_a * delta[..., 0] - sin_a * delta[..., 1]
    local_y = sin_a * delta[..., 0] + cos_a * delta[..., 1]
    qx = jnp.abs(local_x) - hx
    qy = jnp.abs(local_y) - hy
    outside = jnp.sqrt(jnp.maximum(qx, 0.0) ** 2 + jnp.maximum(qy, 0.0) ** 2)
    inside = jnp.minimum(jnp.maximum(qx, qy), 0.0)
    return outside + inside


def primitive_sdf(points: jnp.ndarray, primitive: Primitive) -> jnp.ndarray:
    """Dispatch to the SDF of a single primitive."""
    kind, *params = primitive
    if kind == "circle":
        return _sdf_circle(points, *params)
    if kind == "box":
        return _sdf_box(points, *params)
    raise ValueError(f"unknown primitive '{kind}'")


def scene_sdf(points: jnp.ndarray, scene: Scene) -> jnp.ndarray:
    """Return the union SDF of the scene (pointwise minimum over primitives)."""
    return jnp.min(jnp.stack([primitive_sdf(points, p) for p in scene], axis=0), axis=0)


def inside_mask(points: np.ndarray, scene: Scene) -> np.ndarray:
    """Return a boolean array that is True for points inside any obstacle."""
    return np.asarray(scene_sdf(jnp.asarray(points, dtype=jnp.float32), scene)) < 0.0


def smooth_slowness(points: jnp.ndarray, scene: Scene, slowness_max: float, width: float) -> jnp.ndarray:
    """Return a smooth slowness field: ~1 in free space, rising to ``slowness_max`` inside obstacles.

    The transition follows the SDF through a sigmoid of width ``width``. A smooth (rather than
    discontinuous) rise keeps the field representable by a smooth splat, so obstacle-rim error
    stays small and does not compound as objects are added, while the high interior cost is the
    global signal that forces the wave to route around.
    """
    distance = scene_sdf(points, scene)
    return 1.0 + (slowness_max - 1.0) * jax.nn.sigmoid(-distance / width)


def _primitive_surface(primitive: Primitive, samples: int) -> np.ndarray:
    """Sample points along the surface of a single primitive."""
    kind, *params = primitive
    if kind == "circle":
        cx, cy, r = params
        angle = np.linspace(0.0, 2 * np.pi, samples, endpoint=False)
        return np.stack([cx + r * np.cos(angle), cy + r * np.sin(angle)], axis=1)
    cx, cy, hx, hy, rot = params
    t = np.linspace(-1.0, 1.0, samples // 4, endpoint=False)
    edges = np.concatenate(
        [
            np.stack([t * hx, np.full_like(t, hy)], axis=1),
            np.stack([t * hx, np.full_like(t, -hy)], axis=1),
            np.stack([np.full_like(t, hx), t * hy], axis=1),
            np.stack([np.full_like(t, -hx), t * hy], axis=1),
        ],
        axis=0,
    )
    cos_a, sin_a = np.cos(rot), np.sin(rot)
    rotated = np.stack([cos_a * edges[:, 0] - sin_a * edges[:, 1], sin_a * edges[:, 0] + cos_a * edges[:, 1]], axis=1)
    return rotated + np.array([cx, cy])


def surface_samples(scene: Scene, samples_per_object: int, tol: float = 1e-3) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return boundary points on the union surface with their outward unit normals.

    Points swallowed inside another primitive are dropped, so only the true outer
    boundary of the union remains; normals come from ``∇(scene_sdf)``.
    """
    candidates = np.concatenate([_primitive_surface(p, samples_per_object) for p in scene], axis=0)
    on_boundary = np.asarray(scene_sdf(jnp.asarray(candidates, dtype=jnp.float32), scene)) > -tol
    points = jnp.asarray(candidates[on_boundary], dtype=jnp.float32)
    gradients = jax.vmap(lambda x: jax.grad(lambda p: scene_sdf(p, scene))(x))(points)
    normals = gradients / (jnp.linalg.norm(gradients, axis=1, keepdims=True) + 1e-9)
    return points, normals


def random_scene(
    seed: int,
    num_objects: int,
    half_width: float = 1.0,
    box_fraction: float = 0.4,
    radius_range: tuple[float, float] = (0.15, 0.28),
    avoid: tuple[float, float] | None = None,
    clearance: float = 0.12,
) -> Scene:
    """Generate a reproducible scene of random circles and boxes clear of ``avoid``."""
    rng = np.random.default_rng(seed)
    scene: list[Primitive] = []
    attempts = 0
    while len(scene) < num_objects and attempts < 1000:
        attempts += 1
        size = float(rng.uniform(*radius_range))
        centre = rng.uniform(-half_width + size, half_width - size, size=2)
        if rng.uniform() < box_fraction:
            hx, hy = size, float(rng.uniform(0.6, 1.4)) * size
            candidate: Primitive = ("box", float(centre[0]), float(centre[1]), hx, hy, float(rng.uniform(0, np.pi)))
        else:
            candidate = ("circle", float(centre[0]), float(centre[1]), size)
        if avoid is not None and float(scene_sdf(jnp.asarray(avoid, dtype=jnp.float32), (candidate,))) < clearance:
            continue
        scene.append(candidate)
    return tuple(scene)


def named_scene(name: str) -> Scene:
    """Return a fixed intricate scene by name (non-convex unions of primitives)."""
    if name == "single":
        return (("circle", 0.1, 0.1, 0.35),)
    if name == "wall":
        return (("box", 0.1, 0.0, 0.05, 0.75, 0.0),)
    if name == "cross":
        return (("box", 0.15, 0.1, 0.12, 0.5, 0.0), ("box", 0.15, 0.1, 0.5, 0.12, 0.0))
    if name == "u_trap":
        return (
            ("box", 0.1, 0.35, 0.45, 0.1, 0.0),
            ("box", -0.3, 0.0, 0.1, 0.45, 0.0),
            ("box", 0.5, 0.0, 0.1, 0.45, 0.0),
        )
    raise ValueError(f"unknown named scene '{name}'")
