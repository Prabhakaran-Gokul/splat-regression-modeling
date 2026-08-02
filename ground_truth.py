"""Ground-truth geodesic time-to-go fields for splat path-planning experiments.

Two ways to obtain the ground truth are provided:

* **Analytic** closed forms on obstacle-free domains — the flat square plane
  (``‖x − start‖``) and the unit sphere (great-circle ``arccos`` distance).
* **Numerical** fast marching (`fast_marching_2d`) which solves the Eikonal
  equation ``|∇T| = 1/speed`` on a grid for arbitrary speed fields, so it also
  covers obstacles where no closed form exists.

Each domain is exposed as an indexable ``PlanningProblem`` that yields
``(start, goal, time)`` queries and can reshape a flat field back into an image.
Running this module renders the ground-truth maps and checks the numerical solver
against the analytic plane.
"""

from __future__ import annotations

import dataclasses
import heapq
from typing import NamedTuple

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import tyro
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from matplotlib.patches import Circle as CirclePatch

Circle = tuple[float, float, float]


class Query(NamedTuple):
    """A single start/goal query together with its ground-truth time-to-go."""

    start: np.ndarray
    goal: np.ndarray
    time: float


@dataclasses.dataclass(frozen=True)
class PlanningProblem:
    """An obstacle-free-or-not domain with a time-to-go field measured from ``start``.

    Attributes:
        name: Domain identifier, ``"plane"`` or ``"sphere"``.
        dim: Ambient input dimension of the splat model (2 for plane, 3 for sphere).
        grid_shape: ``(H, W)`` used to reshape a flat field into an image.
        points: ``[N, dim]`` flattened grid of goal positions.
        start: ``[dim]`` source position that every time is measured from.
        ground_truth: ``[N, 1]`` time-to-go from ``start`` to each point (NaN inside obstacles).
        extent: ``(left, right, bottom, top)`` axes extent for ``imshow``.
        axis_labels: ``(x_label, y_label)`` for plots.
        start_marker: ``(x, y)`` location of the source in axes coordinates.
        obstacles: circles ``(cx, cy, r)`` blocking the plane (empty on free domains).
        valid: ``[N]`` boolean mask of free-space points, or ``None`` when all are free.
    """

    name: str
    dim: int
    grid_shape: tuple[int, int]
    points: jnp.ndarray
    start: jnp.ndarray
    ground_truth: jnp.ndarray
    extent: tuple[float, float, float, float]
    axis_labels: tuple[str, str]
    start_marker: tuple[float, float]
    obstacles: tuple[Circle, ...] = ()
    valid: np.ndarray | None = None

    def __len__(self) -> int:
        return int(self.points.shape[0])

    def __getitem__(self, index: int) -> Query:
        """Return the ``(start, goal, time)`` query for the ``index``-th grid point."""
        return Query(
            start=np.asarray(self.start),
            goal=np.asarray(self.points[index]),
            time=float(self.ground_truth[index, 0]),
        )

    def to_image(self, values: jnp.ndarray) -> np.ndarray:
        """Reshape a flat ``[N]`` or ``[N, 1]`` field into an ``[H, W]`` image array."""
        return np.asarray(values).reshape(self.grid_shape)

    def mesh(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(X, Y)`` axis-coordinate grids aligned with ``to_image`` for contours."""
        left, right, bottom, top = self.extent
        height, width = self.grid_shape
        return np.meshgrid(np.linspace(left, right, width), np.linspace(bottom, top, height))

    def free_indices(self) -> np.ndarray:
        """Return indices of free-space points (all points when there are no obstacles)."""
        if self.valid is None:
            return np.arange(len(self))
        return np.where(np.asarray(self.valid))[0]


def make_plane(resolution: int, start_xy: tuple[float, float], half_width: float = 1.0) -> PlanningProblem:
    """Build the square plane ``[-half_width, half_width]^2`` with analytic Euclidean time-to-go."""
    axis = jnp.linspace(-half_width, half_width, resolution)
    grid_x, grid_y = jnp.meshgrid(axis, axis, indexing="xy")
    points = jnp.stack([grid_x.ravel(), grid_y.ravel()], axis=-1).astype(jnp.float32)
    start = jnp.asarray(start_xy, dtype=jnp.float32)
    time = jnp.linalg.norm(points - start, axis=-1, keepdims=True)
    return PlanningProblem(
        name="plane",
        dim=2,
        grid_shape=(resolution, resolution),
        points=points,
        start=start,
        ground_truth=time,
        extent=(-half_width, half_width, -half_width, half_width),
        axis_labels=("x", "y"),
        start_marker=(float(start_xy[0]), float(start_xy[1])),
    )


def _lonlat_to_xyz(lon: jnp.ndarray, lat: jnp.ndarray) -> jnp.ndarray:
    """Convert longitude/latitude in radians to unit 3-vectors on the sphere."""
    cos_lat = jnp.cos(lat)
    return jnp.stack([cos_lat * jnp.cos(lon), cos_lat * jnp.sin(lon), jnp.sin(lat)], axis=-1)


def make_sphere(resolution: int, start_latlon_deg: tuple[float, float]) -> PlanningProblem:
    """Build the unit sphere with great-circle (arccos) time-to-go, mapped to lon/lat."""
    lon = jnp.linspace(-jnp.pi, jnp.pi, 2 * resolution)
    lat = jnp.linspace(-jnp.pi / 2, jnp.pi / 2, resolution)
    grid_lon, grid_lat = jnp.meshgrid(lon, lat, indexing="xy")
    points = _lonlat_to_xyz(grid_lon.ravel(), grid_lat.ravel()).astype(jnp.float32)
    lat0, lon0 = jnp.deg2rad(jnp.asarray(start_latlon_deg, dtype=jnp.float32))
    start = _lonlat_to_xyz(lon0, lat0).astype(jnp.float32)
    cos_angle = jnp.clip(points @ start, -1.0, 1.0)
    time = jnp.arccos(cos_angle)[:, None]
    return PlanningProblem(
        name="sphere",
        dim=3,
        grid_shape=(resolution, 2 * resolution),
        points=points,
        start=start,
        ground_truth=time,
        extent=(-180.0, 180.0, -90.0, 90.0),
        axis_labels=("longitude (deg)", "latitude (deg)"),
        start_marker=(float(start_latlon_deg[1]), float(start_latlon_deg[0])),
    )


def fast_marching_2d(speed: np.ndarray, axis: np.ndarray, start_xy: tuple[float, float]) -> np.ndarray:
    """Solve ``|∇T| = 1/speed`` on a square grid by first-order fast marching.

    Args:
        speed: ``[H, W]`` speed field with ``speed[j, i]`` at ``(axis[i], axis[j])``.
        axis: shared 1-D coordinate array for both axes (uniform spacing).
        start_xy: source location; nearby nodes are seeded with their Euclidean distance.

    Returns:
        ``[H, W]`` array of arrival times (``inf`` in unreachable regions).
    """
    n = axis.shape[0]
    step = float(axis[1] - axis[0])
    grid_x, grid_y = np.meshgrid(axis, axis)
    slowness = 1.0 / speed
    time = np.full((n, n), np.inf)
    accepted = np.zeros((n, n), dtype=bool)
    heap: list[tuple[float, int, int]] = []

    radius = np.sqrt((grid_x - start_xy[0]) ** 2 + (grid_y - start_xy[1]) ** 2)
    for j, i in zip(*np.where(radius <= step)):
        time[j, i] = float(radius[j, i])
        heapq.heappush(heap, (float(time[j, i]), int(j), int(i)))

    while heap:
        _, j, i = heapq.heappop(heap)
        if accepted[j, i]:
            continue
        accepted[j, i] = True
        for dj, di in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nj, ni = j + dj, i + di
            if not (0 <= nj < n and 0 <= ni < n) or accepted[nj, ni]:
                continue
            along_x = min(
                time[nj, ni - 1] if ni > 0 and accepted[nj, ni - 1] else np.inf,
                time[nj, ni + 1] if ni < n - 1 and accepted[nj, ni + 1] else np.inf,
            )
            along_y = min(
                time[nj - 1, ni] if nj > 0 and accepted[nj - 1, ni] else np.inf,
                time[nj + 1, ni] if nj < n - 1 and accepted[nj + 1, ni] else np.inf,
            )
            cost = slowness[nj, ni] * step
            if np.isinf(along_x) and np.isinf(along_y):
                continue
            if np.isinf(along_x):
                candidate = along_y + cost
            elif np.isinf(along_y):
                candidate = along_x + cost
            else:
                low, high = min(along_x, along_y), max(along_x, along_y)
                if high - low >= cost:
                    candidate = low + cost
                else:
                    candidate = 0.5 * (low + high + np.sqrt(max(0.0, 2 * cost**2 - (high - low) ** 2)))
            if candidate < time[nj, ni]:
                time[nj, ni] = candidate
                heapq.heappush(heap, (float(candidate), int(nj), int(ni)))
    return time


def obstacle_mask(axis: np.ndarray, obstacles: tuple[Circle, ...]) -> np.ndarray:
    """Return an ``[H, W]`` boolean mask that is True inside any obstacle circle."""
    grid_x, grid_y = np.meshgrid(axis, axis)
    inside = np.zeros(grid_x.shape, dtype=bool)
    for cx, cy, radius in obstacles:
        inside |= (grid_x - cx) ** 2 + (grid_y - cy) ** 2 <= radius**2
    return inside


def make_plane_fmm(
    resolution: int,
    start_xy: tuple[float, float],
    obstacles: tuple[Circle, ...] = (),
    half_width: float = 1.0,
    obstacle_speed: float = 1e-3,
) -> PlanningProblem:
    """Build the square plane with a numerical (fast-marching) time-to-go around obstacles."""
    axis = np.linspace(-half_width, half_width, resolution)
    inside = obstacle_mask(axis, obstacles)
    speed = np.where(inside, obstacle_speed, 1.0)
    time = fast_marching_2d(speed, axis, start_xy)
    time = np.where(inside, np.nan, time)
    grid_x, grid_y = np.meshgrid(axis, axis)
    points = jnp.asarray(np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1), dtype=jnp.float32)
    return PlanningProblem(
        name="plane",
        dim=2,
        grid_shape=(resolution, resolution),
        points=points,
        start=jnp.asarray(start_xy, dtype=jnp.float32),
        ground_truth=jnp.asarray(time.reshape(-1, 1), dtype=jnp.float32),
        extent=(-half_width, half_width, -half_width, half_width),
        axis_labels=("x", "y"),
        start_marker=(float(start_xy[0]), float(start_xy[1])),
        obstacles=tuple(obstacles),
        valid=~inside.ravel(),
    )


def make_plane_scene(
    resolution: int,
    start_xy: tuple[float, float],
    scene,
    half_width: float = 1.0,
    slowness_max: float = 12.0,
    width: float = 0.04,
) -> PlanningProblem:
    """Build the square plane with a fast-marching time-to-go through a smooth SDF slowness field.

    The speed field is ``1 / smooth_slowness`` — the same field the self-supervised solver uses —
    so the ground truth and the solved field solve identical physics. Obstacle interiors are
    masked to NaN for display and scoring.
    """
    import scenes

    axis = np.linspace(-half_width, half_width, resolution)
    grid_x, grid_y = np.meshgrid(axis, axis)
    points_np = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)
    points = jnp.asarray(points_np, dtype=jnp.float32)
    inside = scenes.inside_mask(points_np, scene).reshape(resolution, resolution)
    slow = np.asarray(scenes.smooth_slowness(points, scene, slowness_max, width)).reshape(resolution, resolution)
    time = fast_marching_2d(1.0 / slow, axis, start_xy)
    time = np.where(inside, np.nan, time)
    return PlanningProblem(
        name="plane",
        dim=2,
        grid_shape=(resolution, resolution),
        points=jnp.asarray(points_np, dtype=jnp.float32),
        start=jnp.asarray(start_xy, dtype=jnp.float32),
        ground_truth=jnp.asarray(time.reshape(-1, 1), dtype=jnp.float32),
        extent=(-half_width, half_width, -half_width, half_width),
        axis_labels=("x", "y"),
        start_marker=(float(start_xy[0]), float(start_xy[1])),
        valid=~inside.ravel(),
    )


def make_problem(name: str, resolution: int, start: tuple[float, float]) -> PlanningProblem:
    """Dispatch to the plane or sphere builder; ``start`` is interpreted per domain."""
    if name == "plane":
        return make_plane(resolution, start)
    if name == "sphere":
        return make_sphere(resolution, start)
    raise ValueError(f"unknown problem '{name}', expected 'plane' or 'sphere'")


def draw_obstacles(ax: Axes, obstacles: tuple[Circle, ...]) -> None:
    """Overlay obstacle circles as hatched grey patches."""
    for cx, cy, radius in obstacles:
        ax.add_patch(CirclePatch((cx, cy), radius, facecolor="0.5", edgecolor="black", hatch="///", zorder=5))


def draw_field(
    ax: Axes, problem: PlanningProblem, image: np.ndarray, cmap: str, vmin: float, vmax: float, contours: int = 12
) -> AxesImage:
    """Draw a field as an image with iso-time contour lines, obstacles, and the source marker."""
    ax.set_facecolor("lightgray")
    handle = ax.imshow(image, origin="lower", extent=problem.extent, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    grid_x, grid_y = problem.mesh()
    if contours:
        ax.contour(grid_x, grid_y, image, levels=contours, colors="white", linewidths=0.6, alpha=0.7)
    if problem.valid is not None:
        blocked = (~np.asarray(problem.valid)).reshape(problem.grid_shape).astype(float)
        ax.contour(grid_x, grid_y, blocked, levels=[0.5], colors="black", linewidths=1.2)
    draw_obstacles(ax, problem.obstacles)
    ax.plot(*problem.start_marker, "*", color="red", markersize=14, markeredgecolor="white", zorder=6)
    ax.set_xlabel(problem.axis_labels[0])
    ax.set_ylabel(problem.axis_labels[1])
    return handle


def render_ground_truth(problem: PlanningProblem, path: str) -> None:
    """Save a single-panel image of a problem's ground-truth time-to-go with contours."""
    image = problem.to_image(problem.ground_truth)
    fig, ax = plt.subplots(figsize=(6, 5))
    handle = draw_field(ax, problem, image, "viridis", float(np.nanmin(image)), float(np.nanmax(image)))
    ax.set_title(f"ground-truth time-to-go — {problem.name}")
    fig.colorbar(handle, ax=ax, label="time-to-go")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


@dataclasses.dataclass
class RenderConfig:
    """Configuration for rendering and validating the ground-truth maps."""

    resolution: int = 200
    plane_start: tuple[float, float] = (0.0, 0.0)
    sphere_start: tuple[float, float] = (0.0, 0.0)
    out_dir: str = "figures"


def main(cfg: RenderConfig) -> None:
    """Render ground-truth maps and validate fast marching against the analytic plane."""
    plane = make_plane(cfg.resolution, cfg.plane_start)
    sphere = make_sphere(cfg.resolution, cfg.sphere_start)
    obstacle = make_plane_fmm(cfg.resolution, (-0.6, -0.6), obstacles=((0.1, 0.1, 0.35),))
    render_ground_truth(plane, f"{cfg.out_dir}/gt_plane.png")
    render_ground_truth(sphere, f"{cfg.out_dir}/gt_sphere.png")
    render_ground_truth(obstacle, f"{cfg.out_dir}/gt_obstacle.png")

    axis = np.linspace(-1.0, 1.0, cfg.resolution)
    numerical = fast_marching_2d(np.ones((cfg.resolution, cfg.resolution)), axis, cfg.plane_start)
    analytic = plane.to_image(plane.ground_truth)
    print(f"plane:  time in [{float(plane.ground_truth.min()):.3f}, {float(plane.ground_truth.max()):.3f}]")
    print(f"sphere: time in [{float(sphere.ground_truth.min()):.3f}, {float(sphere.ground_truth.max()):.3f}]")
    print(f"fast-marching vs analytic plane: max abs diff = {float(np.abs(numerical - analytic).max()):.4f}")


if __name__ == "__main__":
    main(tyro.cli(RenderConfig))
