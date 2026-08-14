"""Rendering / scoring for a splat time-to-go prediction against ground truth."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def _draw_field(
    ax,
    env,
    img: np.ndarray,
    shape: tuple[int, int],
    coords: tuple[np.ndarray, np.ndarray] | None,
    edges: tuple[np.ndarray, np.ndarray] | None,
    cmap: str,
    vmin: float,
    vmax: float,
):
    """Draw ``img`` (shape ``shape``) on ``ax`` over ``env``'s chart; returns ``(handle, mesh1,
    mesh2)`` (the last two for ``contour``, which needs cell-*centre* coordinates matching
    ``img``'s shape regardless of which branch below is taken).

    ``coords``, if given, is a pair of 2D (X, Y) curvilinear cell-centre coordinate arrays
    matching ``shape`` (e.g. ``HyperbolicEnvironment.render_grid_xy``'s Poincaré-disk
    coordinates): drawn via ``pcolormesh`` + a unit-circle boundary instead of
    ``imshow(extent=...)``, since a curvilinear chart isn't a rectangle in its own coordinates.
    ``edges``, if given, is the same chart's cell-*edge* coordinates
    ((shape[0]+1, shape[1]+1); e.g. ``render_grid_edges_xy``) for ``pcolormesh`` specifically —
    ``pcolormesh``'s own centre-to-edge inference (``shading="auto"``) silently breaks on a
    periodic chart at the wraparound seam, so a curvilinear environment should supply real edges;
    without them this falls back to plotting the centres with ``shading="auto"`` regardless.
    torus/sphere never pass either (callers only do for environments that define
    ``render_grid_xy``/``render_grid_edges_xy``).
    """
    if coords is None:
        extent = env.render_extent
        mesh1, mesh2 = np.meshgrid(
            np.linspace(extent[0], extent[1], shape[1]), np.linspace(extent[2], extent[3], shape[0])
        )
        handle = ax.imshow(img, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        return handle, mesh1, mesh2

    mesh1, mesh2 = coords
    if edges is None:
        handle = ax.pcolormesh(mesh1, mesh2, img, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        handle = ax.pcolormesh(edges[0], edges[1], img, shading="flat", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.add_patch(plt.Circle((0.0, 0.0), 1.0, fill=False, edgecolor="black", linewidth=1.0))
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    return handle, mesh1, mesh2


def render(
    env,
    cfg,
    gt: np.ndarray,
    prediction: np.ndarray,
    inside: np.ndarray,
    shape: tuple[int, int],
    out_name: str = "torus_obstacles.png",
    coords: tuple[np.ndarray, np.ndarray] | None = None,
    edges: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict:
    """Save a [GT | prediction | error] figure over the environment's chart and return metrics.
    See ``_draw_field`` for what ``coords``/``edges`` do."""
    extent = env.render_extent
    marker = env.render_marker_deg()
    gt_img = np.where(inside, np.nan, gt).reshape(shape)
    pred_img = np.where(inside, np.nan, prediction).reshape(shape)
    error_img = pred_img - gt_img
    vmax = float(np.nanmax(gt_img))
    mesh1, mesh2 = np.meshgrid(np.linspace(extent[0], extent[1], shape[1]), np.linspace(extent[2], extent[3], shape[0]))
    blocked = inside.reshape(shape).astype(float)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    panels = [
        # Each environment names its own marcher; they genuinely differ.
        (getattr(env, "gt_label", "ground truth (fast marching)"), gt_img, "viridis", 0.0, vmax, 14),
        ("splat prediction", pred_img, "viridis", 0.0, vmax, 14),
        ("error (pred − GT)", error_img, "bwr", -cfg.error_clip, cfg.error_clip, 0),
    ]
    for ax, (title, img, cmap, lo, hi, levels) in zip(axes, panels):
        ax.set_facecolor("lightgray")
        handle, mesh1, mesh2 = _draw_field(ax, env, img, shape, coords, edges, cmap, lo, hi)
        if levels:
            ax.contour(mesh1, mesh2, img, levels=levels, colors="white", linewidths=0.6, alpha=0.7)
        ax.contour(mesh1, mesh2, blocked, levels=[0.5], colors="black", linewidths=1.2)
        ax.plot(*marker, "*", color="red", markersize=14, markeredgecolor="white")
        ax.set_title(title)
        ax.set_xlabel(env.axis_labels[0])
        ax.set_ylabel(env.axis_labels[1])
        fig.colorbar(handle, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"{env.title} — {out_name}")
    fig.tight_layout()
    fig.savefig(f"{cfg.out_dir}/{out_name}", dpi=140)
    plt.close(fig)

    rms = float(np.sqrt(np.nanmean(error_img**2)))
    return {"rms": rms, "max_abs": float(np.nanmax(np.abs(error_img))), "rel_rms": rms / (np.nanstd(gt_img) + 1e-12)}


def _slice_2d(field: np.ndarray, shape: tuple[int, ...], axis: int = 2, index: int | None = None):
    """Take a 2-D slice of a raveled field defined on a 2-D or 3-D grid.

    A 3-D manifold has no single image, so every 3-D panel in this repo is a *declared slice*: the
    grid is reshaped and one axis is fixed. Returns (2-D image, the slice index used).
    """
    img = field.reshape(shape)
    if len(shape) == 2:
        return img, None
    index = shape[axis] // 2 if index is None else index
    return np.take(img, index, axis=axis), index


def render_prediction(env, cfg, prediction, inside, shape, out_name="prediction.png") -> None:
    """Save a ground-truth-free view of the predicted field (used for mid-training checkpoints).

    Kept separate from ``render`` on purpose: anything called from inside the training loop must not
    be able to see ground truth, so this function has no parameter for it (see ``run.main``).
    """
    img, idx = _slice_2d(np.where(inside, np.nan, prediction), shape)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_facecolor("0.85")
    handle = ax.imshow(img, origin="lower", extent=env.render_extent, cmap="viridis", aspect="auto")
    ax.contour(
        *np.meshgrid(
            np.linspace(env.render_extent[0], env.render_extent[1], img.shape[1]),
            np.linspace(env.render_extent[2], env.render_extent[3], img.shape[0]),
        ),
        _slice_2d(inside.astype(float), shape)[0],
        levels=[0.5],
        colors="black",
        linewidths=1.2,
    )
    ax.set_title(f"{env.title} — prediction" + (f" (slice {idx})" if idx is not None else ""), fontsize=10)
    ax.set_xlabel(env.axis_labels[0])
    ax.set_ylabel(env.axis_labels[1])
    fig.colorbar(handle, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(f"{cfg.out_dir}/{out_name}", dpi=130)
    plt.close(fig)


def plot_speed_field(env, resolution: int, out_path: str) -> None:
    """Save a standalone visualization of the speed field s(x) = 1/slowness(x) over env's domain,
    with obstacle boundaries and the source marked — no training or prediction involved, purely
    the scene definition (``env.slowness``/``env.sdf``). Useful for sanity-checking a scene
    (obstacle placement, ``slowness_max``/``slow_width`` scale) before spending a training budget
    on it. Uses the same curvilinear-chart dispatch as ``render()`` (see ``_draw_field``):
    environments with ``render_grid_xy``/``render_grid_edges_xy`` (e.g.
    ``HyperbolicEnvironment``'s Poincaré disk) get a ``pcolormesh`` + unit-circle boundary; others
    (torus, sphere) fall back to ``imshow(extent=env.render_extent)``. Only meaningful at dim=2,
    same restriction as ``env.grid()`` itself (raises ``NotImplementedError`` otherwise).
    """
    points, shape = env.grid(resolution)
    speed = (1.0 / np.asarray(env.slowness(points))).reshape(shape)
    inside = np.asarray(env.sdf(points)).reshape(shape) < 0.0
    speed_img = np.where(inside, np.nan, speed)
    blocked = inside.astype(float)
    marker = env.render_marker_deg()

    coords = env.render_grid_xy(resolution) if hasattr(env, "render_grid_xy") else None
    edges = env.render_grid_edges_xy(resolution) if hasattr(env, "render_grid_edges_xy") else None

    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.set_facecolor("lightgray")
    handle, mesh1, mesh2 = _draw_field(ax, env, speed_img, shape, coords, edges, "viridis", 0.0, 1.0)
    ax.contour(mesh1, mesh2, blocked, levels=[0.5], colors="black", linewidths=1.4)
    ax.plot(*marker, "*", color="red", markersize=16, markeredgecolor="white")
    ax.set_xlabel(env.axis_labels[0])
    ax.set_ylabel(env.axis_labels[1])
    ax.set_title(f"{env.title} — speed field s(x) = 1/slowness(x)")
    fig.colorbar(handle, ax=ax, fraction=0.046, pad=0.04, label="speed (1 = free space)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
