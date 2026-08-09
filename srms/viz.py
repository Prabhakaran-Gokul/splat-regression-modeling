"""Rendering / scoring for a splat time-to-go prediction against ground truth."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def render(
    env,
    cfg,
    gt: np.ndarray,
    prediction: np.ndarray,
    inside: np.ndarray,
    shape: tuple[int, int],
    out_name: str = "torus_obstacles.png",
) -> dict:
    """Save a [GT | prediction | error] figure over the environment's chart and return metrics."""
    extent = env.render_extent
    marker = env.render_marker_deg()
    gt_img = np.where(inside, np.nan, gt).reshape(shape)
    pred_img = np.where(inside, np.nan, prediction).reshape(shape)
    error_img = pred_img - gt_img
    vmax = float(np.nanmax(gt_img))
    mesh1, mesh2 = np.meshgrid(
        np.linspace(extent[0], extent[1], shape[1]), np.linspace(extent[2], extent[3], shape[0])
    )
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
        ax.set_xlabel(env.axis_labels[0])
        ax.set_ylabel(env.axis_labels[1])
        fig.colorbar(handle, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"{env.title} — {out_name}")
    fig.tight_layout()
    fig.savefig(f"{cfg.out_dir}/{out_name}", dpi=140)
    plt.close(fig)

    rms = float(np.sqrt(np.nanmean(error_img**2)))
    return {"rms": rms, "max_abs": float(np.nanmax(np.abs(error_img))), "rel_rms": rms / (np.nanstd(gt_img) + 1e-12)}
