"""Zoom into one obstacle and visualise how the splat field behaves at its rim.

Loads a trained NTFields-style splat (``splat.pkl`` written by ``torus.py``) and renders, over a
window around the largest obstacle: the time field ``T``, the Eikonal ratio ``‖∇T‖_g / s`` (which
should be ~1 everywhere; oscillations away from 1 at the rim are the C∞ splats *ringing* against the
steep slowness wall), and the correction ``τ``. Splat centres are overlaid so we can see whether
Gaussians pile up and distort at the boundary.
"""

from __future__ import annotations

import pickle

import jax
import jax.numpy as jnp
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import tyro

from torus import eval_splat_torus, geodesic, metric_inv, torus_slowness, wrap


def main(pkl: str, out: str = "figures/diagnose_obstacle.png", margin: float = 0.9, res: int = 200) -> None:
    """Render the near-obstacle diagnostic from a saved splat and print the rim Eikonal-ratio spread."""
    with open(pkl, "rb") as f:
        blob = pickle.load(f)
    splat = tuple(jnp.asarray(a) for a in blob["splat"])
    obstacles = tuple(tuple(o) for o in blob["obstacles"])
    cfg = blob["cfg"]
    start = jnp.asarray(cfg["start"])
    tau_bias, tau_min = cfg["tau_bias"], cfg["tau_min"]

    c1, c2, r = max(obstacles, key=lambda o: o[2])  # largest obstacle
    axis1 = np.linspace(c1 - r - margin, c1 + r + margin, res)
    axis2 = np.linspace(c2 - r - margin, c2 + r + margin, res)
    g1, g2 = np.meshgrid(axis1, axis2)
    thetas = jnp.asarray(np.stack([g1.ravel(), g2.ravel()], axis=-1), dtype=jnp.float32)

    def field(t: jnp.ndarray) -> jnp.ndarray:
        tau = tau_min + (1.0 - tau_min) * jax.nn.sigmoid(eval_splat_torus(t[None, :], splat)[0, 0] + tau_bias)
        return jnp.linalg.norm(wrap(t - start)) / tau

    tau = tau_min + (1.0 - tau_min) * jax.nn.sigmoid(eval_splat_torus(thetas, splat).ravel() + tau_bias)
    tfield = np.asarray(geodesic(thetas, start)) / np.asarray(tau)
    grad = jax.vmap(jax.grad(field))(thetas)
    gnorm = np.asarray(jax.vmap(lambda gr, t: jnp.sqrt(gr @ metric_inv(t) @ gr))(grad, thetas))
    slow = np.asarray(torus_slowness(thetas, obstacles, cfg["slowness_max"], cfg["slow_width"]))
    ratio = (gnorm / slow).reshape(res, res)  # Eikonal ratio: 1 = satisfied, ringing = oscillation

    shape = (res, res)
    extent = (float(axis1[0]), float(axis1[-1]), float(axis2[0]), float(axis2[-1]))
    panels = [
        ("time field T", tfield.reshape(shape), "viridis", None, None),
        ("Eikonal ratio ‖∇T‖/s  (1 = ok, ring = ✗)", ratio, "bwr", 0.0, 2.0),
        ("correction τ  (∈ [τ_min, 1])", np.asarray(tau).reshape(shape), "viridis", 0.0, 1.0),
    ]
    centres = np.asarray(splat[2])
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))
    for ax, (title, img, cmap, lo, hi) in zip(axes, panels):
        h = ax.imshow(img, origin="lower", extent=extent, cmap=cmap, vmin=lo, vmax=hi, aspect="auto")
        ax.add_patch(mpatches.Circle((c1, c2), r, fill=False, color="black", lw=1.6))
        in_win = (np.abs(centres[:, 0] - c1) < r + margin) & (np.abs(centres[:, 1] - c2) < r + margin)
        ax.scatter(centres[in_win, 0], centres[in_win, 1], s=8, c="white", edgecolors="k", linewidths=0.3)
        ax.set_title(title)
        ax.set_xlabel("θ1 (rad)")
        ax.set_ylabel("θ2 (rad)")
        fig.colorbar(h, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"near-obstacle diagnostic — {pkl}")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)

    rim = np.abs(np.sqrt((g1 - c1) ** 2 + (g2 - c2) ** 2).ravel() - r) < 0.15  # thin annulus at the rim
    print(f"saved {out}")
    print(f"rim Eikonal-ratio: mean={ratio.ravel()[rim].mean():.3f} std={ratio.ravel()[rim].std():.3f} "
          f"(std≫0 ⇒ ringing at the wall)")


if __name__ == "__main__":
    tyro.cli(main)
