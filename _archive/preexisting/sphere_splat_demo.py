#!/usr/bin/env python3
"""
Visualize wrapped Gaussian SRM on S².

Produces figures/sphere_splat.png with two panels:
  Left:  3D sphere colored by SRM density
  Right: equirectangular (lat/lon) heatmap of the same field

Usage:
    python sphere_splat_demo.py
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

jax.config.update("jax_platform_name", "gpu")

from lib.manifold_splat import eval_sphere_splat_jit, init_sphere_splat

# ---------------------------------------------------------------------------
# Build the evaluation grid on S²
# ---------------------------------------------------------------------------

GRID_N = 200   # lat × lon resolution

lat = np.linspace(-np.pi / 2, np.pi / 2, GRID_N)    # -π/2 … π/2
lon = np.linspace(-np.pi, np.pi, GRID_N)             # -π … π
LON, LAT = np.meshgrid(lon, lat)                     # both [N, N]

# Sphere embedding
X_grid = np.stack([
    np.cos(LAT) * np.cos(LON),   # x
    np.cos(LAT) * np.sin(LON),   # y
    np.sin(LAT),                  # z
], axis=-1)  # [N, N, 3]

X_flat = jnp.array(X_grid.reshape(-1, 3), dtype=jnp.float32)  # [N², 3]

# ---------------------------------------------------------------------------
# Construct a demo SRM with 5 components, varied centers & anisotropy
# ---------------------------------------------------------------------------

key = jr.PRNGKey(0)
k1, k2 = jr.split(key)

# Manually placed centers: spread around the sphere
centers_raw = jnp.array([
    [ 1.0,  0.0,  0.0],  # equator 0°
    [-1.0,  0.0,  0.0],  # equator 180°
    [ 0.0,  1.0,  0.5],  # equator-ish 90°
    [ 0.0, -0.5,  1.0],  # northern high lat
    [ 0.5,  0.5, -0.8],  # southern
], dtype=jnp.float32)
B = centers_raw / jnp.linalg.norm(centers_raw, axis=-1, keepdims=True)

# Covariance factors A [k, 2, 2] — wrapped Gaussian in tangent plane at each center.
# Varying sigma gives components with different angular spread.
sigma = 0.35
s = [sigma * f for f in [0.5, 1.0, 1.4, 0.7, 1.0]]  # different widths
A = jnp.array([
    jnp.diag(jnp.array([s[0], s[0]])),
    jnp.diag(jnp.array([s[1], s[1]])),
    jnp.diag(jnp.array([s[2], s[2]])),
    jnp.diag(jnp.array([s[3], s[3]])),
    jnp.diag(jnp.array([s[4], s[4]])),
], dtype=jnp.float32)

# Positive weights (we want to see peaks, not cancellation)
V = jnp.array([[1.0], [0.8], [1.2], [0.9], [1.1]], dtype=jnp.float32)

splatnn = (V, A, B)

# ---------------------------------------------------------------------------
# Evaluate the SRM on the grid
# ---------------------------------------------------------------------------

print("Evaluating sphere SRM on grid...")
Y_flat = eval_sphere_splat_jit(X_flat, splatnn)        # [N², 1]
Y_grid = np.array(Y_flat).reshape(GRID_N, GRID_N)      # [N, N]

print(f"  min={Y_grid.min():.4f}  max={Y_grid.max():.4f}  "
      f"mean={Y_grid.mean():.4f}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig = plt.figure(figsize=(14, 6))
fig.suptitle("Wrapped Gaussian SRM on S²", fontsize=14)

colormap = cm.plasma
Y_norm = (Y_grid - Y_grid.min()) / (Y_grid.max() - Y_grid.min() + 1e-12)

# --- Panel 1: 3D sphere ---
ax3d = fig.add_subplot(1, 2, 1, projection="3d")
facecolors = colormap(Y_norm)   # [N, N, 4] RGBA

ax3d.plot_surface(
    X_grid[..., 0], X_grid[..., 1], X_grid[..., 2],
    facecolors=facecolors,
    rstride=1, cstride=1,
    antialiased=False,
    shade=False,
)

# Mark centers
B_np = np.array(B)
ax3d.scatter(B_np[:, 0], B_np[:, 1], B_np[:, 2],
             color="white", s=40, zorder=5, edgecolors="black", linewidths=0.8)

ax3d.set_title("3D view")
ax3d.set_box_aspect([1, 1, 1])
ax3d.set_axis_off()

# Add colorbar via a proxy ScalarMappable
sm = plt.cm.ScalarMappable(cmap=colormap,
                            norm=plt.Normalize(Y_grid.min(), Y_grid.max()))
sm.set_array([])
plt.colorbar(sm, ax=ax3d, shrink=0.5, pad=0.05, label="SRM value")

# --- Panel 2: equirectangular projection ---
ax2d = fig.add_subplot(1, 2, 2)
im = ax2d.imshow(
    Y_grid,
    extent=[-180, 180, -90, 90],
    origin="lower",
    cmap=colormap,
    aspect="auto",
)
ax2d.set_xlabel("Longitude (°)")
ax2d.set_ylabel("Latitude (°)")
ax2d.set_title("Equirectangular projection")
plt.colorbar(im, ax=ax2d, label="SRM value")

# Mark center projections on equirectangular
B_np = np.array(B)
B_lat = np.degrees(np.arcsin(np.clip(B_np[:, 2], -1, 1)))
B_lon = np.degrees(np.arctan2(B_np[:, 1], B_np[:, 0]))
ax2d.scatter(B_lon, B_lat, color="white", s=40, zorder=5,
             edgecolors="black", linewidths=0.8)

plt.tight_layout()
out_path = "figures/sphere_splat.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved to {out_path}")
plt.show()
