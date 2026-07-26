#!/usr/bin/env python3
"""
Dubins car wrapped Gaussian SRM demo on SE(2).

Visualises how the sub-Riemannian (SR) constraint shapes the distribution:
  - Isotropic A = diag(σ, σ, σ): density spreads equally in all directions
  - SR        A = diag(σ_fwd, σ_lat, σ_rot) with σ_lat ≪ σ_fwd:
      density is elongated in the forward direction and rotates with heading

Panels: 2 × 4 grid of density slices in the (x,y) plane at fixed heading θ.
Plus a bottom row: marginal ∫dθ  density  and  a quiver/configuration cloud.

Output: figures/dubins_splat.png
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

jax.config.update("jax_platform_name", "gpu")

from lib.manifold_splat import eval_se2_rho_single, eval_se2_splat_jit, init_se2_splat

# ── Grid and covariance definitions ───────────────────────────────────────────

N      = 200          # spatial grid resolution
EXTENT = 3.5          # grid half-width [m]
N_ANG  = 48           # θ samples for marginal integration

x_vec = np.linspace(-EXTENT, EXTENT, N)
y_vec = np.linspace(-EXTENT, EXTENT, N)
XX, YY = np.meshgrid(x_vec, y_vec)

# Heading orientations to show (radians)
THETAS  = [-np.pi / 2, -np.pi / 4, 0.0, np.pi / 4]
LABELS  = ["-90°", "-45°", "0°", "+45°"]

# Covariance factors
A_ISO = jnp.diag(jnp.array([0.8, 0.8, 0.8]))       # isotropic
A_SR  = jnp.diag(jnp.array([1.4, 0.15, 0.6]))       # sub-Riemannian

# ── Helper: evaluate density on a 2D (x,y) grid at fixed θ ───────────────────

def density_xy_slice(center, A, theta_query):
    """
    Density N_w(x, y, theta_query | center, A) on the (x,y) grid.
    Returns [N, N] array.
    """
    G = jnp.array(
        np.column_stack([XX.ravel(), YY.ravel(),
                         np.full(N * N, float(theta_query))]),
        dtype=jnp.float32,
    )
    h = jnp.array(center, dtype=jnp.float32)
    rho = jax.vmap(lambda g: eval_se2_rho_single(g, h, A))(G)
    return np.array(rho).reshape(N, N)


def marginal_density(center, A):
    """
    Marginal density ∫ N_w(x,y,θ|center,A) dθ  ≈ Σ_θ N_w * dθ.
    Returns [N, N] array.
    """
    dtheta = 2.0 * np.pi / N_ANG
    total  = np.zeros((N, N), dtype=np.float32)
    for theta in np.linspace(-np.pi, np.pi, N_ANG, endpoint=False):
        total += np.array(density_xy_slice(center, A, theta), dtype=np.float32)
    return total * dtheta


# ── Draw a "car" arrow at (x0, y0, θ0) ───────────────────────────────────────

def draw_car(ax, x0, y0, theta, length=0.35, color="red", alpha=1.0, lw=2):
    dx = length * np.cos(theta)
    dy = length * np.sin(theta)
    ax.annotate(
        "", xy=(x0 + dx, y0 + dy), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", color=color,
                        lw=lw, mutation_scale=14 * alpha),
        alpha=alpha,
    )


# ── Figure ────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor("#f8f8f8")

CMAP = "magma"

# Titles for rows
row_labels = [
    "Isotropic Gaussian   A = diag(0.8, 0.8, 0.8)",
    "Sub-Riemannian (Dubins)   A = diag(1.4, 0.15, 0.6)   [σ_lat ≪ σ_fwd]",
]

# --- Rows 0 & 1: density slices at 4 headings ---

for row, (A, A_label) in enumerate(zip([A_ISO, A_SR], row_labels)):
    for col, (theta0, ang_label) in enumerate(zip(THETAS, LABELS)):
        ax = fig.add_subplot(3, 4, row * 4 + col + 1)

        center = [0.0, 0.0, theta0]
        Z = density_xy_slice(center, A, theta0)

        vmax = Z.max()
        ax.contourf(x_vec, y_vec, Z, levels=30, cmap=CMAP, vmin=0, vmax=vmax)
        ax.contour (x_vec, y_vec, Z, levels=8,  colors="white",
                    linewidths=0.5, alpha=0.5)

        # Draw the car at center
        draw_car(ax, 0, 0, theta0, length=0.5, color="cyan")

        ax.set_xlim(-EXTENT, EXTENT)
        ax.set_ylim(-EXTENT, EXTENT)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

        title = f"θ = {ang_label}"
        if col == 0:
            title = f"[{chr(65+row*4+col)}]  θ = {ang_label}"
        ax.set_title(title, fontsize=9)

        if col == 0:
            ax.set_ylabel(f"Row {row}: " + ("Isotropic" if row == 0 else "Sub-Riemannian"),
                          fontsize=8, labelpad=4)

# --- Row 2 left: marginal density comparison ---

ax_marg_iso = fig.add_subplot(3, 4, 9)
ax_marg_sr  = fig.add_subplot(3, 4, 10)

print("Computing marginal densities (integrating over θ)...")
center0 = [0.0, 0.0, 0.0]
Z_iso = marginal_density(center0, A_ISO)
Z_sr  = marginal_density(center0, A_SR)

vmax_iso = Z_iso.max()
vmax_sr  = Z_sr.max()

ax_marg_iso.contourf(x_vec, y_vec, Z_iso, levels=30, cmap=CMAP, vmin=0, vmax=vmax_iso)
ax_marg_iso.contour (x_vec, y_vec, Z_iso, levels=8,  colors="white",
                     linewidths=0.5, alpha=0.5)
draw_car(ax_marg_iso, 0, 0, 0.0, color="cyan")
ax_marg_iso.set_xlim(-EXTENT, EXTENT)
ax_marg_iso.set_ylim(-EXTENT, EXTENT)
ax_marg_iso.set_aspect("equal")
ax_marg_iso.set_xticks([])
ax_marg_iso.set_yticks([])
ax_marg_iso.set_title("[I] Marginal ∫dθ — Isotropic", fontsize=9)

ax_marg_sr.contourf(x_vec, y_vec, Z_sr, levels=30, cmap=CMAP, vmin=0, vmax=vmax_sr)
ax_marg_sr.contour (x_vec, y_vec, Z_sr, levels=8,  colors="white",
                    linewidths=0.5, alpha=0.5)
draw_car(ax_marg_sr, 0, 0, 0.0, color="cyan")
ax_marg_sr.set_xlim(-EXTENT, EXTENT)
ax_marg_sr.set_ylim(-EXTENT, EXTENT)
ax_marg_sr.set_aspect("equal")
ax_marg_sr.set_xticks([])
ax_marg_sr.set_yticks([])
ax_marg_sr.set_title("[J] Marginal ∫dθ — Sub-Riemannian", fontsize=9)

# --- Row 2 right: configuration cloud (quiver) for SR Gaussian ---

ax_quiv = fig.add_subplot(3, 4, (11, 12))

print("Computing configuration cloud...")
N_QUIV = 20   # grid resolution for quiver
N_ANG_Q = 8   # angular samples
x_q = np.linspace(-EXTENT * 0.8, EXTENT * 0.8, N_QUIV)
y_q = np.linspace(-EXTENT * 0.8, EXTENT * 0.8, N_QUIV)
theta_q = np.linspace(-np.pi, np.pi, N_ANG_Q, endpoint=False)

h_quiv = jnp.array([0.0, 0.0, 0.0])
max_rho = 0.0

# First pass: find max density for normalization
rho_grid = np.zeros((N_QUIV, N_QUIV, N_ANG_Q))
for ti, th in enumerate(theta_q):
    G = jnp.array(
        np.column_stack([
            np.tile(x_q, N_QUIV),
            np.repeat(y_q, N_QUIV),
            np.full(N_QUIV * N_QUIV, float(th)),
        ]),
        dtype=jnp.float32,
    )
    rho_vals = np.array(jax.vmap(lambda g: eval_se2_rho_single(g, h_quiv, A_SR))(G))
    rho_grid[:, :, ti] = rho_vals.reshape(N_QUIV, N_QUIV)

max_rho = rho_grid.max()

# Plot: at each (x,y), draw all θ arrows with opacity ∝ density
ax_quiv.set_facecolor("k")
cmap_q = plt.get_cmap("plasma")

for ti, th in enumerate(theta_q):
    for xi, xv in enumerate(x_q):
        for yi, yv in enumerate(y_q):
            rho_here = rho_grid[xi, yi, ti]
            alpha    = float(rho_here / (max_rho + 1e-12))
            if alpha < 0.02:
                continue
            color = cmap_q(alpha)
            dx = 0.18 * np.cos(th)
            dy = 0.18 * np.sin(th)
            ax_quiv.annotate(
                "", xy=(xv + dx, yv + dy), xytext=(xv, yv),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=0.8, mutation_scale=8 * alpha),
            )

draw_car(ax_quiv, 0, 0, 0.0, length=0.6, color="cyan", lw=2.5)
ax_quiv.set_xlim(-EXTENT * 0.85, EXTENT * 0.85)
ax_quiv.set_ylim(-EXTENT * 0.85, EXTENT * 0.85)
ax_quiv.set_aspect("equal")
ax_quiv.set_xticks([])
ax_quiv.set_yticks([])
ax_quiv.set_title("[K-L] SE(2) configuration cloud — SR Gaussian\n"
                  "(arrows: all θ, opacity ∝ density)", fontsize=9)

# ── Titles and layout ─────────────────────────────────────────────────────────

fig.suptitle(
    "Dubins Car Wrapped Gaussian on SE(2)\n"
    r"$N_w(g;\,h,A) = N(\mathrm{Log}_h(g);\,0,A)\cdot\left(\frac{\Delta\theta}{"
    r"2\sin(\Delta\theta/2)}\right)^{\!2}$   "
    r"|   Cyan arrow = center heading",
    fontsize=13, y=1.01,
)

# Add row annotations
fig.text(0.01, 0.82, row_labels[0], fontsize=9, va="center", rotation=90, color="#333")
fig.text(0.01, 0.52, row_labels[1], fontsize=9, va="center", rotation=90, color="#333")
fig.text(0.01, 0.18, "Row 2: marginals + configuration cloud", fontsize=9,
         va="center", rotation=90, color="#333")

plt.tight_layout(rect=[0.02, 0, 1, 1])
out_path = "figures/dubins_splat.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved {out_path}")
