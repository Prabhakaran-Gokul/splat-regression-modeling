#!/usr/bin/env python3
"""
Eikonal equation on S² with uniform speed — physics-informed SRM.

    |∇_{S²} u(x)| = 1,    x ∈ S² \ {x_src}
    u(x_src) = 0
    Exact solution: u(x) = arccos(⟨x, x_src⟩)   (geodesic distance)

Density: proper wrapped Gaussian N_w(x; μ, A) = N(Log_μ(x); 0, A) · (θ/sin θ)
with A ∈ R^{2×2} in the tangent plane (textbook pushforward definition).

Output: figures/sphere_eikonal.png
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import optax
from tqdm import trange

jax.config.update("jax_platform_name", "gpu")
jax.config.update("jax_enable_x64", False)

from lib.manifold_splat import eval_sphere_splat, eval_sphere_splat_jit, init_sphere_splat

# ── Config ────────────────────────────────────────────────────────────────────

K              = 200
N_BATCH        = 2000
N_SOURCE       = 300
EPS            = 0.10       # source ring radius [rad ≈ 5.7°]
NUM_STEPS      = 10_000
LR             = 3e-3
PHYSICS_WEIGHT = 5.0
SIGMA_INIT     = 0.35
LOG_EVERY      = 500
SEED           = 42

X_SRC = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)   # north pole

# ── Sampling ──────────────────────────────────────────────────────────────────

def sample_source_ring(key, x_src, eps, n):
    """n points on S² at geodesic distance eps from x_src."""
    t = jr.normal(key, (n, 3))
    t = t - jnp.dot(t, x_src)[:, None] * x_src
    t /= jnp.linalg.norm(t, axis=-1, keepdims=True)
    return jnp.cos(eps) * x_src + jnp.sin(eps) * t


def sample_interior(key, x_src, eps, n):
    """n uniform points on S², strictly outside the source eps-cap."""
    raw = np.array(jr.normal(key, (n * 6, 3)), dtype=np.float32)
    raw /= np.linalg.norm(raw, axis=-1, keepdims=True)
    dist = np.arccos(np.clip(raw @ np.array(x_src), -1.0, 1.0))
    raw  = raw[dist > eps * 1.1][:n]
    return jnp.array(raw)


# ── PINN loss ─────────────────────────────────────────────────────────────────

def eikonal_loss_sphere(params, interior, source_pts, source_vals, pw):
    """
    L = L_bc + pw · L_pde

    L_pde enforces |∇_{S²} u| = 1 at interior collocation points.
    Tangential gradient: ∇_{S²}u = ∇_{R³}u − (∇_{R³}u · x̂) x̂
    """
    u_bc    = eval_sphere_splat(source_pts, params).squeeze(-1)
    bc_loss = jnp.mean((u_bc - source_vals) ** 2)

    def u_fn(x):
        return eval_sphere_splat(x[None, :], params)[0, 0]

    grad_R3  = jax.vmap(jax.grad(u_fn))(interior)
    dot_n    = jnp.sum(grad_R3 * interior, axis=-1, keepdims=True)
    grad_S2  = grad_R3 - dot_n * interior
    gnorm    = jnp.sqrt(jnp.sum(grad_S2 ** 2, axis=-1) + 1e-12)
    pde_loss = jnp.mean((gnorm - 1.0) ** 2)

    return bc_loss + pw * pde_loss, (bc_loss, pde_loss)


# ── Training loop ─────────────────────────────────────────────────────────────

def train(params, source_pts, source_vals, rng):
    warmup = max(1, NUM_STEPS // 10)
    sched  = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=LR,
        warmup_steps=warmup, decay_steps=NUM_STEPS, end_value=LR * 0.05,
    )
    opt   = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(sched, weight_decay=1e-4),
    )
    state = opt.init(params)
    hist  = []

    @jax.jit
    def step(params, state, interior):
        (loss, (bc, pde)), grads = jax.value_and_grad(
            eikonal_loss_sphere, has_aux=True
        )(params, interior, source_pts, source_vals, PHYSICS_WEIGHT)
        upd, new_state = opt.update(grads, state, params)
        return optax.apply_updates(params, upd), new_state, loss, bc, pde

    for i in trange(NUM_STEPS, desc="sphere eikonal"):
        rng, sk  = jr.split(rng)
        interior = sample_interior(sk, X_SRC, EPS, N_BATCH)

        params, state, loss, bc, pde = step(params, state, interior)

        # Retract centers back onto S² after each gradient step
        V, A, B = params
        B       = B / jnp.linalg.norm(B, axis=-1, keepdims=True)
        params  = (V, A, B)

        if i % LOG_EVERY == 0 or i == NUM_STEPS - 1:
            hist.append(dict(step=i, total=float(loss), bc=float(bc), pde=float(pde)))
            print(f"  {i:5d}  loss={float(loss):.5f}  "
                  f"bc={float(bc):.5f}  pde={float(pde):.5f}")

    return params, hist


# ── Initialise ────────────────────────────────────────────────────────────────

rng = jr.PRNGKey(SEED)
k1, k2, k3 = jr.split(rng, 3)

V0, A0, B0 = init_sphere_splat(k1, K, p=1, sigma=SIGMA_INIT)
# Warm-start: scale weights by geodesic distance from source
r_B = jnp.arccos(jnp.clip(B0 @ X_SRC, -1.0, 1.0))[:, None]
V0  = jnp.abs(V0) * r_B + EPS

source_pts  = sample_source_ring(k2, X_SRC, EPS, N_SOURCE)
source_vals = jnp.full((N_SOURCE,), EPS)

print(f"Sphere Eikonal SRM  K={K}  N_BATCH={N_BATCH}  "
      f"steps={NUM_STEPS}  physics_weight={PHYSICS_WEIGHT}")
params, hist = train((V0, A0, B0), source_pts, source_vals, k3)

# ── Evaluate on a lat/lon grid ────────────────────────────────────────────────

GRID_N = 300
lat    = np.linspace(-np.pi / 2, np.pi / 2, GRID_N)
lon    = np.linspace(-np.pi,      np.pi,     GRID_N)
LON_g, LAT_g = np.meshgrid(lon, lat)
X_g = np.stack([np.cos(LAT_g) * np.cos(LON_g),
                np.cos(LAT_g) * np.sin(LON_g),
                np.sin(LAT_g)], axis=-1)          # [N, N, 3]

X_flat  = jnp.array(X_g.reshape(-1, 3), dtype=jnp.float32)
U_srm   = np.array(eval_sphere_splat_jit(X_flat, params)).reshape(GRID_N, GRID_N)
U_exact = np.arccos(np.clip((X_flat @ np.array(X_SRC)).ravel(), -1, 1)
                    ).reshape(GRID_N, GRID_N)
error   = np.abs(U_srm - U_exact)

rmse = float(np.sqrt(np.mean(error ** 2)))
print(f"\nRMSE = {rmse:.4f} rad   max err = {error.max():.4f} rad")

# ── Visualise ─────────────────────────────────────────────────────────────────

LON_deg = np.degrees(lon)
LAT_deg = np.degrees(lat)
LEVELS  = np.linspace(0.2, np.pi - 0.2, 12)
VMIN, VMAX = 0.0, np.pi
CMAP_U = "viridis"
CMAP_E = "hot_r"


def _sphere_contour_lines(U_grid):
    """Extract iso-contour paths as (xs,ys,zs) tuples for 3D plotting."""
    fig_tmp, ax_tmp = plt.subplots()
    cs = ax_tmp.contour(LON_g, LAT_g, U_grid, levels=LEVELS)
    plt.close(fig_tmp)
    lines = []
    for path in cs.get_paths():
        v = path.vertices
        lo, la = v[:, 0], v[:, 1]
        lines.append((np.cos(la)*np.cos(lo),
                      np.cos(la)*np.sin(lo),
                      np.sin(la)))
    return lines


def _plot_sphere(ax3d, U_grid, title, elev=30, azim=-60):
    """Colored sphere surface + white contour rings + red source dot."""
    U_n = np.clip((U_grid - VMIN) / (VMAX - VMIN), 0.0, 1.0)
    fc  = plt.get_cmap(CMAP_U)(U_n)
    ax3d.plot_surface(X_g[..., 0], X_g[..., 1], X_g[..., 2],
                      facecolors=fc, rstride=2, cstride=2,
                      antialiased=False, shade=False)
    for xs, ys, zs in _sphere_contour_lines(U_grid):
        ax3d.plot(xs, ys, zs, "w-", lw=0.8, alpha=0.9)
    src = np.array(X_SRC)
    ax3d.scatter([src[0]], [src[1]], [src[2]], color="red", s=60, zorder=6)
    ax3d.set_title(title, fontsize=10)
    ax3d.set_box_aspect([1, 1, 1])
    ax3d.set_axis_off()
    ax3d.view_init(elev=elev, azim=azim)


def _equirect(ax, U_grid, title):
    im = ax.contourf(LON_deg, LAT_deg, U_grid, levels=40,
                     cmap=CMAP_U, vmin=VMIN, vmax=VMAX)
    ax.contour(LON_deg, LAT_deg, U_grid, levels=LEVELS,
               colors="white", linewidths=0.6, alpha=0.7)
    plt.colorbar(im, ax=ax, label="u  [rad]")
    ax.set_title(title)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")


fig = plt.figure(figsize=(17, 9))
fig.suptitle(
    f"Eikonal SRM on S²  (uniform speed, source = north pole,  "
    f"K={K},  RMSE = {rmse:.3f} rad)\n"
    f"Density: wrapped Gaussian  N_w = N(Log_μ(x); 0, A) · θ/sin θ",
    fontsize=11,
)

# Row 0 ──────────────────────────────────────────────────────────────────────
# Standard globe view (elev=30) comparing SRM vs exact
ax1 = fig.add_subplot(2, 3, 1, projection="3d")
_plot_sphere(ax1, U_srm,   "SRM solution", elev=30, azim=-60)

ax2 = fig.add_subplot(2, 3, 2, projection="3d")
_plot_sphere(ax2, U_exact, "Exact solution", elev=30, azim=-60)

ax3 = fig.add_subplot(2, 3, 3)
steps_ = [h["step"] for h in hist]
ax3.semilogy(steps_, [h["total"] for h in hist],       label="total")
ax3.semilogy(steps_, [h["bc"]    for h in hist], "--", label="BC")
ax3.semilogy(steps_, [h["pde"]   for h in hist], ":",  label="PDE")
ax3.set_xlabel("Step")
ax3.set_ylabel("Loss (log scale)")
ax3.set_title("Training loss")
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)

# Row 1 ──────────────────────────────────────────────────────────────────────
# Equatorial view (elev=0): contour rings appear as equator-like circles
ax4 = fig.add_subplot(2, 3, 4, projection="3d")
_plot_sphere(ax4, U_srm,
             "SRM — side view\n(contour rings as equators)",
             elev=0, azim=0)

ax5 = fig.add_subplot(2, 3, 5)
_equirect(ax5, U_srm, "SRM  (equirectangular)")

ax6 = fig.add_subplot(2, 3, 6)
im6 = ax6.contourf(LON_deg, LAT_deg, error, levels=40, cmap=CMAP_E)
plt.colorbar(im6, ax=ax6, label="|u_SRM − u_exact|  [rad]")
ax6.set_title(f"|error|   RMSE = {rmse:.3f} rad")
ax6.set_xlabel("Longitude (°)")
ax6.set_ylabel("Latitude (°)")

plt.tight_layout()
out_path = "figures/sphere_eikonal.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved {out_path}")
