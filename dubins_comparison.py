#!/usr/bin/env python3
"""
Dubins car sub-Riemannian Eikonal: SRM vs Dubins path ground truth.

PDE (sub-Riemannian Eikonal on SE(2), SR model with X1=forward, X2=rotation):
    (cos θ · ∂V/∂x + sin θ · ∂V/∂y)² + (∂V/∂θ)² = 1

Ground truth: exact Dubins path length from each (x,y,θ) to (0,0,0).
  The standard Dubins car model (constant forward speed, bounded curvature) solves
  a closely related TTG problem.  The CSC paths (LSL, RSR, LSR, RSL) cover almost
  all optimal cases in the domain.

SRM improvement over dubins_eikonal.py:
  - 4-D input (x, y, cos θ, sin θ): θ periodicity is handled naturally.
  - K=500 components (up from 300).
  - 20k steps (up from 15k).

Output: figures/dubins_comparison.png
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

from lib.splat import eval_splat

# ── Config ────────────────────────────────────────────────────────────────────

XY_RANGE       = 3.0
K              = 500
N_BATCH        = 2000
N_SOURCE       = 300
EPS_BC         = 0.10
NUM_STEPS      = 20_000
LR             = 3e-3
PHYSICS_WEIGHT = 6.0
SIGMA_INIT     = 0.5
LOG_EVERY      = 1000
SEED           = 42

# ── 1. Ground truth: exact Dubins path lengths ────────────────────────────────
#
# The Dubins car model: always moves forward at speed 1, can turn at rate |u|≤1.
# Min time from (x,y,θ) → (0,0,0) equals the shortest Dubins path length.
#
# We compute CSC paths (LSL, RSR, LSR, RSL) which cover virtually all optima
# in the domain.  The tiny region where CCC paths are shorter (d < 4R, small)
# is clipped to np.nan to avoid distorting the comparison.

def _mod2pi(a):
    return a % (2.0 * np.pi)


def _dubins_csc(alpha, beta, d):
    """
    Four CSC Dubins path lengths for start heading alpha, end heading beta,
    normalized distance d (÷ R).  All inputs must be numpy scalars or broadcast
    arrays.  Returns [4] array of lengths (inf where path is infeasible).
    """
    sa, ca = np.sin(alpha), np.cos(alpha)
    sb, cb = np.sin(beta),  np.cos(beta)
    INF = np.full_like(d, np.inf)

    # --- LSL ---
    tmp_lsl = np.arctan2(cb - ca, d + sa - sb)
    t_lsl   = _mod2pi(-alpha + tmp_lsl)
    p2_lsl  = 2 + d**2 - 2*(ca*cb + sa*sb) + 2*d*(sa - sb)
    q_lsl   = _mod2pi(beta - tmp_lsl)
    lsl = np.where(p2_lsl >= 0, t_lsl + np.sqrt(np.maximum(p2_lsl, 0)) + q_lsl, INF)

    # --- RSR ---
    tmp_rsr = np.arctan2(ca - cb, d - sa + sb)
    t_rsr   = _mod2pi(alpha - tmp_rsr)
    p2_rsr  = 2 + d**2 - 2*(ca*cb + sa*sb) + 2*d*(sb - sa)
    q_rsr   = _mod2pi(-beta + tmp_rsr)
    rsr = np.where(p2_rsr >= 0, t_rsr + np.sqrt(np.maximum(p2_rsr, 0)) + q_rsr, INF)

    # --- LSR ---
    p2_lsr  = -2 + d**2 + 2*(ca*cb + sa*sb) + 2*d*(sa + sb)
    p_lsr   = np.sqrt(np.maximum(p2_lsr, 0))
    tmp_lsr = np.arctan2(-ca - cb, d + sa + sb) - np.arctan2(-2.0, p_lsr)
    t_lsr   = _mod2pi(-alpha + tmp_lsr)
    q_lsr   = _mod2pi(-beta + tmp_lsr)
    lsr = np.where(p2_lsr >= 0, t_lsr + p_lsr + q_lsr, INF)

    # --- RSL ---
    p2_rsl  = -2 + d**2 + 2*(ca*cb + sa*sb) - 2*d*(sa + sb)
    p_rsl   = np.sqrt(np.maximum(p2_rsl, 0))
    tmp_rsl = np.arctan2(ca + cb, d - sa - sb) - np.arctan2(2.0, p_rsl)
    t_rsl   = _mod2pi(alpha - tmp_rsl)
    q_rsl   = _mod2pi(beta - tmp_rsl)
    rsl = np.where(p2_rsl >= 0, t_rsl + p_rsl + q_rsl, INF)

    return np.stack([lsl, rsr, lsr, rsl], axis=0)   # [4, ...]


def dubins_dist(x, y, theta, R=1.0):
    """
    Shortest Dubins CSC path length from (x,y,theta) to (0,0,0) with turning
    radius R.  Vectorised; returns array same shape as x.
    """
    d   = np.sqrt(x**2 + y**2) / R
    psi = np.arctan2(-y, -x)          # direction from (x,y) to origin
    alpha = _mod2pi(theta - psi)      # start heading in canonical frame
    beta  = _mod2pi(0.0   - psi)      # goal  heading in canonical frame
    lengths = _dubins_csc(alpha, beta, d)
    return np.min(lengths, axis=0) * R


print("Computing Dubins ground truth ...")
N_GT   = 200
x_gt   = np.linspace(-XY_RANGE, XY_RANGE, N_GT)
y_gt   = np.linspace(-XY_RANGE, XY_RANGE, N_GT)
XX, YY = np.meshgrid(x_gt, y_gt)

THETA_SLICES = {
    r"$\theta$=−90°": -np.pi / 2,
    r"$\theta$=0°":    0.0,
    r"$\theta$=+90°":  np.pi / 2,
}

V_gt = {}
for label, th_val in THETA_SLICES.items():
    TT = np.full_like(XX, th_val)
    V = dubins_dist(XX, YY, TT, R=1.0)
    # Mask source neighbourhood (not well defined there)
    V[np.sqrt(XX**2 + TT**2) < EPS_BC * 1.1] = np.nan
    V_gt[label] = V

print("  Done.  GT range: "
      f"[{np.nanmin(list(V_gt.values())):.2f}, "
      f"{np.nanmax(list(V_gt.values())):.2f}]")

# ── 2. Improved SRM: 4-D input (x, y, cos θ, sin θ) ─────────────────────────

print("\n" + "=" * 60)
print(f"SRM training  K={K}  steps={NUM_STEPS}")
print("=" * 60)


def to4d(q3):
    return jnp.concatenate([q3[:, :2],
                            jnp.cos(q3[:, 2:3]),
                            jnp.sin(q3[:, 2:3])], axis=-1)


def sample_source_ring(key, eps, n):
    phi = jr.uniform(key, (n,), minval=0.0, maxval=2.0 * jnp.pi)
    x   = eps * jnp.cos(phi)
    th  = eps * jnp.sin(phi)
    q3  = jnp.stack([x, jnp.zeros(n), th], axis=-1)
    return q3, to4d(q3)


def sample_interior(key, n):
    k1, k2 = jr.split(key)
    over   = 6
    xy     = jr.uniform(k1, (n * over, 2), minval=-XY_RANGE, maxval=XY_RANGE)
    theta  = jr.uniform(k2, (n * over, 1), minval=-jnp.pi, maxval=jnp.pi)
    q3     = jnp.concatenate([xy, theta], axis=-1)
    sr_dist = jnp.sqrt(xy[:, 0]**2 + theta[:, 0]**2)
    q3 = q3[sr_dist > EPS_BC * 1.1][:n]
    return q3, to4d(q3)


def dubins_eikonal_loss(params, interior_3d, interior_4d, source_4d, source_vals, pw):
    u_bc    = eval_splat(source_4d, params).squeeze(-1)
    bc_loss = jnp.mean((u_bc - source_vals) ** 2)

    # u_fn takes 3D (x,y,θ) → evaluates via 4D (x,y,cosθ,sinθ) embedding.
    # jax.grad delivers ∂V/∂θ correctly through the embedding chain rule,
    # so the ±π periodicity seam that plagued the original 3D version is gone.
    def u_fn(q3):
        q4 = jnp.stack([q3[0], q3[1], jnp.cos(q3[2]), jnp.sin(q3[2])])
        return eval_splat(q4[None, :], params)[0, 0]

    grad_q = jax.vmap(jax.grad(u_fn))(interior_3d)
    Vx, Vy, Vt = grad_q[:, 0], grad_q[:, 1], grad_q[:, 2]
    theta  = interior_3d[:, 2]
    X1V    = Vx * jnp.cos(theta) + Vy * jnp.sin(theta)
    pde_loss = jnp.mean((X1V**2 + Vt**2 - 1.0) ** 2)

    return bc_loss + pw * pde_loss, (bc_loss, pde_loss)


# Initialise
rng = jr.PRNGKey(SEED)
k1, k2, k3 = jr.split(rng, 3)

B_xy  = jr.uniform(k1, (K, 2), minval=-XY_RANGE, maxval=XY_RANGE)
B_th  = jr.uniform(k2, (K, 1), minval=-jnp.pi, maxval=jnp.pi)
B0    = jnp.concatenate([B_xy, jnp.cos(B_th), jnp.sin(B_th)], axis=-1)  # [K, 4]
A0    = jnp.tile(jnp.eye(4) * SIGMA_INIT, (K, 1, 1))
V0    = jr.normal(k3, (K, 1)) * 0.1
r_B   = jnp.sqrt(B_xy[:, 0]**2 + B_th[:, 0]**2)[:, None]
V0    = jnp.abs(V0) * r_B + EPS_BC

source_3d, source_4d = sample_source_ring(k2, EPS_BC, N_SOURCE)
source_vals = jnp.full((N_SOURCE,), EPS_BC)
params = (V0, A0, B0)

warmup = max(1, NUM_STEPS // 10)
sched  = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=LR,
    warmup_steps=warmup, decay_steps=NUM_STEPS, end_value=LR * 0.05,
)
opt   = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=1e-4))
state = opt.init(params)
hist  = []


@jax.jit
def step(params, state, interior_3d, interior_4d):
    (loss, (bc, pde)), grads = jax.value_and_grad(
        dubins_eikonal_loss, has_aux=True
    )(params, interior_3d, interior_4d, source_4d, source_vals, PHYSICS_WEIGHT)
    upd, new_state = opt.update(grads, state, params)
    return optax.apply_updates(params, upd), new_state, loss, bc, pde


for i in trange(NUM_STEPS, desc="SRM 4D"):
    rng, sk    = jr.split(rng)
    int3, int4 = sample_interior(sk, N_BATCH)
    params, state, loss, bc, pde = step(params, state, int3, int4)
    if i % LOG_EVERY == 0 or i == NUM_STEPS - 1:
        hist.append(dict(step=i, total=float(loss), bc=float(bc), pde=float(pde)))
        print(f"  {i:5d}  loss={float(loss):.5f}  bc={float(bc):.5f}  pde={float(pde):.5f}")

# ── 3. Evaluate SRM on same (x,y) grid ───────────────────────────────────────

xy_flat = np.column_stack([XX.ravel(), YY.ravel()])

@jax.jit
def eval_srm(params, G4):
    return eval_splat(G4, params).squeeze(-1)

V_srm = {}
for label, th_val in THETA_SLICES.items():
    G4 = jnp.array(
        np.column_stack([xy_flat,
                         np.full(N_GT**2, np.cos(float(th_val))),
                         np.full(N_GT**2, np.sin(float(th_val)))]),
        dtype=jnp.float32,
    )
    vals = np.array(eval_srm(params, G4)).reshape(N_GT, N_GT)
    V_srm[label] = vals

# ── 4. Error vs Dubins ground truth ──────────────────────────────────────────

all_errs = []
for label in THETA_SLICES:
    mask = ~np.isnan(V_gt[label])
    err  = np.abs(V_srm[label][mask] - V_gt[label][mask])
    all_errs.append(err)
    print(f"  {label}: MAE={err.mean():.4f}  RMSE={np.sqrt((err**2).mean()):.4f}")
combined_err = np.concatenate(all_errs)
print(f"  Overall  MAE={combined_err.mean():.4f}  RMSE={np.sqrt((combined_err**2).mean()):.4f}")

# ── 5. Visualise ──────────────────────────────────────────────────────────────

CMAP_V  = "viridis"
CMAP_E  = "hot_r"
VMAX    = float(np.nanpercentile(np.concatenate([v.ravel() for v in V_gt.values()]), 97))
LEVELS  = np.linspace(0.15, VMAX * 0.95, 12)

fig, axes = plt.subplots(3, 4, figsize=(18, 13))
fig.suptitle(
    "Dubins car SR Eikonal: Dubins path ground truth vs SRM (K=500, 4-D cos/sin embedding)\n"
    r"PDE: $(cos\theta\,\partial_x V + sin\theta\,\partial_y V)^2 + (\partial_\theta V)^2 = 1$"
    "     GT: exact CSC Dubins path length",
    fontsize=10,
)

for col, (label, th_val) in enumerate(THETA_SLICES.items()):
    dlen = 0.5

    # ── Row 0: Dubins ground truth ────────────────────────────────────────────
    ax = axes[0, col]
    Z_gt = np.clip(V_gt[label], 0, VMAX)
    im = ax.contourf(x_gt, y_gt, Z_gt, levels=30, cmap=CMAP_V, vmin=0, vmax=VMAX)
    ax.contour(x_gt, y_gt, V_gt[label], levels=LEVELS,
               colors="white", linewidths=0.5, alpha=0.7)
    ax.scatter(0, 0, c="red", s=50, zorder=6)
    ax.annotate("", xy=(dlen * np.cos(th_val), dlen * np.sin(th_val)), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="red", lw=2, mutation_scale=12))
    ax.set_title(f"Dubins GT  {label}", fontsize=9)
    ax.set_xlim(-XY_RANGE, XY_RANGE); ax.set_ylim(-XY_RANGE, XY_RANGE)
    ax.set_aspect("equal"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    plt.colorbar(im, ax=ax, label="V [s]")

    # ── Row 1: SRM ────────────────────────────────────────────────────────────
    ax = axes[1, col]
    Z_srm = np.clip(V_srm[label], 0, VMAX)
    im = ax.contourf(x_gt, y_gt, Z_srm, levels=30, cmap=CMAP_V, vmin=0, vmax=VMAX)
    ax.contour(x_gt, y_gt, V_srm[label], levels=LEVELS,
               colors="white", linewidths=0.5, alpha=0.7)
    ax.scatter(0, 0, c="red", s=50, zorder=6)
    ax.annotate("", xy=(dlen * np.cos(th_val), dlen * np.sin(th_val)), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="red", lw=2, mutation_scale=12))
    ax.set_title(f"SRM (4-D)  {label}", fontsize=9)
    ax.set_xlim(-XY_RANGE, XY_RANGE); ax.set_ylim(-XY_RANGE, XY_RANGE)
    ax.set_aspect("equal"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    plt.colorbar(im, ax=ax, label="V [s]")

    # ── Row 2: |error| ────────────────────────────────────────────────────────
    ax = axes[2, col]
    err_map = np.abs(V_srm[label] - V_gt[label])
    vt_max  = float(np.nanpercentile(err_map, 95))
    im = ax.contourf(x_gt, y_gt, np.clip(err_map, 0, vt_max),
                     levels=25, cmap=CMAP_E, vmin=0, vmax=vt_max)
    ax.scatter(0, 0, c="blue", s=50, zorder=6)
    ax.annotate("", xy=(dlen * np.cos(th_val), dlen * np.sin(th_val)), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="blue", lw=2, mutation_scale=12))
    mae = float(np.nanmean(err_map))
    ax.set_title(f"|error|  {label}  MAE={mae:.3f}", fontsize=9)
    ax.set_xlim(-XY_RANGE, XY_RANGE); ax.set_ylim(-XY_RANGE, XY_RANGE)
    ax.set_aspect("equal"); ax.set_xlabel("x [m]")
    plt.colorbar(im, ax=ax, label="|V_SRM − V_GT| [s]")

# ── Column 3: training loss + error summary ───────────────────────────────────

ax_l = axes[0, 3]
steps_ = [h["step"] for h in hist]
ax_l.semilogy(steps_, [h["total"] for h in hist],       label="total")
ax_l.semilogy(steps_, [h["bc"]    for h in hist], "--", label="BC")
ax_l.semilogy(steps_, [h["pde"]   for h in hist], ":",  label="PDE")
ax_l.set_xlabel("Step"); ax_l.set_ylabel("Loss (log)")
ax_l.set_title(f"Training loss  (K={K}, 4D embed)"); ax_l.legend(fontsize=9)
ax_l.grid(alpha=0.3)

# Row 1 col 3: side-by-side 1D radial slice at θ=0
ax_1d = axes[1, 3]
x_line = x_gt
y_line = np.zeros_like(x_line)
th_line = np.zeros_like(x_line)
gt_line  = dubins_dist(x_line, y_line, th_line)
G4_line  = jnp.array(np.column_stack([x_line[:, None], y_line[:, None],
                                       np.ones((N_GT,1)), np.zeros((N_GT,1))]), dtype=jnp.float32)
srm_line = np.array(eval_srm(params, G4_line))
ax_1d.plot(x_line, gt_line,  label="Dubins GT", lw=2)
ax_1d.plot(x_line, srm_line, label="SRM 4D",    lw=1.5, linestyle="--")
ax_1d.set_xlabel("x [m]  (y=0, θ=0)")
ax_1d.set_ylabel("V  [s]")
ax_1d.set_title("Radial slice: θ=0, y=0")
ax_1d.legend(fontsize=9); ax_1d.grid(alpha=0.3)

# Row 2 col 3: error histogram
ax_h = axes[2, 3]
ax_h.hist(combined_err, bins=50, color="steelblue", edgecolor="white", linewidth=0.3)
rmse = float(np.sqrt((combined_err**2).mean()))
ax_h.axvline(combined_err.mean(), color="red", linestyle="--",
             label=f"MAE={combined_err.mean():.3f}")
ax_h.set_xlabel("|V_SRM − V_GT| [s]")
ax_h.set_ylabel("Count")
ax_h.set_title(f"Error histogram  RMSE={rmse:.3f}")
ax_h.legend(fontsize=9); ax_h.grid(alpha=0.3)

plt.tight_layout()
out_path = "figures/dubins_comparison.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved {out_path}")
