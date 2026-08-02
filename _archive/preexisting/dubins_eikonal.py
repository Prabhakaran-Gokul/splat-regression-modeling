#!/usr/bin/env python3
"""
Dubins car sub-Riemannian Eikonal using standard (Euclidean) Gaussian SRM.

PDE (sub-Riemannian Eikonal on SE(2)):
    (cos θ · ∂V/∂x + sin θ · ∂V/∂y)² + (∂V/∂θ)² = 1
    V(0, 0, 0) = 0   [goal at origin, heading θ=0]

This corresponds to minimum time from (x,y,θ) to origin for a unit-speed
car that can turn at rate |u| ≤ 1 simultaneously with forward motion.

The model is a flat 3-D Gaussian SRM in (x, y, θ) ∈ R³.  θ is treated as
a plain Euclidean coordinate, so periodicity at θ=±π is NOT enforced.
The hypothesis: sharp directional features ("banana" reachable sets) and
the θ=±π seam will be hard to represent with isotropic Gaussian basis
functions.

Output: figures/dubins_eikonal.png
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

K              = 300          # splat components
N_BATCH        = 2000
N_SOURCE       = 300
EPS            = 0.10         # source ring radius  (≈ SR distance from ring to origin)
XY_RANGE       = 3.0          # (x,y) domain half-width [m]
NUM_STEPS      = 15_000
LR             = 3e-3
PHYSICS_WEIGHT = 5.0
SIGMA_INIT     = 0.5          # initial Gaussian width [m or rad]
LOG_EVERY      = 500
SEED           = 42

# ── Sampling ──────────────────────────────────────────────────────────────────

def sample_source_ring(key, eps, n):
    """
    n points on the ring sqrt(x²+θ²) = eps, y=0.
    x and θ are the two directly accessible directions from origin
    (X₁ = forward, X₂ = rotation), so this ring ≈ SR ball of radius eps.
    """
    phi = jr.uniform(key, (n,), minval=0.0, maxval=2.0 * jnp.pi)
    x   = eps * jnp.cos(phi)
    th  = eps * jnp.sin(phi)
    return jnp.stack([x, jnp.zeros(n), th], axis=-1)   # [n, 3]


def sample_interior(key, n):
    """n uniform points in [-XY_RANGE,XY_RANGE]² × [-π,π], excluding source region."""
    k1, k2 = jr.split(key)
    over   = 6
    xy     = jr.uniform(k1, (n * over, 2), minval=-XY_RANGE, maxval=XY_RANGE)
    theta  = jr.uniform(k2, (n * over, 1), minval=-jnp.pi, maxval=jnp.pi)
    q      = jnp.concatenate([xy, theta], axis=-1)
    # Exclude a cylinder around the source ring
    sr_dist = jnp.sqrt(xy[:, 0] ** 2 + theta[:, 0] ** 2)
    return q[sr_dist > EPS * 1.1][:n]


# ── PINN loss ─────────────────────────────────────────────────────────────────

def dubins_eikonal_loss(params, interior, source_pts, source_vals, pw):
    """
    L = L_bc + pw · L_pde
    L_pde: (X₁V)² + (X₂V)² = 1
      X₁V = cos θ · V_x + sin θ · V_y   (forward directional derivative)
      X₂V = V_θ                           (rotation)
    """
    u_bc    = eval_splat(source_pts, params).squeeze(-1)
    bc_loss = jnp.mean((u_bc - source_vals) ** 2)

    def u_fn(q):
        return eval_splat(q[None, :], params)[0, 0]

    grad_q   = jax.vmap(jax.grad(u_fn))(interior)       # [n, 3]
    Vx       = grad_q[:, 0]
    Vy       = grad_q[:, 1]
    Vt       = grad_q[:, 2]
    theta    = interior[:, 2]

    X1V      = Vx * jnp.cos(theta) + Vy * jnp.sin(theta)
    pde_loss = jnp.mean((X1V ** 2 + Vt ** 2 - 1.0) ** 2)

    return bc_loss + pw * pde_loss, (bc_loss, pde_loss)


# ── Initialise ────────────────────────────────────────────────────────────────

rng    = jr.PRNGKey(SEED)
k1, k2, k3 = jr.split(rng, 3)

# Centers: spread uniformly over domain
B0_xy  = jr.uniform(k1, (K, 2), minval=-XY_RANGE, maxval=XY_RANGE)
B0_th  = jr.uniform(k2, (K, 1), minval=-jnp.pi, maxval=jnp.pi)
B0     = jnp.concatenate([B0_xy, B0_th], axis=-1)      # [K, 3]
A0     = jnp.tile(jnp.eye(3) * SIGMA_INIT, (K, 1, 1))  # [K, 3, 3]
V0     = jr.normal(k3, (K, 1)) * 0.1

# Warm-start: scale weights by approximate SR distance from origin
r_B    = jnp.sqrt(B0[:, 0] ** 2 + B0[:, 2] ** 2)[:, None]   # sqrt(x²+θ²)
V0     = jnp.abs(V0) * r_B + EPS

source_pts  = sample_source_ring(k2, EPS, N_SOURCE)
source_vals = jnp.full((N_SOURCE,), EPS)

params = (V0, A0, B0)

print(f"Dubins Eikonal SRM  K={K}  N_BATCH={N_BATCH}  "
      f"steps={NUM_STEPS}  physics_weight={PHYSICS_WEIGHT}")

# ── Training ──────────────────────────────────────────────────────────────────

warmup = max(1, NUM_STEPS // 10)
sched  = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=LR,
    warmup_steps=warmup, decay_steps=NUM_STEPS, end_value=LR * 0.05,
)
opt    = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(sched, weight_decay=1e-4),
)
state  = opt.init(params)
hist   = []


@jax.jit
def step(params, state, interior):
    (loss, (bc, pde)), grads = jax.value_and_grad(
        dubins_eikonal_loss, has_aux=True
    )(params, interior, source_pts, source_vals, PHYSICS_WEIGHT)
    upd, new_state = opt.update(grads, state, params)
    return optax.apply_updates(params, upd), new_state, loss, bc, pde


for i in trange(NUM_STEPS, desc="dubins eikonal"):
    rng, sk  = jr.split(rng)
    interior = sample_interior(sk, N_BATCH)
    params, state, loss, bc, pde = step(params, state, interior)

    if i % LOG_EVERY == 0 or i == NUM_STEPS - 1:
        hist.append(dict(step=i, total=float(loss), bc=float(bc), pde=float(pde)))
        print(f"  {i:5d}  loss={float(loss):.5f}  "
              f"bc={float(bc):.5f}  pde={float(pde):.5f}")

# ── Evaluate on a 2-D (x,y) grid at several fixed θ ──────────────────────────

N_VIS   = 200
x_vis   = np.linspace(-XY_RANGE, XY_RANGE, N_VIS)
y_vis   = np.linspace(-XY_RANGE, XY_RANGE, N_VIS)
XX, YY  = np.meshgrid(x_vis, y_vis)
xy_flat = np.column_stack([XX.ravel(), YY.ravel()])    # [N², 2]

THETA_SLICES = {
    r"$\theta=-90°$": -np.pi / 2,
    r"$\theta=0°$":    0.0,
    r"$\theta=90°$":   np.pi / 2,
    r"$\theta=180°$":  np.pi * 0.999,   # near ±π seam
}

V_slices   = {}   # value function slices
Vt_slices  = {}   # ∂V/∂θ slices

@jax.jit
def eval_and_grad_theta(params, G):
    def u_fn(q):
        return eval_splat(q[None, :], params)[0, 0]
    val  = jax.vmap(u_fn)(G)
    grad = jax.vmap(jax.grad(u_fn))(G)
    return val, grad

for label, theta_val in THETA_SLICES.items():
    G = jnp.array(
        np.column_stack([xy_flat, np.full(N_VIS ** 2, float(theta_val))]),
        dtype=jnp.float32,
    )
    val, grad = eval_and_grad_theta(params, G)
    V_slices[label]  = np.array(val).reshape(N_VIS, N_VIS)
    Vt_slices[label] = np.array(grad[:, 2]).reshape(N_VIS, N_VIS)

# ── Visualise ─────────────────────────────────────────────────────────────────

CMAP_V  = "viridis"
CMAP_VT = "RdBu"
VMAX    = np.pi + 1.0    # rough maximum SR distance in domain
LEVELS  = np.linspace(EPS, VMAX, 15)

fig = plt.figure(figsize=(18, 10))
fig.suptitle(
    "Dubins Car SR Eikonal — standard Gaussian SRM\n"
    r"PDE: $(cos\,\theta\;\partial_x V + sin\,\theta\;\partial_y V)^2 + (\partial_\theta V)^2 = 1$",
    fontsize=12,
)

# ── Row 0: value function slices ──────────────────────────────────────────────
for col, (label, theta_val) in enumerate(THETA_SLICES.items()):
    ax = fig.add_subplot(2, 4, col + 1)
    Z  = V_slices[label]
    im = ax.contourf(x_vis, y_vis, Z, levels=30, cmap=CMAP_V, vmin=0, vmax=VMAX)
    ax.contour(x_vis, y_vis, Z, levels=LEVELS, colors="white", linewidths=0.5, alpha=0.6)
    # Goal marker + heading arrow
    ax.scatter(0, 0, c="red", s=60, zorder=6, label="goal")
    dlen = 0.5
    ax.annotate("", xy=(dlen * np.cos(theta_val), dlen * np.sin(theta_val)),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="red", lw=2, mutation_scale=15))
    ax.set_title(f"V(x,y)  {label}", fontsize=10)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_xlim(-XY_RANGE, XY_RANGE)
    ax.set_ylim(-XY_RANGE, XY_RANGE)
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax, label="V [s]")

# ── Row 1: ∂V/∂θ slices + training loss ──────────────────────────────────────

for col, (label, theta_val) in enumerate(THETA_SLICES.items()):
    if col >= 3:
        break
    ax  = fig.add_subplot(2, 4, 5 + col)
    Zt  = Vt_slices[label]
    vt_abs = np.max(np.abs(Zt))
    im2 = ax.contourf(x_vis, y_vis, Zt, levels=30, cmap=CMAP_VT,
                      vmin=-vt_abs, vmax=vt_abs)
    ax.contour(x_vis, y_vis, Zt, levels=[0], colors="k", linewidths=1.2)
    ax.scatter(0, 0, c="red", s=60, zorder=6)
    ax.annotate("", xy=(dlen * np.cos(theta_val), dlen * np.sin(theta_val)),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="red", lw=2, mutation_scale=15))
    ax.set_title(rf"$\partial_\theta V$  {label}  [black = switching curve]", fontsize=9)
    ax.set_xlabel("x [m]")
    ax.set_xlim(-XY_RANGE, XY_RANGE)
    ax.set_ylim(-XY_RANGE, XY_RANGE)
    ax.set_aspect("equal")
    plt.colorbar(im2, ax=ax)

# Training loss
ax_loss = fig.add_subplot(2, 4, 8)
steps_  = [h["step"] for h in hist]
ax_loss.semilogy(steps_, [h["total"] for h in hist],       label="total")
ax_loss.semilogy(steps_, [h["bc"]    for h in hist], "--", label="BC")
ax_loss.semilogy(steps_, [h["pde"]   for h in hist], ":",  label="PDE")
ax_loss.set_xlabel("Step")
ax_loss.set_ylabel("Loss (log)")
ax_loss.set_title("Training loss")
ax_loss.legend(fontsize=9)
ax_loss.grid(alpha=0.3)

plt.tight_layout()
out_path = "figures/dubins_eikonal.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved {out_path}")
