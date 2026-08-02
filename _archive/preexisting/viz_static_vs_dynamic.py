#!/usr/bin/env python3
"""
Visual comparison: static (8 000 fixed pts) vs dynamic (1 000 fresh pts/step)
collocation for the uniform-speed Eikonal |∇u| = 1, u = ‖x‖.

Dimensions: d=2 (top row) and d=3 xy midplane slice (bottom row).
Each row: [static pred | dynamic pred | analytical | |error| static | |error| dynamic]

Output: static_vs_dynamic.png
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

jax.config.update("jax_platform_name", "gpu")
jax.config.update("jax_enable_x64", False)

from eikonal_splat import (
    source_sphere_nd,
    _uniform_collocation_nd,
    init_splat_params,
    train_eikonal_splat,
)
from lib.splat import eval_splat

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

K         = 200
N_INT     = 8000      # static strategy — fixed collocation points
N_BATCH   = 1000      # dynamic strategy — fresh points per step
NUM_STEPS = 5000
LR        = 5e-3
EPS       = 0.10
N_SPHERE  = 200
SEED      = 42
SPEED_FN  = lambda x: jnp.ones((x.shape[0], 1))
Ng        = 80        # grid resolution for heatmaps


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------

def train_one(key, d, dynamic=False):
    domain = [(-1.0, 1.0)] * d
    x_src  = jnp.zeros(d)
    src_pts, src_vals = source_sphere_nd(x_src, EPS, N_SPHERE, c_at_src=1.0)

    key, sk = jr.split(key)
    params = init_splat_params(sk, K, d, domain, scale=0.25)

    if dynamic:
        def resample_fn(sk):
            return _uniform_collocation_nd(sk, N_BATCH, domain,
                                           exclude_center=[0.] * d, min_dist=EPS)
        key, train_key  = jr.split(key)
        interior_pts    = src_pts   # placeholder — not used by train loop
    else:
        key, sk          = jr.split(key)
        interior_pts     = _uniform_collocation_nd(sk, N_INT, domain,
                                                    exclude_center=[0.] * d, min_dist=EPS)
        resample_fn = None
        train_key   = None

    params, history = train_eikonal_splat(
        params, interior_pts, src_pts, src_vals, SPEED_FN,
        num_steps=NUM_STEPS, lr=LR, physics_weight=1.0,
        log_interval=999_999, resample_fn=resample_fn, key=train_key,
    )
    return params, history


# ---------------------------------------------------------------------------
# Slice evaluation on the x1-x2 plane (all other coords = 0)
# ---------------------------------------------------------------------------

def eval_xy_slice(params, d, Ng=80):
    g = jnp.linspace(-1.0, 1.0, Ng)
    Gx, Gy = jnp.meshgrid(g, g)
    xy     = jnp.stack([Gx.ravel(), Gy.ravel()], axis=1)   # [Ng², 2]
    if d > 2:
        zeros = jnp.zeros((Ng * Ng, d - 2))
        pts   = jnp.concatenate([xy, zeros], axis=1)
    else:
        pts   = xy
    u_pred = eval_splat(pts, params).reshape(Ng, Ng)
    u_true = jnp.sqrt(Gx ** 2 + Gy ** 2)
    return u_pred, u_true, Gx, Gy


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

keys = jr.split(jr.PRNGKey(SEED), 4)

print("Training d=2 static  (8 000 fixed pts) ...")
params_2s, hist_2s = train_one(keys[0], d=2, dynamic=False)

print("Training d=2 dynamic (1 000 pts/step) ...")
params_2d, hist_2d = train_one(keys[1], d=2, dynamic=True)

print("Training d=3 static  (8 000 fixed pts) ...")
params_3s, hist_3s = train_one(keys[2], d=3, dynamic=False)

print("Training d=3 dynamic (1 000 pts/step) ...")
params_3d, hist_3d = train_one(keys[3], d=3, dynamic=True)


# ---------------------------------------------------------------------------
# Evaluate slices
# ---------------------------------------------------------------------------

u2s, u2t, Gx2, Gy2 = eval_xy_slice(params_2s, d=2, Ng=Ng)
u2d, _,   _,   _   = eval_xy_slice(params_2d, d=2, Ng=Ng)

u3s, u3t, Gx3, Gy3 = eval_xy_slice(params_3s, d=3, Ng=Ng)
u3d, _,   _,   _   = eval_xy_slice(params_3d, d=3, Ng=Ng)

mse_2s = float(jnp.mean((u2s - u2t) ** 2))
mse_2d = float(jnp.mean((u2d - u2t) ** 2))
mse_3s = float(jnp.mean((u3s - u3t) ** 2))
mse_3d = float(jnp.mean((u3d - u3t) ** 2))

print(f"\nd=2  static MSE={mse_2s:.3e}  dynamic MSE={mse_2d:.3e}")
print(f"d=3  static MSE={mse_3s:.3e}  dynamic MSE={mse_3d:.3e}")


# ---------------------------------------------------------------------------
# Plot  (2 rows × 5 columns + 1 narrow column for loss curves)
# ---------------------------------------------------------------------------

LEVELS = jnp.linspace(0.1, 1.3, 13)
EXT    = [-1, 1, -1, 1]


def fill_row(axes, u_s, u_d, u_t, Gx, Gy, d, mse_s, mse_d):
    err_s = jnp.abs(u_s - u_t)
    err_d = jnp.abs(u_d - u_t)
    emax  = float(jnp.maximum(jnp.max(err_s), jnp.max(err_d)))
    vmax  = float(jnp.max(u_t))

    cols = [
        (u_s,   "viridis", 0, vmax,  f"Static 8k\nxy-MSE={mse_s:.2e}"),
        (u_d,   "viridis", 0, vmax,  f"Dynamic 1k/step\nxy-MSE={mse_d:.2e}"),
        (u_t,   "viridis", 0, vmax,  r"Analytical $\|\mathbf{x}_{1:2}\|$"),
        (err_s, "Reds",    0, emax,  r"$|$error$|$ — static"),
        (err_d, "Reds",    0, emax,  r"$|$error$|$ — dynamic"),
    ]
    for ax, (data, cmap, vlo, vhi, title) in zip(axes, cols):
        im = ax.imshow(data, origin="lower", extent=EXT,
                       vmin=vlo, vmax=vhi, cmap=cmap, aspect="equal")
        if cmap == "viridis":
            ax.contour(Gx, Gy, data, levels=LEVELS,
                       colors="white", linewidths=0.6, alpha=0.7)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=8.5)
        ax.set_xlabel("$x_1$", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel(f"d={d}   $x_2$", fontsize=9)


fig, axes = plt.subplots(2, 5, figsize=(22, 9))

fill_row(axes[0], u2s, u2d, u2t, Gx2, Gy2, d=2,
         mse_s=mse_2s, mse_d=mse_2d)
fill_row(axes[1], u3s, u3d, u3t, Gx3, Gy3, d=3,
         mse_s=mse_3s, mse_d=mse_3d)

# Row labels
for row, (d, mse_s, mse_d) in enumerate([(2, mse_2s, mse_2d),
                                           (3, mse_3s, mse_3d)]):
    axes[row][0].set_ylabel(
        f"d={d}   $x_2$\n(static MSE={mse_s:.2e})", fontsize=8.5
    )

fig.suptitle(
    f"Static (8 000 fixed pts) vs Dynamic (1 000 fresh pts/step) collocation\n"
    f"Eikonal $|\\nabla u|=1$, $u=\\|x\\|$  —  k={K}, {NUM_STEPS} steps, lr={LR}",
    fontsize=12,
)
plt.tight_layout()
plt.savefig("static_vs_dynamic.png", dpi=150, bbox_inches="tight")
print("\nSaved static_vs_dynamic.png")
plt.show()
