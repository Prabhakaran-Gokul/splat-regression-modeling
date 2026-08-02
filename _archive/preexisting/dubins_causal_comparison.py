#!/usr/bin/env python3
"""
dubins_causal_comparison.py

Compares causal (GT-TTG-ordered curriculum) vs standard (uniform) PINN training
for the Dubins HJB on SE(2) Wrapped Gaussian SRM.

CAUSAL TRAINING IDEA
--------------------
The Eikonal equation has a causal structure: V(x) depends only on V at
neighbouring states *closer to the goal*.  Standard PINN training violates
this by sampling the PDE residual uniformly across the domain, including
states that are "causally downstream" of states not yet learned correctly.

Causal training fixes this by ordering the PDE collocation points by their
ground-truth TTG.  A schedule expands the training front from near-goal
(small TTG) to far-field (large TTG), ensuring that when the PDE residual
is first applied to a state, the model already has accurate values for all
states with smaller TTG.

This is especially relevant for the Dubins car, which has a genuine jump
discontinuity in V at the cut locus (TTG ≈ 2π).  Causal training means the
front reaches the discontinuity only after the near-goal region is well learned.

EXPERIMENT DESIGN
-----------------
1. Pre-sample a large pool of interior points; compute their GT TTG.
2. Shared BC warm-up (Phase 1) for both models — identical.
3. Fork: from the same post-warmup params:
   - Standard Phase 2: PDE residual on uniform interior points.
   - Causal Phase 2:   PDE residual on expanding-TTG-front points.
4. Compare: MAE tables, error maps, 1D slice through the discontinuity.

Causal TTG schedule (linear):
   ttg_thresh(step) = TTG_START + (TTG_END − TTG_START) × step/STEPS_PINN

Output: figures/dubins_comparison.png
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import optax
from tqdm import trange

jax.config.update("jax_platform_name", "gpu")
jax.config.update("jax_enable_x64", False)

from lib.manifold_splat import eval_se2_splat, init_se2_splat

# ── Config (matches best config from dubins_se2_pinn.py) ─────────────────────

K              = 600
SIGMA          = 1.0
SIGMA_ROT      = 0.5
N_BATCH        = 2048
N_SOURCE       = 600
EPS_BC         = 0.20
XY_RANGE       = 3.0

STEPS_BC       = 48_000
STEPS_PINN     = 26_000
NUM_STEPS      = STEPS_BC + STEPS_PINN

LR             = 5e-4
PHYSICS_WEIGHT = 4.0
BC_WEIGHT      = 15.0
POS_WEIGHT     = 2.0
LOG_EVERY      = 2000
SEED           = 42

# Causal schedule
N_POOL          = 300_000   # candidate pool size
TTG_CAUSAL_START = 2.0      # initial TTG threshold (Phase 2 step 0)
TTG_CAUSAL_END   = 15.0     # final TTG threshold   (Phase 2 last step)

# ── Ground truth ──────────────────────────────────────────────────────────────

def _mod2pi(a):
    return a % (2.0 * np.pi)


def dubins_csc_dist(x, y, theta, R=1.0):
    """Shortest Dubins CSC path length from (x,y,θ) to origin. Vectorised numpy."""
    d     = np.sqrt(x**2 + y**2) / R
    psi   = np.arctan2(-y, -x)
    alpha = _mod2pi(theta - psi)
    beta  = _mod2pi(-psi)
    sa, ca = np.sin(alpha), np.cos(alpha)
    sb, cb = np.sin(beta),  np.cos(beta)
    INF = np.full_like(d, np.inf)

    tmp = np.arctan2(cb - ca, d + sa - sb)
    p2  = 2 + d**2 - 2*(ca*cb + sa*sb) + 2*d*(sa - sb)
    lsl = np.where(p2 >= 0,
                   _mod2pi(-alpha + tmp) + np.sqrt(np.maximum(p2, 0)) + _mod2pi(beta - tmp),
                   INF)
    tmp = np.arctan2(ca - cb, d - sa + sb)
    p2  = 2 + d**2 - 2*(ca*cb + sa*sb) + 2*d*(sb - sa)
    rsr = np.where(p2 >= 0,
                   _mod2pi(alpha - tmp) + np.sqrt(np.maximum(p2, 0)) + _mod2pi(-beta + tmp),
                   INF)
    p2  = -2 + d**2 + 2*(ca*cb + sa*sb) + 2*d*(sa + sb)
    p   = np.sqrt(np.maximum(p2, 0))
    tmp = np.arctan2(-ca - cb, d + sa + sb) - np.arctan2(-2.0, p)
    lsr = np.where(p2 >= 0,
                   _mod2pi(-alpha + tmp) + p + _mod2pi(-beta + tmp),
                   INF)
    p2  = -2 + d**2 + 2*(ca*cb + sa*sb) - 2*d*(sa + sb)
    p   = np.sqrt(np.maximum(p2, 0))
    tmp = np.arctan2(ca + cb, d - sa - sb) - np.arctan2(2.0, p)
    rsl = np.where(p2 >= 0,
                   _mod2pi(alpha - tmp) + p + _mod2pi(beta - tmp),
                   INF)

    return np.min(np.stack([lsl, rsr, lsr, rsl], axis=0), axis=0) * R


# ── Model ─────────────────────────────────────────────────────────────────────

_ALPHA_FIXED = 1.0
_EPS_HEADING = 0.5
_EPS_HJB     = 0.05


def eval_model(G, params):
    """G: [n,3] → [n,1]."""
    V, A, H = params
    srm  = eval_se2_splat(G, (V, A, H))
    x    = G[:, 0];  y = G[:, 1];  theta = G[:, 2]
    r_sq = x**2 + y**2
    r    = jnp.sqrt(r_sq + 1e-6)
    cos_th_psi = -(x * jnp.cos(theta) + y * jnp.sin(theta)) / r
    taper      = r_sq / (r_sq + _EPS_HEADING**2)
    heading    = (jnp.pi / 2.0) * (1.0 - cos_th_psi) * taper
    anchor     = (_ALPHA_FIXED * r + heading)[:, None]
    return anchor + srm


eval_model_jit = jax.jit(eval_model)


# ── Losses ────────────────────────────────────────────────────────────────────

def bc_loss_fn(params, source_pts):
    u = eval_model(source_pts, params).squeeze(-1)
    return jnp.mean(u ** 2)


def pde_loss_fn(params, interior):
    def v_fn(q):
        return eval_model(q[None, :], params)[0, 0]
    grad_q = jax.vmap(jax.grad(v_fn))(interior)
    theta  = interior[:, 2]
    X1V    = grad_q[:, 0] * jnp.cos(theta) + grad_q[:, 1] * jnp.sin(theta)
    X2V    = grad_q[:, 2]
    hjb    = jnp.sqrt(X2V ** 2 + _EPS_HJB ** 2) - X1V - 1.0
    return jnp.mean(hjb ** 2)


def pos_loss_fn(params, interior):
    u = eval_model(interior, params).squeeze(-1)
    return jnp.mean(jax.nn.relu(-u) ** 2)


def phase1_loss(params, source_pts, interior):
    bc  = bc_loss_fn(params, source_pts)
    pos = pos_loss_fn(params, interior)
    return BC_WEIGHT * bc + POS_WEIGHT * pos, (bc, jnp.zeros(()), pos)


def phase2_loss(params, source_pts, interior, pw):
    bc  = bc_loss_fn(params, source_pts)
    pde = pde_loss_fn(params, interior)
    pos = pos_loss_fn(params, interior)
    return BC_WEIGHT * bc + pw * pde + POS_WEIGHT * pos, (bc, pde, pos)


# ── Trainable param helpers ───────────────────────────────────────────────────

def get_trainable(p):
    V, A, H = p
    return (V, H)


def set_trainable(p, t):
    _, A, _ = p
    V, H = t
    return (V, A, H)


# ── Sampling ──────────────────────────────────────────────────────────────────

def sample_source_ball(key, eps, n):
    k1, k2 = jr.split(key)
    dirs = jr.normal(k1, (n, 3))
    dirs = dirs / jnp.linalg.norm(dirs, axis=-1, keepdims=True)
    r    = eps * jr.uniform(k2, (n,)) ** (1.0 / 3.0)
    return dirs * r[:, None]


def sample_interior_jax(key, n):
    k1, k2 = jr.split(key)
    xy    = jr.uniform(k1, (n, 2), minval=-XY_RANGE, maxval=XY_RANGE)
    theta = jr.uniform(k2, (n, 1), minval=-jnp.pi, maxval=jnp.pi)
    q     = jnp.concatenate([xy, theta], axis=-1)
    mask  = jnp.sqrt(jnp.sum(q**2, axis=-1)) > EPS_BC * 1.2
    return q[mask]


# ── Causal pool: pre-sample + sort by GT TTG ─────────────────────────────────

print(f"Pre-sampling causal pool of {N_POOL:,} points and computing GT distances…")
rng_pool = np.random.default_rng(SEED + 1)
pool_xy    = rng_pool.uniform(-XY_RANGE, XY_RANGE, (N_POOL, 2)).astype(np.float32)
pool_theta = rng_pool.uniform(-np.pi, np.pi, (N_POOL, 1)).astype(np.float32)
pool_all   = np.hstack([pool_xy, pool_theta])

# Exclude points inside the source ball
pool_dist_eucl = np.sqrt(np.sum(pool_all**2, axis=1))
pool_valid     = pool_all[pool_dist_eucl > EPS_BC * 1.2]

pool_gt = dubins_csc_dist(pool_valid[:, 0], pool_valid[:, 1], pool_valid[:, 2])
fin_mask  = np.isfinite(pool_gt)
pool_valid = pool_valid[fin_mask]
pool_gt    = pool_gt[fin_mask]

# Sort ascending by GT TTG
sort_idx   = np.argsort(pool_gt)
pool_valid = pool_valid[sort_idx]
pool_gt    = pool_gt[sort_idx]

print(f"  Valid pool size: {len(pool_valid):,}  "
      f"TTG range [{pool_gt[0]:.2f}, {pool_gt[-1]:.2f}]")

ttg_cutoff_steps = np.searchsorted(pool_gt, TTG_CAUSAL_START)
print(f"  Points with TTG ≤ {TTG_CAUSAL_START}: {ttg_cutoff_steps:,}")


def sample_causal_interior(rng_np, ttg_thresh):
    """Sample N_BATCH points from pool with GT TTG ≤ ttg_thresh (numpy, CPU)."""
    n_avail = int(np.searchsorted(pool_gt, ttg_thresh, side="right"))
    n_avail = max(n_avail, 64)
    replace = n_avail < N_BATCH
    idx = rng_np.choice(n_avail, N_BATCH, replace=replace)
    return jnp.array(pool_valid[idx], dtype=jnp.float32)


# ── Init shared params ────────────────────────────────────────────────────────

rng = jr.PRNGKey(SEED)
k1, k2, k3, k4, k5 = jr.split(rng, 5)

_, A0, H0 = init_se2_splat(k1, K, p=1,
                            sigma_fwd=SIGMA,
                            sigma_lat=SIGMA,
                            sigma_rot=SIGMA_ROT,
                            xy_range=XY_RANGE)
A_fixed   = A0
V0        = jr.normal(k2, (K, 1)) * 0.01

source_pts   = sample_source_ball(k3, EPS_BC, N_SOURCE)
params_init  = (V0, A_fixed, H0)

opt = optax.chain(
    optax.clip_by_global_norm(0.3),
    optax.adamw(LR, weight_decay=2e-5),
)

print(f"\nSE(2) SRM  K={K}  σ_xy={SIGMA}  σ_rot={SIGMA_ROT}  (A fixed, α=1 fixed)")
print(f"Phase 1 (shared BC warmup): {STEPS_BC} steps")
print(f"Phase 2 (fork):             {STEPS_PINN} steps  PDE ramp 0→{PHYSICS_WEIGHT}")

# ── Phase 1: shared BC warmup ─────────────────────────────────────────────────

print("\n=== Phase 1: Shared BC warm-up ===")

trainable_bc  = get_trainable(params_init)
state_bc      = opt.init(trainable_bc)
hist_bc       = []

@jax.jit
def step_bc(trainable, state, source_pts, interior):
    (loss, aux), grads = jax.value_and_grad(
        lambda t: phase1_loss(set_trainable(params_init, t), source_pts, interior),
        has_aux=True,
    )(trainable)
    upd, state = opt.update(grads, state, trainable)
    return optax.apply_updates(trainable, upd), state, loss, aux

rng_bc = k4
for i in trange(STEPS_BC, desc="BC warmup"):
    rng_bc, sk = jr.split(rng_bc)
    interior  = sample_interior_jax(sk, N_BATCH)
    trainable_bc, state_bc, loss, (bc, _, pos) = step_bc(
        trainable_bc, state_bc, source_pts, interior)
    if i % LOG_EVERY == 0 or i == STEPS_BC - 1:
        hist_bc.append(dict(step=i, bc=float(bc), pos=float(pos)))
        if i % (LOG_EVERY * 5) == 0:
            print(f"  {i:5d}  bc={float(bc):.5f}  pos={float(pos):.5f}")

params_after_bc = set_trainable(params_init, trainable_bc)
print(f"  BC warmup done.  Final bc={hist_bc[-1]['bc']:.5f}")


# ── Phase 2 helper ────────────────────────────────────────────────────────────

def run_phase2(params_start, rng_key, rng_np_seed, causal: bool):
    label     = "Causal" if causal else "Standard"
    hist      = []
    trainable = get_trainable(params_start)
    state     = opt.init(trainable)
    rng_np    = np.random.default_rng(rng_np_seed)

    # Need to close over params_start for the JIT'd step function
    _params_ref = params_start

    @jax.jit
    def step_p2(trainable, state, source_pts, interior, pw):
        (loss, aux), grads = jax.value_and_grad(
            lambda t: phase2_loss(set_trainable(_params_ref, t), source_pts, interior, pw),
            has_aux=True,
        )(trainable)
        upd, state = opt.update(grads, state, trainable)
        return optax.apply_updates(trainable, upd), state, loss, aux

    print(f"\n=== Phase 2: {label} ===")
    for i in trange(STEPS_PINN, desc=label):
        rng_key, sk = jr.split(rng_key)

        if causal:
            frac      = i / max(1, STEPS_PINN - 1)
            ttg_t     = TTG_CAUSAL_START + (TTG_CAUSAL_END - TTG_CAUSAL_START) * frac
            interior  = sample_causal_interior(rng_np, ttg_t)
        else:
            interior  = sample_interior_jax(sk, N_BATCH)

        frac_pde = i / max(1, int(STEPS_PINN * 0.5))
        pw       = jnp.array(float(PHYSICS_WEIGHT * min(1.0, frac_pde)), dtype=jnp.float32)

        trainable, state, loss, (bc, pde, pos) = step_p2(
            trainable, state, source_pts, interior, pw)

        step_g = STEPS_BC + i
        if step_g % LOG_EVERY == 0 or i == STEPS_PINN - 1:
            ttg_display = (TTG_CAUSAL_START + (TTG_CAUSAL_END - TTG_CAUSAL_START)
                           * i / max(1, STEPS_PINN - 1)) if causal else -1
            rec = dict(step=step_g, bc=float(bc), pde=float(pde),
                       pos=float(pos), pw=float(pw))
            if causal:
                rec["ttg_thresh"] = ttg_display
            hist.append(rec)
            if step_g % (LOG_EVERY * 5) == 0 or i == STEPS_PINN - 1:
                ttg_str = f"  ttg≤{ttg_display:.1f}" if causal else ""
                print(f"  {step_g:5d}  bc={float(bc):.5f}  pde={float(pde):.5f}"
                      f"  pos={float(pos):.5f}  pw={float(pw):.2f}{ttg_str}")

    params_final = set_trainable(params_start, trainable)
    return params_final, hist


# Run both Phase 2 variants (sequentially — one GPU)
params_std,    hist_std    = run_phase2(params_after_bc, k5,          SEED + 10, causal=False)
params_causal, hist_causal = run_phase2(params_after_bc, jr.PRNGKey(SEED + 99), SEED + 20, causal=True)


# ── Evaluate ──────────────────────────────────────────────────────────────────

N_VIS  = 200
x_vis  = np.linspace(-XY_RANGE, XY_RANGE, N_VIS)
y_vis  = np.linspace(-XY_RANGE, XY_RANGE, N_VIS)
XX, YY = np.meshgrid(x_vis, y_vis)
xy_flat = np.column_stack([XX.ravel(), YY.ravel()])

THETA_SLICES = {
    r"$\theta{=}{-}90°$": -np.pi / 2,
    r"$\theta{=}0°$":      0.0,
    r"$\theta{=}{+}90°$":  np.pi / 2,
}

print("\nEvaluating models…")
V_gt, V_std, V_caus = {}, {}, {}
for label, th_val in THETA_SLICES.items():
    G = jnp.array(
        np.column_stack([xy_flat, np.full(N_VIS**2, float(th_val))]),
        dtype=jnp.float32,
    )
    V_std[label]  = np.array(eval_model_jit(G, params_std).squeeze(-1)).reshape(N_VIS, N_VIS)
    V_caus[label] = np.array(eval_model_jit(G, params_causal).squeeze(-1)).reshape(N_VIS, N_VIS)

    TT = np.full_like(XX, th_val)
    gt = dubins_csc_dist(XX, YY, TT)
    gt[np.sqrt(XX**2 + YY**2 + TT**2) < EPS_BC * 1.1] = np.nan
    V_gt[label] = gt

all_gt_fin = np.concatenate([v[np.isfinite(v)] for v in V_gt.values()])
VMAX   = float(np.percentile(all_gt_fin, 97))
LEVELS = np.linspace(0.2, VMAX * 0.95, 12)

print(f"\nDomain VMAX (97-pct): {VMAX:.2f}")
print(f"{'Slice':30s}  {'Standard MAE':>14}  {'Causal MAE':>12}")
print("-" * 60)
mae_std_all, mae_caus_all = [], []
for label in THETA_SLICES:
    e_std  = np.abs(V_std[label]  - V_gt[label])
    e_caus = np.abs(V_caus[label] - V_gt[label])
    mae_s  = float(np.nanmean(e_std))
    mae_c  = float(np.nanmean(e_caus))
    mae_std_all.append(mae_s);  mae_caus_all.append(mae_c)
    winner = "←" if mae_c < mae_s else "  "
    print(f"  {label:28s}  {mae_s:>14.4f}  {mae_c:>12.4f}  {winner}")
print(f"  {'Average':28s}  {np.mean(mae_std_all):>14.4f}  {np.mean(mae_caus_all):>12.4f}")


# ── Figure ────────────────────────────────────────────────────────────────────

CMAP_V = "viridis"
CMAP_E = "hot_r"

fig = plt.figure(figsize=(22, 18))
fig.suptitle(
    f"Dubins HJB PINN — Causal vs Standard training  (K={K}, σ_xy={SIGMA}, σ_rot={SIGMA_ROT})\n"
    r"Causal: PDE front expands by GT TTG from "
    f"TTG≤{TTG_CAUSAL_START} → TTG≤{TTG_CAUSAL_END} over {STEPS_PINN//1000}k steps",
    fontsize=11,
)

gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.40, wspace=0.35)

# ── Rows 0–2: GT | Standard error | Causal error for each theta slice ─────────
row_labels = ["GT", "Standard error", "Causal error"]
for col, (slice_label, th_val) in enumerate(THETA_SLICES.items()):
    dlen = 0.5

    # Row 0: GT
    ax = fig.add_subplot(gs[0, col])
    Z  = np.clip(V_gt[slice_label], 0, VMAX)
    im = ax.contourf(x_vis, y_vis, Z, levels=30, cmap=CMAP_V, vmin=0, vmax=VMAX)
    ax.contour(x_vis, y_vis, V_gt[slice_label], levels=LEVELS,
               colors="white", linewidths=0.4, alpha=0.5)
    ax.scatter(0, 0, c="red", s=50, zorder=6)
    ax.annotate("", xy=(dlen*np.cos(th_val), dlen*np.sin(th_val)), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="red", lw=2, mutation_scale=11))
    ax.set_title(f"GT  {slice_label}", fontsize=9)
    ax.set_xlim(-XY_RANGE, XY_RANGE); ax.set_ylim(-XY_RANGE, XY_RANGE)
    ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, label="V [s]")

    # Row 1: Standard error
    ax = fig.add_subplot(gs[1, col])
    err = np.abs(V_std[slice_label] - V_gt[slice_label])
    vt  = float(np.nanpercentile(err, 95))
    im  = ax.contourf(x_vis, y_vis, np.clip(err, 0, vt),
                      levels=25, cmap=CMAP_E, vmin=0, vmax=vt)
    ax.scatter(0, 0, c="blue", s=50, zorder=6)
    ax.annotate("", xy=(dlen*np.cos(th_val), dlen*np.sin(th_val)), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="blue", lw=2, mutation_scale=11))
    mae_s = float(np.nanmean(err))
    ax.set_title(f"Standard |err|  {slice_label}  MAE={mae_s:.3f}", fontsize=9)
    ax.set_xlim(-XY_RANGE, XY_RANGE); ax.set_ylim(-XY_RANGE, XY_RANGE)
    ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, label="|err| [s]")

    # Row 2: Causal error
    ax = fig.add_subplot(gs[2, col])
    err_c = np.abs(V_caus[slice_label] - V_gt[slice_label])
    im    = ax.contourf(x_vis, y_vis, np.clip(err_c, 0, vt),
                        levels=25, cmap=CMAP_E, vmin=0, vmax=vt)
    ax.scatter(0, 0, c="blue", s=50, zorder=6)
    ax.annotate("", xy=(dlen*np.cos(th_val), dlen*np.sin(th_val)), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="blue", lw=2, mutation_scale=11))
    mae_c = float(np.nanmean(err_c))
    ax.set_title(f"Causal |err|  {slice_label}  MAE={mae_c:.3f}", fontsize=9)
    ax.set_xlim(-XY_RANGE, XY_RANGE); ax.set_ylim(-XY_RANGE, XY_RANGE)
    ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, label="|err| [s]")

# ── Column 3: supplementary panels ───────────────────────────────────────────

# Row 0 col 3: 1D slice y=0, θ=0 through the discontinuity
ax1d = fig.add_subplot(gs[0, 3])
x_line   = x_vis
gt_line  = dubins_csc_dist(x_line, np.zeros_like(x_line), np.zeros_like(x_line))
G_line   = jnp.array(
    np.column_stack([x_line[:, None], np.zeros((N_VIS, 1)), np.zeros((N_VIS, 1))]),
    dtype=jnp.float32,
)
std_line  = np.array(eval_model_jit(G_line, params_std).squeeze(-1))
caus_line = np.array(eval_model_jit(G_line, params_causal).squeeze(-1))
ax1d.plot(x_line, gt_line,   lw=2.0, color="black",      label="Dubins GT")
ax1d.plot(x_line, std_line,  lw=1.5, color="royalblue",  linestyle="--", label="Standard PINN")
ax1d.plot(x_line, caus_line, lw=1.5, color="darkorange",  linestyle="--", label="Causal PINN")
ax1d.axvline(0, color="gray", lw=0.8, linestyle=":")
ax1d.set_xlabel("x [m]  (y=0, θ=0)"); ax1d.set_ylabel("V [s]")
ax1d.set_title("1D slice through cut-locus  (y=0, θ=0)", fontsize=9)
ax1d.legend(fontsize=8); ax1d.grid(alpha=0.3)

# Row 1 col 3: Training loss — Standard
ax_ls = fig.add_subplot(gs[1, 3])
s2 = [h["step"] for h in hist_std]
ax_ls.semilogy(s2, [h["pde"] for h in hist_std], color="royalblue",  label="PDE")
ax_ls.semilogy(s2, [h["bc"]  for h in hist_std], color="royalblue",  linestyle="--", label="BC")
ax_ls.set_xlabel("Step"); ax_ls.set_ylabel("Loss (log)")
ax_ls.set_title("Standard: Phase 2 losses", fontsize=9)
ax_ls.legend(fontsize=8); ax_ls.grid(alpha=0.3)

# Row 2 col 3: Training loss — Causal + TTG schedule
ax_lc = fig.add_subplot(gs[2, 3])
s2c = [h["step"] for h in hist_causal]
ax_lc.semilogy(s2c, [h["pde"] for h in hist_causal], color="darkorange", label="PDE")
ax_lc.semilogy(s2c, [h["bc"]  for h in hist_causal], color="darkorange", linestyle="--", label="BC")
ax_lc.set_xlabel("Step"); ax_lc.set_ylabel("Loss (log)")
ax_lc.set_title("Causal: Phase 2 losses", fontsize=9)
ax_lc.legend(fontsize=8); ax_lc.grid(alpha=0.3)

# Add TTG threshold as twin axis
ax_ttg = ax_lc.twinx()
ttg_vals = [h["ttg_thresh"] for h in hist_causal if "ttg_thresh" in h]
if ttg_vals:
    ax_ttg.plot(s2c[:len(ttg_vals)], ttg_vals, color="green", lw=1.2,
                linestyle=":", alpha=0.7)
    ax_ttg.axhline(2 * np.pi, color="green", lw=0.8, linestyle="-.", alpha=0.5)
    ax_ttg.set_ylabel("TTG threshold [s]", color="green", fontsize=8)
    ax_ttg.tick_params(axis="y", colors="green")
    ax_ttg.annotate("2π (cut locus)", xy=(s2c[len(ttg_vals)//2], 2*np.pi),
                    fontsize=7, color="green", va="bottom")

# Row 3: Summary MAE bar chart + TTG front coverage info
ax_bar = fig.add_subplot(gs[3, :2])
slice_names = [r"θ=−90°", r"θ=0°", r"θ=+90°", "Average"]
maes_s  = mae_std_all  + [np.mean(mae_std_all)]
maes_c  = mae_caus_all + [np.mean(mae_caus_all)]
x_pos   = np.arange(len(slice_names))
width   = 0.35
bars_s  = ax_bar.bar(x_pos - width/2, maes_s, width, label="Standard",
                     color="royalblue", alpha=0.8, edgecolor="white")
bars_c  = ax_bar.bar(x_pos + width/2, maes_c, width, label="Causal",
                     color="darkorange", alpha=0.8, edgecolor="white")
for bar, val in zip(list(bars_s) + list(bars_c),
                    maes_s + maes_c):
    ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8)
ax_bar.set_xticks(x_pos); ax_bar.set_xticklabels(slice_names)
ax_bar.set_ylabel("MAE [s]"); ax_bar.set_title("MAE comparison", fontsize=9)
ax_bar.legend(fontsize=9); ax_bar.grid(axis="y", alpha=0.3)

# Row 3 col 2–3: Causal front coverage at various training steps
ax_cov = fig.add_subplot(gs[3, 2:])
steps_show = np.linspace(0, STEPS_PINN - 1, 200)
ttg_sched  = TTG_CAUSAL_START + (TTG_CAUSAL_END - TTG_CAUSAL_START) * steps_show / (STEPS_PINN - 1)
n_pts_sched = np.array([np.searchsorted(pool_gt, t, side="right") for t in ttg_sched])
ax_cov.plot(STEPS_BC + steps_show, ttg_sched, color="green", lw=1.5, label="TTG threshold")
ax_cov.axhline(2 * np.pi, color="red", lw=1.0, linestyle="--", alpha=0.7,
               label=f"2π ≈ {2*np.pi:.2f} (cut locus)")
ax_cov.set_xlabel("Step"); ax_cov.set_ylabel("TTG threshold [s]", color="green")
ax_cov.set_title("Causal front schedule", fontsize=9)

ax_cov2 = ax_cov.twinx()
ax_cov2.fill_between(STEPS_BC + steps_show, n_pts_sched, alpha=0.2, color="purple")
ax_cov2.plot(STEPS_BC + steps_show, n_pts_sched, color="purple", lw=1.0, linestyle=":")
ax_cov2.set_ylabel("Pool points available", color="purple", fontsize=8)
ax_cov2.tick_params(axis="y", colors="purple")
ax_cov.legend(fontsize=8, loc="upper left"); ax_cov.grid(alpha=0.3)

plt.savefig("figures/dubins_comparison.png", dpi=150, bbox_inches="tight")
print("\nSaved figures/dubins_comparison.png")
