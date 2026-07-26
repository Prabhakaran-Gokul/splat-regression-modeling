#!/usr/bin/env python3
"""
dubins_se2_pinn.py

Pure PINN for the Dubins car sub-Riemannian Eikonal using SE(2) wrapped
Gaussian SRM.

WHY SE(2) KERNELS
-----------------
The state space is SE(2) = R² × S¹.  A flat Gaussian in (x,y,θ) treats θ
as a real line and breaks at ±π.  The (cosθ, sinθ) embedding fixes periodicity
but uses the chord distance.

The SE(2) wrapped Gaussian uses the LEFT-INVARIANT LOG MAP:

    v = Log_h(g) = (v_fwd, v_lat, v_rot)  ∈  se(2)

where v_rot = wrap(θ − θ₀) is the angular arc.  The density is:

    N_w(g; h, A) = N(v; 0, A) · (Δθ / 2sin(Δθ/2))²

This is smooth and periodic in θ with no seam.

KERNEL SHAPE
------------
Isotropic: A = σ·I₃.  The SE(2) log map carries the topology; anisotropy
is not needed to represent the S¹ part correctly.

SCALE ANCHOR
------------
Fix: V(g) = α · r(x,y) + SRM(g),  r = √(x²+y²+ε)

α·r is the first-order approximation to any goal-reaching distance.  The SRM
learns heading-dependent corrections.  α starts at 1.0 and is trained.

PDE (Dubins HJB: cost-to-go to origin)
---------------------------------------
    |X₂V| − X₁V = 1      (forward Dubins, unit speed, unit turn radius)
    X₁V = cosθ ∂_xV + sinθ ∂_yV
    X₂V = ∂_θV
    Smooth form used in training: sqrt(X₂V² + ε²) − X₁V = 1

BC: V = 0 on the 3-D Euclidean ball  |(x,y,θ)| ≤ EPS  around (0,0,0).

TRAINING CURRICULUM
-------------------
Phase 1 (BC warm-up): minimise only the BC loss so the model first learns
V≈0 near the goal.  With the radial anchor, this corrects SRM(source) ≈ 0.

Phase 2 (BC + PDE): add the PDE loss.  The model already respects the BC, so
the PDE shapes the value function in the interior without the loss landscape
pulling the solution away from the BC.

PDE weight is linearly ramped from 0 to PHYSICS_WEIGHT over Phase 2 to give
the model time to relax into the right viscosity solution.

Output: figures/dubins_se2_srm.png, figures/dubins_se2_theta0.png
"""

import os
import pickle
import time

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optax
from tqdm import trange

jax.config.update("jax_platform_name", "gpu")
jax.config.update("jax_enable_x64", False)

from lib.manifold_splat import eval_se2_splat, init_se2_splat

# ── MLflow ────────────────────────────────────────────────────────────────────

MLFLOW_URI        = "sqlite:////home/tassos/.local/share/mlflow/runs.db"
MLFLOW_EXPERIMENT = "dubins-pinn-srm"
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT)
os.makedirs("logs",    exist_ok=True)
os.makedirs("figures", exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────

K              = 600        # splat components
SIGMA          = 1.0        # spatial kernel width
SIGMA_ROT      = 0.5        # rotational kernel width
N_BATCH        = 2048
N_SOURCE       = 600        # BC collocation points
EPS_BC         = 0.20       # goal ball radius in (x,y,θ) Euclidean space
XY_RANGE       = 3.0

# Training schedule
STEPS_BC       = 48_000     # Phase 1: BC warm-up only
STEPS_PINN     = 26_000     # Phase 2: BC + PDE
NUM_STEPS      = STEPS_BC + STEPS_PINN
STEPS_SUP      = 15_000     # Supervised (GT labels) training
N_SUP_SAMPLE   = 100_000    # Supervised pre-sampled training set size

LR             = 5e-4
PHYSICS_WEIGHT = 4.0        # PDE weight at end of ramp
BC_WEIGHT      = 15.0       # BC weight throughout
POS_WEIGHT     = 2.0        # positivity weight
LOG_EVERY      = 2000
SEED           = 42

# ── Ground truth for comparison only (not used in training) ───────────────────

def _mod2pi(a):
    return a % (2.0 * np.pi)


def dubins_csc_dist(x, y, theta, R=1.0):
    """Shortest Dubins CSC path from (x,y,θ) to (0,0,0). Vectorised numpy."""
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
#
# params = (V: [k,1], A: [k,3,3], H: [k,3])
#
# A is FIXED throughout training (isotropic σ·I).
# α = 1 is FIXED (radial anchor scale); NOT trained.
# Only V and H are optimised.
#
# Fixing α=1 prevents the optimizer from collapsing the scale: with trainable α,
# the PDE gradient prefers shrinking α rather than building the complex heading-
# dependent SRM correction, converging to a degenerate V≈0.5r solution.

def eval_model(G, params):
    """G: [n,3] → [n,1].  V = α·r(x,y) + SRM,  r = sqrt(x²+y²+ε)."""
    V, A, H = params
    r = jnp.sqrt(G[:, 0]**2 + G[:, 1]**2 + _EPS_R)
    anchor = (_ALPHA_FIXED * r)[:, None]
    return anchor + eval_se2_splat(G, (V, A, H))


eval_model_jit = jax.jit(eval_model)


# ── Sampling ──────────────────────────────────────────────────────────────────

def sample_source_ball(key, eps, n):
    """Uniform 3-D ball {|(x,y,θ)| ≤ eps} around origin."""
    k1, k2 = jr.split(key)
    dirs = jr.normal(k1, (n, 3))
    dirs = dirs / jnp.linalg.norm(dirs, axis=-1, keepdims=True)
    r    = eps * jr.uniform(k2, (n,)) ** (1.0 / 3.0)
    return dirs * r[:, None]       # [n,3]: (x,y,θ)


def sample_interior(key, n):
    k1, k2 = jr.split(key)
    xy    = jr.uniform(k1, (n, 2), minval=-XY_RANGE, maxval=XY_RANGE)
    theta = jr.uniform(k2, (n, 1), minval=-jnp.pi, maxval=jnp.pi)
    q     = jnp.concatenate([xy, theta], axis=-1)
    # Exclude the source ball
    dist  = jnp.sqrt(jnp.sum(q**2, axis=-1))
    mask  = dist > EPS_BC * 1.2
    return q[mask]


# ── PINN loss ─────────────────────────────────────────────────────────────────

def bc_loss_fn(params, source_pts):
    """V(source) = 0."""
    u = eval_model(source_pts, params).squeeze(-1)
    return jnp.mean(u ** 2)


_EPS_HJB    = 0.05   # smooth |X₂V| ≈ sqrt(X₂V² + ε²)
_ALPHA_FIXED = 1.0   # radial anchor scale (fixed, not trained)
_EPS_R       = 1e-3  # prevents r singularity at origin

def pde_loss_fn(params, interior):
    """Dubins HJB: sqrt(X₂V² + ε²) - X₁V = 1  (smooth forward-only HJB).

    This is the correct PDE for the Dubins TTG, not the CC-symmetric SR Eikonal.
    The smoothed |X₂V| avoids a non-differentiable absolute value.
    """
    def v_fn(q):
        return eval_model(q[None, :], params)[0, 0]

    grad_q = jax.vmap(jax.grad(v_fn))(interior)
    theta  = interior[:, 2]
    X1V    = grad_q[:, 0] * jnp.cos(theta) + grad_q[:, 1] * jnp.sin(theta)
    X2V    = grad_q[:, 2]
    hjb = jnp.sqrt(X2V ** 2 + _EPS_HJB ** 2) - X1V - 1.0
    return jnp.mean(hjb ** 2)


def pos_loss_fn(params, interior):
    """V ≥ 0 soft penalty."""
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


# ── Initialise ────────────────────────────────────────────────────────────────

rng = jr.PRNGKey(SEED)
k1, k2, k3 = jr.split(rng, 3)

_, A0, H0 = init_se2_splat(k1, K, p=1,
                            sigma_fwd=SIGMA,
                            sigma_lat=SIGMA,
                            sigma_rot=SIGMA_ROT,
                            xy_range=XY_RANGE)
# A is FIXED throughout training — diagonal diag(σ, σ, σ_rot)
A_fixed   = A0

# Small SRM weights; radial anchor carries the scale
V0        = jr.normal(k2, (K, 1)) * 0.01
log_alpha = jnp.array(0.0)   # α = 1.0

source_pts = sample_source_ball(k3, EPS_BC, N_SOURCE)

# params = (V, A_fixed, H) — only V and H are trained, A and α are frozen
params = (V0, A_fixed, H0)

# Trainable = (V, H); A_fixed is excluded
def get_trainable(p):
    V, A, H = p
    return (V, H)

def set_trainable(p, t):
    _, A, _ = p
    V, H = t
    return (V, A, H)


print(f"SE(2) SRM PINN  K={K}  σ_xy={SIGMA} σ_rot={SIGMA_ROT} (FIXED)  α=1 (FIXED)  steps={NUM_STEPS}")
print(f"  Phase 1 (BC warmup): {STEPS_BC} steps")
print(f"  Phase 2 (BC+PDE):    {STEPS_PINN} steps  PDE weight ramp 0→{PHYSICS_WEIGHT}")
print(f"  BC_WEIGHT={BC_WEIGHT}  POS_WEIGHT={POS_WEIGHT}  LR={LR}")

_run_name = f"dubins-se2-K{K}-bc{STEPS_BC}-pinn{STEPS_PINN}"
mlflow.start_run(run_name=_run_name)
mlflow.log_params({
    "K":              K,
    "sigma":          SIGMA,
    "sigma_rot":      SIGMA_ROT,
    "n_batch":        N_BATCH,
    "n_source":       N_SOURCE,
    "eps_bc":         EPS_BC,
    "xy_range":       XY_RANGE,
    "steps_bc":       STEPS_BC,
    "steps_pinn":     STEPS_PINN,
    "lr":             LR,
    "physics_weight": PHYSICS_WEIGHT,
    "bc_weight":      BC_WEIGHT,
    "pos_weight":     POS_WEIGHT,
    "seed":           SEED,
    "eps_hjb":        _EPS_HJB,
    "model":          "se2-srm-radial-anchor",
    "loss":           "symmetric",
})
_t0_train = time.time()

# ── Optimizer ─────────────────────────────────────────────────────────────────

opt   = optax.chain(
    optax.clip_by_global_norm(0.3),
    optax.adamw(LR, weight_decay=2e-5),
)
hist  = []

# ── Phase 1: BC warmup ────────────────────────────────────────────────────────

print("\n--- Phase 1: BC warmup ---")

trainable0 = get_trainable(params)
state1 = opt.init(trainable0)

@jax.jit
def step_bc(trainable, state, source_pts, interior):
    (loss, aux), grads = jax.value_and_grad(
        lambda t: phase1_loss(set_trainable(params, t), source_pts, interior),
        has_aux=True,
    )(trainable)
    upd, state = opt.update(grads, state, trainable)
    trainable  = optax.apply_updates(trainable, upd)
    return trainable, state, loss, aux

trainable = trainable0
for i in trange(STEPS_BC, desc="BC warmup"):
    rng, sk = jr.split(rng)
    interior = sample_interior(sk, N_BATCH)
    trainable, state1, loss, (bc, _, pos) = step_bc(
        trainable, state1, source_pts, interior)

    if i % LOG_EVERY == 0 or i == STEPS_BC - 1:
        hist.append(dict(step=i, phase=1, total=float(loss), bc=float(bc),
                         pde=0.0, pos=float(pos)))
        print(f"  {i:5d}  loss={float(loss):.5f}  bc={float(bc):.5f}"
              f"  pos={float(pos):.5f}")
        mlflow.log_metrics({
            "loss/total": float(loss),
            "loss/bc":    float(bc),
            "loss/pos":   float(pos),
            "loss/pde":   0.0,
        }, step=i)

params = set_trainable(params, trainable)

# ── Phase 2: BC + PDE with ramp ───────────────────────────────────────────────

print("\n--- Phase 2: BC + PDE (PDE ramp) ---")

state2 = opt.init(get_trainable(params))

@jax.jit
def step_pinn(trainable, state, source_pts, interior, pw):
    (loss, aux), grads = jax.value_and_grad(
        lambda t: phase2_loss(set_trainable(params, t), source_pts, interior, pw),
        has_aux=True,
    )(trainable)
    upd, state = opt.update(grads, state, trainable)
    trainable  = optax.apply_updates(trainable, upd)
    return trainable, state, loss, aux

trainable = get_trainable(params)
for i in trange(STEPS_PINN, desc="PINN"):
    rng, sk = jr.split(rng)
    interior = sample_interior(sk, N_BATCH)

    frac = i / max(1, int(STEPS_PINN * 0.5))
    pw   = jnp.array(float(PHYSICS_WEIGHT * min(1.0, frac)), dtype=jnp.float32)

    trainable, state2, loss, (bc, pde, pos) = step_pinn(
        trainable, state2, source_pts, interior, pw)

    step_global = STEPS_BC + i
    if step_global % LOG_EVERY == 0 or i == STEPS_PINN - 1:
        hist.append(dict(step=step_global, phase=2, total=float(loss), bc=float(bc),
                         pde=float(pde), pos=float(pos), pw=float(pw)))
        print(f"  {step_global:5d}  loss={float(loss):.5f}  bc={float(bc):.5f}"
              f"  pde={float(pde):.5f}  pos={float(pos):.5f}  pw={float(pw):.2f}")
        mlflow.log_metrics({
            "loss/total": float(loss),
            "loss/bc":    float(bc),
            "loss/pde":   float(pde),
            "loss/pos":   float(pos),
            "loss/pw":    float(pw),
        }, step=step_global)

params = set_trainable(params, trainable)

# ── Supervised training (GT labels, same architecture) ────────────────────────

print("\n--- Supervised training (GT labels, no anchor) ---")

# Pre-compute a large dataset of GT Dubins distances
_rng_sup = jr.PRNGKey(SEED + 77)
_rng_sup, _ks1, _ks2, _ks3 = jr.split(_rng_sup, 4)

_xy_s  = np.random.default_rng(SEED + 77).uniform(-XY_RANGE, XY_RANGE,
                                                    (N_SUP_SAMPLE, 2)).astype(np.float32)
_th_s  = np.random.default_rng(SEED + 78).uniform(-np.pi, np.pi,
                                                    N_SUP_SAMPLE).astype(np.float32)
_X_all = np.column_stack([_xy_s, _th_s])
_Y_all = dubins_csc_dist(_X_all[:, 0], _X_all[:, 1], _X_all[:, 2]).astype(np.float32)

_src_ok = np.sqrt(np.sum(_X_all**2, axis=1)) > EPS_BC * 1.5
_fin_ok = np.isfinite(_Y_all)
_mask   = _src_ok & _fin_ok
_X_sup  = jnp.array(_X_all[_mask])
_Y_sup  = jnp.array(_Y_all[_mask, None])
_N_SUP  = int(_X_sup.shape[0])
print(f"  Training set: {_N_SUP} points")

# Fresh init — same kernel geometry, independent weights
_, _A_sup, _H_sup0 = init_se2_splat(_ks1, K, p=1,
                                     sigma_fwd=SIGMA, sigma_lat=SIGMA,
                                     sigma_rot=SIGMA_ROT, xy_range=XY_RANGE)
_V_sup0   = jr.normal(_ks2, (K, 1)) * 0.01
params_sup = (_V_sup0, _A_sup, _H_sup0)

def _get_t_sup(p): V, A, H = p; return (V, H)
def _set_t_sup(p, t): _, A, _ = p; V, H = t; return (V, A, H)

_opt_sup  = optax.chain(optax.clip_by_global_norm(0.3),
                        optax.adamw(LR, weight_decay=2e-5))

@jax.jit
def _step_sup(trainable, state, X_b, Y_b):
    def loss_fn(t):
        Y_pred = eval_model(X_b, _set_t_sup(params_sup, t))
        return jnp.mean((Y_pred - Y_b) ** 2)
    loss, grads = jax.value_and_grad(loss_fn)(trainable)
    upd, state  = _opt_sup.update(grads, state, trainable)
    return optax.apply_updates(trainable, upd), state, loss

_t_sup  = _get_t_sup(params_sup)
_st_sup = _opt_sup.init(_t_sup)
hist_sup = []

for i in trange(STEPS_SUP, desc="Supervised"):
    _rng_sup, _sk = jr.split(_rng_sup)
    _idx   = jr.randint(_sk, (N_BATCH,), 0, _N_SUP)
    _X_b   = _X_sup[_idx]
    _Y_b   = _Y_sup[_idx]
    _t_sup, _st_sup, _loss_sup = _step_sup(_t_sup, _st_sup, _X_b, _Y_b)

    if i % LOG_EVERY == 0 or i == STEPS_SUP - 1:
        hist_sup.append(dict(step=i, loss=float(_loss_sup)))
        print(f"  {i:5d}  mse={float(_loss_sup):.6f}")
        mlflow.log_metrics({"sup/mse": float(_loss_sup)}, step=i)

params_sup = _set_t_sup(params_sup, _t_sup)

# ── Evaluate ──────────────────────────────────────────────────────────────────

N_VIS  = 200
x_vis  = np.linspace(-XY_RANGE, XY_RANGE, N_VIS)
y_vis  = np.linspace(-XY_RANGE, XY_RANGE, N_VIS)
XX, YY = np.meshgrid(x_vis, y_vis)
xy_flat = np.column_stack([XX.ravel(), YY.ravel()])

THETA_SLICES = {
    r"$\theta{=}{-90°}$": -np.pi / 2,
    r"$\theta{=}0°$":      0.0,
    r"$\theta{=}{+90°}$":  np.pi / 2,
    r"$\theta{=}180°$":    np.pi,
}

V_srm, V_gt, V_sup_eval = {}, {}, {}
for label, th_val in THETA_SLICES.items():
    G = jnp.array(
        np.column_stack([xy_flat, np.full(N_VIS**2, float(th_val))]),
        dtype=jnp.float32,
    )
    V_srm[label]     = np.array(eval_model_jit(G, params).squeeze(-1)).reshape(N_VIS, N_VIS)
    V_sup_eval[label] = np.array(eval_model_jit(G, params_sup).squeeze(-1)).reshape(N_VIS, N_VIS)

    TT = np.full_like(XX, th_val)
    gt = dubins_csc_dist(XX, YY, TT)
    gt[np.sqrt(XX**2 + YY**2 + TT**2) < EPS_BC * 1.1] = np.nan
    V_gt[label] = gt

# Print statistics
ph1_end_idx = STEPS_BC // LOG_EVERY - 1
_train_elapsed = time.time() - _t0_train
print(f"\nPhase 1 final bc loss: {hist[ph1_end_idx]['bc']:.5f}")
print(f"Total training wall time: {_train_elapsed:.1f}s")
all_gt_fin = np.concatenate([v[np.isfinite(v)] for v in V_gt.values()])
VMAX   = float(np.percentile(all_gt_fin, 97))
LEVELS = np.linspace(0.2, VMAX * 0.95, 24)

print(f"Domain VMAX (97-pct): {VMAX:.2f}")
_mae_vals_pinn = []
_mae_vals_sup  = []
for label in THETA_SLICES:
    err_p   = np.abs(V_srm[label]      - V_gt[label])
    err_s   = np.abs(V_sup_eval[label] - V_gt[label])
    mae_p   = float(np.nanmean(err_p))
    mae_s   = float(np.nanmean(err_s))
    rmse_p  = float(np.sqrt(np.nanmean(err_p**2)))
    rmse_s  = float(np.sqrt(np.nanmean(err_s**2)))
    _mae_vals_pinn.append(mae_p)
    _mae_vals_sup.append(mae_s)
    print(f"  {label:28s}  PINN MAE={mae_p:.4f}  Sup MAE={mae_s:.4f}")
    _key = (label.replace("$","").replace("{","").replace("}","")
               .replace("\\","").replace("=","_").replace("°","deg")
               .replace("+","pos").replace("-","neg").strip())
    mlflow.log_metrics({
        f"pinn/mae/{_key}":  mae_p, f"pinn/rmse/{_key}":  rmse_p,
        f"sup/mae/{_key}":   mae_s, f"sup/rmse/{_key}":   rmse_s,
    })

CMAP_V = "viridis"
CMAP_E = "hot_r"

# ── Main 5×5 comparison figure (GT / PINN / PINN-err / Sup / Sup-err) ────────

fig, axes = plt.subplots(5, 5, figsize=(24, 22))
pde_str = r"$|X_2V|-X_1V=1$"
fig.suptitle(
    f"Dubins Car HJB — SE(2) Wrapped Gaussian SRM  K={K}, σ_xy={SIGMA}, σ_rot={SIGMA_ROT}\n"
    r"PINN: $V{=}r{+}\mathrm{SRM}$ (radial anchor, no GT)   " +
    r"Supervised: $V{=}r{+}\mathrm{SRM}$ fit to GT labels",
    fontsize=11,
)

_labels = list(THETA_SLICES.items())
_dlen   = 0.5

for col, (label, th_val) in enumerate(_labels):
    _cos, _sin = np.cos(th_val), np.sin(th_val)

    def _arrow(ax, color):
        ax.scatter(0, 0, c=color, s=50, zorder=6)
        ax.annotate("", xy=(_dlen * _cos, _dlen * _sin), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                    lw=2, mutation_scale=12))

    # Row 0: GT
    ax = axes[0, col]
    Z  = np.clip(V_gt[label], 0, VMAX)
    im = ax.contourf(x_vis, y_vis, Z, levels=30, cmap=CMAP_V, vmin=0, vmax=VMAX)
    ax.contour(x_vis, y_vis, V_gt[label], levels=LEVELS,
               colors="white", linewidths=0.5, alpha=0.65)
    _arrow(ax, "red")
    ax.set_title(f"GT  {label}", fontsize=8)
    ax.set_xlim(-XY_RANGE, XY_RANGE); ax.set_ylim(-XY_RANGE, XY_RANGE)
    ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, label="V [s]")

    # Row 1: PINN
    _err_p = np.abs(V_srm[label] - V_gt[label])
    ax = axes[1, col]
    Z  = np.clip(V_srm[label], 0, VMAX)
    im = ax.contourf(x_vis, y_vis, Z, levels=30, cmap=CMAP_V, vmin=0, vmax=VMAX)
    ax.contour(x_vis, y_vis, V_srm[label], levels=LEVELS,
               colors="white", linewidths=0.5, alpha=0.65)
    _arrow(ax, "red")
    ax.set_title(f"PINN  {label}  MAE={np.nanmean(_err_p):.2f}", fontsize=8)
    ax.set_xlim(-XY_RANGE, XY_RANGE); ax.set_ylim(-XY_RANGE, XY_RANGE)
    ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, label="V [s]")

    # Row 2: PINN error
    _vt_p = float(np.nanpercentile(_err_p, 95))
    ax = axes[2, col]
    im  = ax.contourf(x_vis, y_vis, np.clip(_err_p, 0, _vt_p),
                      levels=25, cmap=CMAP_E, vmin=0, vmax=_vt_p)
    ax.contour(x_vis, y_vis, V_gt[label], levels=LEVELS,
               colors="gray", linewidths=0.4, alpha=0.5)
    _arrow(ax, "deepskyblue")
    ax.set_title(f"|PINN error|  {label}", fontsize=8)
    ax.set_xlim(-XY_RANGE, XY_RANGE); ax.set_ylim(-XY_RANGE, XY_RANGE)
    ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, label="|err| [s]")

    # Row 3: Supervised
    _err_s = np.abs(V_sup_eval[label] - V_gt[label])
    ax = axes[3, col]
    Z  = np.clip(V_sup_eval[label], 0, VMAX)
    im = ax.contourf(x_vis, y_vis, Z, levels=30, cmap=CMAP_V, vmin=0, vmax=VMAX)
    ax.contour(x_vis, y_vis, V_sup_eval[label], levels=LEVELS,
               colors="white", linewidths=0.5, alpha=0.65)
    _arrow(ax, "red")
    ax.set_title(f"Supervised  {label}  MAE={np.nanmean(_err_s):.2f}", fontsize=8)
    ax.set_xlim(-XY_RANGE, XY_RANGE); ax.set_ylim(-XY_RANGE, XY_RANGE)
    ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, label="V [s]")

    # Row 4: Supervised error
    _vt_s = float(np.nanpercentile(_err_s, 95))
    ax = axes[4, col]
    im  = ax.contourf(x_vis, y_vis, np.clip(_err_s, 0, _vt_s),
                      levels=25, cmap=CMAP_E, vmin=0, vmax=_vt_s)
    ax.contour(x_vis, y_vis, V_gt[label], levels=LEVELS,
               colors="gray", linewidths=0.4, alpha=0.5)
    _arrow(ax, "deepskyblue")
    ax.set_title(f"|Sup error|  {label}", fontsize=8)
    ax.set_xlim(-XY_RANGE, XY_RANGE); ax.set_ylim(-XY_RANGE, XY_RANGE)
    ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, label="|err| [s]")

# ── Summary column (col 4) ────────────────────────────────────────────────────

# Row 0: PINN training loss
ax_l = axes[0, 4]
_ph1 = [h for h in hist if h["phase"] == 1]
_ph2 = [h for h in hist if h["phase"] == 2]
if _ph1:
    _s1 = [h["step"] for h in _ph1]
    ax_l.semilogy(_s1, [h["total"] for h in _ph1], label="BC-only total", lw=1.5)
    ax_l.semilogy(_s1, [h["bc"]    for h in _ph1], "--", lw=1, label="BC")
if _ph2:
    _s2 = [h["step"] for h in _ph2]
    ax_l.semilogy(_s2, [h["pde"]   for h in _ph2], label="PDE", lw=1.5)
    ax_l.semilogy(_s2, [h["bc"]    for h in _ph2], "--", lw=1, label="BC (ph2)")
ax_l.axvline(STEPS_BC, color="gray", ls=":", alpha=0.6, label="PDE on")
ax_l.set_xlabel("Step"); ax_l.set_ylabel("Loss")
ax_l.set_title("PINN training loss"); ax_l.legend(fontsize=7); ax_l.grid(alpha=0.3)

# Row 1: Supervised training loss
ax_sl = axes[1, 4]
_ss   = [h["step"] for h in hist_sup]
ax_sl.semilogy(_ss, [h["loss"] for h in hist_sup], lw=1.5, color="C2")
ax_sl.set_xlabel("Step"); ax_sl.set_ylabel("MSE")
ax_sl.set_title("Supervised training loss"); ax_sl.grid(alpha=0.3)

# Row 2: 1D slice y=0 (GT + PINN + Sup)
ax_1d = axes[2, 4]
_x_line = x_vis
for _lbl, _th in _labels:
    _gt_l  = dubins_csc_dist(_x_line, np.zeros_like(_x_line),
                              np.full_like(_x_line, _th))
    _G_l   = jnp.array(np.column_stack([
        _x_line[:, None], np.zeros((N_VIS, 1)),
        np.full((N_VIS, 1), float(_th))]), dtype=jnp.float32)
    _pinn_l = np.array(eval_model_jit(_G_l, params).squeeze(-1))
    _sup_l  = np.array(eval_model_jit(_G_l, params_sup).squeeze(-1))
    _sh     = _lbl.replace("$","").replace("{","").replace("}","")
    ax_1d.plot(_x_line, _gt_l,   lw=1.8, label=f"GT {_sh}")
    ax_1d.plot(_x_line, _pinn_l, lw=1,   ls="--")
    ax_1d.plot(_x_line, _sup_l,  lw=1,   ls=":")
ax_1d.set_xlabel("x [m]  (y=0)"); ax_1d.set_ylabel("V [s]")
ax_1d.set_title("1D y=0  (—GT  --PINN  ··Sup)")
ax_1d.legend(fontsize=6); ax_1d.grid(alpha=0.3)

# Row 3: PINN error histogram
ax_hp = axes[3, 4]
_errs_p = np.concatenate([np.abs(V_srm[l] - V_gt[l]).ravel()
                          for l in THETA_SLICES])
_errs_p = _errs_p[np.isfinite(_errs_p)]
ax_hp.hist(_errs_p, bins=60, color="steelblue", edgecolor="white", lw=0.3)
ax_hp.axvline(_errs_p.mean(), color="red", ls="--",
              label=f"PINN MAE={_errs_p.mean():.3f}")
ax_hp.set_xlabel("|err| [s]"); ax_hp.set_title("PINN error hist")
ax_hp.legend(fontsize=8); ax_hp.grid(alpha=0.3)

# Row 4: Supervised error histogram
ax_hs = axes[4, 4]
_errs_s = np.concatenate([np.abs(V_sup_eval[l] - V_gt[l]).ravel()
                          for l in THETA_SLICES])
_errs_s = _errs_s[np.isfinite(_errs_s)]
ax_hs.hist(_errs_s, bins=60, color="darkorange", edgecolor="white", lw=0.3)
ax_hs.axvline(_errs_s.mean(), color="red", ls="--",
              label=f"Sup MAE={_errs_s.mean():.3f}")
ax_hs.set_xlabel("|err| [s]"); ax_hs.set_title("Supervised error hist")
ax_hs.legend(fontsize=8); ax_hs.grid(alpha=0.3)

plt.tight_layout()
out_main = "figures/dubins_se2_srm.png"
plt.savefig(out_main, dpi=150, bbox_inches="tight")
print(f"Saved {out_main}")

# ── Min-over-θ figure ─────────────────────────────────────────────────────────

print("Computing min_θ V(x,y,θ)…")
N_THETA_MIN = 72
thetas_min  = np.linspace(-np.pi, np.pi, N_THETA_MIN, endpoint=False)

gt_min  = np.full(N_VIS * N_VIS, np.inf)
srm_min = np.full(N_VIS * N_VIS, np.inf)

for th in thetas_min:
    # GT
    TT  = np.full_like(XX, th)
    gt  = dubins_csc_dist(XX, YY, TT).ravel()
    gt_min = np.minimum(gt_min, gt)
    # SRM
    G   = jnp.array(
        np.column_stack([xy_flat, np.full(N_VIS**2, float(th))]),
        dtype=jnp.float32,
    )
    v   = np.array(eval_model_jit(G, params).squeeze(-1))
    srm_min = np.minimum(srm_min, v)

gt_min  = gt_min.reshape(N_VIS, N_VIS)
srm_min = srm_min.reshape(N_VIS, N_VIS)
gt_min[gt_min == np.inf] = np.nan

err_min = np.abs(srm_min - gt_min)
mae_min = float(np.nanmean(err_min))
vmax_min = float(np.nanpercentile(gt_min[np.isfinite(gt_min)], 97))
levels_min = np.linspace(0.1, vmax_min * 0.95, 24)

fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
fig2.suptitle(
    r"$\min_\theta\, V(x,y,\theta)$  — minimum TTG over all headings, goal $(0,0,0)$" + "\n"
    f"SE(2) wrapped Gaussian SRM — pure PINN  K={K}, σ_xy={SIGMA}, σ_rot={SIGMA_ROT}  ({N_THETA_MIN} θ samples)",
    fontsize=11,
)

for ax, (Z, title, cmap, vmax_) in zip(axes2, [
    (np.clip(gt_min, 0, vmax_min),
     "Dubins GT  min_θ V", CMAP_V, vmax_min),
    (np.clip(srm_min, 0, vmax_min),
     f"SE(2) SRM PINN  min_θ V  MAE={mae_min:.3f}", CMAP_V, vmax_min),
    (np.clip(err_min, 0, float(np.nanpercentile(err_min, 95))),
     f"|error|  MAE={mae_min:.3f}", CMAP_E,
     float(np.nanpercentile(err_min, 95))),
]):
    im = ax.contourf(x_vis, y_vis, Z, levels=30, cmap=cmap, vmin=0, vmax=vmax_)
    ax.contour(x_vis, y_vis, Z, levels=levels_min,
               colors="white", linewidths=0.5, alpha=0.5)
    ax.scatter(0, 0, c="red", s=80, zorder=6)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_xlim(-XY_RANGE, XY_RANGE); ax.set_ylim(-XY_RANGE, XY_RANGE)
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax, label="V [s]")

plt.tight_layout()
out_min = "figures/dubins_se2_min_theta.png"
plt.savefig(out_min, dpi=150, bbox_inches="tight")
print(f"Saved {out_min}")

# ── y-θ slice at fixed x (behind the goal) ────────────────────────────────────
# Shows V(x_fixed, y, θ) as a 2D heatmap.  Reveals where SRM breaks down along
# the heading dimension compared to GT.

X_SLICES  = [-0.25, -0.5, -1.0, -2.0]
N_Y_SL    = 150
N_T_SL    = 150
y_sl      = np.linspace(-XY_RANGE, XY_RANGE, N_Y_SL)
t_sl      = np.linspace(-np.pi, np.pi, N_T_SL)
YY_sl, TT_sl = np.meshgrid(y_sl, t_sl)   # [N_T, N_Y]

fig3, axes3 = plt.subplots(len(X_SLICES), 3,
                            figsize=(14, 4 * len(X_SLICES)))
fig3.suptitle(
    r"SE(2) SRM PINN — $V(x_0, y, \theta)$ slice at fixed $x_0$ behind goal" + "\n"
    f"K={K}, σ_xy={SIGMA}, σ_rot={SIGMA_ROT}  (no anchor, pure PINN)",
    fontsize=11,
)

for row, x0 in enumerate(X_SLICES):
    pts = np.column_stack([
        np.full(N_Y_SL * N_T_SL, x0, dtype=np.float32),
        YY_sl.ravel().astype(np.float32),
        TT_sl.ravel().astype(np.float32),
    ])
    G_sl    = jnp.array(pts)
    V_s     = np.array(eval_model_jit(G_sl, params).squeeze(-1)).reshape(N_T_SL, N_Y_SL)
    V_g     = dubins_csc_dist(pts[:, 0], pts[:, 1], pts[:, 2]).reshape(N_T_SL, N_Y_SL)
    err_s   = np.abs(V_s - V_g)

    vmax_s  = float(np.nanpercentile(V_g[np.isfinite(V_g)], 97))
    vt_e    = float(np.nanpercentile(err_s[np.isfinite(err_s)], 95))
    lev_v   = np.linspace(0, vmax_s * 0.95, 22)

    for col, (Z, title, cmap, vmax_c) in enumerate([
        (np.clip(V_g, 0, vmax_s), f"GT  x={x0:.2f}", CMAP_V, vmax_s),
        (np.clip(V_s, 0, vmax_s), f"SRM  x={x0:.2f}  MAE={np.nanmean(err_s):.3f}",
         CMAP_V, vmax_s),
        (np.clip(err_s, 0, vt_e), f"|error|  x={x0:.2f}", CMAP_E, vt_e),
    ]):
        ax = axes3[row, col]
        im = ax.contourf(y_sl, t_sl, Z, levels=30, cmap=cmap, vmin=0, vmax=vmax_c)
        ax.contour(y_sl, t_sl, Z, levels=lev_v,
                   colors="white", linewidths=0.4, alpha=0.5)
        ax.set_xlabel("y [m]")
        ax.set_ylabel("θ [rad]")
        ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        ax.set_yticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
        ax.set_title(title, fontsize=9)
        plt.colorbar(im, ax=ax, label="V [s]")

plt.tight_layout()
out_yt = "figures/dubins_se2_ytheta.png"
plt.savefig(out_yt, dpi=150, bbox_inches="tight")
print(f"Saved {out_yt}")

# ── MLflow: final metrics + artifacts ─────────────────────────────────────────

mlflow.log_metrics({
    "pinn/mae/avg_slices": float(np.mean(_mae_vals_pinn)),
    "sup/mae/avg_slices":  float(np.mean(_mae_vals_sup)),
    "mae/min_theta":       mae_min,
    "final/wall_time_s":   _train_elapsed,
})

# Save model parameters as a pickle artifact
_params_path = "logs/dubins_se2_params.pkl"
with open(_params_path, "wb") as _f:
    pickle.dump({
        "params": jax.tree_util.tree_map(np.array, params),
        "config": dict(K=K, sigma=SIGMA, sigma_rot=SIGMA_ROT,
                       steps_bc=STEPS_BC, steps_pinn=STEPS_PINN,
                       lr=LR, bc_weight=BC_WEIGHT,
                       physics_weight=PHYSICS_WEIGHT,
                       eps_bc=EPS_BC, xy_range=XY_RANGE,
                       model="se2-srm-no-anchor", loss="symmetric"),
        "history": hist,
    }, _f)
mlflow.log_artifact(_params_path, artifact_path="checkpoints")
print(f"Saved model params → {_params_path}")

for _fig_path in [out_main, out_min, out_yt]:
    mlflow.log_artifact(_fig_path, artifact_path="figures")

mlflow.end_run()
print(f"MLflow run '{_run_name}' complete.")
