"""
dubins_lifted_pinn.py

Bounded-control Dubins car: ẋ=cosθ, ẏ=sinθ, θ̇=u, |u|≤1.
Goal: (0,0,π). Value function V(x,y,θ) = minimum time to goal.

Lifted formulation on T*M: Ṽ(x,y,θ,h₁,h₂,h₃) is smooth on each PMP sheet.
Recovery: V̂(q) = min_λ Ṽ(q,λ).

Two PDEs are enforced jointly:

  1. Dubins eikonal:   X₁Ṽ + |X₂Ṽ| = 1
       X₁ = cosθ ∂_x + sinθ ∂_y  (forward motion)
       X₂ = ∂_θ                   (turning)
     This is the correct bounded-control HJB: H(q,∇V) = h₁+|h₂| = 1.
     Without the transport this alone has many solutions (both V=|x| and
     V=2π+|x| satisfy it at (−x,0,π)), including the wrong branch.

  2. PMP transport:    X_H[Ṽ] = −1
       X_H = cosθ ∂_x + sinθ ∂_y + u* ∂_θ
           + h₃u* ∂_h₁ + h₃ ∂_h₂ + h₁u* ∂_h₃
       u* = sign(h₂)  (optimal turning direction from PMP)
     This constrains the λ-variation of Ṽ: each sheet must decrease at
     rate 1 along the Hamiltonian characteristics. It propagates the BC
     outward along the CORRECT geodesic, selecting the right branch.

Lambda parameterisation (PMP constraint h₁+|h₂|=1):
  h₁ = cosφ,  h₂ = s(1−cosφ),  φ∈[0,2π],  s∈{±1},  h₃∈R (free).

Model: Ṽ(z) = α·r(x,y) + SRM(z),   r = √(x²+y²+ε).
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from tqdm import trange

jax.config.update('jax_platform_name', 'gpu')

# ── Model ──────────────────────────────────────────────────────────────────────

def eval_srm(Z, W, log_prec, mu):
    """
    Z:        [n, 6]  = (x, y, θ, h₁, h₂, h₃)
    W:        [k]
    log_prec: [k, 6]
    mu:       [k, 6]
    Returns:  [n]
    """
    prec = jnp.exp(log_prec)               # [k, 6]
    diff = Z[:, None, :] - mu[None, :, :]  # [n, k, 6]

    # Geodesic angular distance for θ (index 2)
    d2_theta = (2.0 * (1.0 - jnp.cos(diff[:, :, 2]))
                * prec[None, :, 2] ** 2)    # [n, k]

    # Euclidean for (x, y, h₁, h₂, h₃) — indices 0,1,3,4,5
    idx     = jnp.array([0, 1, 3, 4, 5])
    scaled  = diff[:, :, idx] * prec[None, :, idx]
    d2_eucl = jnp.sum(scaled ** 2, axis=-1)

    phi = jnp.exp(-0.5 * (d2_eucl + d2_theta))
    return phi @ W                                    # [n]


def eval_model(Z, params):
    """Ṽ(z) = α·r(x,y) + SRM(z). Returns [n, 1]."""
    W, log_prec, mu, alpha = params
    r   = jnp.sqrt(Z[:, 0] ** 2 + Z[:, 1] ** 2 + 1e-6)
    srm = eval_srm(Z, W, log_prec, mu)
    return (alpha * r + srm)[:, None]


eval_model_jit = jax.jit(eval_model)


def init_params(key, k=2000):
    k1, k2, k3, k4 = jr.split(key, 4)
    xy    = jr.uniform(k1, (k, 2), minval=-5.0, maxval=5.0)
    theta = jr.uniform(k2, (k, 1), minval=0.0, maxval=2.0 * jnp.pi)

    # Lambda centres: (h₁,h₂) on PMP constraint, h₃ free.
    phi = jr.uniform(k3, (k,), minval=0.0, maxval=2.0 * jnp.pi)
    s   = 2.0 * jr.bernoulli(k4, shape=(k,)).astype(jnp.float32) - 1.0
    h1  = jnp.cos(phi)
    h2  = s * (1.0 - jnp.cos(phi))
    k5  = jr.PRNGKey(1234)
    h3  = jr.normal(k5, (k,)) * 2.0
    lam = jnp.stack([h1, h2, h3], axis=1)
    mu  = jnp.concatenate([xy, theta, lam], axis=-1)

    lp_xy  = jnp.full((k, 2), -jnp.log(2.0))   # σ_xy ≈ 2
    lp_th  = jnp.zeros((k, 1))                   # σ_θ ≈ 1
    lp_lam = jnp.full((k, 3),  jnp.log(2.0))    # σ_λ ≈ 0.5
    log_prec = jnp.concatenate([lp_xy, lp_th, lp_lam], axis=-1)

    W     = jr.normal(jr.PRNGKey(999), (k,)) * 0.01
    alpha = jnp.array(1.0)
    return (W, log_prec, mu, alpha)


# ── Loss functions ─────────────────────────────────────────────────────────────

def _v_scalar(z, params):
    return eval_model(z[None, :], params)[0, 0]


def eikonal_loss(params, interior):
    """
    Dubins eikonal: X₁Ṽ + |X₂Ṽ| − 1 = 0.
    This is the correct HJB for bounded-control (forward-only) Dubins car.
    Unlike the sub-Riemannian (X₁V)²+(X₂V)²=1, this forbids backward motion
    and produces the correct cut-locus structure.
    """
    gz    = jax.vmap(jax.grad(lambda z: _v_scalar(z, params)))(interior)
    theta = interior[:, 2]
    X1V   = gz[:, 0] * jnp.cos(theta) + gz[:, 1] * jnp.sin(theta)
    X2V   = gz[:, 2]
    return jnp.mean((X1V + jnp.abs(X2V) - 1.0) ** 2)


def transport_loss(params, interior, beta=5.0):
    """
    PMP transport: X_H[Ṽ] = −1.

    X_H = cosθ ∂_x + sinθ ∂_y + u* ∂_θ + h₃u* ∂_h₁ + h₃ ∂_h₂ + h₁u* ∂_h₃
    u* = sign(h₂) ≈ tanh(β·h₂)  (optimal turning direction)

    This enforces that Ṽ decreases at unit rate along the PMP characteristics.
    It selects the correct geodesic sheet for each λ, removing the wrong-branch
    degeneracy that the eikonal alone cannot break.
    """
    gz    = jax.vmap(jax.grad(lambda z: _v_scalar(z, params)))(interior)
    theta = interior[:, 2]
    h1    = interior[:, 3]
    h2    = interior[:, 4]
    h3    = interior[:, 5]
    u     = jnp.tanh(beta * h2)   # smooth sign(h₂)

    X_H_V = (gz[:, 0] * jnp.cos(theta) + gz[:, 1] * jnp.sin(theta)
           + gz[:, 2] * u
           + gz[:, 3] * h3 * u
           + gz[:, 4] * h3
           + gz[:, 5] * h1 * u)
    return jnp.mean((X_H_V + 1.0) ** 2)


def bc_loss(params, lam_samples):
    """Ṽ(0,0,π,λ) = 0 for all λ."""
    m    = lam_samples.shape[0]
    goal = jnp.tile(jnp.array([0.0, 0.0, jnp.pi]), (m, 1))
    z_bc = jnp.concatenate([goal, lam_samples], axis=1)
    return jnp.mean(eval_model(z_bc, params)[:, 0] ** 2)


def total_loss(params, interior, lam_bc, w_transport=1.0, beta=5.0):
    """Standard (non-causal) loss: eikonal + transport + BC + positivity."""
    eik   = eikonal_loss(params, interior)
    tr    = transport_loss(params, interior, beta=beta)
    V_int = eval_model(interior, params)[:, 0]
    pos   = jnp.mean(jax.nn.relu(-V_int) ** 2)
    return eik + w_transport * tr + 10.0 * bc_loss(params, lam_bc) + pos


def total_loss_causal(params, interior_sorted, lam_bc,
                      eps_causal=1e-2, w_transport=1.0, beta=5.0):
    """
    Causal version (Wang et al. 2022): interior_sorted must be pre-sorted by
    ascending reference Dubins distance. Causal weights suppress gradients from
    far-from-goal points until near-goal regions are well-fitted.
    """
    gz    = jax.vmap(jax.grad(lambda z: _v_scalar(z, params)))(interior_sorted)
    theta = interior_sorted[:, 2]
    X1V   = gz[:, 0] * jnp.cos(theta) + gz[:, 1] * jnp.sin(theta)
    X2V   = gz[:, 2]
    eik_r = (X1V + jnp.abs(X2V) - 1.0) ** 2

    res_sg  = jax.lax.stop_gradient(eik_r)
    cumsum  = jnp.concatenate([jnp.zeros(1), jnp.cumsum(res_sg[:-1])])
    weights = jax.lax.stop_gradient(jnp.exp(-eps_causal * cumsum))
    eik     = jnp.mean(weights * eik_r)

    tr    = transport_loss(params, interior_sorted, beta=beta)
    V_int = eval_model(interior_sorted, params)[:, 0]
    pos   = jnp.mean(jax.nn.relu(-V_int) ** 2)
    return eik + w_transport * tr + 10.0 * bc_loss(params, lam_bc) + pos


# ── Sampling ───────────────────────────────────────────────────────────────────

def _sample_lambda_dubins(key, n):
    """
    Sample λ on the Dubins PMP constraint h₁+|h₂|=1.
      h₁ = cosφ,  h₂ = s(1−cosφ),  φ∈[0,2π],  s∈{±1}
    h₃ is free (represents out-of-plane momentum), sampled ~N(0,4).
    """
    k1, k2, k3 = jr.split(key, 3)
    phi = jr.uniform(k1, (n,), minval=0.0, maxval=2.0 * jnp.pi)
    s   = 2.0 * jr.bernoulli(k2, shape=(n,)).astype(jnp.float32) - 1.0
    h1  = jnp.cos(phi)
    h2  = s * (1.0 - jnp.cos(phi))
    h3  = jr.normal(k3, (n,)) * 2.0
    return jnp.stack([h1, h2, h3], axis=1)   # [n, 3]


def _sample_interior(key, n):
    k1, k2, k3 = jr.split(key, 3)
    xy    = jr.uniform(k1, (n, 2), minval=-5.0, maxval=5.0)
    theta = jr.uniform(k2, (n, 1), minval=0.0, maxval=2.0 * jnp.pi)
    lam   = _sample_lambda_dubins(k3, n)
    return jnp.concatenate([xy, theta, lam], axis=1)


# ── Vectorised reference Dubins distances (for causal sorting) ─────────────────

_2PI = 2.0 * np.pi


def _dubins_batch_np(xyt, goal=(0.0, 0.0, np.pi)):
    """Vectorised Dubins distance from each row of xyt=[n,3] to goal."""
    x, y, theta = xyt[:, 0], xyt[:, 1], xyt[:, 2]
    D   = np.hypot(goal[0] - x, goal[1] - y)
    psi = np.arctan2(goal[1] - y, goal[0] - x)
    a   = (theta - psi) % _2PI
    b   = (goal[2] - psi) % _2PI
    sa, ca, sb, cb = np.sin(a), np.cos(a), np.sin(b), np.cos(b)
    cosab = np.cos(a - b)
    INF   = np.full_like(D, np.inf)

    p2  = 2 + D**2 - 2*cosab + 2*D*(sa - sb)
    t1  = np.arctan2(cb - ca, D + sa - sb)
    lsl = np.where(p2 >= 0,
                   (-a+t1)%_2PI + np.sqrt(np.maximum(p2, 0)) + (b-t1)%_2PI, INF)

    p2  = 2 + D**2 - 2*cosab + 2*D*(sb - sa)
    t1  = np.arctan2(ca - cb, D - sa + sb)
    rsr = np.where(p2 >= 0,
                   (a-t1)%_2PI + np.sqrt(np.maximum(p2, 0)) + (-b+t1)%_2PI, INF)

    p2   = -2 + D**2 + 2*cosab + 2*D*(sa + sb)
    p    = np.sqrt(np.maximum(p2, 0))
    t0   = np.arctan2(-ca-cb, D+sa+sb) - np.arctan2(-2.0, p)
    lsr  = np.where(p2 >= 0, (-a+t0)%_2PI + p + (-b+t0)%_2PI, INF)

    p2   = D**2 - 2 + 2*cosab - 2*D*(sa + sb)
    p    = np.sqrt(np.maximum(p2, 0))
    t0   = np.arctan2(ca+cb, D-sa-sb) - np.arctan2(2.0, p)
    rsl  = np.where(p2 >= 0, (a-t0)%_2PI + p + (b-t0)%_2PI, INF)

    tmp  = (6 - D**2 + 2*cosab + 2*D*(sa - sb)) / 8.0
    p    = (2*np.pi - np.arccos(np.clip(tmp, -1, 1))) % _2PI
    t    = (a - np.arctan2(ca-cb, D-sa+sb) + (p/2)%_2PI) % _2PI
    q    = (a - b - t + p%_2PI) % _2PI
    rlr  = np.where(np.abs(tmp) <= 1, t + p + q, INF)

    tmp  = (6 - D**2 + 2*cosab + 2*D*(-sa + sb)) / 8.0
    p    = (2*np.pi - np.arccos(np.clip(tmp, -1, 1))) % _2PI
    t    = (-a - np.arctan2(ca-cb, D+sa-sb) + p/2) % _2PI
    q    = (b - a - t + p) % _2PI
    lrl  = np.where(np.abs(tmp) <= 1, t + p + q, INF)

    return np.min(np.stack([lsl, rsr, lsr, rsl, rlr, lrl], axis=1), axis=1)


def dubins_distance(q0, q1=(0.0, 0.0, np.pi), R=1.0):
    """Scalar Dubins distance from q0 to q1."""
    res = _dubins_batch_np(
        np.array([[q0[0]/R, q0[1]/R, q0[2]]]),
        goal=(q1[0]/R, q1[1]/R, q1[2]),
    )
    return float(res[0]) * R


# ── Training ───────────────────────────────────────────────────────────────────

def train(seed=42, k=2000, lr=1e-3, n_steps=10_000, batch_size=1024,
          eps_causal=1e-2, warmup_steps=500, w_transport=1.0, beta=5.0):
    """
    Train lifted Dubins SRM with causal eikonal + PMP transport.

    No supervised anchors needed: the transport PDE uniquely identifies
    which geodesic sheet each λ selects, propagating the BC outward along
    the correct characteristics and producing the right cut-locus structure.
    """
    key     = jr.PRNGKey(seed)
    k0, key = jr.split(key)
    params    = init_params(k0, k=k)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    _loss_std = jax.jit(
        jax.value_and_grad(
            lambda p, z, l: total_loss(p, z, l, w_transport=w_transport, beta=beta)
        )
    )
    _loss_causal = jax.jit(
        jax.value_and_grad(
            lambda p, z, l: total_loss_causal(
                p, z, l, eps_causal=eps_causal, w_transport=w_transport, beta=beta
            )
        )
    )

    losses = []
    pbar   = trange(n_steps, desc='Training lifted Dubins SRM')
    for i in pbar:
        key, k1, k2 = jr.split(key, 3)
        interior = _sample_interior(k1, batch_size)
        lam_bc   = _sample_lambda_dubins(k2, 256)

        if i < warmup_steps:
            loss, grads = _loss_std(params, interior, lam_bc)
            phase = 'warmup'
        else:
            interior_np     = np.array(interior)
            d_ref           = _dubins_batch_np(interior_np[:, :3])
            sort_idx        = np.argsort(d_ref)
            interior_sorted = jnp.array(interior_np[sort_idx])
            loss, grads     = _loss_causal(params, interior_sorted, lam_bc)
            phase = 'causal'

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        losses.append(float(loss))

        if i % 500 == 0:
            pbar.set_postfix({
                'loss':  f'{float(loss):.4f}',
                'α':     f'{float(params[3]):.3f}',
                'phase': phase,
            })

    return params, losses


# ── Projected value function ───────────────────────────────────────────────────

@jax.jit
def _min_over_lam(params, spatial_pts, lam_grid):
    """min_λ Ṽ(q,λ) via lax.scan. spatial_pts:[n,3], lam_grid:[m,3]."""
    n = spatial_pts.shape[0]

    def step(v_hat, lam):
        lam_rep = jnp.tile(lam[None, :], (n, 1))
        z   = jnp.concatenate([spatial_pts, lam_rep], axis=1)
        V_l = eval_model(z, params)[:, 0]
        return jnp.minimum(v_hat, V_l), None

    v_init  = jnp.full(n, jnp.inf)
    v_final, _ = jax.lax.scan(step, v_init, lam_grid)
    return v_final


def compute_projected_value(params, n_xy=100, n_phi=32, n_h3=8, h3_max=3.0):
    """
    V̂(x,y,π) = min_λ Ṽ(x,y,π,λ).
    Lambda grid: 2·n_phi φ-values × n_h3 h₃-values, covering the full
    PMP constraint surface h₁+|h₂|=1 with free h₃∈[−h3_max, h3_max].
    """
    x_vals = np.linspace(-5, 5, n_xy)
    y_vals = np.linspace(-5, 5, n_xy)
    XX, YY = np.meshgrid(x_vals, y_vals, indexing='ij')
    theta_pi    = np.full(n_xy ** 2, np.pi)
    spatial_pts = jnp.array(np.column_stack([XX.ravel(), YY.ravel(), theta_pi]))

    phi_vals = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    h3_vals  = np.linspace(-h3_max, h3_max, n_h3)
    rows = []
    for s in [1.0, -1.0]:
        for phi in phi_vals:
            h1 = np.cos(phi)
            h2 = s * (1.0 - np.cos(phi))
            for h3 in h3_vals:
                rows.append([h1, h2, h3])
    lam_grid = jnp.array(rows, dtype=jnp.float32)   # [2·n_phi·n_h3, 3]

    print(f"  evaluating {len(lam_grid)} λ pts × {n_xy**2} spatial pts …")
    V_hat = np.array(_min_over_lam(params, spatial_pts, lam_grid))
    return V_hat.reshape(n_xy, n_xy), x_vals, y_vals


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_heatmap(V_hat, x_vals, y_vals,
                 path='figures/dubins_lifted_heatmap.png'):
    XX, YY = np.meshgrid(x_vals, y_vals, indexing='ij')
    fig, ax = plt.subplots(figsize=(7, 6))
    cf = ax.contourf(XX, YY, V_hat, levels=30, cmap='viridis')
    ax.contour(XX, YY, V_hat, levels=30, colors='white', alpha=0.4, linewidths=0.5)
    plt.colorbar(cf, ax=ax, label='$\\hat{V}(x,y,\\theta=\\pi)$')
    ax.plot(0, 0, '*', color='white', markersize=14,
            markeredgecolor='gold', markeredgewidth=1.5, label='Goal $(0,0,\\pi)$')
    ax.set_xlabel('x');  ax.set_ylabel('y')
    ax.set_title('Learned Dubins Value Function $\\hat{V}(x,y,\\theta=\\pi)$')
    ax.legend(loc='upper right');  ax.set_aspect('equal')
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=150);  plt.close()
    print(f"Saved {path}")


def plot_slice(x_vals, V_hat_slice,
               path='figures/dubins_lifted_slice.png'):
    gt = np.array([dubins_distance((x, 0.0, np.pi)) for x in x_vals])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x_vals, V_hat_slice, lw=2, label='Learned $\\hat{V}(x,0,\\pi)$')
    ax.plot(x_vals, gt, '--', lw=2, color='tab:orange',
            label='Dubins distance (reference)')
    ax.set_xlabel('x');  ax.set_ylabel('Value / Distance')
    ax.set_title('Slice comparison $y=0$, $\\theta=\\pi$')
    ax.legend();  ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=150);  plt.close()
    print(f"Saved {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k',           type=int,   default=2000)
    ap.add_argument('--steps',       type=int,   default=10_000)
    ap.add_argument('--lr',          type=float, default=1e-3)
    ap.add_argument('--batch',       type=int,   default=1024)
    ap.add_argument('--seed',        type=int,   default=42)
    ap.add_argument('--n-phi',       type=int,   default=32,
                    help='φ values in projection λ grid')
    ap.add_argument('--n-h3',        type=int,   default=8,
                    help='h₃ values in projection λ grid')
    ap.add_argument('--h3-max',      type=float, default=3.0,
                    help='h₃ range for projection: [−h3_max, h3_max]')
    ap.add_argument('--n-xy',        type=int,   default=100)
    ap.add_argument('--eps',         type=float, default=1e-2,
                    help='Causal decay rate')
    ap.add_argument('--warmup',      type=int,   default=500,
                    help='Warmup steps (standard loss) before causal kicks in')
    ap.add_argument('--w-transport', type=float, default=1.0,
                    help='Weight on PMP transport loss')
    ap.add_argument('--beta',        type=float, default=5.0,
                    help='Sharpness of tanh(β·h₂) approximation to sign(h₂)')
    args = ap.parse_args()

    print(f"Training: k={args.k}, steps={args.steps}, lr={args.lr}, "
          f"batch={args.batch}, seed={args.seed}\n"
          f"         eps={args.eps}, warmup={args.warmup}, "
          f"w_transport={args.w_transport}, beta={args.beta}")

    params, losses = train(
        seed=args.seed, k=args.k, lr=args.lr,
        n_steps=args.steps, batch_size=args.batch,
        eps_causal=args.eps, warmup_steps=args.warmup,
        w_transport=args.w_transport, beta=args.beta,
    )
    print(f"Final loss: {losses[-1]:.5f}  |  alpha = {float(params[3]):.4f}")

    print("Computing projected value function …")
    V_hat, x_vals, y_vals = compute_projected_value(
        params, n_xy=args.n_xy,
        n_phi=args.n_phi, n_h3=args.n_h3, h3_max=args.h3_max,
    )

    plot_heatmap(V_hat, x_vals, y_vals)
    y0_idx = np.argmin(np.abs(y_vals))
    plot_slice(x_vals, V_hat[:, y0_idx])
    print("Done.")


if __name__ == '__main__':
    main()
