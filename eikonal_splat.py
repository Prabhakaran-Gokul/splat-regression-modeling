#!/usr/bin/env python3
"""
Eikonal equation solver using Splat Regression Models (SRM).

Solves:
    |∇u(x)| = 1/c(x)        (slowness form; 1/c is the travel-time gradient magnitude)
    u(x) = u_bc(x)   on Γ   (known travel time on a boundary/source set)

Physics-informed training: minimise
    L = L_bc + physics_weight * L_pde
    L_bc  = mean( (u(x_bc) − u_bc)² )
    L_pde = mean( (|∇u(x)| − 1/c(x))² )

Source singularity strategy
----------------------------
The viscosity solution u = T(x, x_src) has a kink (non-differentiable point)
at a point source.  Enforcing u(x_src) = 0 gives the optimiser no scale
information; the trivial u ≡ 0 is a near-local-minimum.

Instead we enforce the BC on a small circle (ring) of radius ε surrounding the
source, where the travel time is known analytically:

    u(x) = ε / c(x_src)    for ‖x − x_src‖ = ε   (uniform-speed approx.)

Collocation points are sampled strictly outside this ring, so the model only
needs to represent the smooth part of the solution.

Gradients of u w.r.t. x are computed with jax.grad through eval_splat
(the autodiff path — NOT eval_splat_grad, which is the analytic closed-form
gradient for least-squares MSE and is unrelated to PDE constraints).
"""

import argparse

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax
from tqdm import tqdm

from lib.splat import eval_splat


# ---------------------------------------------------------------------------
# Core solver
# ---------------------------------------------------------------------------

def eikonal_loss(params, interior_pts, source_pts, source_vals, speed_fn, physics_weight):
    """
    Physics-informed loss for the Eikonal equation.

    L = L_bc  +  physics_weight * L_pde

    L_bc  = mean( (u(x_bc) − source_vals)² )
    L_pde = mean( (|∇u(x)| − 1/c(x))² )

    The (|∇u| − f) form (rather than |∇u|² − f²) gives better gradient
    conditioning near the solution.  A small ε inside the sqrt avoids NaN
    when ∇u ≈ 0 during the first few steps.

    Args:
        params:       (V, A, B) splat parameters
        interior_pts: [n_int, d]  PDE collocation points
        source_pts:   [n_src, d]  BC points
        source_vals:  [n_src]     target u values at source_pts
        speed_fn:     c(x): [n, d] → [n, 1], JAX-traceable
        physics_weight: scalar weight on PDE term
    """
    def u_single(x):
        return eval_splat(x[None, :], params)[0, 0]

    # Boundary condition
    u_bc = eval_splat(source_pts, params).squeeze(-1)         # [n_src]
    bc_loss = jnp.mean((u_bc - source_vals) ** 2)

    # PDE residual at interior collocation points
    grad_u = jax.vmap(jax.grad(u_single))(interior_pts)          # [n_int, d]
    grad_norm = jnp.sqrt(jnp.sum(grad_u ** 2, axis=-1) + 1e-12)  # [n_int]
    slowness  = 1.0 / speed_fn(interior_pts).squeeze(-1)          # [n_int]
    pde_loss  = jnp.mean((grad_norm - slowness) ** 2)

    total = bc_loss + physics_weight * pde_loss
    return total, (bc_loss, pde_loss)


def train_eikonal_splat(
    init_params,
    interior_pts,
    source_pts,
    source_vals,
    speed_fn,
    *,
    num_steps=5000,
    lr=1e-3,
    physics_weight=1.0,
    log_interval=500,
):
    """
    Train a splat model to solve the Eikonal equation.

    Args:
        init_params:    (V, A, B) initial splat parameters; see init_splat_params
        interior_pts:   [n_int, d] collocation points (PDE residual evaluated here)
        source_pts:     [n_src, d] BC points
        source_vals:    [n_src]    target u at source_pts (e.g. ε on a ring of radius ε)
        speed_fn:       c(x): [n, d] → [n, 1], JAX-traceable speed field
        num_steps:      Adam iterations
        lr:             learning rate
        physics_weight: weight on PDE loss relative to BC loss
        log_interval:   console print frequency

    Returns:
        params:   final trained (V, A, B)
        history:  list of {'total', 'bc', 'pde'} dicts, one per step
    """
    warmup_steps = max(1, num_steps // 10)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=lr, warmup_steps=warmup_steps,
        decay_steps=num_steps, end_value=lr * 0.01,
    )
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(schedule))
    opt_state = optimizer.init(init_params)

    # speed_fn and physics_weight captured from closure; step is JIT-compiled
    # once per call.  int_pts/src_pts/src_vals are dynamic arguments so they
    # can be swapped (e.g. for mini-batches) without recompilation.
    @jax.jit
    def step(params, opt_state, int_pts, src_pts, src_vals):
        (total, aux), grads = jax.value_and_grad(
            eikonal_loss, has_aux=True
        )(params, int_pts, src_pts, src_vals, speed_fn, physics_weight)
        updates, new_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), new_state, total, aux

    params = init_params
    history = []

    for i in tqdm(range(num_steps), desc="Eikonal SRM"):
        params, opt_state, total, (bc, pde) = step(
            params, opt_state, interior_pts, source_pts, source_vals
        )
        history.append({"total": float(total), "bc": float(bc), "pde": float(pde)})
        if (i + 1) % log_interval == 0:
            tqdm.write(
                f"  {i+1:5d}/{num_steps}  "
                f"total={total:.3e}  bc={bc:.3e}  pde={pde:.3e}"
            )

    return params, history


def init_splat_params(key, k, d, domain_bounds, *, scale=0.3):
    """
    Initialise splat parameters for an Eikonal solver.

    V ~ N(0, 0.01),  A = scale * I,  B uniform over domain.

    Args:
        key:           JAX random key
        k:             number of splat components
        d:             spatial dimension
        domain_bounds: list of (lo, hi) per dimension, e.g. [(-1,1), (-1,1)]
        scale:         diagonal entry of initial A matrices (spatial spread per splat)
    """
    kv, kb = jr.split(key)
    V = jr.normal(kv, (k, 1)) * 0.01
    A = jnp.tile(jnp.eye(d)[None] * scale, (k, 1, 1))
    ks = jr.split(kb, d)
    B = jnp.hstack([
        jr.uniform(ks[i], (k, 1), minval=lo, maxval=hi)
        for i, (lo, hi) in enumerate(domain_bounds)
    ])
    return V, A, B


# ---------------------------------------------------------------------------
# Source-ring helpers
# ---------------------------------------------------------------------------

def source_ring_2d(x_src, eps, n_pts, c_at_src):
    """
    Build a ring of source BC points at radius eps around x_src in 2D.

    Returns:
        src_pts:  [n_pts, 2]  points on the ring
        src_vals: [n_pts]     u = eps / c(x_src) at each ring point
                              (first-order approximation of travel time at radius eps)
    """
    theta = jnp.linspace(0, 2 * jnp.pi, n_pts, endpoint=False)
    pts = x_src + eps * jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=1)
    vals = jnp.full(n_pts, eps / c_at_src)
    return pts, vals


def source_segment_1d(x_src, eps, c_at_src):
    """
    Two BC points at x_src ± eps in 1D.

    Returns:
        src_pts:  [2, 1]
        src_vals: [2]   u = eps / c(x_src)
    """
    pts  = jnp.array([[x_src - eps], [x_src + eps]])
    vals = jnp.full(2, eps / c_at_src)
    return pts, vals


# ---------------------------------------------------------------------------
# Demo utilities
# ---------------------------------------------------------------------------

def _uniform_collocation_2d(key, n, domain_bounds, exclude_center, min_dist):
    lo = jnp.array([b[0] for b in domain_bounds])
    hi = jnp.array([b[1] for b in domain_bounds])
    key, sk = jr.split(key)
    cands = jr.uniform(sk, (n * 4, 2)) * (hi - lo) + lo
    dist  = jnp.linalg.norm(cands - jnp.array(exclude_center), axis=-1)
    return cands[dist > min_dist][:n]


def _eval_grid_2d(params, domain_bounds, Ng=80):
    g1 = jnp.linspace(*domain_bounds[0], Ng)
    g2 = jnp.linspace(*domain_bounds[1], Ng)
    X1, X2 = jnp.meshgrid(g1, g2)
    pts = jnp.stack([X1.ravel(), X2.ravel()], axis=1)
    u = eval_splat(pts, params).reshape(Ng, Ng)
    return u, X1, X2


# ---------------------------------------------------------------------------
# Demo 1: 2D uniform speed — analytical solution available
# ---------------------------------------------------------------------------

def demo_uniform_speed_2d(key, k=100, n_int=1000, num_steps=2000, lr=1e-3,
                           eps=0.08, n_ring=32):
    """
    2D Eikonal |∇u| = 1 (c=1), source at origin.
    Analytical solution: u(x) = ‖x‖.

    BC is enforced on a ring of radius eps with known values u = eps,
    avoiding the kink singularity at the origin.
    """
    print("=" * 60)
    print("Demo 1: 2D Eikonal  |∇u| = 1  (c=1, source at origin)")
    print(f"        Ring BC: radius={eps}, {n_ring} points, u_ring={eps:.3f}")
    print("=" * 60)

    domain = [(-1.0, 1.0), (-1.0, 1.0)]
    x_src  = jnp.array([0.0, 0.0])

    # Source ring: u = eps on a circle of radius eps (c=1 near origin)
    source_pts, source_vals = source_ring_2d(x_src, eps, n_ring, c_at_src=1.0)

    # Collocation points strictly outside the ring
    key, sk = jr.split(key)
    interior_pts = _uniform_collocation_2d(sk, n_int, domain,
                                           exclude_center=[0.0, 0.0],
                                           min_dist=eps)

    speed_fn = lambda x: jnp.ones((x.shape[0], 1))

    key, sk = jr.split(key)
    init_params = init_splat_params(sk, k, 2, domain, scale=0.35)

    params, history = train_eikonal_splat(
        init_params, interior_pts, source_pts, source_vals, speed_fn,
        num_steps=num_steps, lr=lr, physics_weight=1.0, log_interval=500,
    )

    u_pred, X1, X2 = _eval_grid_2d(params, domain)
    u_true = jnp.sqrt(X1 ** 2 + X2 ** 2)
    mse = float(jnp.mean((u_pred - u_true) ** 2))
    print(f"  Grid MSE vs analytical u=‖x‖: {mse:.4e}")

    levels = jnp.linspace(0.1, 1.3, 13)
    extent = [*domain[0], *domain[1]]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    vmin, vmax = 0.0, float(jnp.max(u_true))

    for ax, data, title in zip(
        axes[:2],
        [u_true, u_pred],
        [r"Analytical $\|x\|$  (isochrones = circles)", f"SRM k={k}  MSE={mse:.1e}"],
    ):
        im = ax.imshow(data, origin="lower", extent=extent,
                       vmin=vmin, vmax=vmax, cmap="viridis", alpha=0.85)
        ax.contour(X1, X2, data, levels=levels, colors="white",
                   linewidths=0.8, alpha=0.9)
        ax.set_title(title)
        ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
        plt.colorbar(im, ax=ax, label="u(x)")
        ax.plot(*x_src, "r*", markersize=8, label="source")
        ax.legend(fontsize=8)

    axes[2].semilogy([h["total"] for h in history], lw=1.2, label="total")
    axes[2].semilogy([h["bc"]    for h in history], lw=1.2, label="BC")
    axes[2].semilogy([h["pde"]   for h in history], lw=1.2, label="PDE")
    axes[2].set_xlabel("step"); axes[2].set_ylabel("loss")
    axes[2].set_title("Loss history"); axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("eikonal_uniform_speed.png", dpi=150, bbox_inches="tight")
    print("  Saved eikonal_uniform_speed.png")
    return params, history


# ---------------------------------------------------------------------------
# Demo 2: 2D variable speed — no analytical solution
# ---------------------------------------------------------------------------

def demo_variable_speed_2d(key, k=100, n_int=1000, num_steps=2000, lr=1e-3,
                            eps=0.05, n_ring=32):
    """
    2D Eikonal with c(x) = 1 + 0.8·exp(−‖x‖²/0.2), source at origin.
    Faster near the centre → tighter isochrones near origin.
    """
    print("=" * 60)
    print("Demo 2: 2D Eikonal  variable speed  c(x)=1+0.8·exp(−‖x‖²/0.2)")
    print("=" * 60)

    domain = [(-1.0, 1.0), (-1.0, 1.0)]
    x_src  = jnp.array([0.0, 0.0])

    def speed_fn(x):
        r2 = jnp.sum(x ** 2, axis=-1, keepdims=True)
        return 1.0 + 0.8 * jnp.exp(-r2 / 0.2)

    # Speed at source for ring BC: c(0,0) = 1.8
    c_src = float(speed_fn(x_src[None, :])[0, 0])
    source_pts, source_vals = source_ring_2d(x_src, eps, n_ring, c_at_src=c_src)

    key, sk = jr.split(key)
    interior_pts = _uniform_collocation_2d(sk, n_int, domain,
                                           exclude_center=[0.0, 0.0],
                                           min_dist=eps)

    key, sk = jr.split(key)
    init_params = init_splat_params(sk, k, 2, domain, scale=0.35)

    params, history = train_eikonal_splat(
        init_params, interior_pts, source_pts, source_vals, speed_fn,
        num_steps=num_steps, lr=lr, physics_weight=1.0, log_interval=500,
    )

    u_pred, X1, X2 = _eval_grid_2d(params, domain)
    grid = jnp.stack([X1.ravel(), X2.ravel()], axis=1)
    c_grid = speed_fn(grid).reshape(X1.shape)

    # PDE residual check on training collocation points
    def u_single(x):
        return eval_splat(x[None, :], params)[0, 0]

    grad_u    = jax.vmap(jax.grad(u_single))(interior_pts)
    grad_norm = jnp.sqrt(jnp.sum(grad_u ** 2, axis=-1))
    slowness  = 1.0 / speed_fn(interior_pts).squeeze(-1)
    pde_err   = float(jnp.mean((grad_norm - slowness) ** 2))
    print(f"  PDE residual L² on collocation pts: {pde_err:.4e}")

    levels = jnp.linspace(float(source_vals[0]) + 0.05, 1.2, 14)
    extent = [*domain[0], *domain[1]]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    im0 = axes[0].imshow(c_grid, origin="lower", extent=extent, cmap="hot")
    axes[0].set_title("Speed field $c(x)$  (faster = brighter)")
    axes[0].set_xlabel("$x_1$"); axes[0].set_ylabel("$x_2$")
    plt.colorbar(im0, ax=axes[0], label="c(x)")

    im1 = axes[1].imshow(u_pred, origin="lower", extent=extent, cmap="viridis", alpha=0.85)
    axes[1].contour(X1, X2, u_pred, levels=levels, colors="white",
                    linewidths=0.8, alpha=0.9)
    axes[1].set_title(f"SRM travel time (k={k})  PDE-res={pde_err:.1e}\n"
                      r"(isochrones bunch near fast region)")
    axes[1].set_xlabel("$x_1$")
    plt.colorbar(im1, ax=axes[1], label="u(x)")
    axes[1].plot(*x_src, "r*", markersize=8, label="source")
    axes[1].legend(fontsize=8)

    axes[2].semilogy([h["total"] for h in history], lw=1.2, label="total")
    axes[2].semilogy([h["bc"]    for h in history], lw=1.2, label="BC")
    axes[2].semilogy([h["pde"]   for h in history], lw=1.2, label="PDE")
    axes[2].set_xlabel("step"); axes[2].set_ylabel("loss")
    axes[2].set_title("Loss history"); axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("eikonal_variable_speed.png", dpi=150, bbox_inches="tight")
    print("  Saved eikonal_variable_speed.png")
    return params, history


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eikonal SRM solver demo")
    parser.add_argument(
        "--demo", choices=["uniform", "variable", "both"], default="both",
        help="which demo to run",
    )
    parser.add_argument("--k",     type=int,   default=100,  help="splat components")
    parser.add_argument("--steps", type=int,   default=2000, help="training steps")
    parser.add_argument("--lr",    type=float, default=1e-3,  help="learning rate")
    parser.add_argument("--seed",  type=int,   default=42)
    parser.add_argument("--gpu",   action="store_true", help="force GPU backend")
    args = parser.parse_args()

    if args.gpu:
        jax.config.update("jax_platform_name", "gpu")
    jax.config.update("jax_enable_x64", False)

    key = jr.PRNGKey(args.seed)

    if args.demo in ("uniform", "both"):
        key, sk = jr.split(key)
        demo_uniform_speed_2d(sk, k=args.k, num_steps=args.steps, lr=args.lr)

    if args.demo in ("variable", "both"):
        key, sk = jr.split(key)
        demo_variable_speed_2d(sk, k=args.k, num_steps=args.steps, lr=args.lr)

    plt.show()
