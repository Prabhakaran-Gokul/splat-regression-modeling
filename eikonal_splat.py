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
from flax import linen as nn
from tqdm import tqdm

from lib.splat import eval_splat


# ---------------------------------------------------------------------------
# Mother functions
# ---------------------------------------------------------------------------

def gaussian_rho(x):
    """Standard multivariate Gaussian (default). Decays to zero away from center."""
    return jnp.exp(-0.5 * jnp.sum(x ** 2, axis=-1)) / (2 * jnp.pi)


def distance_rho(x):
    """Euclidean distance RBF. Grows linearly away from center.
    Exact single-splat representation of the uniform-speed Eikonal solution:
    with k=1, B=source, A=αI, V=α³ → f(x) = ‖x−B‖ exactly."""
    return jnp.sqrt(jnp.sum(x ** 2, axis=-1) + 1e-12)


# ---------------------------------------------------------------------------
# Core solver
# ---------------------------------------------------------------------------

def eikonal_loss(params, interior_pts, source_pts, source_vals, speed_fn,
                 physics_weight, rho=None):
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
        rho:          mother function (None = Gaussian)
    """
    def u_single(x):
        return eval_splat(x[None, :], params, rho=rho)[0, 0]

    # Boundary condition
    u_bc = eval_splat(source_pts, params, rho=rho).squeeze(-1)   # [n_src]
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
    rho=None,
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
        decay_steps=num_steps, end_value=lr * 0.05,
    )
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(init_params)

    # speed_fn, physics_weight, and rho captured from closure; step is
    # JIT-compiled once per call.  int_pts/src_pts/src_vals are dynamic.
    def _loss(params, int_pts, src_pts, src_vals):
        return eikonal_loss(params, int_pts, src_pts, src_vals,
                            speed_fn, physics_weight, rho)

    @jax.jit
    def step(params, opt_state, int_pts, src_pts, src_vals):
        (total, aux), grads = jax.value_and_grad(
            _loss, has_aux=True
        )(params, int_pts, src_pts, src_vals)
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
    A = jnp.tile(jnp.eye(d)[None] * scale, (k, 1, 1))
    ks = jr.split(kb, d)
    B = jnp.hstack([
        jr.uniform(ks[i], (k, 1), minval=lo, maxval=hi)
        for i, (lo, hi) in enumerate(domain_bounds)
    ])
    # Scale V proportional to ‖B‖: splat at radius r should contribute ~r to u=‖x‖
    r_B = jnp.linalg.norm(B, axis=-1, keepdims=True)   # [k, 1]
    V = jnp.abs(jr.normal(kv, (k, 1))) * 0.3 * (r_B + 0.1)
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


def _eval_grid_2d(params, domain_bounds, Ng=80, rho=None):
    g1 = jnp.linspace(*domain_bounds[0], Ng)
    g2 = jnp.linspace(*domain_bounds[1], Ng)
    X1, X2 = jnp.meshgrid(g1, g2)
    pts = jnp.stack([X1.ravel(), X2.ravel()], axis=1)
    u = eval_splat(pts, params, rho=rho).reshape(Ng, Ng)
    return u, X1, X2


# ---------------------------------------------------------------------------
# Demo 1: 2D uniform speed — analytical solution available
# ---------------------------------------------------------------------------

def demo_uniform_speed_2d(key, k=100, n_int=1000, num_steps=2000, lr=1e-3,
                           eps=0.08, n_ring=32, rho=None):
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
    init_params = init_splat_params(sk, k, 2, domain, scale=0.20)

    params, history = train_eikonal_splat(
        init_params, interior_pts, source_pts, source_vals, speed_fn,
        num_steps=num_steps, lr=lr, physics_weight=1.0, log_interval=500,
        rho=rho,
    )

    u_pred, X1, X2 = _eval_grid_2d(params, domain, rho=rho)
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
# MLP baseline
# ---------------------------------------------------------------------------

class EikonalMLP(nn.Module):
    """Fully-connected network with tanh activations for PINN use.

    tanh is smooth everywhere and infinitely differentiable, which is
    required for jax.grad to produce meaningful ∇u.
    """
    hidden: tuple = (32, 32)

    @nn.compact
    def __call__(self, x):
        for h in self.hidden:
            x = nn.Dense(h)(x)
            x = nn.tanh(x)
        return nn.Dense(1)(x)


def count_params(params) -> int:
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


def train_eikonal_mlp(
    mlp,
    interior_pts,
    source_pts,
    source_vals,
    speed_fn,
    init_key,
    *,
    num_steps=2000,
    lr=1e-3,
    physics_weight=1.0,
    log_interval=500,
):
    """Train a Flax Linen MLP as a PINN on the Eikonal equation.

    Identical loss, optimizer, and schedule to train_eikonal_splat so the
    comparison is fair.
    """
    dummy = jnp.ones((1, interior_pts.shape[-1]))
    params = mlp.init(init_key, dummy)
    print(f"  MLP params: {count_params(params):,}")

    warmup_steps = max(1, num_steps // 10)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=lr, warmup_steps=warmup_steps,
        decay_steps=num_steps, end_value=lr * 0.05,
    )
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(params)

    def loss_fn(params):
        def u_single(x):
            return mlp.apply(params, x[None, :])[0, 0]

        u_bc = mlp.apply(params, source_pts).squeeze(-1)
        bc_loss = jnp.mean((u_bc - source_vals) ** 2)

        grad_u = jax.vmap(jax.grad(u_single))(interior_pts)
        grad_norm = jnp.sqrt(jnp.sum(grad_u ** 2, axis=-1) + 1e-12)
        slowness = 1.0 / speed_fn(interior_pts).squeeze(-1)
        pde_loss = jnp.mean((grad_norm - slowness) ** 2)

        total = bc_loss + physics_weight * pde_loss
        return total, (bc_loss, pde_loss)

    @jax.jit
    def step(params, opt_state):
        (total, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), new_state, total, aux

    history = []
    for i in tqdm(range(num_steps), desc="Eikonal MLP"):
        params, opt_state, total, (bc, pde) = step(params, opt_state)
        history.append({"total": float(total), "bc": float(bc), "pde": float(pde)})
        if (i + 1) % log_interval == 0:
            tqdm.write(
                f"  {i+1:5d}/{num_steps}  "
                f"total={total:.3e}  bc={bc:.3e}  pde={pde:.3e}"
            )

    return params, history


# ---------------------------------------------------------------------------
# Fourier-feature MLP (ntrl-demo inspired)
# ---------------------------------------------------------------------------

class EikonalMLPFourier(nn.Module):
    """MLP with random Fourier feature encoding to mitigate spectral bias.

    Projects input x through a random matrix B, then applies sin/cos to get
    a 2*fourier_dim dimensional feature vector before the MLP layers.
    B is stored as a Flax param (survives serialization) and updated by the
    optimizer — letting it adapt the frequency basis slightly.
    """
    hidden: tuple = (64, 64, 64)
    fourier_dim: int = 128          # sin + cos → 2*fourier_dim features
    fourier_scale: float = 1.0

    @nn.compact
    def __call__(self, x):
        B = self.param('fourier_B',
                       nn.initializers.normal(self.fourier_scale),
                       (x.shape[-1], self.fourier_dim))
        proj = x @ B                # [..., fourier_dim]
        x = jnp.concatenate([jnp.sin(2 * jnp.pi * proj),
                              jnp.cos(2 * jnp.pi * proj)], axis=-1)
        for h in self.hidden:
            x = nn.Dense(h)(x)
            x = nn.tanh(x)
        return nn.Dense(1)(x)


# ---------------------------------------------------------------------------
# Enhanced Eikonal loss (ntrl-demo inspired)
# ---------------------------------------------------------------------------

def eikonal_loss_enhanced(params, interior_pts, source_pts, source_vals,
                           speed_fn, physics_weight, apply_fn,
                           char_weight=1e-3, normal_weight=1e-3,
                           char_step=0.03):
    """
    4-term Eikonal PINN loss inspired by the ntrl-demo methodology.

    Terms:
      1. BC loss  (standard mean-squared error on source ring)
      2. Sqrt-form PDE residual  (√(|∇u|·c) − 1)²  weighted by exp(−0.5·u)
         so near-source points (small u) dominate early training
      3. Characteristic consistency  u(x) − u(x − δ·∇u/|∇u|) = δ
         skipped where u < δ to avoid the singularity at the source
      4. Normal loss at BC ring  ∇u/|∇u| ≈ x/|x|  (radially outward)

    Args:
        apply_fn: apply_fn(params, x: [d]) → scalar  (single-point evaluation)
    """
    def u_single(x):
        return apply_fn(params, x)

    # Interior evaluations
    u_int     = jax.vmap(u_single)(interior_pts)            # [n]
    grad_u    = jax.vmap(jax.grad(u_single))(interior_pts)  # [n, d]
    grad_norm = jnp.sqrt(jnp.sum(grad_u ** 2, axis=-1) + 1e-12)
    speed     = speed_fn(interior_pts).squeeze(-1)           # [n]

    # 1. BC loss
    u_bc    = jax.vmap(u_single)(source_pts)
    bc_loss = jnp.mean((u_bc - source_vals) ** 2)

    # 2. Sqrt-form Eikonal + exponential proximity weighting
    eik_res  = (jnp.sqrt(jnp.clip(grad_norm * speed, 0.0)) - 1.0) ** 2
    weights  = jnp.exp(-0.5 * jnp.abs(u_int))
    pde_loss = jnp.mean(eik_res * weights)

    # 3. Characteristic consistency
    grad_dir = grad_u / grad_norm[:, None]
    x_char   = interior_pts - char_step * grad_dir
    u_char   = jax.vmap(u_single)(x_char)
    char_res = (u_int - u_char - char_step) ** 2
    tau_loss = jnp.mean(jnp.where(u_int < char_step, 0.0, char_res))

    # 4. Normal loss at BC ring
    grad_bc      = jax.vmap(jax.grad(u_single))(source_pts)
    grad_bc_norm = jnp.sqrt(jnp.sum(grad_bc ** 2, axis=-1) + 1e-12)
    grad_bc_dir  = grad_bc / grad_bc_norm[:, None]
    radial_dir   = source_pts / (
        jnp.linalg.norm(source_pts, axis=-1, keepdims=True) + 1e-12
    )
    normal_loss = jnp.mean(jnp.sum((grad_bc_dir - radial_dir) ** 2, axis=-1))

    total = (bc_loss
             + physics_weight * pde_loss
             + char_weight * tau_loss
             + normal_weight * normal_loss)
    return total, (bc_loss, pde_loss, tau_loss, normal_loss)


def train_eikonal_mlp_enhanced(
    mlp,
    interior_pts,
    source_pts,
    source_vals,
    speed_fn,
    init_key,
    *,
    num_steps=2000,
    lr=5e-4,
    physics_weight=1.0,
    char_weight=1e-3,
    normal_weight=1e-3,
    char_step=0.03,
    log_interval=500,
):
    """Train a Flax Linen MLP with the enhanced 4-term Eikonal loss.

    Uses AdamW (weight_decay=0.01) with warmup+cosine schedule, matching the
    spirit of ntrl-demo's training setup.
    """
    dummy  = jnp.ones((1, interior_pts.shape[-1]))
    params = mlp.init(init_key, dummy)
    print(f"  Enhanced MLP params: {count_params(params):,}")

    warmup_steps = max(1, num_steps // 10)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=lr, warmup_steps=warmup_steps,
        decay_steps=num_steps, end_value=lr * 0.05,
    )
    optimizer  = optax.adamw(schedule, weight_decay=0.01)
    opt_state  = optimizer.init(params)

    def apply_fn(p, x):
        return mlp.apply(p, x[None, :])[0, 0]

    def loss_fn(p):
        return eikonal_loss_enhanced(
            p, interior_pts, source_pts, source_vals, speed_fn,
            physics_weight, apply_fn, char_weight, normal_weight, char_step,
        )

    @jax.jit
    def step(params, opt_state):
        (total, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), new_state, total, aux

    history = []
    for i in tqdm(range(num_steps), desc="Enhanced MLP"):
        params, opt_state, total, (bc, pde, tau, norm) = step(params, opt_state)
        history.append({
            "total": float(total), "bc": float(bc),
            "pde": float(pde), "tau": float(tau), "norm": float(norm),
        })
        if (i + 1) % log_interval == 0:
            tqdm.write(
                f"  {i+1:5d}/{num_steps}  "
                f"total={total:.3e}  bc={bc:.3e}  pde={pde:.3e}"
                f"  tau={tau:.3e}  norm={norm:.3e}"
            )

    return params, history


# ---------------------------------------------------------------------------
# SRM vs MLP comparison demo
# ---------------------------------------------------------------------------

def demo_srm_vs_mlp(key, num_steps=2000, lr=1e-3, k=50,
                     eps=0.08, n_ring=32, n_int=1000,
                     mlp_hidden=(32, 32)):
    """Compare Gaussian SRM (k splats) vs MLP PINN on 2D uniform-speed Eikonal.

    Both models use identical Adam+warmup+cosine schedule and the same
    collocation / BC points.
    """
    print("=" * 70)
    print(f"SRM vs MLP — 2D Eikonal |∇u|=1  (analytical: u=‖x‖)")
    print(f"  SRM k={k}  |  MLP hidden={mlp_hidden}  |  steps={num_steps}")
    print("=" * 70)

    domain   = [(-1.0, 1.0), (-1.0, 1.0)]
    x_src    = jnp.array([0.0, 0.0])
    speed_fn = lambda x: jnp.ones((x.shape[0], 1))

    source_pts, source_vals = source_ring_2d(x_src, eps, n_ring, c_at_src=1.0)
    key, sk = jr.split(key)
    interior_pts = _uniform_collocation_2d(sk, n_int, domain,
                                           exclude_center=[0.0, 0.0],
                                           min_dist=eps)

    # --- SRM -----------------------------------------------------------------
    print("\n--- Gaussian SRM ---")
    key, sk = jr.split(key)
    init_params = init_splat_params(sk, k, 2, domain, scale=0.35)
    n_srm = sum(x.size for x in jax.tree_util.tree_leaves(init_params))
    print(f"  SRM params: {n_srm:,}")
    srm_params, srm_hist = train_eikonal_splat(
        init_params, interior_pts, source_pts, source_vals, speed_fn,
        num_steps=num_steps, lr=lr, physics_weight=1.0,
        log_interval=num_steps // 4,
    )
    srm_u, X1, X2 = _eval_grid_2d(srm_params, domain)
    u_true = jnp.sqrt(X1 ** 2 + X2 ** 2)
    srm_mse = float(jnp.mean((srm_u - u_true) ** 2))
    print(f"  Grid MSE: {srm_mse:.4e}")

    # --- MLP -----------------------------------------------------------------
    print("\n--- MLP ---")
    mlp = EikonalMLP(hidden=mlp_hidden)
    key, sk = jr.split(key)
    mlp_params, mlp_hist = train_eikonal_mlp(
        mlp, interior_pts, source_pts, source_vals, speed_fn, sk,
        num_steps=num_steps, lr=lr, physics_weight=1.0,
        log_interval=num_steps // 4,
    )
    g1 = jnp.linspace(*domain[0], 80)
    g2 = jnp.linspace(*domain[1], 80)
    G1, G2 = jnp.meshgrid(g1, g2)
    grid_pts = jnp.stack([G1.ravel(), G2.ravel()], axis=1)
    mlp_u = mlp.apply(mlp_params, grid_pts).reshape(80, 80)
    mlp_mse = float(jnp.mean((mlp_u - u_true) ** 2))
    print(f"  Grid MSE: {mlp_mse:.4e}")

    # --- Plot ----------------------------------------------------------------
    levels = jnp.linspace(0.1, 1.3, 13)
    extent = [-1, 1, -1, 1]
    vmin, vmax = 0.0, float(jnp.max(u_true))

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    rows = [
        (f"SRM Gaussian k={k}  ({n_srm} params)\nMSE={srm_mse:.2e}", srm_u, srm_hist),
        (f"MLP tanh {mlp_hidden}  ({count_params(mlp_params)} params)\nMSE={mlp_mse:.2e}", mlp_u, mlp_hist),
    ]
    for row_i, (title, u_pred, hist) in enumerate(rows):
        ax_pred = axes[row_i][0]
        im = ax_pred.imshow(u_pred, origin="lower", extent=extent,
                            vmin=vmin, vmax=vmax, cmap="viridis", alpha=0.85)
        ax_pred.contour(X1, X2, u_pred, levels=levels, colors="white",
                        linewidths=0.8, alpha=0.9)
        ax_pred.set_title(title, fontsize=9)
        ax_pred.set_xlabel("$x_1$"); ax_pred.set_ylabel("$x_2$")
        plt.colorbar(im, ax=ax_pred)
        ax_pred.plot(*x_src, "r*", markersize=8)

        ax_err = axes[row_i][1]
        err = jnp.abs(u_pred - u_true)
        im2 = ax_err.imshow(err, origin="lower", extent=extent, cmap="hot", vmin=0)
        ax_err.set_title("|error| vs u=‖x‖")
        ax_err.set_xlabel("$x_1$"); ax_err.set_ylabel("$x_2$")
        plt.colorbar(im2, ax=ax_err)

        ax_loss = axes[row_i][2]
        ax_loss.semilogy([h["total"] for h in hist], lw=1.2, label="total")
        ax_loss.semilogy([h["bc"]    for h in hist], lw=1.2, label="BC")
        ax_loss.semilogy([h["pde"]   for h in hist], lw=1.2, label="PDE")
        ax_loss.set_xlabel("step"); ax_loss.set_ylabel("loss")
        ax_loss.set_title("Loss history"); ax_loss.legend(fontsize=8)
        ax_loss.grid(True, alpha=0.3)

    fig.suptitle(
        "Gaussian SRM vs MLP — 2D Eikonal PINN |∇u|=1  (analytical: u=‖x‖)",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig("eikonal_srm_vs_mlp.png", dpi=150, bbox_inches="tight")
    print("\n  Saved eikonal_srm_vs_mlp.png")
    return srm_params, srm_hist, mlp_params, mlp_hist


# ---------------------------------------------------------------------------
# RHO comparison demo
# ---------------------------------------------------------------------------

def demo_rho_comparison(key, num_steps=2000, lr=1e-3, eps=0.08, n_ring=32,
                         n_int=1000):
    """
    Side-by-side comparison of Gaussian vs distance-RBF mother functions on
    the uniform-speed Eikonal (analytical solution u=‖x‖).

    Runs three configurations:
      (A) Gaussian,     k=50
      (B) Distance RBF, k=50
      (C) Distance RBF, k=1   — theoretically exact with correct V, B
    """
    print("=" * 70)
    print("RHO COMPARISON: Gaussian vs Distance RBF on 2D Eikonal |∇u|=1")
    print("=" * 70)

    domain  = [(-1.0, 1.0), (-1.0, 1.0)]
    x_src   = jnp.array([0.0, 0.0])
    speed_fn = lambda x: jnp.ones((x.shape[0], 1))

    source_pts, source_vals = source_ring_2d(x_src, eps, n_ring, c_at_src=1.0)
    key, sk = jr.split(key)
    interior_pts = _uniform_collocation_2d(sk, n_int, domain,
                                           exclude_center=[0.0, 0.0],
                                           min_dist=eps)
    u_true_fn = lambda X1, X2: jnp.sqrt(X1 ** 2 + X2 ** 2)

    configs = [
        ("Gaussian k=50",      gaussian_rho, 50),
        ("Distance RBF k=50",  distance_rho, 50),
        ("Distance RBF k=1",   distance_rho,  1),
    ]

    results = []
    for label, rho, k in configs:
        print(f"\n--- {label} ---")
        key, sk = jr.split(key)
        init_params = init_splat_params(sk, k, 2, domain, scale=0.35)
        params, history = train_eikonal_splat(
            init_params, interior_pts, source_pts, source_vals, speed_fn,
            num_steps=num_steps, lr=lr, physics_weight=1.0,
            log_interval=num_steps // 4, rho=rho,
        )
        u_pred, X1, X2 = _eval_grid_2d(params, domain, rho=rho)
        u_true = u_true_fn(X1, X2)
        mse = float(jnp.mean((u_pred - u_true) ** 2))
        print(f"  Grid MSE: {mse:.4e}")
        results.append((label, rho, k, params, history, u_pred, mse))

    # ---- Plot ---------------------------------------------------------------
    levels = jnp.linspace(0.1, 1.3, 13)
    extent = [-1, 1, -1, 1]
    _, X1, X2 = _eval_grid_2d(results[0][3], domain, rho=results[0][1])
    u_true = u_true_fn(X1, X2)
    vmin, vmax = 0.0, float(jnp.max(u_true))

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    # Row 0: analytical truth + the three predictions
    for col, (label, rho, k, params, history, u_pred, mse) in enumerate(results):
        ax_pred = axes[col][0]
        im = ax_pred.imshow(u_pred, origin="lower", extent=extent,
                            vmin=vmin, vmax=vmax, cmap="viridis", alpha=0.85)
        ax_pred.contour(X1, X2, u_pred, levels=levels, colors="white",
                        linewidths=0.8, alpha=0.9)
        ax_pred.set_title(f"{label}\nMSE={mse:.2e}")
        ax_pred.set_xlabel("$x_1$"); ax_pred.set_ylabel("$x_2$")
        plt.colorbar(im, ax=ax_pred)
        ax_pred.plot(*x_src, "r*", markersize=8)

        ax_err = axes[col][1]
        err = jnp.abs(u_pred - u_true)
        im2 = ax_err.imshow(err, origin="lower", extent=extent,
                             cmap="hot", vmin=0)
        ax_err.set_title("|error|")
        ax_err.set_xlabel("$x_1$"); ax_err.set_ylabel("$x_2$")
        plt.colorbar(im2, ax=ax_err)

        ax_loss = axes[col][2]
        ax_loss.semilogy([h["total"] for h in history], lw=1.2, label="total")
        ax_loss.semilogy([h["bc"]    for h in history], lw=1.2, label="BC")
        ax_loss.semilogy([h["pde"]   for h in history], lw=1.2, label="PDE")
        ax_loss.set_xlabel("step"); ax_loss.set_ylabel("loss")
        ax_loss.set_title("Loss history"); ax_loss.legend(fontsize=7)
        ax_loss.grid(True, alpha=0.3)

    # Row labels
    for row, (label, *_) in enumerate(results):
        axes[row][0].set_ylabel(label + "\n$x_2$", fontsize=8)

    fig.suptitle(
        "Mother function comparison — 2D Eikonal |∇u|=1  (analytical: u=‖x‖)",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig("eikonal_rho_comparison.png", dpi=150, bbox_inches="tight")
    print("\n  Saved eikonal_rho_comparison.png")
    return results


# ---------------------------------------------------------------------------
# 3-way enhanced comparison demo
# ---------------------------------------------------------------------------

def demo_enhanced_comparison(key, num_steps=2000, lr=1e-3, k=50,
                              eps=0.08, n_ring=32, n_int=1000,
                              mlp_hidden=(32, 32),
                              fourier_hidden=(64, 64, 64), fourier_dim=128):
    """
    Compare on 2D uniform-speed Eikonal (analytical u=‖x‖):
      - Gaussian SRM (k splats, standard 2-term loss)
      - Vanilla MLP  (tanh, standard 2-term loss)
      - Fourier MLP  (random Fourier features, 4-term ntrl-demo loss)
    """
    print("=" * 70)
    print("ENHANCED COMPARISON: SRM vs Vanilla MLP vs Fourier MLP")
    print(f"  steps={num_steps}  k={k}  lr={lr}")
    print("=" * 70)

    domain   = [(-1.0, 1.0), (-1.0, 1.0)]
    x_src    = jnp.array([0.0, 0.0])
    speed_fn = lambda x: jnp.ones((x.shape[0], 1))

    source_pts, source_vals = source_ring_2d(x_src, eps, n_ring, c_at_src=1.0)
    key, sk = jr.split(key)
    interior_pts = _uniform_collocation_2d(sk, n_int, domain,
                                           exclude_center=[0.0, 0.0],
                                           min_dist=eps)

    g1 = jnp.linspace(*domain[0], 80)
    g2 = jnp.linspace(*domain[1], 80)
    G1, G2 = jnp.meshgrid(g1, g2)
    grid_pts = jnp.stack([G1.ravel(), G2.ravel()], axis=1)
    u_true = jnp.sqrt(G1 ** 2 + G2 ** 2)

    # --- Gaussian SRM ---
    print("\n--- Gaussian SRM ---")
    key, sk = jr.split(key)
    init_params = init_splat_params(sk, k, 2, domain, scale=0.35)
    n_srm = sum(x.size for x in jax.tree_util.tree_leaves(init_params))
    print(f"  SRM params: {n_srm:,}")
    srm_params, srm_hist = train_eikonal_splat(
        init_params, interior_pts, source_pts, source_vals, speed_fn,
        num_steps=num_steps, lr=lr, physics_weight=1.0,
        log_interval=num_steps // 4,
    )
    srm_u   = eval_splat(grid_pts, srm_params).reshape(80, 80)
    srm_mse = float(jnp.mean((srm_u - u_true) ** 2))
    print(f"  Grid MSE: {srm_mse:.4e}")

    # --- Vanilla MLP ---
    print("\n--- Vanilla MLP ---")
    vanilla_mlp = EikonalMLP(hidden=mlp_hidden)
    key, sk = jr.split(key)
    vanilla_params, vanilla_hist = train_eikonal_mlp(
        vanilla_mlp, interior_pts, source_pts, source_vals, speed_fn, sk,
        num_steps=num_steps, lr=lr, physics_weight=1.0,
        log_interval=num_steps // 4,
    )
    vanilla_u   = vanilla_mlp.apply(vanilla_params, grid_pts).reshape(80, 80)
    vanilla_mse = float(jnp.mean((vanilla_u - u_true) ** 2))
    print(f"  Grid MSE: {vanilla_mse:.4e}")

    # --- Fourier MLP with enhanced loss ---
    print("\n--- Fourier MLP (4-term enhanced loss) ---")
    fourier_mlp = EikonalMLPFourier(hidden=fourier_hidden, fourier_dim=fourier_dim)
    key, sk = jr.split(key)
    fourier_params, fourier_hist = train_eikonal_mlp_enhanced(
        fourier_mlp, interior_pts, source_pts, source_vals, speed_fn, sk,
        num_steps=num_steps, lr=5e-4, physics_weight=1.0,
        char_weight=1e-3, normal_weight=1e-3, char_step=0.03,
        log_interval=num_steps // 4,
    )
    fourier_u   = fourier_mlp.apply(fourier_params, grid_pts).reshape(80, 80)
    fourier_mse = float(jnp.mean((fourier_u - u_true) ** 2))
    print(f"  Grid MSE: {fourier_mse:.4e}")

    # --- Plot ---
    levels  = jnp.linspace(0.1, 1.3, 13)
    extent  = [-1, 1, -1, 1]
    vmin, vmax = 0.0, float(jnp.max(u_true))
    n_vanilla = count_params(vanilla_params)
    n_fourier = count_params(fourier_params)

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    rows = [
        (f"Gaussian SRM k={k}  ({n_srm} params)\nMSE={srm_mse:.2e}",
         srm_u, srm_hist, ["total", "bc", "pde"]),
        (f"Vanilla MLP {mlp_hidden}  ({n_vanilla} params)\nMSE={vanilla_mse:.2e}",
         vanilla_u, vanilla_hist, ["total", "bc", "pde"]),
        (f"Fourier MLP {fourier_hidden}  ({n_fourier} params)\nMSE={fourier_mse:.2e}",
         fourier_u, fourier_hist, ["total", "bc", "pde", "tau", "norm"]),
    ]
    for row_i, (title, u_pred, hist, loss_keys) in enumerate(rows):
        ax_pred = axes[row_i][0]
        im = ax_pred.imshow(u_pred, origin="lower", extent=extent,
                            vmin=vmin, vmax=vmax, cmap="viridis", alpha=0.85)
        ax_pred.contour(G1, G2, u_pred, levels=levels, colors="white",
                        linewidths=0.8, alpha=0.9)
        ax_pred.set_title(title, fontsize=9)
        ax_pred.set_xlabel("$x_1$"); ax_pred.set_ylabel("$x_2$")
        plt.colorbar(im, ax=ax_pred)
        ax_pred.plot(*x_src, "r*", markersize=8)

        ax_err = axes[row_i][1]
        err = jnp.abs(u_pred - u_true)
        im2 = ax_err.imshow(err, origin="lower", extent=extent, cmap="hot", vmin=0)
        ax_err.set_title("|error| vs u=‖x‖")
        ax_err.set_xlabel("$x_1$"); ax_err.set_ylabel("$x_2$")
        plt.colorbar(im2, ax=ax_err)

        ax_loss = axes[row_i][2]
        for lk in loss_keys:
            vals = [h[lk] for h in hist if lk in h]
            if vals:
                ax_loss.semilogy(vals, lw=1.2, label=lk)
        ax_loss.set_xlabel("step"); ax_loss.set_ylabel("loss")
        ax_loss.set_title("Loss history"); ax_loss.legend(fontsize=7)
        ax_loss.grid(True, alpha=0.3)

    fig.suptitle(
        "SRM vs Vanilla MLP vs Fourier MLP (ntrl-demo enhanced)\n"
        "2D Eikonal |∇u|=1  (analytical: u=‖x‖)",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig("eikonal_enhanced_comparison.png", dpi=150, bbox_inches="tight")
    print("\n  Saved eikonal_enhanced_comparison.png")
    return srm_params, vanilla_params, fourier_params


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eikonal SRM solver demo")
    parser.add_argument(
        "--demo", choices=["uniform", "variable", "both", "compare", "mlp", "enhanced"],
        default="both", help="which demo to run",
    )
    parser.add_argument("--mlp-hidden", type=int, nargs="+", default=[32, 32],
                        help="MLP hidden layer widths (e.g. --mlp-hidden 32 32 32)")
    parser.add_argument("--k",     type=int,   default=100,  help="splat components")
    parser.add_argument("--steps", type=int,   default=2000, help="training steps")
    parser.add_argument("--lr",    type=float, default=3e-3,  help="learning rate")
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

    if args.demo == "compare":
        key, sk = jr.split(key)
        demo_rho_comparison(sk, num_steps=args.steps, lr=args.lr)

    if args.demo == "mlp":
        key, sk = jr.split(key)
        demo_srm_vs_mlp(sk, num_steps=args.steps, lr=args.lr, k=args.k,
                         mlp_hidden=tuple(args.mlp_hidden))

    if args.demo == "enhanced":
        key, sk = jr.split(key)
        demo_enhanced_comparison(sk, num_steps=args.steps, lr=args.lr, k=args.k,
                                  mlp_hidden=tuple(args.mlp_hidden))

    plt.show()
