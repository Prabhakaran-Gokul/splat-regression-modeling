"""
Wrapped Gaussian SRM on Riemannian manifolds.

Pushforward definition (Said et al. / Chevallier et al.):

    N_w(x) = N(ψ⁻¹(x)) · |det ∂ψ⁻¹/∂x|

where ψ = Exp_μ is the exponential map at the mean, so ψ⁻¹ = Log_μ and
the Jacobian factor corrects for the curvature of the manifold.

Currently implements S² (2-sphere embedded in R³).

Parameter convention:
    V: [k, p]    output weights
    A: [k, d, d] covariance factor in tangent space  (d = manifold dimension)
    B: [k, 3]    centers on S² (unit vectors in R³)
"""

import jax
import jax.numpy as jnp
import jax.random as jr


# ── General wrapped Gaussian ────────────────────────────────────────────────────

def eval_wrapped_gaussian(x, mu, A, log_map_fn, jac_factor_fn, dim):
    """
    N_w(x) = N(ψ⁻¹(x); 0, A) · |det ∂ψ⁻¹/∂x|

    To design a WGD it is sufficient to know the diffeomorphism ψ (here
    Exp_μ), its inverse ψ⁻¹ = Log_μ, and the corresponding Jacobian.

    Args:
        x:             point on the manifold
        mu:            mean / center
        A:             [dim, dim] covariance factor in tangent space at mu
        log_map_fn:    (mu, x) → [dim]   tangent coordinates  (= ψ⁻¹)
        jac_factor_fn: (mu, x) → scalar  |det ∂ψ⁻¹/∂x|
        dim:           manifold dimension (= size of log_map output)

    Returns:
        scalar density value
    """
    v     = log_map_fn(mu, x)                                          # [dim]
    z     = jnp.linalg.solve(A, v)
    det_A = jnp.abs(jnp.linalg.det(A))
    N_v   = jnp.exp(-0.5 * jnp.dot(z, z)) / (
                (2.0 * jnp.pi) ** (dim / 2.0) * (det_A + 1e-12))
    return N_v * jac_factor_fn(mu, x)


# ── S² geometry primitives ──────────────────────────────────────────────────────

def sphere_frame(mu):
    """Orthonormal frame (e1, e2) for T_μS². Uses jnp.maximum for AD safety."""
    x_hat = jnp.array([1.0, 0.0, 0.0])
    y_hat = jnp.array([0.0, 1.0, 0.0])
    near_x = jnp.abs(jnp.dot(mu, x_hat)) > 0.9
    ref    = jnp.where(near_x, y_hat, x_hat)
    e1_raw = ref - jnp.dot(ref, mu) * mu
    e1     = e1_raw / jnp.maximum(jnp.linalg.norm(e1_raw), 1e-8)
    e2     = jnp.cross(mu, e1)
    return e1, e2


def sphere_exp_map(mu, v):
    """Exp map on S²: from base mu along 3D tangent vector v."""
    norm_v   = jnp.linalg.norm(v)
    safe_dir = jnp.where(norm_v > 1e-8, v / norm_v, jnp.zeros_like(v))
    return jnp.cos(norm_v) * mu + jnp.sin(norm_v) * safe_dir


def sphere_log_map(mu, x):
    """3D Log map on S² (post-processing / geodesic distance only).

    NOTE: contains arccos singularity at x = mu.  For density computation
    use sphere_log_map_2d which is safe for second-order AD.
    """
    cos_theta = jnp.clip(jnp.dot(mu, x), -1.0 + 1e-7, 1.0 - 1e-7)
    theta     = jnp.arccos(cos_theta)
    perp      = x - jnp.dot(x, mu) * mu
    return theta * (perp / jnp.maximum(jnp.linalg.norm(perp), 1e-8))


def sphere_log_map_2d(mu, x):
    """Log_μ(x) expressed as 2D coordinates in the tangent frame at mu.

    Returns v = [⟨Log_μ(x), e₁⟩, ⟨Log_μ(x), e₂⟩] where (e₁,e₂) comes
    from sphere_frame(mu).  The norm ‖v‖ = θ = d(μ,x) (geodesic distance),
    so the wrapped Gaussian distinguishes θ from π−θ — no hemisphere
    symmetry problem.

    AD safety: arccos input clipped away from ±1; perp normalised with
    jnp.maximum (no jnp.where in the differentiable path).
    """
    # Clip at ±(1-1e-3) bounds θ/sinθ ≤ ~70, preventing large gradients when
    # a center is near-antipodal to a collocation point.
    cos_theta = jnp.clip(jnp.dot(mu, x), -1.0 + 1e-3, 1.0 - 1e-3)
    theta     = jnp.arccos(cos_theta)
    perp      = x - cos_theta * mu
    e_perp    = perp / jnp.maximum(jnp.linalg.norm(perp), 1e-8)
    e1, e2    = sphere_frame(mu)
    return theta * jnp.array([jnp.dot(e_perp, e1), jnp.dot(e_perp, e2)])


def sphere_log_jac_factor(mu, x):
    """Jacobian correction |det ∂Log_μ/∂x| = θ / sin θ for S².

    Inverse of the Riemannian volume element: det d Exp_μ|_v = sin(‖v‖)/‖v‖.
    Limit at θ→0 is 1; clip keeps θ away from the cut locus (θ=π).
    """
    cos_theta = jnp.clip(jnp.dot(mu, x), -1.0 + 1e-3, 1.0 - 1e-3)
    theta     = jnp.arccos(cos_theta)
    return theta / jnp.maximum(jnp.sin(theta), 1e-8)


# ── S² wrapped Gaussian density ─────────────────────────────────────────────────

def eval_sphere_rho_single(x, mu, A):
    """
    Wrapped Gaussian on S²:  N_w(x; μ, A) = N(Log_μ(x); 0, A) · (θ/sin θ)

    A is [2, 2] — covariance factor in the tangent plane at μ (2D manifold).
    θ = arccos(⟨μ,x⟩) is the geodesic distance; the factor θ/sin θ is the
    inverse Riemannian volume element that corrects for sphere curvature.

    Args:
        x:   (3,) unit vector on S²
        mu:  (3,) unit vector on S² (center)
        A:   (2, 2) covariance factor in the tangent plane
    """
    return eval_wrapped_gaussian(
        x, mu, A,
        log_map_fn    = sphere_log_map_2d,
        jac_factor_fn = sphere_log_jac_factor,
        dim           = 2,
    )


# ── Sphere SRM ──────────────────────────────────────────────────────────────────

def eval_sphere_splat(X, splatnn):
    """
    Evaluate Sphere SRM at X.

        f(x) = Σ_j V[j] · N_w(x | B[j], A[j])

    Args:
        X:       [n, 3] unit vectors on S²
        splatnn: (V [k,p], A [k,2,2], B [k,3])

    Returns:
        [n, p]
    """
    V, A, B = splatnn

    def rho_at_x(x):
        return jax.vmap(lambda mu, Ak: eval_sphere_rho_single(x, mu, Ak))(B, A)

    return jax.vmap(rho_at_x)(X) @ V


eval_sphere_splat_jit = jax.jit(eval_sphere_splat)


def init_sphere_splat(key, k, p=1, sigma=0.3):
    """
    Initialize sphere SRM parameters.

    A: [k, 2, 2] — isotropic covariance σ·I₂ in the tangent plane at each center.
    B: [k, 3]    — random unit vectors on S².
    V: [k, p]    — small random weights.
    """
    k1, k2 = jr.split(key)
    B_raw = jr.normal(k1, (k, 3))
    B     = B_raw / jnp.linalg.norm(B_raw, axis=-1, keepdims=True)
    A     = jnp.tile(jnp.eye(2) * sigma, (k, 1, 1))   # [k, 2, 2]
    V     = jr.normal(k2, (k, p)) * 0.1
    return V, A, B


# ── SE(2) = Dubins car configuration space ─────────────────────────────────────
#
# SE(2) = R² ⋊ S¹ parametrised as g = (x, y, θ).  The Dubins car has a
# sub-Riemannian structure: only forward motion (X₁) and rotation (X₂) are
# directly controllable.  The wrapped Gaussian uses the LEFT-INVARIANT Lie
# group Log map, which gives a 3D embedding of SE(2) onto its Lie algebra
# se(2) ≅ R³ with coordinates (v_fwd, v_lat, v_rot).  A sub-Riemannian
# Gaussian is then obtained by setting σ_lat ≪ σ_fwd in the 3×3 covariance.
#
# Jacobian derivation (det ∂Log_h/∂g = (dt / (2·sin(dt/2)))²):
#   The SE(2) Exp map has Jacobian det = (2·sin(dt/2)/dt)², so by the inverse
#   function theorem the Log map Jacobian is the reciprocal squared.
#   For dt ∈ (−π, π) this is bounded: min = 1 at dt=0, max = (π/2)² ≈ 2.47.


def _wrap_angle(a):
    """Wrap angle to (−π, π]."""
    return (a + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


def se2_log_map(h, g):
    """
    Left-invariant Log map on SE(2): Log_h(g) → v ∈ se(2).

    Returns (v_fwd, v_lat, v_rot): the Lie algebra vector such that
    se2_exp_map(h, v) = g.

    h, g: (3,) arrays [x, y, θ]

    AD note: uses the JAX-safe "substitute safe input" pattern inside jnp.where
    so that BOTH branches have bounded gradients everywhere — necessary for the
    nested second-order differentiation that appears in PINN training.
    """
    x0, y0, t0 = h[0], h[1], h[2]
    x1, y1, t1 = g[0], g[1], g[2]

    dt  = _wrap_angle(t1 - t0)
    dx  = x1 - x0
    dy  = y1 - y0

    # Relative displacement in the body frame of h
    dxb =  jnp.cos(t0) * dx + jnp.sin(t0) * dy
    dyb = -jnp.sin(t0) * dx + jnp.cos(t0) * dy

    # g1 = (dt/2)·cot(dt/2) = half_dt·cos_half / sin_half   →  1 as dt→0
    # g2 = dt/2
    #
    # JAX-safe evaluation: for |dt| < THRESH use the Taylor series (always smooth),
    # for |dt| ≥ THRESH use the exact formula with "safe" denominator that keeps
    # the non-selected branch's gradient finite (avoids the classic jnp.where NaN
    # gradient bug that appears when one branch blows up even if not selected).
    _THRESH = 1e-2
    half_dt  = dt / 2.0
    cos_half = jnp.cos(half_dt)
    sin_half = jnp.sin(half_dt)

    cond = jnp.abs(dt) > _THRESH
    safe_sin  = jnp.where(cond, sin_half, jnp.ones_like(sin_half))  # non-NaN in formula
    g1_exact  = half_dt * cos_half / safe_sin                        # exact when cond
    g1_taylor = 1.0 - dt ** 2 / 12.0 + dt ** 4 / 720.0              # Taylor when !cond
    g1 = jnp.where(cond, g1_exact, g1_taylor)
    g2 = half_dt

    v_fwd =  g1 * dxb + g2 * dyb
    v_lat = -g2 * dxb + g1 * dyb
    v_rot = dt

    return jnp.array([v_fwd, v_lat, v_rot])


def se2_log_jac_factor(h, g):
    """
    |det ∂Log_h/∂g| = (dt / (2·sin(dt/2)))²  for SE(2).

    Bounded on (−π, π): equals 1 at dt=0, reaches (π/2)² ≈ 2.47 at |dt|=π.

    AD note: same JAX-safe jnp.where pattern as se2_log_map.
    """
    dt       = _wrap_angle(g[2] - h[2])
    half_dt  = dt / 2.0
    sin_half = jnp.sin(half_dt)

    _THRESH  = 1e-3
    cond     = jnp.abs(dt) > _THRESH
    safe_sin = jnp.where(cond, sin_half, jnp.ones_like(sin_half))
    f_exact  = half_dt / safe_sin                        # (dt/2)/sin(dt/2) when cond
    f_taylor = 1.0 + dt ** 2 / 24.0 + 7.0 * dt ** 4 / 5760.0   # Taylor when !cond
    f        = jnp.where(cond, f_exact, f_taylor)
    return f * f


def se2_exp_map(h, v):
    """
    Left-invariant Exp map on SE(2): Exp_h(v_fwd, v_lat, v_rot).

    Computes the endpoint of the arc: forward speed v_fwd, lateral v_lat,
    turning rate v_rot, driven for unit time starting from h.
    """
    x0, y0, t0 = h[0], h[1], h[2]
    v1, v2, v3 = v[0], v[1], v[2]

    s  = jnp.sin(v3)
    c  = jnp.cos(v3)
    sv = jnp.where(jnp.abs(v3) > 1e-8, v3, jnp.sign(v3 + 1e-30) * 1e-8)

    # Body-frame endpoint
    dxb = jnp.where(jnp.abs(v3) > 1e-6, (v1 * s - v2 * (1.0 - c)) / sv, v1)
    dyb = jnp.where(jnp.abs(v3) > 1e-6, (v1 * (1.0 - c) + v2 * s) / sv, v2)

    # Rotate to world frame
    x1 = x0 + jnp.cos(t0) * dxb - jnp.sin(t0) * dyb
    y1 = y0 + jnp.sin(t0) * dxb + jnp.cos(t0) * dyb
    t1 = t0 + v3
    return jnp.array([x1, y1, t1])


def eval_se2_rho_single(g, h, A):
    """
    Wrapped Gaussian on SE(2) / Dubins car:

        N_w(g; h, A) = N(Log_h(g); 0, A) · (dt / (2·sin(dt/2)))²

    A is [3, 3] in Lie algebra coordinates (v_fwd, v_lat, v_rot).
    Sub-Riemannian constraint: use A = diag(σ_fwd, σ_lat, σ_rot) with
    σ_lat ≪ σ_fwd to confine spread to the forward-motion direction.

    Args:
        g: (3,) query configuration  [x, y, θ]
        h: (3,) center configuration [x₀, y₀, θ₀]
        A: (3, 3) Lie algebra covariance factor
    """
    return eval_wrapped_gaussian(
        g, h, A,
        log_map_fn    = se2_log_map,
        jac_factor_fn = se2_log_jac_factor,
        dim           = 3,
    )


def eval_se2_splat(G, splatnn):
    """
    Evaluate SE(2) SRM at configurations G.

        f(g) = Σ_j V[j] · N_w(g | H[j], A[j])

    Args:
        G:       [n, 3] configurations [x, y, θ]
        splatnn: (V [k,p], A [k,3,3], H [k,3])
    """
    V, A, H = splatnn

    def rho_at_g(g):
        return jax.vmap(lambda h, Ak: eval_se2_rho_single(g, h, Ak))(H, A)

    return jax.vmap(rho_at_g)(G) @ V


eval_se2_splat_jit = jax.jit(eval_se2_splat)


def init_se2_splat(key, k, p=1, sigma_fwd=0.5, sigma_lat=0.1, sigma_rot=0.4,
                   xy_range=2.0):
    """
    Initialize SE(2) SRM with sub-Riemannian anisotropy.

    A: [k, 3, 3] — diagonal diag(σ_fwd, σ_lat, σ_rot) in se(2) coordinates.
    H: [k, 3]    — random configurations in [-xy_range,xy_range]² × [−π,π].
    V: [k, p]    — small random weights.
    """
    k1, k2  = jr.split(key)
    xy      = jr.uniform(k1, (k, 2), minval=-xy_range, maxval=xy_range)
    theta   = jr.uniform(k1, (k, 1), minval=-jnp.pi, maxval=jnp.pi)
    H       = jnp.concatenate([xy, theta], axis=-1)
    A_diag  = jnp.array([sigma_fwd, sigma_lat, sigma_rot])
    A       = jnp.tile(jnp.diag(A_diag), (k, 1, 1))
    V       = jr.normal(k2, (k, p)) * 0.1
    return V, A, H
