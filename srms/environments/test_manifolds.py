"""Falsifiable checks on each Environment's manifold plumbing — no ground truth, no training.

This is the *correctness criterion* behind the claim that the SRM ports across manifolds by swapping
two textbook functions. Each check has an exact expected value, so a wrong ``log_map``,
``jac_factor`` or ``metric_inv`` fails here rather than surfacing later as a mediocre RMS that looks
like a tuning problem.

Run: ``python -m srms.environments.test_manifolds`` (exit code 1 on any failure).

    1. round trip      Exp_mu(Log_mu(x)) == x                    log/exp really are inverses
    2. isometry        ‖Log_mu(x)‖ == geodesic(x, mu)            tangent coords carry Riemannian
                                                                 units, not raw chart coordinates
    3. jacobian        jac_factor == |det ∂Log_mu/∂x| (autodiff) the curvature term — the check that
                                                                 separates sin from sinh
    4. eikonal         ‖∇ geodesic(·, start)‖_g == 1             couples log_map and metric_inv into
                                                                 the exact residual the solver minimises
    5. normalisation   ∫ N_w dvol == closed-form contained mass  the wrapped Gaussian is a density,
                                                                 and its shortfall has a named cause

Check 3 needs the determinant of ``∂Log_mu/∂x`` between *orthonormal* frames, which differ per
manifold (the sphere's tangent plane is a rank-n subspace of R^{n+1}; hyperbolic's is R^d rescaled by
the conformal factor). Hand-coding one frame per manifold would undercut the claim being tested, so
the frame is derived generically from ``metric_inv``: an orthonormal frame is g^(-1/2), read off the
eigendecomposition of g^(-1) with null directions dropped. That recovers I for the torus, (1/λ)·I for
hyperbolic and the tangent-plane basis for the sphere, with no per-manifold branch in this file.

Check 5 runs at two scales and compares against ``containment_mass``'s closed form rather than
against 1, because at a wide scale none of the three integrates to 1 and the *reasons* differ. The
torus and sphere use a single-image (principal-branch) wrapped Gaussian, so they lose tail mass past
a cut locus — a genuine limit of the representation on those manifolds. H^d has no cut locus at all
(Exp is a global diffeomorphism), so nothing is lost to curvature; its shortfall comes only from the
truncation wall this repo chooses, which at the default ``trunc_radius=0.9`` sits at hyperbolic radius
2.94 — *nearer* than the other two manifolds' cut loci at π. So hyperbolic shows the largest measured
shortfall while having the only exact representation, which is exactly why the check reports cause
alongside magnitude instead of ranking the manifolds by a single number.
"""

from __future__ import annotations

import math
import sys

import jax
import jax.numpy as jnp
import numpy as np

from srms.environments import ENVIRONMENTS
from srms.lib.manifold_splat import eval_wrapped_gaussian

jax.config.update("jax_enable_x64", True)  # exactness checks, not training

EXACT_TOL = 1e-5
NARROW, WIDE = 0.4, 1.2  # Gaussian scales: inside vs outside the torus/sphere injectivity radius
MC_SAMPLES = 400_000
MC_TOL = 1.5e-2  # Monte-Carlo noise floor at MC_SAMPLES, not a modelling tolerance


# ---- generic helpers (no per-manifold branches) -----------------------------------------------


def orthonormal_frame(env, x: jnp.ndarray) -> jnp.ndarray:
    """Orthonormal basis of T_x M as columns [dim, tangent_dim], derived from g^(-1) = metric_inv.

    E = g^(-1/2) satisfies EᵀgE = I. Eigen-decomposing the symmetric g^(-1) and keeping the
    ``tangent_dim`` largest eigenvalues drops the null direction an embedded manifold's projector
    carries (the sphere's radial direction), so one expression covers embedded and chart manifolds.
    """
    eigvals, eigvecs = jnp.linalg.eigh(env.metric_inv(x))
    keep = jnp.argsort(-eigvals)[: env.tangent_dim]
    return eigvecs[:, keep] * jnp.sqrt(eigvals[keep])


def region_volume(env) -> float:
    """Riemannian volume of the region ``sample_domain`` covers.

    Every environment here samples uniformly w.r.t. its own Riemannian volume — the torus because it
    is flat, the sphere and SO(3) because a normalised Gaussian is, and hyperbolic because its
    sampler inverts the radial volume CDF. So the Monte-Carlo weight is simply that volume, with no
    per-point density correction. An earlier version applied a chart-uniform correction
    (chart volume x sqrt(det g)), which was right for hyperbolic until its sampler was changed to
    volume-uniform and then double-counted the factor — the mass check read 0.23 instead of 1.00.
    """
    return env.volume


def points_near_source(env, n: int, seed: int, lo: float, hi: float) -> np.ndarray:
    """``n`` points whose geodesic distance to ``env.start`` lies in (lo, hi).

    Keeps check 4 away from the source (where ∇d is undefined) and the cut locus (where it flips).
    """
    rng = np.random.default_rng(seed)
    pool = env.sample_domain(rng, 40 * n)
    dist = np.linalg.norm(env.displacement_np(np.asarray(env.start, dtype=float), pool), axis=-1)
    return pool[(dist > lo) & (dist < hi)][:n]


def sample_pairs(env, n: int, seed: int, max_sep: float) -> tuple[np.ndarray, np.ndarray]:
    """``n`` (mu, x) pairs separated by more than 0.2 and less than ``max_sep`` (inside the cut locus)."""
    rng = np.random.default_rng(seed)
    mu = env.sample_domain(rng, 40 * n)
    x = env.sample_domain(rng, 40 * n)
    sep = np.linalg.norm(env.displacement_np(mu, x), axis=-1)
    ok = (sep > 0.2) & (sep < max_sep)
    return mu[ok][:n], x[ok][:n]


# ---- the five checks --------------------------------------------------------------------------


def check_round_trip(env, mu, x) -> float:
    """1. Exp_mu(Log_mu(x)) == x, measured as a *geodesic distance on the manifold*.

    Comparing coordinates componentwise would be wrong for a quotient manifold: on SO(3) = S³/±1,
    ``log_map`` takes the short way round, so ``exp_map`` legitimately returns ``−q`` — a different
    4-vector denoting the *same rotation*. That registered as an error of 2.0 under a componentwise
    test. The geodesic distance is the coordinate-free statement of "same point" and is correct for
    every manifold here, quotient or not.

    The measurement is taken *relative to the distance function's own resolution*: several
    ``geodesic`` implementations clip their ``arccos`` argument away from 1 for AD safety, so they
    report a nonzero distance from a point to itself (4.5e-4 on S², 8.9e-4 on SO(3) where the angle is
    doubled). Subtracting ``geodesic(x, x)`` measures the round-trip *excess* over that floor rather
    than re-reporting the floor as an error.
    """
    recon = jax.vmap(lambda m, p: env.exp_map(m, env.log_map(m, p)))(jnp.asarray(mu), jnp.asarray(x))
    dist = jax.vmap(lambda r, p: env.geodesic(r[None, :], p)[0])(recon, jnp.asarray(x))
    floor = jax.vmap(lambda p: env.geodesic(p[None, :], p)[0])(jnp.asarray(x))
    return float(jnp.max(jnp.abs(dist - floor)))


def check_isometry(env, mu, x) -> float:
    """2. ‖Log_mu(x)‖ == geodesic(x, mu)."""
    norms = jax.vmap(lambda m, p: jnp.linalg.norm(env.log_map(m, p)))(jnp.asarray(mu), jnp.asarray(x))
    geo = jax.vmap(lambda m, p: env.geodesic(p[None, :], m)[0])(jnp.asarray(mu), jnp.asarray(x))
    return float(jnp.max(jnp.abs(norms - geo)))


def check_jacobian(env, mu, x) -> float:
    """3. jac_factor(mu, x) == |det ∂Log_mu/∂x| between orthonormal frames (autodiff)."""

    def one(m, p):
        jac = jax.jacobian(lambda q: env.log_map(m, q))(p)  # [tangent_dim, dim]
        return jnp.abs(jnp.linalg.det(jac @ orthonormal_frame(env, p))) - env.jac_factor(m, p)

    return float(jnp.max(jnp.abs(jax.vmap(one)(jnp.asarray(mu), jnp.asarray(x)))))


def check_eikonal(env, x) -> float:
    """4. ‖∇ geodesic(·, start)‖_g == 1 — the analytic distance solves the unit-speed Eikonal PDE."""
    start = jnp.asarray(env.start, dtype=jnp.float64)

    def one(p):
        grad = jax.grad(lambda q: env.geodesic(q[None, :], start)[0])(p)
        return jnp.sqrt(jnp.einsum("i,ij,j->", grad, env.metric_inv(p), grad)) - 1.0

    return float(jnp.max(jnp.abs(jax.vmap(one)(jnp.asarray(x)))))


def check_normalisation(env, scale: float, seed: int) -> float:
    """5. ∫ N_w(·; start, scale·I) dvol, by Monte Carlo against the Riemannian volume measure."""
    rng = np.random.default_rng(seed)
    points = jnp.asarray(env.sample_domain(rng, MC_SAMPLES))
    weight = region_volume(env)
    mu = jnp.asarray(env.start, dtype=jnp.float64)
    A = scale * jnp.eye(env.tangent_dim, dtype=jnp.float64)
    density = jax.lax.map(
        lambda p: eval_wrapped_gaussian(p, mu, A, env.log_map, env.jac_factor, env.tangent_dim), points
    )
    return float(jnp.mean(density * weight))


def containment_mass(env, scale: float) -> tuple[float, str]:
    """Closed-form N(0, scale²·I) mass inside the tangent region that maps injectively onto the
    *integrated* part of the manifold, plus the name of what bounds that region.

    This is what ``check_normalisation`` must return: substituting x = Exp_mu(v) cancels
    ``jac_factor`` exactly, so ∫ N_w dvol = ∫_D N(v) dv with D the injectivity domain. Comparing
    against it turns check 5 from "is it roughly 1" into an exact identity, and separates the three
    manifolds by *why* D is bounded — which is the substantive difference between them.

    H^d is the case worth reading carefully: it has **no cut locus** (Exp is a global
    diffeomorphism), so D would be all of R^d and the wrapped Gaussian is exact at any scale. What
    bounds it is only the truncation this repo chooses, which at ``trunc_radius=0.9`` sits at
    hyperbolic radius 2.94 — nearer than the other two manifolds' cut loci at π. Hyperbolic therefore
    shows the *largest* shortfall in the table while being the only exact representation of the
    three, which is why cause is reported alongside magnitude.

    dim=2 closed forms; this file only constructs dim=2 environments.
    """
    if hasattr(env, "trunc_radius"):
        # Hyperbolic: bounded by our own truncation, not by curvature (see docstring).
        return 1.0 - math.exp(-(env.wall_distance**2) / (2 * scale**2)), "domain truncation (our choice)"
    if env.tangent_dim < env.dim:
        return 1.0 - math.exp(-(math.pi**2) / (2 * scale**2)), "antipodal cut locus (curvature)"
    return math.erf(math.pi / (scale * math.sqrt(2.0))) ** env.tangent_dim, "half-period cut locus"


def main() -> int:
    rows: list[tuple[str, str, float, str]] = []
    mass_rows: list[tuple[str, float, float, float, float, str]] = []
    failures = 0
    for name in ("torus", "sphere", "hyperbolic"):
        env = ENVIRONMENTS[name](n=2) if name == "sphere" else ENVIRONMENTS[name](dim=2)
        mu, x = sample_pairs(env, 200, seed=0, max_sep=1.8)
        near_source = points_near_source(env, 200, seed=1, lo=0.4, hi=1.8)
        exact = {
            "round trip": check_round_trip(env, mu, x),
            "isometry": check_isometry(env, mu, x),
            "jacobian": check_jacobian(env, mu, x),
            "eikonal ‖∇d‖_g=1": check_eikonal(env, near_source),
        }
        for label, value in exact.items():
            ok = abs(value) < EXACT_TOL
            failures += not ok
            rows.append((name, label, value, "PASS" if ok else "FAIL"))
        for scale in (NARROW, WIDE):
            measured = check_normalisation(env, scale, seed=2)
            predicted, cause = containment_mass(env, scale)
            failures += abs(measured - predicted) >= MC_TOL
            mass_rows.append((name, scale, measured, predicted, measured - predicted, cause))

    width = max(len(r[1]) for r in rows)
    print(f"\nExact identities\n{'manifold':<12} {'check':<{width}} {'error':>13}  verdict")
    print("-" * (12 + width + 24))
    for name, label, value, verdict in rows:
        print(f"{name:<12} {label:<{width}} {value:>13.2e}  {verdict}")

    print(f"\n∫ N_w dvol vs closed form (MC, {MC_SAMPLES:,} samples, tol {MC_TOL})")
    print(f"{'manifold':<12} {'σ':>5} {'measured':>10} {'predicted':>10} {'Δ':>10}  what bounds the domain")
    print("-" * 82)
    for name, scale, measured, predicted, delta, cause in mass_rows:
        print(f"{name:<12} {scale:>5.1f} {measured:>10.4f} {predicted:>10.4f} {delta:>10.1e}  {cause}")

    print(f"\n{failures} failure(s) across {3 * 4} exact identities and {3 * 2} mass checks.")
    print(
        "Read the mass table by *cause*, not by size: the torus and sphere lose tail mass to a cut\n"
        "locus, which is a limit of the wrapped-Gaussian representation on those manifolds. Hyperbolic\n"
        "has no cut locus at all (Exp is a global diffeomorphism on H^d) — its shortfall is only the\n"
        "domain bound at ‖x‖ ≤ 0.9, which is where we stop integrating, not a representation limit."
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
