"""Environment interface shared by every Eikonal training strategy.

A concrete Environment (e.g. ``TorusEnvironment``) owns the manifold geometry,
the obstacle/slowness field, and ground truth, so that ``srms/methods``
(backends and strategies) never import a specific manifold's functions by
name. Methods here mirror the ``(log_map_fn, jac_factor_fn, dim)`` interface
already used by ``srms/lib/manifold_splat.py``'s ``eval_wrapped_gaussian``, extended
with the pieces the Eikonal residuals, adaptive sampling, and RRT* prior need.

This is a duck-typed ``Protocol``, not an ABC — matching the un-opinionated
style already used by ``ground_truth.py``'s ``PlanningProblem``. Concrete
environments (e.g. ``TorusEnvironment``) satisfy it structurally; there is no
need to inherit from it.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import jax.numpy as jnp
import numpy as np


@runtime_checkable
class Environment(Protocol):
    dim: int  # point-representation dimension (= ambient dimension; equals tangent_dim for flat charts)
    tangent_dim: int  # dimension of the tangent space / splat scale matrix A (= dim for flat charts)
    domain: tuple[float, float]  # (low, high) for uniform sampling over the manifold's chart
    obstacles: tuple
    axis_labels: tuple[str, str]
    render_extent: tuple[float, float, float, float]
    title: str

    # ---- manifold geometry (feeds methods/backends/srm.py's eval_wrapped_gaussian) --------------

    def log_map(self, mu: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Intrinsic tangent-frame coordinates of x at base mu, size tangent_dim (= psi^-1(x), psi = Exp_mu).

        Used only for splat evaluation (matches the A scale matrix's shape); for a flat chart this
        coincides with ``log_map_ambient``, for an embedded manifold (e.g. the sphere) it does not.
        """
        ...

    def log_map_ambient(self, mu: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Ambient tangent vector (size dim) at mu pointing toward x; ‖·‖ = geodesic distance.

        Used where the tangent vector needs to stay addable to an ambient point (e.g. the RRT*-roadmap
        last-hop interpolation in weak_supervision.py). Identical to log_map for flat charts.
        """
        ...

    def jac_factor(self, mu: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """|det d(log_map)/dx| Riemannian-volume correction; 1.0 for flat manifolds."""
        ...

    def wrap_point(self, x: jnp.ndarray) -> jnp.ndarray:
        """Canonicalize a point into the manifold's chart (jax). Identity for non-periodic charts."""
        ...

    def wrap_point_np(self, x: np.ndarray) -> np.ndarray:
        """NumPy counterpart of wrap_point, for host-side sampling (RRT*)."""
        ...

    def displacement_np(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """NumPy ambient tangent vector pointing from a to b, ‖·‖ = geodesic distance; broadcasts."""
        ...

    def boundary_ring_np(self, rng: np.random.Generator, eps: float, n: int) -> np.ndarray:
        """``n`` points at exact geodesic distance ``eps`` from ``self.start`` (the Eikonal BC ring)."""
        ...

    # ---- PDE ingredients -----------------------------------------------------------------------

    def metric_inv(self, x: jnp.ndarray) -> jnp.ndarray:
        """Inverse metric g^{ij}(x), as a [dim, dim] quadratic form on ambient gradients.

        Identity for a flat chart; a tangent-space projector for an embedded manifold (so that
        ``grad @ metric_inv(x) @ grad`` recovers the squared intrinsic gradient norm of a function
        that was only ever evaluated through ambient coordinates). Used by every Eikonal residual.
        """
        ...

    def geodesic(self, x: jnp.ndarray, start: jnp.ndarray) -> jnp.ndarray:
        """Analytic base distance (the known base / free-space geodesic)."""
        ...

    def slowness(self, x: jnp.ndarray) -> jnp.ndarray:
        """Smooth cost field s(x) >= 1 (jax)."""
        ...

    def sdf(self, x: jnp.ndarray) -> jnp.ndarray:
        """Signed distance to the obstacle union, negative inside (jax)."""
        ...

    def slowness_np(self, points: np.ndarray) -> np.ndarray:
        """NumPy counterpart of slowness, for RRT*'s hot loop."""
        ...

    def sdf_np(self, points: np.ndarray) -> np.ndarray:
        """NumPy counterpart of sdf, for RRT*'s hot loop."""
        ...

    # ---- sampling / ground truth ---------------------------------------------------------------

    def sample_domain(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Uniform host-side samples over the domain, as a NumPy array [n, dim]."""
        ...

    def grid(self, resolution: int) -> tuple[jnp.ndarray, tuple[int, int]]:
        """Dense grid over the domain, raveled to [resolution**dim, dim], plus its per-axis shape."""
        ...

    def ground_truth(self, resolution: int, start: tuple[float, ...] | None = None) -> np.ndarray:
        """Dense ground-truth field on a resolution-per-axis grid, raveled to [resolution**dim]."""
        ...

    def render_marker_deg(self) -> tuple[float, float]:
        """(x, y) position of the source in the render chart's degree units (see viz.py)."""
        ...
