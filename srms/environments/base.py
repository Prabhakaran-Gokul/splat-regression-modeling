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
    dim: int
    domain: tuple[float, float]  # (low, high) for uniform sampling over the manifold's chart
    obstacles: tuple
    axis_labels: tuple[str, str]
    render_extent: tuple[float, float, float, float]
    title: str

    # ---- manifold geometry (feeds methods/backends/srm.py's eval_wrapped_gaussian) --------------

    def log_map(self, mu: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Tangent-space coordinates of x at base mu (= psi^-1(x), psi = Exp_mu)."""
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
        """NumPy tangent vector pointing from a to b (= wrap_point_np(b - a)); broadcasts."""
        ...

    # ---- PDE ingredients -----------------------------------------------------------------------

    def metric_inv(self, x: jnp.ndarray) -> jnp.ndarray:
        """Inverse metric g^{ij}(x); used by every Eikonal residual."""
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
