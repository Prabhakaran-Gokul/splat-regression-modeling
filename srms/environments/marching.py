"""One 3-D fast marcher for every manifold in this repo.

The four 3-D manifolds need what looks like four different solvers — a periodic flat grid for T³, a
conformally-rescaled ball for H³, and geodesic-polar grids for S³ and SO(3). They are all the same
solver once you let the *grid spacing vary per cell and per axis*: a fast marcher only ever needs the
physical distance between a cell and each neighbour, and that is exactly what a diagonal metric in
orthogonal coordinates gives.

    ds² = Σ_i h_i(x)² dξ_i²        (orthogonal coordinates, diagonal metric)

so a step of one cell along grid axis ``i`` covers physical distance ``h_i``. Passing ``spacing``
with shape ``[n1, n2, n3, 3]`` therefore specialises this to:

- **T³**   ``h_i = 2π/n`` uniform, all three axes periodic — the flat case.
- **H³**   ``h_i = λ(x)·Δx`` with ``λ = 2/(1−‖x‖²)``; the Poincaré metric is *conformal*, so the same
  factor multiplies every axis. This is the 3-D form of the trick already verified at 2-D: the
  Euclidean marcher run with conformally rescaled slowness returns exact hyperbolic travel time, and
  ``λ`` has no dimension dependence, so nothing changes going up.
- **S³**   geodesic-polar ``ds² = dr² + sin²r·dΩ²``.
- **SO(3)** geodesic-polar ``ds² = dr² + 4sin²(r/2)·dΩ²`` — the same shape with the constant-curvature
  ¼ radius, covering the whole group at uniform radial resolution (unlike the Gibbs chart, whose
  ``‖g‖ = tan(θ/2)`` blows up at the cut locus and would need either a truncation that discards ~20%
  of the group or a grid ~10x coarser near the identity).

The update is the standard upwind Eikonal quadratic generalised to three anisotropic terms: with
accepted upwind neighbours contributing ``(a_i, w_i = 1/h_i²)``, solve

    Σ_i w_i (T − a_i)²  =  f²        over the subset with T > a_i

by trying the axes in increasing ``a_i`` and keeping the largest root that stays consistent with the
subset used — the standard Rouy–Tourin construction. Falls back to the one-sided update when no
consistent multi-axis root exists.

Accuracy is first order, the same as the 2-D marchers here; ``ground truth`` means "the reference we
score against", and its own discretisation error is measured per manifold rather than assumed
(see ``results/manifolds.md``).
"""

from __future__ import annotations

import heapq

import numpy as np


def solve_cell(values: np.ndarray, weights: np.ndarray, f: float) -> float:
    """Largest root of ``Σ_i w_i (T − a_i)² = f²`` over the consistent upwind subset.

    ``values`` are accepted upwind neighbour times (``inf`` where unavailable) and ``weights`` are
    ``1/h_i²`` for the corresponding axes. Axes are added in increasing time; a candidate root is
    accepted only if it exceeds every value in the subset that produced it, which is what makes the
    scheme upwind (a wavefront cannot arrive before the neighbours it came from).
    """
    order = np.argsort(values)
    best = np.inf
    acc_w = acc_wa = acc_wa2 = 0.0
    for idx in order:
        a, w = values[idx], weights[idx]
        if not np.isfinite(a):
            break
        acc_w += w
        acc_wa += w * a
        acc_wa2 += w * a * a
        disc = acc_wa * acc_wa - acc_w * (acc_wa2 - f * f)
        if disc < 0.0:
            continue
        root = (acc_wa + np.sqrt(disc)) / acc_w
        if root >= a:  # consistent: arrival is later than every upwind neighbour used
            best = root
    if np.isfinite(best):
        return best
    finite = np.isfinite(values)
    if not finite.any():
        return np.inf
    return float(np.min(values[finite] + f / np.sqrt(weights[finite])))  # one-sided fallback


def fast_march_3d(
    spacing: np.ndarray,
    slowness: np.ndarray,
    blocked: np.ndarray,
    seed_time: np.ndarray,
    periodic: tuple[bool, bool, bool],
    seed_tol: float | None = None,
    wrap_fn=None,
) -> np.ndarray:
    """Fast marching of ``‖∇T‖_g = slowness`` on a 3-D orthogonal grid.

    Args:
        spacing: ``[n1, n2, n3, 3]`` physical distance to the next cell along each grid axis.
        slowness: ``[n1, n2, n3]`` cost per unit *physical* length.
        blocked: ``[n1, n2, n3]`` cells excluded from the domain (never accepted).
        seed_time: ``[n1, n2, n3]`` known arrival times used to seed the front (``inf`` elsewhere);
            typically the analytic distance to the source, finite only in a small neighbourhood.
        periodic: whether each grid axis wraps.
        seed_tol: seed cells whose ``seed_time`` is below this; defaults to twice the largest seed-cell
            spacing, matching the 2-D marchers' one-cell seeding.
        wrap_fn: ``(idx, axis, step) -> idx | None`` resolving a step that leaves the array. A grid
            edge is not always a manifold boundary: on the SO(3) polar grid the direction-sphere poles
            join azimuths half a turn apart, and — the one that actually dominated the error — the
            shells r=0 and r=π are single points and an RP² respectively, because the quaternion at
            r=π is ``[0, u]`` and ``[0, u] = [0, −u]``. Leaving those unjoined forced the front to
            detour, giving 14.1% mean error with a maximum of 3.66 (larger than SO(3)'s diameter π),
            growing with r exactly as a missing outer-shell identification predicts. Return ``None``
            for a genuine boundary.

    Returns the arrival-time field, ``inf`` where unreachable.
    """
    shape = slowness.shape
    time = np.full(shape, np.inf)
    accepted = blocked.copy()
    heap: list[tuple[float, int, int, int]] = []

    if seed_tol is None:
        seed_tol = 2.0 * float(np.max(spacing))
    seeds = np.argwhere(np.isfinite(seed_time) & (seed_time <= seed_tol) & ~blocked)
    for i, j, k in seeds:
        time[i, j, k] = float(seed_time[i, j, k])
        heapq.heappush(heap, (float(time[i, j, k]), int(i), int(j), int(k)))

    def neighbour(idx: tuple[int, int, int], axis: int, step: int):
        """Neighbour index along ``axis``, honouring periodicity and pole crossing."""
        nxt = list(idx)
        nxt[axis] += step
        if periodic[axis]:
            nxt[axis] %= shape[axis]
        elif not (0 <= nxt[axis] < shape[axis]):
            return wrap_fn(idx, axis, step) if wrap_fn is not None else None
        return tuple(nxt)

    values = np.empty(3)
    weights = np.empty(3)
    while heap:
        _, i, j, k = heapq.heappop(heap)
        if accepted[i, j, k]:
            continue
        accepted[i, j, k] = True
        for axis in range(3):
            for step in (-1, 1):
                nb = neighbour((i, j, k), axis, step)
                if nb is None or accepted[nb]:
                    continue
                for ax in range(3):
                    best = np.inf
                    for s in (-1, 1):
                        up = neighbour(nb, ax, s)
                        if up is not None and accepted[up] and time[up] < best:
                            best = time[up]
                    values[ax] = best
                    # spacing is the distance from `nb` to its neighbour along `ax`
                    weights[ax] = 1.0 / max(spacing[nb][ax], 1e-12) ** 2
                candidate = solve_cell(values, weights, float(slowness[nb]))
                if candidate < time[nb]:
                    time[nb] = candidate
                    heapq.heappush(heap, (float(candidate), int(nb[0]), int(nb[1]), int(nb[2])))
    return time


def polar_grid_3d(resolution: int, r_max: float, radial_profile):
    """Geodesic-polar grid for a 3-D manifold with metric ``ds² = dr² + f(r)²·dΩ²_{S²}``.

    Both S³ (``f = sin r``) and SO(3) (``f = 2 sin(r/2)``) have this form — they are the constant
    curvature 1 and ¼ members of the same family — so one grid serves both. The chart is centred on a
    reference point; radial resolution is uniform, which is what the Gibbs/stereographic charts fail
    to give (``tan(θ/2)`` diverges at the cut locus).

    Cell-centred in ``r`` and in the S² colatitude, periodic in azimuth, so no sample sits on a
    coordinate singularity. Returns ``(points_in_polar[n,3], shape, spacing[n1,n2,n3,3])`` where the
    polar coordinates are ``(r, colatitude, azimuth)`` and ``spacing`` is the physical distance to the
    next cell along each axis — exactly what ``fast_march_3d`` consumes.
    """
    n = resolution
    d_r, d_col, d_az = r_max / n, np.pi / n, 2.0 * np.pi / n
    r = (np.arange(n) + 0.5) * d_r
    col = (np.arange(n) + 0.5) * d_col
    az = -np.pi + np.arange(n) * d_az
    R, C, A = np.meshgrid(r, col, az, indexing="ij")

    f = radial_profile(R)  # metric coefficient multiplying the S² directions
    spacing = np.stack([np.full_like(R, d_r), f * d_col, f * np.sin(C) * d_az], axis=-1)
    points = np.stack([R.ravel(), C.ravel(), A.ravel()], axis=-1)
    return points, (n, n, n), np.maximum(spacing, 1e-9)


def polar_to_unit3(points: np.ndarray) -> np.ndarray:
    """(r, colatitude, azimuth) -> (r, unit direction in R³) — the tangent direction of the geodesic."""
    r, c, a = points[:, 0], points[:, 1], points[:, 2]
    return r, np.stack([np.sin(c) * np.cos(a), np.sin(c) * np.sin(a), np.cos(c)], axis=-1)


def polar_topology(shape: tuple[int, int, int]):
    """Edge identifications for a geodesic-polar (r, colatitude, azimuth) grid.

    Shared by S³ and SO(3): both are geodesic-polar charts around a point, and although their outer
    shells differ (S³'s r=π is the single antipodal point; SO(3)'s is an RP², since the quaternion
    ``[0,u]`` equals ``[0,−u]``), stepping past *either* radial end reverses the direction ``u → −u``
    in both cases, landing at the same radius one cell away. So one hook serves both.

    - **radial ends**: direction reverses — colatitude reflects *and* azimuth turns by π.
    - **colatitude ends**: a direction-sphere pole — same colatitude ring, azimuth turns by π only.

    Reflecting the colatitude on the *pole* case as well would wire the north pole to the south pole,
    a shortcut across the direction sphere; it shows up as the marcher under-estimating travel time,
    which a consistent upwind scheme can never do. Azimuth is handled by ``periodic``.
    """
    _, n_col, n_az = shape

    def wrap(idx, axis, step):
        r, c, a = idx
        flip_az = (a + n_az // 2) % n_az
        if axis == 0:
            return (r, n_col - 1 - c, flip_az)
        if axis == 1:
            return (r, c, flip_az)
        return None

    return wrap
