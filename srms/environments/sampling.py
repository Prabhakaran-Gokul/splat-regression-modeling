"""RRT* prior over an Environment: mesh-free, dimension-scalable cost-to-come anchors.

Generic over any ``Environment`` (see ``srms/environments/base.py``) — the
algorithm only needs ``sample_domain`` for random configurations and
``displacement_np``/``wrap_point_np``/``slowness_np`` for edge costs, so it
carries over unchanged across manifolds (torus, sphere, ...).

Ported from the original ``torus.py``'s ``rrt_star``/``rrt_star_anchors``/
``rrt_star_anchors_shadow``/``build_roadmap``.
"""

from __future__ import annotations

import heapq

import jax.numpy as jnp
import numpy as np

_RRT_SEED_OFFSET = 11
_ANCHOR_SEED_OFFSET = 12
_ROADMAP_SEED_OFFSET = 13


def rrt_star(
    env, start, iters: int, step: float, radius: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """RRT* on the environment with slowness-weighted edges: returns tree nodes and cost-to-come.

    Edge cost = wrapped length × mean slowness along the edge, matching the soft-obstacle field.
    """
    rng = np.random.default_rng(seed + _RRT_SEED_OFFSET)
    start = np.asarray(start, dtype=float)

    def edge_cost(a: np.ndarray, b: np.ndarray) -> float:
        ts = np.linspace(0.0, 1.0, 6)[:, None]
        disp = env.displacement_np(a, b)
        pts = env.wrap_point_np(a[None, :] + ts * disp[None, :])
        return float(np.linalg.norm(disp) * env.slowness_np(pts).mean())

    nodes = [start]
    costs = [0.0]
    node_arr = np.array(nodes)
    for _ in range(iters):
        q_rand = env.sample_domain(rng, 1)[0]
        distances = np.linalg.norm(env.displacement_np(q_rand, node_arr), axis=1)
        near = int(distances.argmin())
        direction = env.displacement_np(node_arr[near], q_rand)
        length = np.linalg.norm(direction)
        if length < 1e-6:
            continue
        q_new = env.wrap_point_np(node_arr[near] + min(step, length) * direction / length)
        neighbours = np.where(np.linalg.norm(env.displacement_np(q_new, node_arr), axis=1) < radius)[0]
        best_cost = costs[near] + edge_cost(node_arr[near], q_new)
        for i in neighbours:
            best_cost = min(best_cost, costs[i] + edge_cost(node_arr[i], q_new))
        nodes.append(q_new)
        costs.append(best_cost)
        node_arr = np.array(nodes)
        for i in neighbours:
            costs[i] = min(costs[i], best_cost + edge_cost(q_new, node_arr[i]))
    return node_arr, np.array(costs)


def rrt_star_anchors(
    env, start, iters: int, step: float, radius: float, num_weak: int, weak_clearance: float, seed: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return ``num_weak`` sparse (config, cost-to-come) anchors sampled from an RRT* tree."""
    nodes, costs = rrt_star(env, start, iters, step, radius, seed)
    keep = np.asarray(env.sdf(jnp.asarray(nodes, dtype=jnp.float32))) > weak_clearance
    nodes, costs = nodes[keep], costs[keep]
    chosen = np.random.default_rng(seed + _ANCHOR_SEED_OFFSET).choice(
        len(nodes), size=min(num_weak, len(nodes)), replace=False
    )
    return jnp.asarray(nodes[chosen], dtype=jnp.float32), jnp.asarray(costs[chosen], dtype=jnp.float32)


def rrt_star_anchors_shadow(
    env,
    start,
    iters: int,
    step: float,
    radius: float,
    num_weak: int,
    weak_clearance: float,
    shadow_pref: float,
    seed: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Sparse RRT* (config, cost-to-come) anchors biased into obstacle *shadows* (occluded from source).

    The pointwise Eikonal loss constrains slope, not level; behind obstacles the level is
    under-determined and drifts. These anchors pin that free integration constant exactly where it is
    free — the region whose straight source→node ray is blocked.
    """
    nodes, costs = rrt_star(env, start, iters, step, radius, seed)
    clear = np.asarray(env.sdf(jnp.asarray(nodes, dtype=jnp.float32))) > weak_clearance
    nodes, costs = nodes[clear], costs[clear]
    start_arr = np.asarray(start, dtype=float)
    disp = env.displacement_np(start_arr, nodes)
    ts = np.linspace(0.1, 0.95, 8)
    ray = start_arr[None, None, :] + ts[None, :, None] * disp[:, None, :]  # [N, 8, dim] source→node rays
    ray_sdf = np.asarray(env.sdf(jnp.asarray(ray.reshape(-1, env.dim), dtype=jnp.float32))).reshape(len(nodes), -1)
    occluded = ray_sdf.min(axis=1) < 0.0  # blocked ray ⇒ node sits in a routing shadow
    weight = 1.0 + shadow_pref * occluded
    weight = weight / weight.sum()
    k = min(num_weak, len(nodes))
    chosen = np.random.default_rng(seed + _ANCHOR_SEED_OFFSET).choice(len(nodes), size=k, replace=False, p=weight)
    print(f"[anchors] {k} RRT* anchors — {int(occluded[chosen].sum())} in shadow", flush=True)
    return jnp.asarray(nodes[chosen], dtype=jnp.float32), jnp.asarray(costs[chosen], dtype=jnp.float32)


def build_roadmap(
    env, start, iters: int, step: float, radius: float, roadmap_nodes: int, seed: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """RRT* tree subsampled to a fixed node budget — the coarse, obstacle-aware cost-to-come prior.

    Deterministic in ``seed`` so a strategy and its rendering rebuild the identical roadmap. The
    source (cost 0) is always kept.
    """
    nodes, costs = rrt_star(env, start, iters, step, radius, seed)
    if len(nodes) > roadmap_nodes:
        idx = (
            np.random.default_rng(seed + _ROADMAP_SEED_OFFSET).choice(
                len(nodes) - 1, roadmap_nodes - 1, replace=False
            )
            + 1
        )
        idx = np.concatenate([[0], idx])  # keep the source node (cost 0) so T(start)=0
        nodes, costs = nodes[idx], costs[idx]
    return jnp.asarray(nodes, dtype=jnp.float32), jnp.asarray(costs, dtype=jnp.float32)


_SPHERE_ROADMAP_SEED_OFFSET = 14


def _edge_time(env, a: np.ndarray, b: np.ndarray, ksamp: int = 6) -> float:
    """Travel *time* of the straight hop a→b: wrapped length × mean slowness along it."""
    ts = np.linspace(0.0, 1.0, ksamp)[:, None]
    disp = env.displacement_np(a, b)
    pts = env.wrap_point_np(a[None, :] + ts * disp[None, :])
    return float(np.linalg.norm(disp) * env.slowness_np(pts).mean())


def _edge_is_free(env, a: np.ndarray, b: np.ndarray, ksamp: int = 12) -> bool:
    """Straight-line collision check: every sample along the hop must be outside the obstacles."""
    ts = np.linspace(0.0, 1.0, ksamp)[:, None]
    disp = env.displacement_np(a, b)
    pts = env.wrap_point_np(a[None, :] + ts * disp[None, :])
    return bool(np.all(env.sdf_np(pts) > 0.0))


def build_sphere_roadmap(
    env, start, num_nodes: int, pool: int, connect_radius: float, seed: int, max_radius: float = 0.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """H-NTFields' sparse sphere-packing roadmap, plus travel time from ``start`` to each node.

    Implements the roadmap of H-NTFields (Ni, Liu & Qureshi 2026, arXiv 2604.13204, Sec. IV-A):

    - *Free-space volume sampling.* Draw collision-free configurations and give each a maximized
      free sphere (radius = clearance ``env.sdf``). A candidate is accepted only if it lies **outside
      the union of the spheres already accepted**, which is what makes the packing sparse and
      spread-out rather than clustered.
    - *Sparse connectivity.* Node pairs within ``connect_radius`` are joined when the straight line
      between them is collision-free.
    - *Distances.* Dijkstra from ``start`` (always node 0) gives the graph distance later used for
      the travel-time bounds.

    Edge weights are the **slowness-integrated** hop cost (``_edge_time``), not raw length. The
    paper's bounds work because its free-space speed is 1, so distance *is* time; this repo's
    ``env.slowness`` varies smoothly from 1 to ``slowness_max``, so plain lengths would bound
    nothing. Weighting by slowness restores the bound's meaning in travel-time units.

    Returns ``(nodes[N, dim], radii[N], time_from_start[N])`` for the nodes reachable from the
    source; unreachable components are dropped.
    """
    rng = np.random.default_rng(seed + _SPHERE_ROADMAP_SEED_OFFSET)
    start = np.asarray(start, dtype=float)

    cap = max_radius if max_radius > 0 else np.inf  # bound the free sphere; see the note below
    nodes = [start]
    radii = [min(max(float(env.sdf_np(start[None, :])[0]), 1e-3), cap)]
    candidates = env.sample_domain(rng, pool)
    clearance = np.minimum(env.sdf_np(candidates), cap)
    for cand, clear in zip(candidates, clearance):
        if len(nodes) >= num_nodes:
            break
        if clear <= 0.0:  # inside an obstacle
            continue
        node_arr = np.array(nodes)
        if np.any(np.linalg.norm(env.displacement_np(cand, node_arr), axis=1) <= np.array(radii)):
            continue  # already covered by an accepted node's free sphere
        nodes.append(cand)
        radii.append(float(clear))
    node_arr, radius_arr = np.array(nodes), np.array(radii)

    n = len(node_arr)
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for i in range(n):
        near = np.where(np.linalg.norm(env.displacement_np(node_arr[i], node_arr), axis=1) < connect_radius)[0]
        for j in near:
            if j <= i or not _edge_is_free(env, node_arr[i], node_arr[j]):
                continue
            w = _edge_time(env, node_arr[i], node_arr[j])
            adjacency[i].append((int(j), w))
            adjacency[j].append((i, w))

    time_from_start = np.full(n, np.inf)
    time_from_start[0] = 0.0
    queue = [(0.0, 0)]
    while queue:
        d, i = heapq.heappop(queue)
        if d > time_from_start[i]:
            continue
        for j, w in adjacency[i]:
            if d + w < time_from_start[j]:
                time_from_start[j] = d + w
                heapq.heappush(queue, (d + w, j))

    reachable = np.isfinite(time_from_start)
    print(
        f"[sphere-roadmap] {int(reachable.sum())}/{n} nodes reachable "
        f"(radii {radius_arr.min():.2f}–{radius_arr.max():.2f})",
        flush=True,
    )
    return node_arr[reachable], radius_arr[reachable], time_from_start[reachable]
