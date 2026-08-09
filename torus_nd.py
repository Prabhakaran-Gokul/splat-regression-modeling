"""Dimension-scaling test of the roadmap-prior + Eikonal-refinement method (2-D → 3-D → 4-D).

The 2-D result (``torus.py``) showed a sparse RRT* base + Eikonal refinement is the best legitimate
method. The open question is whether it *scales*: at a **fixed sample budget**, as dimension rises the
same nodes cover exponentially less volume (dispersion blows up), so does the physics keep rescuing?

This module generalises the pipeline to the ``d``-torus and, for each ``d``, reports base-alone vs
PDE-refined RMS against a ``d``-D fast-marching ground truth — the honest scaling verdict. FMM is a
vectorised label-correcting sweep (periodic, ``O(n^d)`` grid), feasible to 4-D at coarse resolution.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from tqdm import trange


def wrap(a):
    return (a + jnp.pi) % (2 * jnp.pi) - jnp.pi


def _wrap_np(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


@dataclasses.dataclass
class Config:
    """Fixed-budget dimension-scaling sweep of the roadmap + Eikonal method."""

    dims: tuple[int, ...] = (2, 3, 4)
    resolution: tuple[int, ...] = (100, 46, 30)  # FMM grid per dim (≈1e4/1e5/8e5 cells)
    rrt_nodes_by_dim: tuple[int, ...] = (300, 600, 1200)  # double the RRT* budget each dimension (300·2^(d-2))
    num_obstacles: int = 3
    obstacle_radius: tuple[float, float] = (0.5, 0.9)  # aligned to torus.py (the 2-D reference)
    slowness_max: float = 10.0
    slow_width: float = 0.15
    rrt_nodes: int = 300  # fallback if rrt_nodes_by_dim not used
    rrt_step: float = 0.5
    rrt_radius: float = 0.9
    roadmap_gamma: float = 0.01
    roadmap_hop: int = 5
    base_reg: float = 3.0
    num_splats: int = 384
    num_collocation: int = 2048
    steps: int = 4000
    lr: float = 3e-3
    init_scale: float = 0.4
    causal_strength: float = 5.0
    lambda_init: float = 0.2
    anneal_frac: float = 0.4
    band_width: float = 0.25
    seed: int = 1
    out_dir: str = "figures"


# ---------- scene: d-dimensional angle-space obstacles ----------

def make_obstacles(cfg, d, rng):
    start = np.full(d, -1.5)
    centers, radii = [], []
    while len(centers) < cfg.num_obstacles:
        c = rng.uniform(-np.pi, np.pi, d)
        r = float(rng.uniform(*cfg.obstacle_radius))
        if np.linalg.norm(_wrap_np(c - start)) > r + 0.3:
            centers.append(c)
            radii.append(r)
    return np.asarray(centers), np.asarray(radii), jnp.asarray(start, jnp.float32)


def sdf_j(pts, centers, radii):
    per = jnp.linalg.norm(wrap(pts[:, None, :] - centers[None, :, :]), axis=2) - radii[None, :]
    return jnp.min(per, axis=1)


def slowness_j(pts, centers, radii, smax, width):
    return 1.0 + (smax - 1.0) * jax.nn.sigmoid(-sdf_j(pts, centers, radii) / width)


def slowness_np(pts, centers, radii, smax, width):
    per = np.linalg.norm(_wrap_np(pts[:, None, :] - centers[None, :, :]), axis=2) - radii[None, :]
    return 1.0 + (smax - 1.0) / (1.0 + np.exp(per.min(axis=1) / width))


# ---------- d-D periodic fast marching (vectorised label-correcting sweeps) ----------

def fast_marching_nd(speed, start_coord, n, d):
    """Vectorised Godunov-upwind Eikonal ``|∇T| = 1/speed`` on the periodic ``n^d`` grid (Jacobi sweeps).

    The Godunov update solves ``Σ_i max(T−a_i,0)² = (h·s)²`` per cell (``a_i`` = min of the two
    neighbours along axis ``i``), which gives true **Euclidean** geodesic time — unlike a graph
    Dijkstra, whose axis-only moves overestimate distance by up to √d and would confound the scaling test.
    """
    h = 2 * np.pi / n
    fh = h / speed  # h · slowness (the Eikonal RHS per cell)
    T = np.full(speed.shape, 1e9, dtype=np.float64)
    T[start_coord] = 0.0
    for _ in range(6 * n):  # Jacobi Godunov; info propagates ~1 cell/iter, converges in ~O(diameter)
        A = np.sort(np.stack([np.minimum(np.roll(T, 1, ax), np.roll(T, -1, ax)) for ax in range(d)], 0), axis=0)
        csum, csum2 = np.cumsum(A, axis=0), np.cumsum(A**2, axis=0)
        Tnew = np.full_like(T, 1e9)
        for k in range(1, d + 1):
            s1, s2 = csum[k - 1], csum2[k - 1]
            disc = k * fh**2 - (k * s2 - s1**2)
            cand = np.where(disc >= 0, (s1 + np.sqrt(np.clip(disc, 0, None))) / k, 1e9)
            upper = A[k] if k < d else np.full_like(T, np.inf)
            ok = (cand >= A[k - 1] - 1e-9) & (cand <= upper + 1e-9)
            Tnew = np.where(ok, np.minimum(Tnew, cand), Tnew)
        Tnew = np.minimum(Tnew, T)
        Tnew[start_coord] = 0.0
        if np.abs(Tnew - T).max() < 1e-6 * h:
            T = Tnew
            break
        T = Tnew
    return T


def grid_points(n, d):
    axis = np.linspace(-np.pi, np.pi, n, endpoint=False)
    mesh = np.meshgrid(*([axis] * d), indexing="ij")
    return np.stack([m.ravel() for m in mesh], axis=-1)


# ---------- splat field (d-general) ----------

def eval_splat(thetas, splat):
    weights, scales, centres = splat
    disp = wrap(thetas[:, None, :] - centres[None, :, :])
    solved = jnp.linalg.solve(scales[None], disp[..., None]).squeeze(-1)
    d = thetas.shape[1]
    density = jnp.exp(-0.5 * jnp.sum(solved**2, axis=-1)) / jnp.power(2 * jnp.pi, d / 2.0)
    return (density / jnp.linalg.det(scales)) @ weights


def init_splat(k, scale, d, key):
    centres = jax.random.uniform(key, (k, d), minval=-jnp.pi, maxval=jnp.pi)
    scales = jnp.repeat((scale * jnp.eye(d))[None], k, axis=0)
    return jnp.zeros((k, 1)), scales, centres


# ---------- RRT* (d-general), roadmap base, field, residual ----------

def rrt_star(cfg, centers, radii, start_np, d, rng, num):
    def edge_cost(a, b):
        ts = np.linspace(0.0, 1.0, 6)[:, None]
        pts = _wrap_np(a[None, :] + ts * _wrap_np(b - a)[None, :])
        return float(np.linalg.norm(_wrap_np(b - a)) * slowness_np(pts, centers, radii, cfg.slowness_max, cfg.slow_width).mean())

    nodes = [start_np.copy()]
    costs = [0.0]
    arr = np.array(nodes)
    while len(nodes) < num:
        q = rng.uniform(-np.pi, np.pi, d)
        near = int(np.linalg.norm(_wrap_np(arr - q), axis=1).argmin())
        direction = _wrap_np(q - arr[near])
        length = np.linalg.norm(direction)
        if length < 1e-6:
            continue
        q_new = _wrap_np(arr[near] + min(cfg.rrt_step, length) * direction / length)
        neigh = np.where(np.linalg.norm(_wrap_np(arr - q_new), axis=1) < cfg.rrt_radius)[0]
        best = costs[near] + edge_cost(arr[near], q_new)
        for i in neigh:
            best = min(best, costs[i] + edge_cost(arr[i], q_new))
        nodes.append(q_new)
        costs.append(best)
        arr = np.array(nodes)
        for i in neigh:
            costs[i] = min(costs[i], best + edge_cost(q_new, arr[i]))
    return jnp.asarray(np.array(nodes), jnp.float32), jnp.asarray(np.array(costs), jnp.float32)


def roadmap_base(theta, nodes, costs, gamma, centers, radii, smax, width, ksamp):
    """Soft-min over all tree nodes of ``cost_i + slowness-weighted ‖wrap(θ−node_i)‖`` (standard RRT* roadmap)."""
    d = theta.shape[-1]
    disp = wrap(theta[None, :] - nodes)
    ts = jnp.linspace(0.0, 1.0, ksamp)
    seg = nodes[None, :, :] + ts[:, None, None] * disp[None, :, :]
    mslow = slowness_j(wrap(seg.reshape(-1, d)), centers, radii, smax, width).reshape(ksamp, -1).mean(0)
    wlen = jnp.linalg.norm(disp, axis=1) * mslow
    return -gamma * jax.scipy.special.logsumexp(-(costs + wlen) / gamma)


def field(splat, theta, rm):
    return roadmap_base(theta, *rm) * jnp.exp(eval_splat(theta[None, :], splat)[0, 0])


def residual(splat, thetas, slow, rm):
    grad = jax.vmap(lambda t: jax.grad(field, argnums=1)(splat, t, rm))(thetas)
    q = jnp.sqrt(jnp.clip(jnp.sum(grad**2, axis=1), 1e-12, None)) / slow  # metric = I (isotropic)
    return q + 1.0 / q - 2.0


def predict(splat, thetas, rm, chunk=4000):
    out = [np.asarray(jax.vmap(lambda t: field(splat, t, rm))(thetas[i : i + chunk])) for i in range(0, len(thetas), chunk)]
    return np.concatenate(out)


def solve(cfg, rm, centers, radii, start, d):
    splat = init_splat(cfg.num_splats, cfg.init_scale, d, jax.random.PRNGKey(cfg.seed))
    rng = np.random.default_rng(cfg.seed)
    optimizer = optax.adam(cfg.lr)
    state = optimizer.init(splat)
    rate0 = cfg.causal_strength / cfg.num_collocation

    def loss_fn(params, colloc, slow_full, order, rate, lam):
        slow = 1.0 + lam * (slow_full - 1.0)
        gate = jax.nn.sigmoid(sdf_j(colloc, centers, radii) / cfg.band_width)
        res = residual(params, colloc, slow, rm)
        res_o, gate_o = res[order], gate[order]
        upstream = jnp.cumsum(gate_o * res_o) - gate_o * res_o
        w = jax.lax.stop_gradient(jnp.exp(-rate * upstream)) * gate_o
        pde = jnp.sum(w * res_o) / (jnp.sum(w) + 1e-9)
        return pde + cfg.base_reg * jnp.mean(eval_splat(colloc, params) ** 2)

    @jax.jit
    def step(params, state, colloc, slow_full, order, rate, lam):
        loss, grads = jax.value_and_grad(loss_fn)(params, colloc, slow_full, order, rate, lam)
        updates, state = optimizer.update(grads, state, params)
        return optax.apply_updates(params, updates), state, loss

    def resample():
        pool = rng.uniform(-np.pi, np.pi, size=(cfg.num_collocation, d))
        colloc = jnp.asarray(pool[np.linalg.norm(_wrap_np(pool - np.asarray(start)), axis=1) > 0.25], jnp.float32)
        slow_full = slowness_j(colloc, centers, radii, cfg.slowness_max, cfg.slow_width)
        order = jnp.argsort(jnp.linalg.norm(wrap(colloc - start), axis=1))
        return colloc, slow_full, order

    colloc, slow_full, order = resample()
    prog = trange(cfg.steps, desc=f"d={d} roadmap")
    for i in prog:
        if i % 300 == 0:
            colloc, slow_full, order = resample()
        lam = min(1.0, cfg.lambda_init + (1.0 - cfg.lambda_init) * i / (cfg.anneal_frac * cfg.steps + 1e-9))
        rate = rate0 * (1.0 - i / cfg.steps)
        splat, state, loss = step(splat, state, colloc, slow_full, order, jnp.float32(rate), jnp.float32(lam))
        if i % 100 == 0:
            prog.set_description(f"d={d} roadmap — log10(loss)={float(jnp.log10(loss + 1e-12)):.2f}")
    return splat


def main(cfg: Config) -> None:
    """For each dimension: FMM GT, RRT* base (doubling budget), PDE refine; report prior vs solved RMS/max."""
    rows = []
    for i, (d, n) in enumerate(zip(cfg.dims, cfg.resolution)):
        num = cfg.rrt_nodes_by_dim[i] if i < len(cfg.rrt_nodes_by_dim) else cfg.rrt_nodes
        rng = np.random.default_rng(cfg.seed + d)
        centers, radii, start = make_obstacles(cfg, d, rng)
        cen_j, rad_j = jnp.asarray(centers, jnp.float32), jnp.asarray(radii, jnp.float32)

        pts = grid_points(n, d)
        slow_grid = slowness_np(pts, centers, radii, cfg.slowness_max, cfg.slow_width).reshape((n,) * d)
        start_coord = tuple(int(round((s + np.pi) / (2 * np.pi) * n)) % n for s in np.asarray(start))
        gt = fast_marching_nd(1.0 / slow_grid, start_coord, n, d).ravel()
        inside = np.asarray(sdf_j(jnp.asarray(pts, jnp.float32), cen_j, rad_j)) < 0.0

        nodes, costs = rrt_star(cfg, centers, radii, np.asarray(start), d, rng, num)
        rm = (nodes, costs, cfg.roadmap_gamma, cen_j, rad_j, cfg.slowness_max, cfg.slow_width, cfg.roadmap_hop)

        free = np.where(~inside)[0]  # evaluate error on a random subset of free cells (bounds high-d cost)
        eval_idx = free if len(free) <= 20000 else rng.choice(free, 20000, replace=False)
        eval_pts = jnp.asarray(pts[eval_idx], jnp.float32)
        eval_gt = gt[eval_idx]

        def err(pred):  # (RMS, max|err|) vs FMM ground truth
            e = np.abs(np.asarray(pred) - eval_gt)
            return float(np.sqrt(np.mean(e**2))), float(e.max())

        zero = init_splat(cfg.num_splats, cfg.init_scale, d, jax.random.PRNGKey(0))
        zero = (jnp.zeros_like(zero[0]), zero[1], zero[2])
        base_rms, base_max = err(predict(zero, eval_pts, rm))  # RRT* prior alone
        splat = solve(cfg, rm, cen_j, rad_j, start, d)
        ref_rms, ref_max = err(predict(splat, eval_pts, rm))  # after Eikonal solve
        rows.append((d, len(nodes), base_rms, ref_rms, base_max, ref_max))
        print(f"[d={d}] {len(nodes)} pts | RMS prior={base_rms:.3f} solved={ref_rms:.3f} | max prior={base_max:.3f} solved={ref_max:.3f}", flush=True)

    print("\n===== DIMENSION SWEEP (RRT* points doubling per dimension) =====")
    print(f"{'d':>2} {'points':>7} | {'RMS_prior':>10} {'RMS_solved':>11} | {'max_prior':>10} {'max_solved':>11}")
    for d, k, br, rr, bm, rm in rows:
        print(f"{d:>2} {k:>7} | {br:>10.3f} {rr:>11.3f} | {bm:>10.3f} {rm:>11.3f}")
    print("DIMSWEEP DONE")


if __name__ == "__main__":
    main(tyro.cli(Config))
