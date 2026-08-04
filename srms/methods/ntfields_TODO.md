# NTFields + supervised: deferred migration work

Status: not started. `srms/methods/backends/ntfields.py` is still an empty
stub. This note is a self-contained spec so this can be picked up later
without re-deriving it from `torus.py`.

## What's already done (context)

- `srms/environments/torus.py` — `TorusEnvironment`: `geodesic`, `slowness`,
  `sdf`, `ground_truth`, `log_map`, `metric_inv`, etc.
- `srms/methods/backends/srm.py` — the only implemented backend. Interface
  every backend follows:
  - `init_params(key, env, cfg) -> params` (reads whatever `cfg` fields it
    needs, e.g. `cfg.num_splats`, `cfg.init_scale`)
  - `eval_raw(params, X, env) -> [n, p]` — a raw (unfactored) scalar field
- `srms/methods/strategies/eikonal.py` and `weak_supervision.py` — both
  **backend-agnostic**: they take `backend` as an explicit argument and only
  call `backend.init_params`/`backend.eval_raw`. Both were deliberately
  simplified from the original `torus.py` (no RAD resampling, no causal
  weighting, no curriculum, no RRT* anchors) — plain dynamic random
  collocation each step, following `_archive/preexisting/eikonal_splat.py`'s
  style. This simplification precedent should probably extend to ntfields too
  (see "Open question" below).

## What NTFields actually is, in the original `torus.py`

Four pieces, currently all still living in `torus.py` (not yet ported):

**`field_ntfields`** (torus.py:430-442) — the field itself:
```
T(θ) = base(θ) / τ(θ)
base(θ) = geodesic(θ, start)                      # analytic flat-torus distance
τ(θ)    = τ_min + (1 − τ_min) · σ(raw(θ) + bias)   # τ ∈ [τ_min, 1]
raw(θ)  = eval_splat_torus(θ, splat)               # the SRM raw output
```
Since `τ ≤ 1`, `T ≥ base` always (free-space geodesic is a hard lower bound —
the thing `base·(1+g)` can't guarantee). Since `τ ≥ τ_min`, `T ≤ base/τ_min`
(bounds `T` from above, preventing the `τ→0 ⇒ T→∞` runaway at full obstacle
contrast). `bias` starts `τ` near 1 (free space) at init. `T(start) = 0`
automatically since `base(start) = 0`.

**`ntfields_residual`** (torus.py:445-458) — symmetric speed-match penalty,
not the plain slowness residual eikonal.py uses:
```
q = ‖∇T‖_{g⁻¹} / s          # g⁻¹ = metric_inv(θ), s = slowness(θ)
residual = q + 1/q − 2       # ≥ 0, zero iff q = 1 (i.e. ‖∇T‖ = 1/s exactly)
```
This form is bounded and symmetric (doesn't blow up like `(q²−1)²` does when
`q` is large near obstacles), so free space doesn't get starved by the
obstacle-dominated loss. This is a *different* residual shape than
`eikonal.py`'s `‖∇T‖ − 1/s`; worth keeping as the distinguishing feature of
this strategy rather than reusing eikonal's residual.

**`solve_ntfields`** (torus.py:468-557) — original training loop. In the
sophisticated original this had: RAD resampling, obstacle-band gating, causal
weighting, and a **progressive obstacle-contrast anneal**:
```
slow_λ = 1 + λ·(slow_full − 1)     # λ: λ₀ → 1 over training (anneal_frac)
```
i.e. training starts in an easier, low-contrast version of the obstacle field
and ramps to full contrast. This annealing is NOT present in `eikonal.py`'s
simplified version (which never had it) — it's specific to how `solve_ntfields`
made the harder `base/τ` optimization landscape tractable. Worth testing
whether the simplified (no-anneal, plain dynamic collocation) version still
converges before deciding whether to port the anneal too.

**`solve_supervised`** (torus.py:713-750) — the oracle baseline: regression-fit
`field_ntfields`'s parameters directly to the FMM ground truth (`env.ground_truth`),
restricted to free-space grid points (`env.sdf(...) >= 0`), via plain MSE — no
PDE loss at all. This measures the *representation ceiling* (best any splat of
this budget can do), separate from whether a solver can find it. Note it reuses
`field_ntfields`'s parametrization as the thing being fit — so it's coupled to
whatever the ntfields field ends up looking like post-migration.

## Design tension to resolve first

The original `srms/methods/backends/{kan,mlp,ntfields,srm}.py` stub *names*
suggested NTFields might be a **backend** (an interchangeable function
approximator, parallel to `srm`/`mlp`/`kan`). But looking at what it actually
computes, `field_ntfields` is not a raw-value approximator like
`srm.eval_raw` — it's a **field factorization** (`base(θ)/τ(θ)`) that
*consumes* a raw approximator's output (`eval_splat_torus`, i.e. what is now
`srm.eval_raw`) the same way `weak_supervision.py`'s `field_roadmap` consumes
`backend.eval_raw` to build `roadmap_base(θ)·exp(g(θ))`.

That suggests NTFields is structurally a **strategy**, not a backend — i.e.
it belongs in `srms/methods/strategies/ntfields.py`, generic over whichever
raw backend (`srm`/`mlp`/`kan`) supplies `raw(θ)`, exactly like
`weak_supervision.py` is generic over the backend supplying its correction
term. The empty `srms/methods/backends/ntfields.py` stub would then either be
deleted, or repurposed for something that's actually a raw approximator (e.g.
if "NTFields" is later reinterpreted as a literal MLP architecture from the
P-NTFields paper, distinct from this `base/τ` factorization).

**Recommend confirming this reframing before writing code**: build
`srms/methods/strategies/ntfields.py` (field/residual/solve, generic over
`backend`, same shape as `weak_supervision.py`), and a
`srms/methods/strategies/supervised.py` that fits that same field to
`env.ground_truth(...)` by MSE, also generic over `backend`.

## Open questions for next session

1. Confirm the backend-vs-strategy reframing above.
2. Should `solve_ntfields`'s progressive obstacle-contrast anneal (`λ₀→1`)
   carry over, or should this also get the "simplest, dumbest" treatment like
   `eikonal.py`/`weak_supervision.py` (plain dynamic collocation at full
   contrast from step 0, no anneal, no causal weighting, no RAD resampling)?
   Try the simple version first and see if it still converges reasonably on
   the torus scene before reintroducing the anneal.
3. `solve_supervised` needs `env.ground_truth(cfg.resolution)` restricted to
   free-space points — straightforward with what's already in
   `TorusEnvironment` (`ground_truth`, `sdf`, `grid`), just needs writing.
4. New `cfg` fields the strategy will need (none exist in `srms/run.py`'s
   `Config` yet): `tau_bias`, `tau_min`, and — only if the anneal is kept —
   `lambda_init`, `anneal_frac`.
