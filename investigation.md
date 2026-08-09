# Investigation Log — Path Planning on Manifolds with Splats

Running log of the effort to do **optimal path planning on manifolds using Splat
Regression Models**, trained with a **self-supervised PDE loss** (no labelled
distance field). Theory and justification live in [`theory.md`](theory.md); this
file tracks what we actually try, what we measure, and what we learn.

> **Premise.** A geodesic value function `T` (time-to-go) can be represented by a
> splat field and solved from the Eikonal PDE (via the smooth transform
> `φ = e^{−T/ε}`, `ε²Δφ = s²φ`, `T = −ε log φ`). Paths are then `−∇T`. If this
> works on the plane and on the sphere, it generalizes to manifolds.

---

## Executive summary — state of play (checkpoint)

Main file is [`torus.py`](torus.py) (2-joint arm C-space = 2-torus, obstacles as a slowness field,
FMM ground truth). Canonical scene: seed 1, 3 obstacles, 384 splats, 4000 steps. **Fair baseline
suite** ([`run_baselines.sh`](run_baselines.sh)), RMS vs FMM:

| method | supervision | RMS |
|---|---|---|
| B1 vanilla Eikonal PINN | none | 0.534 |
| B2 NTFields (`base/τ` + speed-match loss) | none | 0.331 |
| B3 P-NTFields (+ progressive + causal) | none | 0.307 |
| B4 300 RRT* samples as **anchors** (trust values) | sparse | 0.393 |
| B5 300 RRT* samples as **base + Eikonal refine** | sparse | **0.109** |
| B6 supervised FMM-fit (oracle) | full GT | 0.003 |

**What we learned (confirmed, honest):**
1. **The splat representation is not the bottleneck** — the oracle (B6) fits the true field to 0.003.
   Everything above it is a *solver/supervision* limit.
2. **Pure physics-informed (P-NTFields analogue) tops out ~0.31** — the Eikonal residual constrains
   *slope*, not *level*; behind obstacles the level is genuinely under-determined.
3. **A prior is necessary, and *how* you use it decides everything.** Same 300 RRT* samples:
   trusting them as anchors (B4, 0.393) is **worse than no supervision** (bakes in RRT*'s suboptimal,
   too-high costs); using them as a rough **base the Eikonal refines** (B5, 0.109) wins by 3.6×,
   because the physics *shaves the planner's suboptimality toward the optimum*. (Anti-MPNet.)
4. **The Eikonal genuinely refines a suboptimal/sparse RRT* prior** (`refine_experiment.py`): −31–41%
   error, and it is the *physics* doing it (physics-led `base_reg=3` beats fit-base `base_reg=30` ~2×).
5. **Graceful, not immune, under thinning** (`thinning_experiment.py`, `figures/thinning.png`): refined
   error `~nodes^-0.67` vs base `~nodes^-0.89` — the physics flattens the slope and helps *most when
   sparsest* (−41% at 81 nodes, −4% at 601), but it is a power law, not flat/log — error still grows
   as samples thin.

**Honest open question (untested):** the true high-D test is error vs sample *dispersion* as dimension
rises at fixed budget (2→3→4-D) — not raw node count. Whether the physics keeps rescuing there is the
make-or-break. A **stronger base construction** at fixed samples (e.g. graph shortest-path over RRT*
*edges* instead of the straight-hop softmin) is the untried lever if it doesn't.

**Fixed-budget dimension sweep** (`torus_nd.py`, isotropic flat torus, 300 RRT* nodes, Godunov-Eikonal
GT). As dimension rises at fixed budget: dispersion 0.19→0.47→0.59, base RMS 0.113→0.201→0.333, refined
0.115→0.169→0.322; physics gap ~0 → +0.032 → +0.011. So error grows ~linearly with dimension (curse is
real, not beaten), fields stay usable through 4-D, and the refinement helps in 3-4D but modestly and
*non-monotonically* (the collocation itself thins in 4-D, so both prior and PDE degrade). Graceful
degradation, consistent with the 2-D thinning slope — supporting evidence the SRM works across
dimensions, not a claim it defeats the curse. `torus_nd.py` is self-contained (does not touch `torus.py`).

**Stress test — does the PDE genuinely help, or is it just the RRT* prior?** (`stress_test.py`, 10 d=2
scenes, obstacles 1→10, 300 RRT* nodes, aligned params.) **Every scene improved: mean +25% RMS, +29%
max-error reduction (range +11% to +33%).** So the physics is not collapsing and is not "just RRT*" —
it robustly refines the prior across complexities. A single easy scene can show ~+4% (base already
near-perfect); the *distribution* is what shows the ~25% contribution. Corroborates the earlier
refine-experiment finding (physics helps most where the base is worst). Note on `torus_nd.py`: it must
use `torus.py`-aligned params (obstacle 0.5–0.9, ramp 0.15, gamma 0.01, 4000 steps) — earlier drift to
harder obstacles + gamma 0.02 muted both the numbers and the PDE, which looked like "PDE not helping".

**3-D base_reg / capacity sweep** (`capacity_3d.py`, fixed 2048 collocation, 600-node base RMS 0.188).
The weak 3-D PDE was the *regularizer* (`base_reg·mean(g²)`, `T=base·exp(g)`) winning the prior↔physics
tug-of-war, **not** diluted collocation (random resampling accumulates coverage) and **not** capacity:

| base_reg | splats | solved RMS | impr | max |
|---|---|---|---|---|
| 3.0 | 384 | 0.176 | +7% | 2.05 |
| 1.0 | 384 | 0.164 | +13% | 1.38 |
| **0.3** | 384 | **0.158** | **+16%** | **0.93** |
| 0.1 | 384 | 0.164 | +13% | 1.13 |
| 1.0 | 768 | 0.169 | +10% | 1.88 |
| 1.0 | 1500 | 0.171 | +9% | 1.64 |

**Lower `base_reg` restores the physics** (+7%→+16%, max 2.05→0.93); ~0.3 is the sweet spot (0.1 drifts).
**Rule: scale `base_reg` DOWN as dimension rises.** Surprise: **more Gaussians does NOT help — it mildly
hurts** (384→1500: 0.164→0.171), so representation capacity is *not* the 3-D bottleneck (384 suffices;
2-D oracle already ~0.003). Training loss converges cleanly (`log10≈−2.9`) — no optimisation failure.
**Bottom line: the PDE genuinely contributes in 2-D (+25%) and 3-D (+16% at the right base_reg); the one
real high-d knob is the regularizer balance, not collocation count or splat count.**

**Planning evaluation — the honest negative result (`planning_eval_full.py`, 2-D→4-D, 6 obstacles, naive
−∇T descent, no guard).** SR / collision-free / path-cost-vs-FMM-optimal / ms-per-query, PDE field vs raw
softmin base vs RRT*:

| d | PDE (SR/cf/opt) | base (SR/cf/opt) | RRT* opt |
|---|---|---|---|
| 2 | 100/100/0.946 | 100/100/0.940 | 1.005 |
| 3 | 100/100/0.910 | 100/100/0.908 | 0.952 |
| 4 | 98/98/0.887 | **100/100**/0.887 | 0.921 |

**The PDE never improves planning and marginally *hurts* at d=4.** Reason: RRT* cost-to-come is already a
valid, monotone, minima-free value function; smoothing it (soft-min / splat) preserves that, so the base
always plans and the PDE has nothing to fix — it only nudges RMS (which diminishes: +25%→+16%→+4% over
d=2→4) and optimality marginally. **Given a roadmap, the physics is redundant for planning.** But the
*field* (either) beats RRT*'s own cost at every dimension (0.89–0.95 vs 0.92–1.005) and plans in
~90–150 ms/query — that win is the *representation*, not the physics.

**Reframe (decision pending with user).** Do NOT frame the paper as "physics-informed planning" (that is
NTFields/H-NTFields' turf, and our physics is light; H-NTFields itself came back to a roadmap). The
defensible contributions: (1) an **interpretable, differentiable, manifold-native splat value function**
with an **anisotropic Riemannian metric** (`metric_inv → M(θ)⁻¹`) that NTFields' *Euclidean* MLP cannot
represent — the uncontested part; (2) building it **cheaply from a sparse roadmap + label-free
self-supervision** (no FMM). Physics = the self-supervision mechanism, not the headline.

**SRM vs MLP on the NTFields lineage — fair head-to-head (2026-08-09, `srms/`).** H-NTFields
(arXiv 2604.13204, the weak-supervision one) and its own PDE-only ablation, 2-D torus, 4000 steps,
2048 collocation, identical scene/loss/sampling/optimizer/lr/seed — **the only CLI difference is
`--backend`**. SRM starts at 64 splats and densifies; MLP is SIREN with Fourier features.

| Baseline | Backend | Params | RMS | max err |
|---|---|---|---|---|
| No weak supervision (`--hnt-lambda-r 0`) | SRM adaptive | 2,779 | 0.291 | 3.15 |
| | MLP w=128 | 33,793 | 0.218 | 1.00 |
| | MLP w=40 | 3,521 | 0.253 | 0.89 |
| **H-NTFields (weak supervision)** | **SRM adaptive** | **2,779** | **0.0637** | 0.395 |
| | MLP w=128 | 33,793 | 0.0667 | 0.343 |
| | MLP w=40 | 3,521 | 0.0650 | 0.328 |

**Both representations land in the same place.** With weak supervision the three runs agree to within
5% (0.0637 / 0.0667 / 0.0650) — almost certainly inside seed noise, so *no ordering should be claimed
from this table without a seed sweep*. Weak supervision helps ~4x on both backends (0.29→0.064,
0.22→0.067), reproducing H-NTFields' qualitative claim and beating this log's previous best sparse-
anchor result (0.177, line ~761) by 2.8x. Pure physics reproduces the historical B2/B3 plateau
(0.291/0.218 here vs 0.331/0.307 then).

**Parameter efficiency is the one real asymmetry.** The SRM matches the 33,793-param MLP at **2,779
params (12x fewer)**, and the matched-size MLP (3,521) at 21% fewer. Caveat: the SRM's size was
*schedule*-limited, not chosen — 64 init + 7 densify passes x 48 spawn = 397 splats, never reaching
`max_splats=512`. A capacity sweep is needed before claiming this is the size it *needs*.

**Open: the SRM's max error.** Pure-physics SRM has max|err| 3.15 vs the MLP's 1.00, but with weak
supervision the gap nearly closes (0.395 vs 0.343). That pattern fits splat **locality**: where no
splat has support, `τ≈σ(bias)` so `T≈base`, the free-space geodesic, which under-estimates in obstacle
shadows — exactly the level (not slope) under-determination this log has documented, and exactly what
level supervision pins. Not yet localized; do that before treating it as established.

**Fidelity note.** The weak-supervision rows required one adaptation, measured not guessed: uncapped
sphere-packing saturates at 78 nodes on an open torus whose roadmap paths run 1.45x optimal, so `T_lb`
sat *above* true travel time at 94% of nodes and training was 6-8x WORSE than PDE-only (RMS 1.49/1.89/
2.07 — void, do not cite). Capping the free-sphere radius at 0.3 rad gives ~300-460 nodes with paths
optimal to within FMM grid noise. Applied to the shared scene, so identical for both backends.

**Retired / do-not-repeat:** dense (~300-node) roadmap base scored 0.070 but is *cheating* (dense RRT* is
impossible in high-D); the screened-Poisson `φ=e^{−T/ε}` route (theory.md §5) was explored but not
adopted — the direct Eikonal with the NTFields-style factored field won. Antipodal cut-locus sampling
*hurts*; tighter `τ_min` hurts; rim-seeded splats are neutral; gentle slowness ramp hurts (spreads the
obstacle). See the dated sections below for the full trail.

---

## Goal & success criterion

- **Goal:** show splats can plan optimal paths on manifolds using a
  self-supervised loss.
- **Done when:** on each test case the recovered `T` and the extracted path match
  a trusted ground truth within tolerance, *including* at least one genuinely
  curved manifold (the **sphere**).
- **Strategy:** 2D is the *flat special case*. Nail the flat cases and the sphere,
  then the exponential-map machinery (see `theory.md` §7) extrapolates to general
  manifolds.

## ★ North star

**Plan a 2-link robot arm from a start pose to a goal pose, around a workspace
obstacle, as a geodesic/Eikonal problem on its configuration-space torus `T²`
with the arm's kinetic-energy (inertia) Riemannian metric** — solved by the splat
value function and executed with feasibility-guarded path extraction.

Why this problem: it exercises every piece on a genuinely curved, non-trivially
metrized manifold, and it's a recognizable robotics result.
- **Manifold:** the arm's C-space is the 2-torus `T² = S¹×S¹` (two joint angles,
  each wrapping) — our first non-sphere manifold, with periodic geodesics.
- **Riemannian metric (the point):** the kinetic-energy metric `g(θ) = M(θ)`
  (the arm's mass/inertia matrix) is anisotropic and configuration-dependent —
  moving the shoulder costs more than the elbow, and the cost changes with pose.
  The Eikonal becomes `gⁱʲ ∂ᵢT ∂ⱼT = s²` with `gⁱʲ = M⁻¹`. This is the concrete
  "Riemannian metric for path planning" the whole project is arguing for.
- **Obstacle → C-obstacle:** a workspace obstacle maps (via forward kinematics +
  collision check) to a region on the torus; encoded as the smooth high-slowness
  field, avoided for real by the feasibility guard.
- **Deliverable:** a dual figure — the torus C-space (C-obstacle + time-to-go +
  planned path) and the workspace (the arm sweeping start→goal around the object).

Related: the **Dubins car** (repo `dubins_*`) is the same idea one step harder —
its C-space is `SE(2)=R²×S¹` and its metric is *sub-Riemannian* (only
forward motion + bounded turn radius; sideways is forbidden, so the metric is
degenerate). The arm is the cleaner *fully*-Riemannian first target; Dubins is a
natural follow-on.

---

## Reuse policy

Reuse as much of the existing repo as possible — the splat core came from the
original SRM repo and is trusted.

| Reuse | File | Role |
|---|---|---|
| **Splat core** | `lib/splat.py` | `eval_splat` (fwd), analytic grad, `gd_splat_regression`; `diag` mode for scaling |
| Eikonal splat scaffolding | `eikonal_splat.py`, `eikonal_nd_dynamic.py` | existing Eikonal-on-splats experiments |
| Manifold splat | `lib/manifold_splat.py` | tangent-space / manifold splat evaluation |
| Sphere Eikonal | `sphere_eikonal.py` | curved-manifold starting point |
| Obstacles | `obstacle_boost.py`, `demo_obstacle_field` | obstacle speed fields + FMM reference |
| PDE / PINN wiring | `physinf_comparison.py` | `PDEProblem`, `compute_pinn_loss`, `compute_derivatives` patterns |
| Ground truth | `fast_marching_2d` in `ground_truth.py` (adapted from `eikonal_splat.py`); closed-form on sphere | validation |

**Rule:** prefer extending these over rewriting. New code only where the readout
(PDE loss / path extraction) genuinely differs.

---

## Ground truth — the most important part

Getting the reference right is what makes results trustworthy. Per case:

| Case | Ground-truth `T` | How |
|---|---|---|
| 2D free space | analytic | `T(x) = ‖x − start‖` (unit speed) — `make_plane` |
| Sphere | closed form | geodesic `T = arccos(⟨x, start⟩)`; cut locus at antipode — `make_sphere` |
| 2D + obstacle(s) | numerical | **fast marching** `|∇T| = 1/speed`, speed→0 in obstacles — `make_plane_fmm` |
| Path check | — | integrate `ẋ = −∇T/|∇T|` from sampled starts; compare length & shape to reference |

`fast_marching_2d` is validated against the analytic plane: max abs diff `0.0155`
(the expected ~1% first-order fast-marching discretisation error). Iso-time
**contour lines** are drawn on every field — they are the level sets `{x : T=c}`,
i.e. the reachable-in-time-`c` wavefronts; around obstacles they bend visibly.

Metrics to log every run: `‖T_splat − T_ref‖` (RMS + max), gradient-direction
error `∠(∇T_splat, ∇T_ref)`, path length ratio vs. optimal, collision count.

---

## Roadmap

Status: ⬜ not started · 🟡 in progress · ✅ done · ⚠️ blocked

Each milestone has two stages: **(a)** validate the ground truth + splat
representation + visualisation by a *supervised* least-squares fit, then **(b)**
replace the targets with the *self-supervised* PDE loss (the real method).

| # | Milestone | Purpose | Ground truth | Status |
|---|---|---|---|---|
| M1 | **2D free grid** (single source) | flat sanity check of representation + viz, then PDE loss + path extraction | analytic `‖x−start‖` | 🟢 (a) done · (b) self-supervised solve done |
| M0 | **Sphere geodesic** (single source, no obstacles) | prove the manifold logic (ambient embed → exp/log map + PDE) on the simplest curved case | closed-form arccos distance | 🟡 (a) done · (b) PDE pending |
| M2 | **2D single obstacle** | handle a boundary / speed field; recover the cut locus | FMM | 🟢 (a) done · (b) self-supervised solve done |
| M3 | **2D multiple obstacles + intricate shapes** | scale the obstacle machinery (SDF scenes) | FMM | 🟢 5–6 objects done · intricate shapes routed (deep pockets improving) |
| M4 | **Generalize to manifolds** | combine curved metric + obstacles | closed-form / FMM-on-manifold | ⬜ |
| M5 | **Flat torus `T²`** (periodic, no obstacles) | validate the periodic-manifold machinery: intrinsic splat (angle space + wrapped log map), wrapping geodesics, Eikonal with the flat metric | analytic flat-torus geodesic | 🟢 done — RMS `2e-5`, intrinsic `d=n` |
| M6 | ★ **2-link arm on `T²`** (inertia metric + C-obstacle) | the north star: Riemannian metric + workspace→C-space + guarded path + dual viz | FMM on the metric torus | ⬜ |

> Note: M0 (sphere) and M1 (plane) can proceed in parallel — the sphere validates
> the manifold logic while the plane validates the PDE/path plumbing in the
> easiest possible setting.

---

## Method (this is what each experiment implements)

Self-supervised loss on the splat field, per `theory.md` §5–6.

- **Field:** fit `φ(x) = Σ_j V_j · ρ_{A_j,B_j}(x)` with `lib/splat.py`.
- **PDE residual (interior):** `‖ ε²Δφ − s²(x) φ ‖²` — *linear*, smooth. (Fallback:
  direct Eikonal residual `‖|∇T|² − s²‖²` on `T = −ε log φ` if the transform
  underperforms.)
- **Boundary/goal:** `φ(goal) = 1` (i.e. `T(goal)=0`); obstacles via `s(x)→∞`
  (speed → 0) or `φ→0` on obstacle interior.
- **Manifold:** evaluate splats at `log_p(x)`; use the metric `g` in the residual.
- **Path extraction:** integrate `ẋ = −∇T/|∇T|` using the closed-form splat
  gradient.

Open method decisions to resolve empirically:
- value of `ε` (accuracy vs. smoothness trade-off);
- transformed (`φ`) vs. direct (`T`) Eikonal residual;
- number of splats `k` and whether to use dynamic allocation.

---

## Experiment log

_Newest first. One entry per meaningful run; record the change, the metric, and
the decision (keep / discard / follow-up)._

### 2026-07-28 — M1(a) plane & M0(a) sphere: representation + GT + viz validated
- **Environment:** replaced Poetry with **uv**; `pyproject.toml` now PEP-621 with
  pinned deps in `uv.lock` (`uv sync`). `tyro` drives all configs.
- **Code:** `ground_truth.py` (analytic time-to-go + indexable `PlanningProblem`
  with `__getitem__ → (start, goal, time)`); `train.py` (splat fit + 3-panel
  `[GT | pred | error(BWR)]` figure, shared error scale). GT verified first:
  `figures/gt_{plane,sphere}.png` — plane max `√2`, sphere max `π`. Correct.
- **Setup:** supervised LS fit; k=256, num_train=4096, steps=3000, Adam lr=5e-3,
  init_scale=0.3, seed=0, resolution=160.
- **Result:**
  - plane — RMS `2.3e-3`, max|err| `4.6e-2`, rel-RMS `0.79%`.
  - sphere — RMS `1.6e-3`, max|err| `4.7e-2`, rel-RMS `0.28%`.
- **Decision:** keep. Representation + GT + viz confirmed on both domains.
- **Learning:** error is `< 0.01` everywhere except the **genuinely non-smooth
  points**, exactly as `theory.md` predicts: the source cone tip (both domains)
  and the **antipodal cut locus** on the sphere (lon=±180°). A smooth splat sum
  rounds these — expected, not a bug. This is the empirical face of the kink that
  motivates the `φ = e^{−T/ε}` transform for stage (b).

### 2026-07-28 — numerical GT, contours, tooling, and M2(a) obstacle batch
- **Numerical ground truth:** added `fast_marching_2d` (first-order FMM, adapted
  from `eikonal_splat._fmm_reference_2d`) and `make_plane_fmm`, so obstacle fields
  now have a real reference. Validated vs analytic plane: max abs diff `0.0155`.
- **Contours:** all fields now render iso-time contour lines; obstacles drawn as
  hatched patches; obstacle interiors masked (NaN → grey) and excluded from
  training and metrics.
- **Tooling:** `ruff` added (dev group), `ruff format` + `ruff check` clean on
  `ground_truth.py`, `train.py`, `experiment.py`; added `[tool.pyright]` (venv)
  so the IDE resolves imports — VS Code diagnostics now empty on both files.
- **M2(a) — single obstacle, batched:** `experiment.py` generates reproducible
  scenarios from `(global_seed, index)` (random start, goal, obstacle), fits a
  splat to the FMM field, writes params+metrics to `logs/obstacle_batch.tsv`.
  Batch of 4 (seed 0, k=256, steps=1500): RMS mean `4.3e-3`, max `5.1e-3`.
- **Result:** splat reproduces the obstacle-routed field well; error `< 0.05`
  except faint residual at the obstacle rim and the cut locus behind it (as
  predicted). Worst case `scenario_seed0_1` archived.
- **Decision:** keep. M2(a) representation validated. Next is the self-supervised
  PDE loss (stage b) and path extraction via `−∇T`.

### 2026-07-28 — stage (b): self-supervised Eikonal solve (`self_supervised.py`)
- **Loss (no targets):** fit the splat to the viscosity Eikonal
  `|∇T|² − εΔT = s²` via a **relative** residual `(|∇T|² − εΔT)/s² − 1`
  (dividing by `s²` balances free-space vs. stiff obstacle collocation points).
  `∇T`, `ΔT` come from autodiff (`jax.grad` / `jnp.trace(jax.hessian)`).
- **Two design choices that made it converge:**
  1. **Factored field** `T(x) = ‖x−start‖·(1 + g(x))`, splat = deviation `g`,
     `g` initialised to **zero**. This bakes in `T(start)=0` and makes the
     free-space distance the *exact* starting point. (Fitting raw `T` from
     `g≈O(1)` init — the `det(A)⁻¹` amplifies `V` — went nowhere: RMS ~1.1.)
  2. **Relative residual.** With the absolute residual the interior
     (`s=10 → |∇T|=10`) dominated the loss (stuck at `~3.3`); dividing by `s²`
     fixed it (`~0.03`).
- **Free plane (no obstacle):** RMS `2.7e-2`, max|err| `3.6e-2`. Error is a
  smooth uniform `+0.03` bias — exactly the systematic `O(ε)` viscosity bias.
- **Single obstacle (`s=10` inside, `(0.1,0.1,0.35)`):** RMS `2.3e-2`,
  rel-RMS `4.1%` vs. the FMM reference, `k=256`, 4000 steps (~8 min CPU). The
  predicted iso-time contours **bend around the obstacle** matching FMM; error is
  `< 0.05` in free space, concentrated at the obstacle rim and the cut-locus
  shadow behind it (max `0.22`) — the genuinely non-smooth region — plus a faint
  `1/r` seam at the source from the factored form. **This meets the goal: the
  self-supervised loss solves the obstacle plane with no ground-truth targets.**
- **Riemannian framing confirmed in code:** the obstacle is encoded purely as a
  high-slowness region `s(x)`, i.e. the conformal metric `g = s²I`; nothing else
  changes. The flat-plane-with-obstacle *is* the Riemannian case (see `theory.md`
  §7 note). Curved manifolds only swap `g` and add the `exp/log` charts.
- **Open:** the `−εΔT` bias (`~ε`) sets an accuracy floor; smaller `ε` or an
  `O(ε)` bias correction would tighten it. Path extraction (`−∇T`) still to do.

### 2026-07-28 — SDF scenes, reflecting-Neumann (negative result), smooth-slowness
- **Geometry (`scenes.py`):** obstacles are now **SDF** primitives (circles + rotated
  boxes); a scene is their union (`min` of SDFs). One representation gives
  membership (`d<0`), boundary normals (`∇d`), surface samples, *and* intricate
  non-convex shapes as unions. `make_plane_scene` builds the FMM ground truth from
  the same field. Rendering is shape-agnostic (obstacle region outlined from the
  mask), so 5–6 objects and shapes like `cross`/`u_trap` all render correctly.
- **Reflecting Neumann boundary — tried, did NOT work (kept as a finding).**
  Imposed `∂T/∂n = 0` on obstacle rims (normals `∇d`), `s≡1` free space, no interior
  collocation. Result: RMS `0.42`, the field stayed **near straight-through** —
  the far side behind the obstacle was uniformly too low. Diagnosis: the "shadow"
  (extra distance from routing) is a **nonlocal** effect; a purely local boundary
  condition under gradient descent does not propagate it, and the factored base
  `‖x−start‖` biases toward the through-solution. Not a capacity/sampling issue —
  a solution-selection failure. (Would need causal/propagation-aware training.)
- **Pivot — smooth SDF slowness (works).** Encode obstacles as a *smooth* high-
  slowness field `s = 1 + (s_max−1)·σ(−d/width)` (`scenes.smooth_slowness`), the
  conformal metric `g = s²I`. The high interior cost is the **global** signal that
  forces routing (which Neumann lacked); the smooth transition (vs. a hard jump)
  keeps the field splat-representable, so rim error stays small. Relative residual
  `(|∇T|²−εΔT)/s² − 1`, collocation over the whole domain.
- **Single circle (`s_max=12`):** RMS `3.8e-2`, **max|err| `0.074`** — down from
  `0.22` with the earlier hard-jump slowness. The smooth ramp ~halved the rim
  error, the compounding-mitigation we were after. Prediction routes around the
  obstacle *and* reproduces the shadow behind it; error is the mild `O(ε)` bias.
- **5 objects (mixed circles+boxes, `k=384`):** RMS `5.6e-2`, max `0.145`. Error
  did **not** compound linearly (≈1.5× the single-object RMS, not 5×) — it is
  dominated by the slowly-accumulating `O(ε)` viscosity bias (grows with distance
  from the source), not per-obstacle rim error. Routing around all five is correct.
  **This is the key scaling evidence: the smooth-metric formulation adds objects
  without per-object error compounding.**
- **Intricate non-convex shape (`cross`, plus-sign):** global routing correct and
  far field accurate; RMS `7.0e-2` (`s_max=12, k=384`). Residual failure mode is
  the **concave reentrant pockets**: to reach a pocket the wave must travel far
  around an arm, and the near-enclosed field there is underestimated (local max
  err ~1.16 in a small pocket; RMS stays ~10% because it is spatially tiny).
- **Negative tuning result:** raising contrast/capacity (`s_max=18, k=512, 2560`
  collocation) made the pockets **worse** (max `1.16 → 1.77`), not better — a
  stronger barrier deepens the true pocket time, so the same under-shoot costs
  more, and the steeper field is harder to resolve. So the deep-concavity error is
  *not* a capacity/contrast problem; it needs **pocket-focused adaptive
  collocation** (sample where residual/curvature is high) — logged as next work.
  For planning this matters little: pockets are near-enclosed dead-ends one would
  not route into. `s_max=12` is the better default.

### 2026-07-28 — adaptive collocation, weak supervision, path extraction
- **Residual-adaptive collocation (RAR/RAD).** Every `resample_every` steps, score
  a fresh candidate pool by the **PDE residual** (self-supervised — no error oracle,
  no obstacle knowledge) and resample collocation with probability
  `∝ residual^p`, blended with a uniform floor for coverage. The residual is the
  proxy for "where the physics isn't satisfied yet." Real-world-honest: it never
  looks at the ground truth or the SDF.
- **Weak supervision (optional).** Adds a sparse anchor term
  `w·mean((T_pred(xₐ) − T̃(xₐ))²)` where `T̃` is a *cheap* coarse solver
  (low-res FMM here; a stand-in for a coarse planner or sparse measured travel
  times). Anchors are drawn biased toward **large distance from the source**,
  where the accumulating `O(ε)` viscosity bias is worst and the PDE residual
  (a gradient constraint) does not pin the absolute level. Slots in as one extra
  loss term; off by default (`weak_weight=0`).
- **Path extraction.** `extract_path` integrates `ẋ = −∇T/|∇T|` from a goal down
  to the source (closed-form splat gradient), giving the optimal source→goal
  route. `figures/path_*.png` overlays the routes on the field.
- **Single circle (adaptive, with paths):** RMS `3.7e-2`; the four extracted paths
  are correct — unobstructed goals go straight, the two behind/beside the obstacle
  **route around it**. First actual planned trajectories from the splat field.
- **5 objects + paths:** all four paths route correctly around the five obstacles
  (weaving through the gap between two of them to reach the far goal) — path
  extraction is robust on a hard scene.
- **Negative result — naive weak supervision hurt (root-caused + fixed).** First
  5-object run with `weak_weight=0.3` regressed (RMS `0.056 → 0.079`, a local blue
  blob of max `0.99` on the near side of an obstacle). Cause: weak anchors were
  placed by distance only, so some landed **near obstacle rims — exactly where the
  cheap coarse (res-60) FMM is least accurate** — and dragged the field wrong
  there. Fix: sample weak anchors only in open space (`weak_clearance` from any
  obstacle), where the cheap solver is reliable. **Validated:** with the fix weak
  supervision now *helps* — RMS `0.056 → 0.048`, max `0.99 → 0.144` (blob gone),
  and the far-field `O(ε)` drift is visibly reduced (anchors pin the level).
  Lesson: weak supervision helps the **far-field open region**; never anchor where
  the cheap signal is itself unreliable. Off by default; a useful knob when on.
- **Adaptive collocation is mild here**, not a silver bullet: it slightly helped
  the single circle and did not, on its own, resolve the intricate-pocket case.
  Its real value is directing capacity by residual without an error oracle; tuning
  (`residual_power`, `resample_uniform_frac`) matters and can destabilise if it
  starves free-space coverage.

### 2026-07-28 — feasibility-guarded path extraction (soft field, hard safety)
- **Why:** the smooth slowness makes obstacles *permeable* (finite cost), so a raw
  `−∇T` path can cut through — worse in high-D where thin C-obstacles are under-
  sampled. Decouple **guidance** (soft field) from **feasibility** (hard SDF check).
- **How:** during path integration, if a step heads into an obstacle (`d<margin`,
  `−∇T·∇d < 0`), remove the inward normal component (slide along the surface) and
  project any penetration back out along `∇d`. **Head-on fix:** when the slide
  tangent degenerates (path perpendicular to the wall), fall back to a consistent
  perpendicular tangent chosen toward the source — without it, dead-centre
  incidence tunnels straight through.
- **Demo (`figures/path_guard_demo.png`, instant, zero-splat obstacle-unaware
  field):** raw paths enter the obstacle **2/4**, guarded **0/4** — the guarded
  paths hug the rim and route around. On properly-solved fields the guard is a
  no-op (the `s_max=12` wall routed around on its own), i.e. it activates exactly
  when the field would otherwise violate feasibility. This is the guarantee the
  soft metric alone can't give.
- **Limits (honest):** the guard fixes *feasibility*, not *optimality* — if the
  field is very wrong, the rerouted path is safe but not shortest. Adequate
  `s_max` + the guard together give safe-and-near-optimal.

### 2026-07-29 — M5 flat torus `T²` (intrinsic representation, known-base + splat)
- **Two placements clarified.** (A) intrinsic: splat in angle space, evaluate at the
  log map `wrap(θ − B)`, dimension `d = n`. (B) ambient: embed `θ→(cos,sin)`,
  `d = 2n`. Both give the identical exact result; **standardised on (A)** — uniform
  across manifolds (only the log map / metric change) and dimension-efficient.
  `torus.eval_splat_torus` is the periodic (wrapped-displacement) splat eval.
- **Self-supervised mis-selection (finding).** Solving the *global* distance field
  from a smooth (chordal) base drifted to a wrong Eikonal solution on the compact
  torus — RMS got *worse* as the residual dropped (`0.39→0.46`). Same class as the
  Neumann-shadow failure: local residual minimisation doesn't fix global structure.
- **Known-base + splat (the agreed pattern).** Use the analytic flat-torus geodesic
  `‖wrap(θ−start)‖` as the base (known global structure), splat learns only *local*
  corrections. Free flat torus → `g≈0`, **RMS `2e-5`** (essentially exact); shape,
  wrapping, and antipodal cut locus all correct. The splat earns its keep on the
  *unknown* parts: obstacles (C-obstacles) and the arm's non-flat metric `M(θ)`.
- **Metric stays separate** from placement: `metric_inv(θ)` (identity now, `M(θ)⁻¹`
  for the arm) enters only the residual's quadratic form. M6 = swap that hook.

### 2026-07-30 — torus `T²` WITH obstacles: debugging the solve (bug → under-determination → weak)
Goal: show the splat learns a *local* correction (obstacle routing) on the manifold, scored
against a **periodic** fast-marching GT. The convergence chase was itself the lesson:

- **Bug (necessary fix).** `fast_marching_torus` expects a *speed* map (`slowness = 1/input`);
  we passed the *slowness* map, so the GT solved the **inverse** problem (obstacles became the
  *fastest* lanes → wave went through them). The training residual solves `|∇T| = s` (obstacles
  slow → routed around) — the opposite. Diagnostic: residual on the GT was `0.20` (inconsistent)
  vs `0.0065` when the FMM is called with `speed = 1/slow`. Fixed by passing `1/slow`.
- **Under-determination (the real obstacle).** Bug-fixed but bare Eikonal → still RMS `0.85`,
  and the splat's loss (`5e-4`) was *lower* than the GT's own residual (`6.5e-3`): it found a
  valid-but-wrong (under-scaled) Eikonal solution. `|∇T| = s` fixes the field's *slope*, not its
  absolute *level* far from the source — a global level ambiguity. Not a local minimum; the
  objective simply doesn't pin the level.
- **Weak supervision fixes it.** Uniform anchors over the whole torus (from a cheap coarse
  periodic FMM, kept clear of rims) pin the level everywhere → **RMS `0.85 → 0.35`**, bulk error
  now ±0.05, global structure/routing correct. Confirms the diagnosis: the degeneracy is a level
  ambiguity, and level-pinning is the cure. (Anchors must be *uniform*, not far-biased, since the
  ambiguity is global.)
- **What remains.** Thin high-error bands hugging obstacle **rims / cut loci** (max `3.09`, tiny
  in area but inflates RMS): where anchors are excluded, the field is non-smooth, and the coarse
  FMM is least accurate. This is the *selection/smoothness* part — the next lever is **viscosity**
  (`−εΔT`), deferred by request. Bug fix + weak supervision are the two confirmed ingredients.

### 2026-08-01 — torus level under-determination: label-free attempts (all insufficient)
The obstacle-free torus is exact (`2e-5`); WITH obstacles the self-supervised solve
has a stubborn **level** error. Root cause (now clear): the Eikonal residual
`|∇T|=s` constrains the *slope*, but the *level* is the integral of the slope, so a
tiny (loss-invisible) uniform slope bias integrates into a large level offset —
severe on the torus because `T_max≈5.6` (vs plane ~2) and the domain wraps.
Attempts to fix it **without anchors**, on the base instance (`start=(-1.5,-1.5)`):

| attempt | mechanism | RMS | failure mode |
|---|---|---|---|
| plain recipe | causal + RAD + curriculum + obstacle/cut-locus bands | **0.44** | shadow *under*-scales |
| causal annealing | relax causal weight → full residual by end | 0.46 | no change |
| DP consistency | `T(x)=stopgrad(T(y))+s·ds` toward source | diverges (6.6, 7.8) | bootstrapping runaway (self-referential target/direction) — **low loss, diverged field** |
| visibility pin | pin `T=‖wrap(x−start)‖` where the source ray is clear | 0.54 | stops runaway but shadow *over*-scales (maxT≈9) |

**Lesson:** local/geometric terms just push the level around (under↔over↔runaway)
without pinning it — the level is a *global* quantity, so only a *global* fix works:
(a) data anchors (weak supervision, `0.22–0.35`), or (b) a formulation **unique by
construction**. Introspection (`maxT` logged per resample) was decisive for seeing
the divergence in real time. Next: the **screened-Poisson** reformulation
(`ε²Δφ = s²φ`, `T=−ε log φ`) — linear, elliptic, unique given `φ(source)=1`; `ε`
trades accuracy/bias vs. `φ` dynamic range (moderate `ε≈0.5` keeps `φ≳10⁻⁵`).

### 2026-08-01 — screened-Poisson (structurally right, but T-recovery capped) + conclusion
- **Screened-Poisson `ε²Δφ = s²φ`, `T = −ε log φ`** (linear, unique). It **killed the
  level ambiguity**: the field *structure* (routing, wrapping, shape) came out
  **exactly right** — the error was a *uniform* `+ε` offset, not the structural
  shadow error of every other attempt. Clean monotone convergence (solvable, as
  expected). `φ(source)=1` made exact by multiplying the correction by the base
  distance (so `T(source)=0`).
- **But the `T`-recovery is log-sensitive and doesn't beat the recipe.** `T=−ε log φ`
  amplifies φ-errors far from the source; smaller `ε` reduces the bias but *worsens*
  the sensitivity (and the `1/φ0` residual weighting over-emphasises the far field,
  causing local blow-ups). Net: `ε=0.5 → RMS 0.795`, `ε=0.3 → 0.85` (worse). No free
  lunch in `ε`.
- **Conclusion (honest).** Best **label-free** torus result remains the
  **importance-sampled + causal recipe, RMS `0.44`** (structurally imperfect but
  lowest error). The full label-free sweep — causal, causal-annealing, DP
  consistency, visibility pin, screened-Poisson — did **not** beat it. Robustly
  reaching `≤0.2` label-free on this compact, larger-scale torus is a genuine open
  problem; the reliable `≤0.2` path is a **handful of sparse anchors** (`0.22–0.35`
  earlier), whose role we now understand precisely: **global level/branch
  selection**, which local self-supervision cannot infer. Recommendation: use a
  minimal sparse-anchor budget for the working 2-D demo and move energy to the
  high-D validation + the arm (M6), the stated goal. `theory.md`/`validation_highd.md`
  hold the transferable pieces.

### 2026-08-02 — RRT* anchors + best formulation: the honest 2-D torus verdict
- **RRT* anchor source built + validated** (`rrt_star`, `rrt_star_anchors`): pure-NumPy,
  mesh-free, dimension-scalable (the intended high-D anchor source, *not* grid FMM).
  Anchor cost-to-come matches FMM to `±0.04` — accurate. Costs ≥ straight-line base;
  detours behind obstacles captured. This is the piece that transfers to high-D.
- **But 20–40 sparse anchors do not reach `≤0.2` on the torus**, across formulations
  and instances: causal recipe + 30 anchors → `0.41`; no-causal + 40 (weak 1.0) → `0.55`;
  simple-uniform + 40 → `0.48`; *moderate* instance (small obstacles) + 40 → `0.37`.
  Historically, dense anchors did better (200 → `0.35`, 400 → `0.22`) — so the **anchor
  count is the limiter**, not their accuracy or the formulation.
- **Why the torus is fundamentally harder than the plane** (the real finding): the plane
  solved obstacles *label-free* to `0.05` because it is **non-compact** — characteristics
  flow outward from the source and never return, so the exact free-space base pins the
  level. The torus is **compact**: characteristics **wrap around the loops**, so the
  value function's level must be **globally consistent around the torus**, and the local
  Eikonal residual under-determines that global level. Sparse anchors pin it at points but
  the level drifts between them. This is a *topological* difficulty, not a tuning failure.
- **Verdict:** `≤0.2` on this compact torus needs either ~150 (still cheap/scalable) RRT*
  anchors (→ ~`0.22`) or a genuinely better global-consistency formulation (open). Best
  with a 40-anchor budget ≈ `0.37`. The high-D value proposition rests on *feasibility +
  bound* certification (`validation_highd.md`), not tight RMS.

<!-- Template for future entries:
### YYYY-MM-DD — <short title>
- **Change:** what was varied vs. previous run.
- **Setup:** case (M#), k, ε, steps, lr, seed, residual type.
- **Result:** T RMS / max err, grad-angle err, path length ratio, collisions.
- **Decision:** keep / discard / follow-up.
- **Learning:** what it tells us.
-->

---

## Open questions & risks

- **Cut-locus accuracy.** Does the `φ` transform recover the kink sharply enough,
  or does it over-smooth the medial axis? (Watch `ε`.)
- **`ε → 0` conditioning.** Small `ε` makes `φ` span many orders of magnitude
  (`e^{−T/ε}`); may need log-space fitting or normalization.
- **Splat coverage.** Do splats migrate to obstacle rims on their own, or do they
  need targeted initialization / dynamic allocation there?
- **Manifold sampling.** Correct collocation-point sampling on the sphere and
  correct `log_p` near the cut locus (antipode is singular).
- **Path integration** near kinks: `∇T` direction is ambiguous on the cut locus.
- **Deep concave pockets** (e.g. `cross` reentrant corners) are under-resolved and
  not fixed by more contrast/capacity — needs **residual-adaptive collocation**
  (densify where the residual/curvature is high). Next concrete task for M3.
- **`O(ε)` viscosity bias** accumulates with distance and is the dominant error on
  clean scenes; try `ε`-annealing or a bias-corrected readout.

---

## Learnings (append as they solidify)

- **Splats represent geodesic fields accurately** on both the plane and the sphere
  (rel-RMS < 1%). Error localises to non-smooth features (source tip, cut locus),
  confirming the theory and motivating the smooth `φ` transform.
- **Reuse caveat:** `lib.splat.gd_splat_regression` re-`jit`s its `variation`
  closure every step (a static arg), so it recompiles per iteration and is
  impractical for long CPU runs. We reuse `eval_splat` (the splat core) and drive
  it with a **jitted Adam + autodiff-MSE loop** in `train.py` instead — ~40 it/s,
  3000 steps in ~75 s. The analytic `eval_splat_grad` and the self-supervised PDE
  loss are separate next steps.
- **Sphere handled in ambient R³** (splat `d=3`, evaluated on the unit sphere)
  rather than via tangent charts — the simplest first embedding. The exp/log-map
  construction from `theory.md` §7 is deferred to stage (b)/M4.
- **Obstacles cost ~2× the free-space error** (RMS `2.3e-3 → ~4.3e-3`) under a
  supervised fit — the extra error is concentrated at the obstacle rim and the
  cut locus behind it, i.e. the new non-smooth features, consistent with the
  free-space finding. No accuracy cliff; the smooth splat handles routed fields.
- **Reproducibility convention:** scenarios derive from `(global_seed, index)` via
  `np.random.default_rng([seed, index])` for geometry and
  `PRNGKey(seed*10000+index)` for the splat. `logs/obstacle_batch.tsv` records
  every scenario's start/goal/obstacles + metrics, so any case is replayable.
- **Obstacles = a metric, not a boundary (what actually works).** Encoding an
  obstacle as a *smooth high-slowness region* (`g = s²I`) is both the correct
  Riemannian view and the robust trainable one: the high interior cost is the
  **global** signal that propagates the routing/shadow. A reflecting Neumann
  boundary is physically exact but only *local*, and under gradient descent it
  fails to establish the nonlocal shadow (field stays near straight-through). Keep
  the metric formulation; it also generalises to curved manifolds by swapping `g`.
- **Smooth beats hard for the splat.** A smooth slowness ramp (SDF sigmoid) vs. a
  hard jump halved the single-obstacle rim error (`0.22 → 0.074`) — a `C∞` splat
  cannot represent a discontinuity, so smoothing the metric is what stops
  per-object error compounding. Adding objects grew RMS sub-linearly (1→5 objects:
  `0.038 → 0.056`); the dominant error is the accumulating `O(ε)` viscosity bias,
  addressable by shrinking `ε` or an `O(ε)` bias correction.
- **SDF is the right obstacle interface** — one representation yields membership
  (`d<0`), the metric ramp, boundary normals (`∇d`), surface samples, intricate
  shapes (unions), and (future) a sensed-environment SDF. See `scenes.py`.

## Checking against NTFields, and the derivation we should have used

We compared to **NTFields** (Ni & Qureshi, ICLR 2023) and **P-NTFields** (progressive
learning) to check for a bias in our own framing. The honest result: our "the torus is
fundamentally hard because it is compact" story was a **rationalisation**. NTFields solves
**4-DOF and 6-DOF manipulator C-spaces — which are compact tori — to 90–96% success** with
FMM-quality time fields. Compactness is not the blocker; our *formulation* had gaps. Three,
all confirmed against their method:

- **Bounded factorisation (the structural fix).** NTFields writes `T = ‖q_s−q_g‖ / τ`
  with `τ ∈ (0,1]`, which **guarantees `T ≥ straight-line distance`** and sends
  `τ→0 ⇒ T→∞` in obstacles. Our `field_value = base·(1+correction)` (`torus.py:187`) leaves
  `correction` unbounded, so it *permits* `T < base` — physically impossible and exactly our
  under-scaling failure. Adopt `T = base_g/τ`, `τ = σ(splat) ∈ (0,1]`: the failure mode
  becomes unrepresentable.
- **Speed-space symmetric loss (conditioning).** We minimise `(‖∇T‖²_{g⁻¹}/s² − 1)²`
  (`torus.py:197`) — quartic in `∇T`, unbounded, and with `s²=100` inside obstacles those
  points dominate the gradient while the level-setting free-space bulk starves. NTFields
  matches the *speed* `Ŝ = 1/‖∇T‖` to the known speed `S = 1/s` with a symmetric, bounded,
  √-smoothed objective. Recommended: `L = mean(Ŝ/S + S/Ŝ − 2)` (≥0, zero iff `Ŝ=S`,
  balanced across free space and obstacles).
- **Progressive obstacles (local minima).** They anneal obstacle influence `λ: 0→1` from the
  free-space solution. Our curriculum was *spatial* (source-outward), not *obstacle-contrast*.
  Use `s_λ = 1 + λ(s−1)`, `λ: 0→1`, starting from the exact free-space field `base` already
  gives.

**The derivation to standardise on** (manifold-general, NTFields-aligned):
`T = base_g/τ` (`τ=σ(splat)∈(0,1]`) → metric dual-norm `Ŝ = 1/√(∇Tᵀ metric_inv ∇T)` →
symmetric speed-match `mean(Ŝ/S + S/Ŝ − 2)` → progressive slowness `λ:0→1` → keep RAD
residual sampling for the cut locus. The `metric_inv` hook (`torus.py:152`, identity now)
carries the anisotropy; the base becomes the metric geodesic `‖wrap(θ−start)‖_g`.

### Isotropic (NTFields) vs anisotropic (ours) — two distinct anisotropies
- **Metric anisotropy `metric_inv(θ)`.** NTFields uses one *scalar* speed `S(q)` per point →
  cost is direction-independent, wavefronts are circles. That is `metric_inv = c·I`. Our arm
  metric is the inertia matrix `M(q)` from `KE = ½q̇ᵀM(q)q̇`: dense, configuration-dependent,
  off-diagonal-coupled — shoulder motion has high inertia (costly), wrist motion low (cheap),
  and cost depends on *direction* in `q̇`. **A scalar speed cannot represent this;
  `metric_inv → M(θ)⁻¹` can.** Note: our *current* runs still set `metric_inv = I`, so they are
  isotropic too — the anisotropy is a live structural hook, not yet exercised.
- **Representation anisotropy (splat covariance).** Each splat `A[j]` is a full scale+rotation
  matrix (`eval_splat_torus`, `torus.py:141`), so every basis bump is an *ellipse* that aligns
  with field ridges/creases (obstacle rims, cut locus) — one splat where an isotropic basis
  needs many. Always on, independent of the metric. An anisotropic basis is exactly what
  represents an anisotropic metric's elliptical equal-cost sets efficiently.

### What the splat representation buys over NTFields (same problem, different representation)
We solve the *same* Eikonal motion-planning problem; the difference is the value-function
representation, and it is why their techniques alone are not the ceiling for us:
1. **Explicit, interpretable geometry** — centres, covariances, weights are readable and
   editable; NTFields is a black-box `(q_s,q_g)→τ` ResNet.
2. **Anisotropic basis functions** — splats represent ridges/creases natively via covariance;
   an MLP with scalar speed must compose many units and still carries no direction structure.
3. **Manifold-native, intrinsic placement** — centres live on the torus, evaluated through the
   flat-torus log map `wrap(θ−B)`, so **periodicity is built in and the metric is pluggable**
   (`metric_inv`); we generalise to curved/anisotropic manifolds by swapping the log-map/metric.
   NTFields embeds the C-space in ambient/Euclidean coordinates and relies on the network to
   learn the wraparound, even though the C-space is topologically a torus.
4. **Known base + sparse correction** — the analytic geodesic *is* the prior; splats model only
   the obstacle-induced deviation. NTFields' `‖Δ‖/τ` is the same idea but with a black-box
   correction.

**Why their tricks are necessary but not sufficient here.** NTFields validated the *isotropic,
identity-metric* case and scaled it dimensionally. A real arm C-space adds a genuinely
*anisotropic* metric (inertia) and nontrivial topology (wraparound, cut locus). We should
adopt their bounded factorisation, speed loss, and progressive schedule (conditioning wins that
apply regardless), but the representation that lets us encode the *metric* and *topology*
directly — anisotropic splats with an intrinsic log-map and a pluggable `metric_inv` — is the
part that is ours, and the part that matters once the problem stops being isotropic.

### Head-to-head: NTFields formulation vs our baseline (same scene, seed 1, 3 obstacles)
Implemented `solve_ntfields` (`torus.py`): `T = base/τ`, `τ = τ_min + (1−τ_min)·σ(splat+bias) ∈
[τ_min, 1]`; symmetric speed-match loss `q + 1/q − 2` (`q = ‖∇T‖_{g⁻¹}/s`); progressive obstacle
contrast `λ: 0.15→1`; **no causal/DP/visibility crutches**, only RAD sampling. Both runs 4000
steps, 384 splats, identical obstacles.

| Metric | Baseline `solve` (causal+vis+RAD) | NTFields `solve_ntfields` (RAD only) |
|---|---|---|
| Final RMS | 0.563 | **0.345** (−39%) |
| max\|err\| | 3.87 | **2.27** |
| loss log10 | −1.6 | **−2.1** |
| RMS @ 1k/2k/3k | 0.55 / 0.53 / — | 0.357 / 0.333 / 0.338 |

**The error changes character, not just magnitude.** Baseline: the whole torus is a uniform blue
**global under-level offset** (pred < GT by ≥0.2 nearly everywhere) — level under-determination
pinned low. NTFields: the global offset is gone and the near-source is **clean white** (the
`base/τ` factorisation forces `T(source)=0` and `τ≈1` in the free near-field — this is the
"start should be pure white" property we kept failing at). Residual error is now *localised* to
two tunable features: (a) an over-shoot behind an obstacle where `τ` hits its floor `τ_min=0.25`
(too much headroom → `T` up to `4·base`, `maxT≈9.9` vs GT `5.6`), and (b) under-leveling in the
far field ~180° from the source — the **cut locus**, genuinely the hardest region.

**Stability lesson (v1→v2).** The first NTFields attempt (unbounded `T=base/τ`, no interior gate)
trained beautifully to loss −2.4 during low contrast, then **blew up at `λ→1`** (RMS 26.7): with
nothing bounding `τ`, the `1/q` penalty in deep obstacle interiors (`s=10`, field can't comply,
`q→0`) drove `τ→0` and `T→∞`. Fix (faithful to NTFields' speed-floor + not-chasing-interiors):
**floor `τ` at `τ_min`** (bounds `T ≤ base/τ_min`) and **gate the loss to the exterior**
(`sigmoid(sdf/w)`, ~0 inside obstacles). Both are physically justified — the field inside a
blocked cell is irrelevant to planning.

**Verdict.** The NTFields *formulation alone* beats our crutch-laden baseline by 39% RMS and,
more importantly, converts a global level failure into two localised, addressable ones. Still
above the 0.2 target; the open levers are a tighter `τ_min` (curb the shadow over-shoot),
stronger cut-locus sampling / more anneal steps for the far field, and optionally re-adding causal
weighting *on top of* the sound formulation. This is the honest confirmation that our earlier
struggle was formulation-limited, not torus-compactness-limited.

### Lever ablation + representation test: the error is level, not representation
Two systematic sweeps on the same scene (all NTFields-style, 4000 steps). Cumulative lever ladder:

| Rung | +lever | RMS | max\|err\| | reading |
|---|---|---|---|---|
| A | v2 (`τ_min=0.25`) | 0.345 | 2.27 | base |
| B | tighter `τ_min=0.40` | 0.377 | 2.13 | **hurts** — wider mismatch with the `s=10` wall |
| C | antipodal cut-locus sampling | 0.377 | 2.39 | **no help** — error is not at the antipode |
| D | causal weighting | 0.345 | **2.04** | tames the worst point (shadow) |

Representation arms (test "Gaussians corrupted by the slowness wall") on the `ref` recipe
(causal + `τ_min=0.25` + **no** antipodal boost + sharp ramp; `ref` itself = **0.307**, the best
label-free result — dropping the default `cutlocus_boost=3.0` was worth ~11%):

| Arm | change | RMS | reading |
|---|---|---|---|
| ref | — | **0.307** | best |
| E | gentle ramp 0.30 **+** rim splats | 0.746 | much worse |
| G | gentle ramp 0.30 only | 0.672 | **the culprit** — widening spreads the obstacle |
| H | rim splats only | 0.339 | ~neutral |

**Diagnostic (`diagnose.py`, near-obstacle zoom).** The violent `‖∇T‖/s` oscillation is *inside*
the gated obstacle (unconstrained, harmless); the **exterior ratio ≈ 1** (rim-annulus std ≈ 0.20)
— the *shape* is locally correct. Yet the full error map shows the **entire far half
under-predicted** (the region behind the central obstacle cluster relative to the source). So the
splats are **not** ringing/capacity-limited near obstacles; they satisfy `|∇T|=s` locally but
accumulate a **detour deficit**: the straight-line `base` prior points *through* the obstacles and
smooth `τ` under-inflates the go-around cost, leaving the shadow **under-leveled**.

**Conclusion.** The pointwise Eikonal residual constrains *slope*, not *level*; behind obstacles the
level is genuinely under-determined and neither smoothing the obstacle (G, hurts) nor adding rim
capacity (H, neutral) touches it — because the error is not at the rim. This is exactly the mode
**sparse anchors are designed to pin** (the integration constant in the shadow), and it argues for
judging the field by **path suboptimality vs RRT\***, not field RMS — the gradient *direction* in
the shadow can be right while the level is off by a constant. Best label-free result: **0.307**.

### Sparse RRT* shadow anchors break the 0.2 target — and prove the diagnosis
Added `rrt_star_anchors_shadow`: RRT* tree → keep cleared nodes → bias selection toward nodes whose
straight source→node ray is **occluded** (in a routing shadow), weighted `1 + shadow_pref·occluded`.
Wired a weak-supervision term `weak_weight·mean((T−cost)²)` into `solve_ntfields` on top of `ref`.
30 anchors (RRT*, **not** FMM; mesh-free, dimension-scalable):

| Run | anchors | weight | RMS | max\|err\| |
|---|---|---|---|---|
| ref | none | — | 0.307 | 1.64 |
| **W1** | 30 shadow (53%) | 0.5 | **0.177** ✅ | 1.00 |
| W2 | 30 shadow | 1.0 | 0.181 ✅ | 1.10 |
| U (control) | 30 **uniform** | 1.0 | 0.229 | 1.47 |

**Two things proven at once.** (1) 30 sparse anchors take us from 0.307 to **0.177 — below the 0.2
target**, matching-ish the Euclidean plane. (2) The **uniform-anchor control U (0.229) is markedly
worse than shadow-targeted W1/W2 (~0.18)** with the *same* 30 anchors — so it is specifically
*placing the anchors in the shadow* that helps, not merely adding supervision. That is direct
confirmation of the level-under-determination diagnosis: the residual supplies the *shape*, and a
handful of anchors supply the *level* exactly where the PDE leaves it free. W1's error map shows the
far-half blue filling in; the residual under-prediction that remains is at the extreme far edges /
wrap seams (near the antipode), not the immediate obstacle shadow. Lighter weight (0.5) beat firm
(1.0) — the anchors only need to pin the constant, not dominate the loss.

**Winning recipe:** NTFields factorisation `T=base/τ` + symmetric speed loss + progressive obstacles
+ causal weighting + **no** antipodal oversampling + **30 shadow-targeted RRT* anchors** (weak_weight
0.5). Obstacles stay modelled as a slowness field throughout — the anchors fix the *level* symptom
that the slowness formulation leaves under-determined, without changing the obstacle model.

### Pushing below 0.1: the straight-line base is the real ceiling; a roadmap base breaks it
Piling capacity + more anchors on the anchor recipe **plateaued** (`PUSH_A` 0.184, `PUSH_B` 0.174) and
`max|err|` *rose* to ~1.5 — adding rim/far anchors created a red over-prediction **wedge** while the far
half still under-predicted. Diagnosis: the straight-line `base` prior points *through* obstacles, so `τ`
must do heavy, conflicting **non-local** work in the shadow, and pinning it with more (biased) anchors
just fights the PDE. This burden only worsens in high-D — so *more anchors is the dimension-fragile road*.

**Fix — replace the prior with an accurate, mesh-free, obstacle-aware base.**
`roadmap_base(θ) = soft-min_i (cost_i + slowness-weighted ‖wrap(θ−node_i)‖)` over an RRT* tree, where the
last hop is weighted by the **mean slowness along it** so hops through obstacles are penalised. Base
quality (zero splat, vs FMM):

| last hop | base RMS | base max (GT 5.53) |
|---|---|---|
| straight (corner-cutting) | 0.517 | 4.42 |
| **slowness-weighted** | **0.065** | **5.54** |

The straight hop underestimates the shadow (cuts corners) — the same failure as the straight-line base.
The slowness-weighted hop makes the base **accurate zero-shot (0.065)** and it is a smooth, differentiable
(soft-min), mesh-free field — it *scales past the grid dimensions* (RRT*, no FMM).

**Two gotchas, both resolved:** (1) pure Eikonal training on top *drifts the accurate base away*
(`maxT→10`) — the same level under-determination — so regularise the splat toward the base
(`base_reg·mean(g²)`, `T=base·exp(g)`) and it only *smooths kinks* instead of re-levelling. (2) the
**soft-min `γ` must be small** (`γ·log N` is the underestimate): `γ=0.05→0.15 RMS`, `γ=0.01→0.099 RMS`,
`max|err| 0.34` (vs 1.0–2.3 for every earlier approach). **Below the 0.1 target, no FMM, no sparse
anchors** — the RRT* roadmap is a *dense mesh-free target* (self-supervised from the known slowness), and
the splat is the compact final representation fit to it + light Eikonal.

**Full sweep result (γ=0.01, 384 splats, 4000 steps):** `base_reg=30` → **RMS 0.070, max|err| 0.27**;
`base_reg=10` → RMS 0.072, max|err| 0.35. The error map is clean across the whole torus (prediction
tracks GT behind the obstacles; no far-half blue, no wedge); the residual is a faint ~0.05 under-haze
plus tiny rim speckles (the sharp `s=10` wall, where max|err| sits). The field is smooth (planning-ready).

**Progression on the same scene:** baseline `solve` 0.563 → NTFields formulation 0.345 → `ref`
(+causal, −antipode) 0.307 → +30 shadow anchors 0.177 → **roadmap base 0.070** (8× the baseline). And
`max|err|` fell from ~3.9 to **0.27**. The decisive move was the *obstacle-aware base*, not more supervision.

**Framing / open decision.** This shifts work from "splat+PDE discovers the shadow" to "splat is the
compact representation of a solution the mesh-free roadmap scaffolds" — accurate and dimension-robust,
but leaning on the roadmap for the level (which the pure PDE genuinely under-determines). Roadmap is
computed from the *known* slowness (self-supervised, RRT* not FMM), so it is legitimate and scalable;
the tradeoff is how much of the "solving" the splat vs the roadmap does. Next: high-D validation battery
(`validation_highd.md`), then the anisotropic metric (`metric_inv → M(θ)⁻¹`) for the arm.

## Fair baseline suite (same scene/seed/budget; only the method varies)
`run_baselines.sh` — canonical scene (seed 1, 3 obstacles), 384 splats, 4000 steps, 2048 collocation,
identical sampling; `cutlocus_boost=0` for all (antipodal sampling was shown to hurt).

| # | Baseline | Supervision | RMS | max\|err\| |
|---|---|---|---|---|
| B1 | Vanilla Eikonal PINN (`base·(1+g)`, residual²) | none | 0.534 | 3.78 |
| B2 | NTFields (`base/τ` + symmetric speed-match loss) | none | 0.331 | 2.02 |
| B3 | P-NTFields (B2 + progressive anneal + causal) | none | 0.307 | 1.64 |
| B4 | 300 RRT* samples as ANCHORS (trust values) | sparse (300) | 0.393 | 2.03 |
| B5 | 300 RRT* samples as BASE + physics refine | sparse (300) | **0.109** | 0.64 |
| B6 | Supervised FMM-fit (oracle) | full GT | **0.0029** | 0.026 |

**B4 vs B5 — the headline (matched ~300-sample budget).** Same RRT* samples, two injections: as trusted
*anchors* (B4) vs a rough *base the Eikonal refines* (B5). **B5 beats B4 by 3.6×** (0.109 vs 0.393),
and — striking — **B4 is *worse than B3* (0.307, no supervision at all).** RRT* costs are upper bounds
(suboptimal → too high); pinning 300 of them bakes the planner's suboptimality into the field, while
using them as a soft prior lets the Eikonal *shave them back toward optimal*. This is the direct,
same-budget measurement of the thesis: **imitating the planner's values is worse than ignoring them;
physics-refining them is what wins** (anti-MPNet, and distinct from pure-physics P-NTFields). Note the
*old* (unfair) suite compared 30-anchor B4 (0.169) to a ~300-node dense-base B5 (0.070) — retired.

### Thinning: how error grows as samples get sparser (`thinning_experiment.py`, `figures/thinning.png`)
B5 (base + physics-led refine) over node budgets 81→601, base-alone vs PDE-refined RMS:

| nodes | base RMS | refined | physics gap |
|---|---|---|---|
| 81 | 0.451 | 0.267 | +0.185 (−41%) |
| 121 | 0.411 | 0.247 | +0.164 |
| 201 | 0.238 | 0.135 | +0.103 |
| 351 | 0.160 | 0.105 | +0.055 |
| 601 | 0.077 | 0.074 | +0.003 |

Fitted: **base `~nodes^-0.89`, refined `~nodes^-0.67`.** Two encouraging signals: (1) the Eikonal
*flattens* the sparsity-degradation slope (−0.89→−0.67, i.e. sub-linear not linear); (2) the physics
contributes **most when samples are sparsest** (−41% at 81 nodes vs −4% at 601) — ideal for high-D where
samples are always sparse. **Honest caveat:** −0.67 is sub-linear but *not* flat/logarithmic — error
still grows as a power law (7× fewer nodes → ~3.6× worse refined RMS). So *graceful degradation, not
immunity*. **This is a node-count curve in 2D; the true scaling test is error vs sample *dispersion* as
dimension rises (fixed budget, 2→3→4-D)** — that, not raw count, is what transfers across dimensions.

**The oracle B6 = 0.003 is the key control: the splat *representation* is not the bottleneck.** A splat
of this budget fits the true field to 0.003 RMS, so the entire B1→B6 gap is **solver + supervision**,
never representation capacity. Clean physics-informed progression: naive 0.53 → factorisation+speed-loss
0.33 → +progressive+causal 0.31 (this is the fair planner-free NTFields/P-NTFields analogue) → sparse
anchors 0.17 → roadmap base 0.07 → representation ceiling 0.003.

**Honest read of B5 vs the literature.** B5 uses a mesh-free RRT* base — *not* imitation (MPNet) and
*not* pure-physics (P-NTFields), but a **noisy-prior + physics-refinement** (multi-fidelity) hybrid. Its
scalability hinges on one open question, tested in `refine_experiment.py`: can the Eikonal *refine a
sparse (few-node, high-D-feasible) base* down, or does it need a dense base (impossible in high-D)? Base
RMS vs RRT* tree size on this scene: 101 nodes→0.43, 251→0.18, 501→0.12, 1001→0.09. **A *dense* base
(~300 nodes) in B5 is effectively cheating and will not scale — discard it as a headline.** The legitimate
method is *sparse* RRT* (anchors or a sparse base) + Eikonal refinement.

### Can the Eikonal refine a suboptimal/sparse RRT* solution? Yes — and it's the physics doing it
`refine_experiment.py`: sparse trees, two base-weights (`base_reg=3` lets the PDE lead; `30` mostly fits
the base). Value RMS (vs FMM) and Eikonal residual (consistency), base → refined:

| tree | base RMS | refined (`base_reg=3`) | refined (`base_reg=30`) | residual (reg 3) |
|---|---|---|---|---|
| 151 nodes | 0.331 | **0.212** (−36%) | 0.279 (−16%) | 0.057 → 0.020 |
| 251 nodes | 0.183 | **0.120** (−34%) | 0.154 (−16%) | 0.035 → 0.019 |
| 501 nodes | 0.114 | **0.079** (−31%) | 0.097 (−15%) | 0.019 → 0.017 |

**(1)** The Eikonal reduces the error in *every* case. **(2)** It is the *physics*, not base-fitting: the
physics-led weight (`base_reg=3`) refines ~2× more than the fit-base weight (`30`) — if the base carried
it, the two columns would match; they don't. **(3)** Physical consistency (residual) drops 30–65%, so the
field becomes much more Eikonal-consistent (better planning gradients) even where value-RMS moves modestly
— i.e. RMS understates the PDE's contribution. **Caveat:** refinement is *partial* — final error tracks
base density (very sparse 151→0.21; you can't refine garbage to perfection). But 251 nodes→0.120 already
beats B4's 30-anchor 0.169, all mesh-free/sparse. **High-D outlook: cautiously positive** — the physics
reliably corrects a rough planner, but there is a floor set by how sparse the planner can be; that
floor-vs-dimension is the next thing to characterise. This vindicates the *noisy-prior + physics-refine*
thesis and, crucially, shows the Eikonal — not the base — is the corrector.
