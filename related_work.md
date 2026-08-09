# Related Work & Positioning (deep-research synthesis, all claims 3-0 verified)

Motivation/related-work for **"suboptimal RRT\* prior + Eikonal PDE refinement, with a splat
(Gaussian-mixture) value function on a manifold."** Every claim below was adversarially verified
(3 independent skeptics, majority-refute to kill; 25/25 confirmed).

## The one that matters most — H-NTFields nearly scoops the paradigm
**H-NTFields (Ni, Liu & Qureshi, 2026, arXiv 2604.13204)** is the single closest prior work: a
weakly-supervised framework combining **a deliberately sparse (sphere-packing) roadmap as weak
supervision — used *not* as the final planner but for upper/lower travel-time bounds — with
physics-informed Eikonal PDE regularization** to learn a neural time field. That is essentially our
paradigm: *cheap suboptimal-planner prior + Eikonal refinement.* From the same lab as NTFields.

It also **corroborates our core empirical finding**: pure PDE-only Eikonal solvers "collapse into
local minima or underestimate long-range travel times without additional structural guidance," and
"simply increasing the number of samples cannot resolve local minima" — exactly the level
under-determination we measured (B1–B3 plateau ~0.31; dense samples don't fix it). So our prior is
demonstrably *not redundant* with the PDE — but the "roadmap-prior + PDE" idea is no longer novel.

**What remains genuinely ours (per the synthesis):** (1) an **RRT\* point-estimate noisy prior**, not
roadmap upper/lower *bounds*; (2) a **splat / Gaussian-mixture value representation** (vs their MLP);
(3) an **anisotropic-metric configuration-space torus / manifold** formulation. Reposition the paper
around these three, not around "planner prior + PDE."

## 1. Physics-informed Eikonal neural planners (the pure-physics line, our B2/B3)
- **EikoNet** (Smith et al. 2020, arXiv 2004.00361): solves the Eikonal PDE self-supervised, "without
  ever needing solutions from a finite-difference algorithm"; origin of the factored travel time.
- **NTFields** (Ni & Qureshi, ICLR 2023, arXiv 2210.00120): factorization `T(q_s,q_g)=‖q_s−q_g‖/τ`,
  `τ∈[0,1]` (source singularity `T→0`, obstacle blow-up `τ→0⇒T→∞`, symmetry). Loss matches a
  predefined obstacle-distance **speed model** against the field's chain-ruled speed — *physics-
  informed, no demonstrations*. Explicitly argues prior methods "require expert data… which limits
  their application." (We reproduce this as B2; our `base/τ` field is this factorization.)
- **P-NTFields** (Ni & Qureshi, RSS 2023, arXiv 2306.00616): adds (a) a **viscosity term**
  `1/S = ‖∇T‖ + ε·ΔT` (vanishing-viscosity ⇒ smooth *unique* solution — the level/shock selector),
  and (b) a **progressive speed-annealing curriculum** `S*_α = (1−α)+α·S*`, `α: 0→1` from the trivial
  constant-speed field toward sharp obstacles. (We reproduce as B3; our `λ:0→1` anneal is this. Note:
  we have an `epsilon` viscosity knob but default 0 — **P-NTFields uses viscosity; worth switching on.**)

## 2. Imitation / demonstration planners (the anti-MPNet contrast, our B4)
- **MPNet** (Qureshi et al., ICRA 2019 / TRO, arXiv 1907.06013): MSE-regresses waypoints from RRT\*
  demonstrations — fast (<1 s) but **inherits demonstrator suboptimality by construction**; recovers
  optimality only by *hybridizing* with a classical planner. Directly supports our B4 result
  (trusting RRT\* values as anchors is worse than refining them).

## 3. Multi-fidelity / residual learning (our methodological lineage)
- **Multi-fidelity PINNs** (Meng & Karniadakis 2020, JCP 401:109020, arXiv 1903.00104): a cheap
  low-fidelity NN + high-fidelity correction branches; "high accuracy based on a very small set of
  high-fidelity data." Establishes the **coarse-prior + learned-correction** template — but for
  PDE/data fusion, *not* a planner prior. Frame our method as multi-fidelity with a *planner* as the
  low-fidelity source.

## 4. Sampling-complexity in high-D (justifies the "noisy prior" framing)
- To get an ε-near-optimal path in `d` dims needs **~O(1/εᵈ) samples** — constant volumetric density
  is exponential, so "RRT\* is no better than grid search in higher dimensions" (Qureshi et al.,
  arXiv 1907.06013). Finite-time RRT\* "prioritizes feasibility over optimality," so its trajectories
  are "a mix of optimal and sub-optimal" (SIL-RRT\*, Dang & Edelkamp 2024, arXiv 2411.17293). **This
  is the formal basis for treating the RRT\* prior as *noisy* and for our fixed-budget scaling test.**

## 5. Eikonal level / viscosity under-determination (our central difficulty)
- The pointwise Eikonal residual leaves the solution non-differentiable and its gradient non-unique
  near obstacles; **vanishing viscosity** (P-NTFields Eq. 6) selects the unique viscosity solution.
  This is the published statement of the "slope-not-level" issue we diagnosed empirically.

## Positioning & baselines to compare against

**The thesis is the *representation*, not the solver.** The contribution is the **Splat Regression
Model as an interpretable value function that learns on (sub-)Riemannian manifolds for robot
planning** — demonstrated first on the arm's C-space torus, then intended for other planning manifolds
(legged robots = constrained tori; non-holonomic motion = sub-Riemannian, e.g. SE(2)/Dubins). The
Eikonal, the RRT\* prior, and the PDE refinement are the *solving vehicle*, not the claim.

- **Do not** frame the paper around "planner prior + Eikonal refinement" — H-NTFields (2026) already
  has that mechanism; it is adjacent *solver* work, not a competitor for the representation thesis.
- **Claim (representation + manifold):** (i) a **splat/Gaussian-mixture value field** — interpretable
  (readable centres/covariances), mesh-free, closed-form ∇T, with an anisotropic basis that aligns
  with field structure; (ii) **intrinsic manifold learning** — splats placed via the log-map, metric
  entering only through `metric_inv`, so the same model covers the **anisotropic inertia metric** of
  the arm and generalises to constrained tori (legged) and sub-Riemannian manifolds (non-holonomic) —
  a manifold generality that a scalar-speed MLP cannot express. (Point-cost RRT\* prior + PDE-refine is
  a secondary solver contribution, distinct from H-NTFields' roadmap bound-boxes.)
- **The manifold claim is under-exercised while `metric_inv = I`.** A flat torus is only trivially
  Riemannian (periodicity). The substantive test is the **anisotropic arm metric** `M(θ)` — that is
  where the splat's anisotropic covariance + intrinsic placement genuinely matter, and where an
  MLP-on-scalar-speed structurally fails. Prioritise turning the metric on.
- **Baselines (same task, different representation):** NTFields, P-NTFields, H-NTFields, and MPNet —
  used as *representation* comparisons (MLP vs splat, and their solver vs ours), plus our own solver
  ablations (pure-PDE B3 vs anchors B4 vs roadmap-refine B5), plus an **MLP-with-same-solver** control
  to isolate the representation effect.
- **Adopt from them:** the P-NTFields **viscosity term** (we default `epsilon=0`) as the level/shock
  selector; benchmark on their environments/metrics (success rate, path cost, planning time).

_Sources: arXiv 2004.00361, 2210.00120, 2306.00616, 1907.06013, 1903.00104, 2411.17293, 2604.13204._
