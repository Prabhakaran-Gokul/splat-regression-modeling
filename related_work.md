# Related Work & Positioning (deep-research synthesis, all claims 3-0 verified)

Motivation/related-work for **"suboptimal RRT\* prior + Eikonal PDE refinement, with a splat
(Gaussian-mixture) value function on a manifold."** Every claim below was adversarially verified
(3 independent skeptics, majority-refute to kill; 25/25 confirmed).

## The one that matters most — H-NTFields nearly scoops the paradigm
**H-NTFields — *Hierarchical* Neural Time Fields (Ni, Liu & Qureshi, 2026, arXiv 2604.13204)** is the
single closest prior work: a weakly-supervised framework combining **a deliberately sparse
(sphere-packing) roadmap as weak supervision — used *not* as the final planner but for upper/lower
travel-time bounds — with physics-informed Eikonal PDE regularization** to learn a neural time field.
That is essentially our paradigm: *cheap suboptimal-planner prior + Eikonal refinement.* From the same
lab as NTFields.

**Correction (2026-08-09, from the paper itself):** H-NTFields extends **TD-NTFields**, *not*
P-NTFields. Its whole PDE half — `L_E`, `L_TD`, `L_N`, the causality weight `L_C`, and the published
weights (λ_E=1e-2, λ_TD=1e-3, λ_N=1e-3, λ_C=0.5, Δt=0.02) — is inherited from that paper; H-NTFields
adds only the roadmap-bound hinge `L_R` and the sphere-packing/perturbation sampling. Its final
objective is `L = (λ_E L_E + λ_TD L_TD + λ_N L_N + λ_R L_R)·L_C` (Eq. 9). Note also that its `L_E` is
the **squared one-directional** `(√(S⋆/S) − 1)²`, a *different* Eikonal loss from NTFields' isotropic
Eq. 4 — the two papers do not share a loss, and conflating them is a fidelity error.

It also **corroborates our core empirical finding**: pure PDE-only Eikonal solvers "collapse into
local minima or underestimate long-range travel times without additional structural guidance," and
"simply increasing the number of samples cannot resolve local minima" — exactly the level
under-determination we measured (B1–B3 plateau ~0.31; dense samples don't fix it). So our prior is
demonstrably *not redundant* with the PDE — but the "roadmap-prior + PDE" idea is no longer novel.

**What remains genuinely ours (per the synthesis):** (1) an **RRT\* point-estimate noisy prior**, not
roadmap upper/lower *bounds*; (2) a **splat / Gaussian-mixture value representation** (vs their MLP);
(3) an **anisotropic-metric configuration-space torus / manifold** formulation. Reposition the paper
around these three, not around "planner prior + PDE."

### TD-NTFields — the missing link in the lineage
- **TD-NTFields** (Ni, Pan & Qureshi, ICLR 2025, arXiv 2505.05691, code `ruiqini/ntrl-demo`): recasts
  travel time as an **optimal value function** and adds Bellman consistency at a *finite* scale
  (`L_TD`), obstacle-normal alignment (`L_N`), and a causality curriculum (`L_C = exp(−λ_C T)`). It also
  **drops the `‖q_s−q_g‖/τ` factorization**, predicting `T` directly as a learned quasimetric
  `D(f(q_s), f(q_g))` — so the NTFields factored field is *not* part of this branch of the lineage.

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
  constant-speed field toward sharp obstacles. **Correction (2026-08-09):** this is linear in *speed*;
  the old `torus.py` annealed the *slowness* (`s_λ = 1 + λ(s−1)`), a different path through field
  space — so the earlier claim that "our `λ:0→1` anneal is this" was wrong.
  `srms/methods/strategies/pntfields.py` now anneals in speed space and defaults `viscosity_eps=0.01` on.

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


## Verified against the released implementations (2026-08-09)

Read directly from the authors' code, not inferred from the papers:

- **`ruiqini/NTFields`, `models/model_3d.py`.** The loss is literally
  `|1−√(S/S⋆)| + |1−√(S⋆/S)|` per endpoint, meaned over the batch — our `ntfields.isotropic_loss`
  matches exactly. Optimizer is **AdamW, weight_decay=0.1**. τ is `sigmoid(0.1·y)` — a *scaled* logit
  with **no bias**, so τ starts at 0.5, whereas our `tau_bias=4.0` starts it near 1. NTFields also uses
  **distance-weighted sampling** (`WeightedRandomSampler`, weights ∝ `max(d)−d` clipped to
  [0.2, 0.95]), favouring nearby start/goal pairs — we sample uniformly. And the epoch-rollback guard
  (`diff_ratio > 1.2` ⇒ reload a random one of the last 5 states) is already in **NTFields**, not just
  P-NTFields.
- **`ruiqini/ntrl-demo`, `models/metric/model_function_metric.py`.** `L_E` is
  `1e-2·(√(S⋆·‖∇T‖) − 1)²` — matches ours. Three `L_TD` details are *not* in the paper's prose: the
  step is `Δt·S⋆·∇T` rather than the unit-normalized `u⋆Δt`; the **entire target is under `no_grad`**,
  not just the policy direction; and the term is **masked off where `T <` the step's time cost**. All
  three are now reproduced. `L_N` weights by `(1.001 − S⋆)` and takes its normal from precomputed
  dataset normals rather than `∇S⋆/‖∇S⋆‖`. `L_C` is **not detached** in their code.
- **Our one forced departure.** Because we keep `T = base/τ` (their quasimetric head is inseparable
  from a two-point field), an attached `L_C` lets the optimizer inflate `T` via `τ→0` and zero out its
  own loss — a degenerate direction their bounded head does not have. Measured on the 2-D torus at 800
  steps: attached **RMS 4.34 / max|err| 170** vs detached **0.63 / 6.4**. We therefore detach by
  default (`--no-hnt-detach-causal` restores their behaviour). This is a *consequence* of the field
  deviation, not an independent one.
