# NTFields baselines: status and fidelity record

Status: **done** for the three published baselines. This note was originally a spec for the
deferred migration; it is now the record of what was built and, more importantly, of **exactly
where these implementations depart from the papers**. Read it before quoting any number as
"NTFields" / "P-NTFields" / "H-NTFields".

Its two original open questions are resolved:

1. *Backend or strategy?* **Strategy.** `field_ntfields` is a field *factorization* that consumes a
   raw approximator's output, exactly as `weak_supervision.py` consumes `backend.eval_raw`. It is
   generic over `srm`/`mlp`. The empty `backends/ntfields.py` stub was never needed.
2. *Port the anneal, or keep it simple?* **Port it, in the right place.** The progressive anneal is
   P-NTFields' contribution, not a training trick — so it lives in `pntfields.py`, and `ntfields.py`
   stays literal. (It is also annealed in *speed* space per the paper, not slowness space as the old
   `torus.py` did.)

## The three strategies

| Method | Module | Paper | Distinguishing mechanism |
|---|---|---|---|
| NTFields | `strategies/ntfields.py` | ICLR 2023, [2210.00120](https://arxiv.org/abs/2210.00120) | `T = ‖q_s−q_g‖/τ`; isotropic speed loss `\|1−√(S⋆/S)\| + \|1−√(S/S⋆)\|` (Eq. 4) |
| P-NTFields | `strategies/pntfields.py` | RSS 2023, [2306.00616](https://arxiv.org/abs/2306.00616) | + viscosity `1/S = ‖∇T‖ + ε·Δτ` (Eq. 6/7); + progressive speed `S⋆_α = (1−α)+α·S⋆` (Eq. 8) |
| H-NTFields | `strategies/hntfields.py` | 2026, [2604.13204](https://arxiv.org/abs/2604.13204) | `(λ_E L_E + λ_TD L_TD + λ_N L_N + λ_R L_R)·L_C` (Eq. 9); sparse sphere-packing roadmap gives travel-time **bounds** |

H-NTFields is *Hierarchical* NTFields and extends **TD-NTFields**
(ICLR 2025, [2505.05691](https://arxiv.org/abs/2505.05691)) — not P-NTFields. Its `L_E`, `L_TD`,
`L_N`, `L_C` and the default weights (λ_E=1e-2, λ_TD=1e-3, λ_N=1e-3, λ_C=0.5, Δt=0.02) come from
that paper. Note `L_E` there is the **squared one-directional** `(√(S⋆/S) − 1)²`, which is *not*
NTFields' isotropic Eq. 4 — the two papers really do use different Eikonal losses.

All three share the `T = base/τ` field and the `srm`/`mlp` backends, so a head-to-head isolates the
objective and the representation rather than the architecture.

## Deviations from the papers

Applies to all three unless noted. **These are why these runs reproduce the papers' *formulations*,
not their *published metrics*.** Getting the latter means running the authors' released code on the
authors' benchmarks; that is a separate track.

| # | Deviation | Why | Affects |
|---|---|---|---|
| 1 | **Fixed source, all goals.** We learn `T(θ)` — source fixed, goal ranging over the whole manifold. Papers learn a two-point `T(q_s,q_g)` and evaluate at *both* endpoints, so only the `q_g` half exists here. | A deliberate problem setting, not a truncation: the field already spans every goal, which is what makes it a value function. A two-point field would additionally rewrite every backend and environment (for the SRM, splats on a product manifold). A different *source* means a different field — the premise of the editability study. | all |
| 2 | **Scene speed model.** Papers use the clipped obstacle-distance ramp `S⋆ = clip(d/d_max, d_min/d_max, 1)`; this repo uses its environments' smooth `1/env.slowness`. | Property of the *scene*, identical across every strategy and backend, so it cannot bias a comparison. | all |
| 3 | **Per-step resampled collocation** instead of a fixed dataset of sampled pairs. | House style of `eikonal.py`/`weak_supervision.py`; also removes the need for epoch bookkeeping. | all |
| 4 | **Backend.** ResNet + workspace-CNN (NTFields/P-NTFields) and PirateNets quasimetric (TD/H-NTFields) replaced by `srm`/`mlp`. | This substitution *is* the experiment. | all |
| 5 | **`cfg.causal` ignored.** | NTFields/P-NTFields have no causal term; H-NTFields has its own `L_C`. The global default is `causal=True`, so honouring it would have silently un-faithfulled every run. Each strategy prints a line when the flag is set. | all |
| 6 | **No η=1.5 loss-threshold rollback** (P-NTFields Algorithm 1 lines 13–16). | Rolls back an epoch over a *fixed* dataset; with per-step resampling there is no epoch to roll back to. | pntfields |
| 7 | **α schedule reparameterized** as fractions of `cfg.steps` (hold, then linear ramp) rather than the paper's literal epoch counts. | Their numbers presuppose their epoch budget; defaults (0.5 / 25% / 1.05) match the paper's shape. | pntfields |
| 8 | **Field is `base/τ`, not TD-NTFields' learned quasimetric** `D(f(q_s),f(q_g))`. | The quasimetric is inseparable from a two-point formulation; holding the field fixed also keeps the comparison on the loss. | hntfields |
| 9 | **Bound slack = slowness-integrated perturbation-hop cost**, not the raw perturbation radius. | The paper's ±radius is valid because its free-space speed is 1 (distance = time). This repo's slowness varies 1→`slowness_max`, so raw radii would bound nothing. Reduces to the paper's form at slowness ≡ 1. | hntfields |
| 10 | **`u⋆` and `L_C` are stop-gradient.** | Papers do not specify; detaching a policy direction and a loss weight is the standard reading, and differentiating through `u⋆` would pull in second derivatives of `T`. | hntfields |
| 11 | **Inference** is dense-grid scoring against fast marching (and `−∇T` descent elsewhere), not the paper's sampling-based MPC. | Out of scope for a field-accuracy comparison. | hntfields |
| 12 | **λ_R defaulted to λ_E's scale.** | Unspecified in the H-NTFields paper. | hntfields |
| 13 | **Δt kept at the paper's literal 0.02**, which is domain-relative. | `u⋆Δt` is a C-space displacement. NTFields normalizes its C-space to unit span, so 0.02 ≈ 2% of the domain; this torus spans 2π, where 2% is ≈0.126. Pass `--hnt-dt 0.126` for the domain-matched setting. | hntfields |

## Reproduction commands

Both backends, same scene, same budget — the only difference is `--backend`:

```bash
for b in srm mlp; do
  python -m srms.run --method ntfields  --backend $b     # NTFields
  python -m srms.run --method pntfields --backend $b     # P-NTFields
  python -m srms.run --method hntfields --backend $b     # H-NTFields
done
```

Paper ablations (P-NTFields Fig. 5): `--viscosity-eps 0` ("w/o viscosity"), `--alpha-init 1.0`
("w/o scheduling"). Setting both recovers `ntfields` exactly. H-NTFields' own ablations (Table I)
are `--hnt-lambda-r 0` (PDE-only) and `--hnt-lambda-e 0 --hnt-lambda-td 0 --hnt-lambda-n 0`
(roadmap-only).

Every run logs to mlflow with a `comparison_group` tag that is identical for the `srm`/`mlp` pair of
a given method and scene, so the head-to-head is a group-by in the mlflow UI.

## Still not ported

`solve_supervised` (the oracle: regression-fit `field_ntfields` to `env.ground_truth` on free-space
grid points, no PDE loss — the representation ceiling, ~0.003 on the 2-D torus in the old
`torus.py`). Straightforward with `env.ground_truth`/`env.sdf`/`env.grid`; useful as the upper bound
in any SRM-vs-MLP table, since it separates "can this representation express the field" from "can
this solver find it".
