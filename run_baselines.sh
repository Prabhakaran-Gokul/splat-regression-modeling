#!/bin/zsh
set -euo pipefail
# B1 vanilla PINN | B2 NTFields | B3 P-NTFields (no RRT*) || B4/B5 use the SAME ~300 RRT* samples,
# injected two ways: B4 as sparse anchors, B5 as a sparse base the PDE refines || B6 supervised.
cd -- "${0:A:h}"
source .venv/bin/activate 2>/dev/null || true
mkdir -p figures/baselines/{B1,B2,B3,B4,B5,B6}

# fairness-controlled shared budget (identical across every baseline)
FAIR="--num_splats 384 --steps 4000 --num_collocation 2048 --resolution 120 --seed 1 --rrt_iters 350 --checkpoint_every 5000"

echo "===== B1: Vanilla Eikonal PINN (base·(1+g), residual², no causal/vis) ====="
python torus.py ${=FAIR} --method eikonal --no-causal --out_dir figures/baselines/B1 > figures/baselines/B1/run.log 2>&1
grep -aE "^RMS=" figures/baselines/B1/run.log

echo "===== B2: NTFields (base/τ + speed-match loss; no progressive, no causal) ====="
python torus.py ${=FAIR} --method ntfields --no-causal --lambda_init 1.0 --out_dir figures/baselines/B2 > figures/baselines/B2/run.log 2>&1
grep -aE "^RMS=" figures/baselines/B2/run.log

echo "===== B3: P-NTFields (B2 + progressive anneal + causal) — planner-free physics ====="
python torus.py ${=FAIR} --method ntfields --causal --lambda_init 0.15 --anneal_frac 0.4 --out_dir figures/baselines/B3 > figures/baselines/B3/run.log 2>&1
grep -aE "^RMS=" figures/baselines/B3/run.log

echo "===== B4: ~300 RRT* samples as sparse ANCHORS + P-NTFields Eikonal ====="
python torus.py ${=FAIR} --method ntfields --causal --lambda_init 0.15 --anneal_frac 0.4 --weak_weight 0.5 --num_weak 300 --shadow_pref 5 --out_dir figures/baselines/B4 > figures/baselines/B4/run.log 2>&1
grep -aE "anchors|^RMS=" figures/baselines/B4/run.log

echo "===== B5: ~300 RRT* samples as sparse BASE + Eikonal refinement (physics-led, base_reg=3) ====="
python torus.py ${=FAIR} --method roadmap --roadmap_gamma 0.01 --base_reg 3 --roadmap_nodes 2000 --roadmap_hop 5 --out_dir figures/baselines/B5 > figures/baselines/B5/run.log 2>&1
grep -aE "^RMS=" figures/baselines/B5/run.log

echo "===== B6: Supervised FMM-fit (oracle — representation ceiling) ====="
python torus.py ${=FAIR} --method supervised --out_dir figures/baselines/B6 > figures/baselines/B6/run.log 2>&1
grep -aE "^RMS=" figures/baselines/B6/run.log

echo "===== BASELINE SUITE SUMMARY (scene: seed 1, 3 obstacles, 384 splats, 4000 steps) ====="
for b in B1 B2 B3 B4 B5 B6; do printf "%s: " $b; grep -aE "^RMS=" figures/baselines/$b/run.log 2>/dev/null || echo "(no result)"; done
echo "BASELINES DONE"
