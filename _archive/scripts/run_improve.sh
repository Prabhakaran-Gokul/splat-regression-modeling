#!/bin/zsh
# Close the gap to the oracle (0.003) at a FIXED ~300 RRT* sample budget, by improving the SOLVER
# (never by adding samples). All arms use rrt_iters=350 (~300 samples), 384 splats, seed 1.
cd /Users/baner121/splat-regression-modeling
source .venv/bin/activate 2>/dev/null || true
mkdir -p figures/improve/{P1,P2,P3,P4}

FAIR="--num_splats 384 --num_collocation 2048 --resolution 120 --seed 1 --rrt_iters 350 --checkpoint_every 99999 --cutlocus_boost 0.0 --num_weak 300 --shadow_pref 5"

echo "===== P1: roadmap base + anchors (w=1.0), base_reg 3, 4000 steps ====="
python torus.py ${=FAIR} --method roadmap --roadmap_gamma 0.01 --base_reg 3 --weak_weight 1.0 --steps 4000 --out_dir figures/improve/P1 > figures/improve/P1/run.log 2>&1
grep -aE "^RMS=" figures/improve/P1/run.log

echo "===== P2: roadmap base + anchors (w=3.0, trust more), base_reg 3, 4000 steps ====="
python torus.py ${=FAIR} --method roadmap --roadmap_gamma 0.01 --base_reg 3 --weak_weight 3.0 --steps 4000 --out_dir figures/improve/P2 > figures/improve/P2/run.log 2>&1
grep -aE "^RMS=" figures/improve/P2/run.log

echo "===== P3: roadmap base + anchors (w=1.0), base_reg 1 (physics-led), 8000 steps (converge) ====="
python torus.py ${=FAIR} --method roadmap --roadmap_gamma 0.01 --base_reg 1 --weak_weight 1.0 --steps 8000 --out_dir figures/improve/P3 > figures/improve/P3/run.log 2>&1
grep -aE "^RMS=" figures/improve/P3/run.log

echo "===== P4: anchors-only (ntfields), trust more (w=3.0) vs B4's 0.5 ====="
python torus.py ${=FAIR} --method ntfields --causal --lambda_init 0.15 --anneal_frac 0.4 --weak_weight 3.0 --steps 4000 --out_dir figures/improve/P4 > figures/improve/P4/run.log 2>&1
grep -aE "^RMS=" figures/improve/P4/run.log

echo "===== IMPROVE SUMMARY (fixed ~300 samples; B4/B5 ~0.10-0.17; oracle 0.003) ====="
for p in P1 P2 P3 P4; do printf "%s: " $p; grep -aE "^RMS=" figures/improve/$p/run.log 2>/dev/null || echo "(no result)"; done
echo "IMPROVE DONE"
