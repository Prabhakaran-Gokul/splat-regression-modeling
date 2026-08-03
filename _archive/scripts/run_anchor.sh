#!/bin/zsh
# Sparse RRT* shadow anchors on the `ref` recipe (0.307): do they pin the shadow level?
# W1/W2 = shadow-targeted (2 weights); U = uniform-anchor control (isolates the targeting).
cd /Users/baner121/splat-regression-modeling
source .venv/bin/activate 2>/dev/null || true
mkdir -p figures/anchor/W1 figures/anchor/W2 figures/anchor/U

REF="--method ntfields --causal --causal_strength 5.0 --tau_min 0.25 --cutlocus_boost 0.0 --slow_width 0.15 --slowness_max 10 --checkpoint_every 2000"

echo "===== W1: shadow anchors, weak_weight 0.5, num_weak 30 ====="
python torus.py ${=REF} --weak_weight 0.5 --num_weak 30 --shadow_pref 8 --out_dir figures/anchor/W1 > figures/anchor/W1/run.log 2>&1
grep -aE "anchors|^RMS=" figures/anchor/W1/run.log

echo "===== W2: shadow anchors, weak_weight 1.0, num_weak 30 ====="
python torus.py ${=REF} --weak_weight 1.0 --num_weak 30 --shadow_pref 8 --out_dir figures/anchor/W2 > figures/anchor/W2/run.log 2>&1
grep -aE "anchors|^RMS=" figures/anchor/W2/run.log

echo "===== U: uniform anchors (control), weak_weight 1.0, num_weak 30 ====="
python torus.py ${=REF} --weak_weight 1.0 --num_weak 30 --shadow_pref 0 --out_dir figures/anchor/U > figures/anchor/U/run.log 2>&1
grep -aE "anchors|^RMS=" figures/anchor/U/run.log

echo "===== ANCHOR SUMMARY (vs ref 0.307) ====="
for r in W1 W2 U; do printf "%s: " $r; grep -aE "^RMS=" figures/anchor/$r/run.log 2>/dev/null || echo "(no result)"; done
echo "ANCHOR DONE"
