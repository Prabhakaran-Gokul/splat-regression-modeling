#!/bin/zsh
# Push below 0.1 from W1 (0.177): add anchor coverage (rim/far), splat capacity, and rim collocation.
cd /Users/baner121/splat-regression-modeling
source .venv/bin/activate 2>/dev/null || true
mkdir -p figures/push/A figures/push/B

REF="--method ntfields --causal --causal_strength 5.0 --tau_min 0.25 --cutlocus_boost 0.0 --slow_width 0.15 --slowness_max 10 --weak_weight 0.5 --shadow_pref 8 --checkpoint_every 2500"
CAP="--num_weak 40 --num_splats 640 --band_boost 4.0 --num_collocation 3072 --steps 5000"

echo "===== PUSH_A: +40 anchors +capacity(640) +rim collocation ====="
python torus.py ${=REF} ${=CAP} --out_dir figures/push/A > figures/push/A/run.log 2>&1
grep -aE "anchors|^RMS=" figures/push/A/run.log

echo "===== PUSH_B: PUSH_A + boundary-Dirichlet + far-field anchors ====="
python torus.py ${=REF} ${=CAP} --rim_pref 8.0 --far_pref 3.0 --out_dir figures/push/B > figures/push/B/run.log 2>&1
grep -aE "anchors|^RMS=" figures/push/B/run.log

echo "===== PUSH SUMMARY (vs W1 0.177) ====="
for r in A B; do printf "%s: " $r; grep -aE "^RMS=" figures/push/$r/run.log 2>/dev/null || echo "(no result)"; done
echo "PUSH DONE"
