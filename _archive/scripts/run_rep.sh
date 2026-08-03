#!/bin/zsh
# Obstacle-representation arms: is the near-obstacle error a splat-vs-slowness-wall problem?
# Causal ON throughout (our idea). Reference = sharp ramp; arms add gentle ramp + rim-seeded splats.
cd /Users/baner121/splat-regression-modeling
source .venv/bin/activate 2>/dev/null || true
mkdir -p figures/rep/ref figures/rep/E figures/rep/F

COM="--method ntfields --causal --causal_strength 5.0 --tau_min 0.25 --cutlocus_boost 0.0 --checkpoint_every 2000"
REP="--slow_width 0.30 --rim_seed_frac 0.4 --rim_scale 0.15"

echo "===== REF: sharp ramp (slow_width=0.15), no rim, s_max=10 ====="
python torus.py ${=COM} --slow_width 0.15 --slowness_max 10 --out_dir figures/rep/ref > figures/rep/ref/run.log 2>&1
grep -aE "^RMS=" figures/rep/ref/run.log

echo "===== E: gentle ramp + rim seeding, s_max=10 (same difficulty) ====="
python torus.py ${=COM} ${=REP} --slowness_max 10 --out_dir figures/rep/E > figures/rep/E/run.log 2>&1
grep -aE "^RMS=" figures/rep/E/run.log

echo "===== F: gentle ramp + rim seeding, s_max=5 (softer contrast) ====="
python torus.py ${=COM} ${=REP} --slowness_max 5 --out_dir figures/rep/F > figures/rep/F/run.log 2>&1
grep -aE "^RMS=" figures/rep/F/run.log

echo "===== DIAGNOSTICS (ringing at the rim) ====="
python diagnose.py --pkl figures/rep/ref/splat.pkl --out figures/rep/diag_ref.png 2>&1 | grep -aE "saved|rim"
python diagnose.py --pkl figures/rep/E/splat.pkl --out figures/rep/diag_E.png 2>&1 | grep -aE "saved|rim"

echo "===== REP SUMMARY (RMS) ====="
for r in ref E F; do printf "%s: " $r; grep -aE "^RMS=" figures/rep/$r/run.log 2>/dev/null || echo "(no result)"; done
echo "REP DONE"
