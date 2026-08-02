#!/bin/zsh
# Disentangle E's two levers on top of the `ref` recipe (RMS 0.307): which half of E hurt?
# ref = causal + tau_min 0.25 + no antipodal boost + sharp ramp + s_max 10.
cd /Users/baner121/splat-regression-modeling
source .venv/bin/activate 2>/dev/null || true
mkdir -p figures/dis/G figures/dis/H

REF="--method ntfields --causal --causal_strength 5.0 --tau_min 0.25 --cutlocus_boost 0.0 --slowness_max 10 --checkpoint_every 2000"

echo "===== G: ref + gentle ramp ONLY (slow_width 0.30, no rim) ====="
python torus.py ${=REF} --slow_width 0.30 --out_dir figures/dis/G > figures/dis/G/run.log 2>&1
grep -aE "^RMS=" figures/dis/G/run.log

echo "===== H: ref + rim seeding ONLY (sharp ramp 0.15, rim 0.4) ====="
python torus.py ${=REF} --slow_width 0.15 --rim_seed_frac 0.4 --rim_scale 0.15 --out_dir figures/dis/H > figures/dis/H/run.log 2>&1
grep -aE "^RMS=" figures/dis/H/run.log

echo "===== DISENTANGLE SUMMARY (vs ref 0.307) ====="
for r in G H; do printf "%s: " $r; grep -aE "^RMS=" figures/dis/$r/run.log 2>/dev/null || echo "(no result)"; done
python diagnose.py --pkl figures/dis/G/splat.pkl --out figures/dis/diag_G.png 2>&1 | grep -aE "saved|rim"
python diagnose.py --pkl figures/dis/H/splat.pkl --out figures/dis/diag_H.png 2>&1 | grep -aE "saved|rim"
echo "DIS DONE"
