#!/bin/zsh
# NTFields ablation ladder: cumulatively add each of the 3 levers, measure RMS.
# Same scene (seed 1, 3 obstacles), 4000 steps, s_max=10 fixed for equal difficulty.
cd /Users/baner121/splat-regression-modeling
source .venv/bin/activate 2>/dev/null || true
mkdir -p figures/ablation/A figures/ablation/B figures/ablation/C figures/ablation/D

C2="--cutlocus_boost 6.0 --cutlocus_width 0.5 --anneal_frac 0.55"

echo "===== A: NTFields v2 recap (tau_min=0.25, no causal) ====="
python torus.py --method ntfields --no-causal --tau_min 0.25 --out_dir figures/ablation/A --checkpoint_every 2000 > figures/ablation/A/run.log 2>&1
grep -aE "^RMS=" figures/ablation/A/run.log

echo "===== B: + lever1 tighter tau_min=0.40 ====="
python torus.py --method ntfields --no-causal --tau_min 0.40 --out_dir figures/ablation/B --checkpoint_every 2000 > figures/ablation/B/run.log 2>&1
grep -aE "^RMS=" figures/ablation/B/run.log

echo "===== C: + lever2 cut-locus sampling + slower anneal ====="
python torus.py --method ntfields --no-causal --tau_min 0.40 ${=C2} --out_dir figures/ablation/C --checkpoint_every 2000 > figures/ablation/C/run.log 2>&1
grep -aE "^RMS=" figures/ablation/C/run.log

echo "===== D: + lever3 causal weighting on top ====="
python torus.py --method ntfields --causal --causal_strength 5.0 --tau_min 0.40 ${=C2} --out_dir figures/ablation/D --checkpoint_every 2000 > figures/ablation/D/run.log 2>&1
grep -aE "^RMS=" figures/ablation/D/run.log

echo "===== ABLATION SUMMARY (RMS) ====="
for r in A B C D; do printf "%s: " $r; grep -aE "^RMS=" figures/ablation/$r/run.log 2>/dev/null || echo "(no result)"; done
echo "ABLATION DONE"
