#!/bin/zsh
# Roadmap-base method: splat fit to an accurate mesh-free RRT* base + light Eikonal. No FMM, no anchors.
cd /Users/baner121/splat-regression-modeling
source .venv/bin/activate 2>/dev/null || true
mkdir -p figures/roadmap/R1 figures/roadmap/R2

COM="--method roadmap --roadmap_gamma 0.01 --roadmap_hop 5 --num_splats 384 --roadmap_nodes 350 --num_collocation 1500 --resolution 120 --steps 4000 --rrt_iters 1500 --checkpoint_every 2000"

echo "===== R1: base_reg 30 (regression-dominant) ====="
python torus.py ${=COM} --base_reg 30 --out_dir figures/roadmap/R1 > figures/roadmap/R1/run.log 2>&1
grep -aE "^RMS=" figures/roadmap/R1/run.log

echo "===== R2: base_reg 10 (more Eikonal smoothing) ====="
python torus.py ${=COM} --base_reg 10 --out_dir figures/roadmap/R2 > figures/roadmap/R2/run.log 2>&1
grep -aE "^RMS=" figures/roadmap/R2/run.log

echo "===== ROADMAP SUMMARY (vs W1-anchors 0.177; target <0.1) ====="
for r in R1 R2; do printf "%s: " $r; grep -aE "^RMS=" figures/roadmap/$r/run.log 2>/dev/null || echo "(no result)"; done
echo "ROADMAP DONE"
