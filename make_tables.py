"""Emit the paper's 2-D result tables as LaTeX, recomputed from the saved models.

Two tables, one per supervision arm, over the three 2-D Riemannian manifolds. Numbers are recomputed
here from each run's ``splat.pkl`` rather than scraped from logs, so a table can never disagree with
the model it claims to describe.

Fairness is asserted, not assumed: ``check_matched`` compares the saved ``cfg`` of every run and
fails loudly if anything other than ``environment``/``method``/``out_dir`` differs between them. A
table is only emitted if that check passes, so the two arms cannot silently drift apart.

Ground truth is the dense fast marcher at the run's own resolution, computed here — never during
training (see ``srms/environments/test_selfsupervised.py``).

Run: ``python make_tables.py > results/tables.tex``
"""

from __future__ import annotations

import dataclasses
import pickle
import sys

import jax
import jax.numpy as jnp
import numpy as np

from srms.methods.backends import BACKENDS
from srms.methods.strategies import eikonal, ntfields
from srms.run import Config, _build_env

MANIFOLDS = [
    ("torus", r"Torus $T^2$", "$0$"),
    ("sphere", r"Sphere $S^2$", "$+1$"),
    ("hyperbolic", r"Hyperbolic $H^2$", "$-1$"),
]
ARMS = [("final", "no weak supervision", "ntfields"), ("weak", "weak supervision", "hntfields")]
_NTFIELDS_FAMILY = ("ntfields", "pntfields", "hntfields")

# Fields allowed to differ between runs: the manifold, the objective, and where output lands.
EXPECTED_DIFFS = {"environment", "method", "out_dir", "start", "dim"}


def load(path: str):
    """Recompute (rms, max_abs, splats, params, cfg) for one run from its saved model.

    Unknown keys in a saved ``cfg`` mean the run predates a config change, so it is not comparable
    with current ones. Rebuilding it silently would let a stale run into a table claiming a matched
    comparison, so this raises instead; ``check_matched`` would catch it later anyway, but failing
    here names the file.
    """
    with open(path, "rb") as handle:
        saved = pickle.load(handle)
    stale = set(saved["cfg"]) - {f.name for f in dataclasses.fields(Config)}
    if stale:
        raise ValueError(f"{path} predates the current Config (unknown keys: {sorted(stale)}) — re-run it")
    cfg = Config(**saved["cfg"])
    env = _build_env(cfg)
    params = jax.tree_util.tree_map(jnp.asarray, saved["splat"])
    points, _ = env.grid(cfg.resolution)
    gt = env.ground_truth(cfg.resolution)
    inside = np.asarray(env.sdf(points)) < 0.0
    if cfg.method in _NTFIELDS_FAMILY:
        pred = np.asarray(ntfields.predict(BACKENDS["srm"], params, points, env, cfg.tau_bias, cfg.tau_min))
    else:
        pred = np.asarray(eikonal.predict(BACKENDS["srm"], params, points, env))
    err = np.where(inside, np.nan, pred - gt)
    return {
        "rms": float(np.sqrt(np.nanmean(err**2))),
        "max_abs": float(np.nanmax(np.abs(err))),
        "splats": len(params[0]),
        "params": int(BACKENDS["srm"].num_params(params)),
        "cfg": saved["cfg"],
    }


def check_matched(runs: dict) -> list[str]:
    """Every run must share one configuration apart from EXPECTED_DIFFS."""
    complaints, keys = [], None
    reference = None
    for name, run in runs.items():
        cfg = run["cfg"]
        if reference is None:
            reference, keys = cfg, set(cfg)
            continue
        for key in sorted(keys):
            if key in EXPECTED_DIFFS:
                continue
            if repr(cfg.get(key)) != repr(reference.get(key)):
                complaints.append(f"{name}: {key} = {cfg.get(key)!r} but reference has {reference.get(key)!r}")
    return complaints


def emit(arm_label: str, rows: list, caption: str, label: str) -> str:
    """One booktabs table for a single supervision arm."""
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \begin{tabular}{llrrr}",
        r"    \toprule",
        r"    Manifold & Curvature $K$ & Splats & RMS error & Max error \\",
        r"    \midrule",
    ]
    for display, curvature, res in rows:
        lines.append(f"    {display} & {curvature} & {res['splats']} & {res['rms']:.4f} & {res['max_abs']:.4f} \\\\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main() -> int:
    runs, missing = {}, []
    for arm, _, _ in ARMS:
        for env_name, _, _ in MANIFOLDS:
            path = f"figures/manifolds/{arm}_{env_name}/splat.pkl"
            try:
                runs[f"{arm}/{env_name}"] = load(path)
            except (FileNotFoundError, ValueError) as exc:
                missing.append(f"{path} — {exc if not isinstance(exc, FileNotFoundError) else 'not run yet'}")
    if missing:
        print("% RUNS MISSING OR STALE — table not emitted:", file=sys.stderr)
        for path in missing:
            print(f"%   {path}", file=sys.stderr)
        return 1

    complaints = check_matched(runs)
    if complaints:
        print("% CONFIGURATIONS NOT MATCHED — refusing to emit a table that claims a fair comparison:", file=sys.stderr)
        for complaint in complaints:
            print(f"%   {complaint}", file=sys.stderr)
        return 1

    ref = next(iter(runs.values()))["cfg"]
    shared = (
        f"Identical configuration across all six runs: SRM backend, {ref['steps']} steps, "
        f"{ref['num_collocation']} collocation points per step, learning rate {ref['lr']}, "
        f"{ref['num_obstacles']} obstacles, seed {ref['seed']}, scored on a fast-marching grid at "
        f"resolution {ref['resolution']}. $K$ is the constant sectional curvature. Splat count is "
        f"chosen by the model, not set: capacity grows "
        f"where the residual is and stops when a densify pass buys less than "
        f"{ref['densify_min_gain']:g} fractional residual reduction per splat added."
    )
    out = []
    for arm, arm_label, method in ARMS:
        rows = [(disp, curv, runs[f"{arm}/{name}"]) for name, disp, curv in MANIFOLDS]
        detail = (
            "PDE residual only; no planner, no roadmap, no ground truth."
            if arm == "final"
            else "PDE residual plus sparse-roadmap travel-time bounds; still no ground truth."
        )
        out.append(
            emit(
                arm_label,
                rows,
                f"Self-supervised Eikonal time-to-go on three 2-D Riemannian manifolds, "
                f"\\textbf{{{arm_label}}} (\\texttt{{{method}}}). {detail} {shared}",
                f"tab:2d-{arm}",
            )
        )
    print("\n\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
