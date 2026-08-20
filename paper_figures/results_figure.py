"""Results figure: SRM vs MLP predicted time-to-go against ground truth, as one 3x3 grid of flat 2D
chart panels — rows are manifolds (torus, sphere, hyperbolic H^2), columns are
(ground truth | SRM | MLP).

Reuses ``paper_figures/intro_figure.py``'s exact ``RUNS`` as the SRM column, and derives the MLP
column from it via ``dataclasses.replace(backend="mlp", ...)`` rather than a separately
hand-specified config. **That's the one place to change a row's obstacle set**: every scene field
(``num_obstacles``, ``obstacle_radius``, ``seed``, ...) comes from ``intro_figure.RUNS``'s Config for
that manifold, so editing it there changes the ground truth *and* both predictions for that row
together — obstacle sampling depends only on those shared scene fields, not on ``backend``/``method``
(verified: the existing ``*_srm``/``*_mlp`` checkpoint pairs this reuses already have identical
sampled obstacles). If a run's checkpoint doesn't exist yet it's trained fresh via
``intro_figure._ensure_trained`` (the normal ``srms.run.main`` pipeline), same as intro_figure.py.

Rendering goes through ``srms.viz``'s own ``_draw_field`` — the same chart ``srms.run.main`` draws
its [GT | prediction | error] figure with: rectangular ``imshow(extent=...)`` for torus/sphere,
curvilinear ``pcolormesh`` (Poincaré disk) for hyperbolic — the manifold's native flat chart, not a
3D embedding (contrast ``intro_figure.py``).

The hyperbolic row is drawn on a much finer grid than it was trained/scored on
(``_CHART_RENDER_RESOLUTION``): ``imshow`` (torus/sphere) embeds its array as a resampled, filtered
raster even in a vector PDF, so the training-resolution grid still looks smooth; ``pcolormesh``
(Poincaré disk) instead draws one flat-shaded polygon per grid cell as real PDF vector paths, so at
that same resolution the panel — and its obstacle silhouette, a marching-squares contour over the
same coarse grid — visibly pixelates once large on the page. Evaluating a trained splat (and
re-running fast marching for ground truth) at a finer grid than it was scored on is just more
points, not a retrain — same trick as ``intro_figure.py``'s own ``_RENDER_RESOLUTION``; the
comparison table below still scores at each run's own ``cfg.resolution``, untouched by this.

Also prints one LaTeX comparison table per row (RSRM vs MLP: final loss, final error, parameter
count, training time), plus each pair's iteration count. Parameters/error are recomputed from the
checkpoint itself (same philosophy as ``make_tables.py``: never scraped from a log that could
disagree with the model). Loss and training time aren't recoverable from the checkpoint alone
(``srms.run.main`` never saves them — see its ``splat.pkl`` dump) so those come from the mlflow run
that produced it, keyed by ``out_dir`` (``_query_training_run``). If a pair's iteration counts
(``cfg.steps``) don't match, that's an unfair comparison and this raises rather than printing a
table that hides it.
"""

from __future__ import annotations

import dataclasses
import math

import matplotlib.pyplot as plt
import mlflow
import numpy as np

from paper_figures.intro_figure import RUNS as SRM_RUNS
from paper_figures.intro_figure import _FILE_SLUGS, _contour_levels, _ensure_trained, _load_splat, _predict_field
from paper_figures.style import savefig, use_latex_fonts
from srms.run import Config
from srms.viz import _draw_field

use_latex_fonts()


def _mlp_variant(cfg: Config) -> Config:
    """``cfg`` with ``backend="mlp"``, everything else (scene, method, resolution, seed, ...)
    untouched — so it's guaranteed to describe the same obstacle set as ``cfg`` itself.
    ``out_dir`` mirrors this repo's own ``*_srm`` -> ``*_mlp`` checkpoint naming convention."""
    out_dir = cfg.out_dir[: -len("srm")] + "mlp" if cfg.out_dir.endswith("srm") else f"{cfg.out_dir}_mlp"
    return dataclasses.replace(cfg, backend="mlp", out_dir=out_dir)


# {manifold: {"srm": Config, "mlp": Config}} — see module docstring: edit intro_figure.RUNS, not this.
RUNS: dict[str, dict[str, Config]] = {name: {"srm": cfg, "mlp": _mlp_variant(cfg)} for name, cfg in SRM_RUNS.items()}

_ROW_TITLES = {"torus": "Torus $T^2$", "sphere": "Sphere $S^2$", "lorentz_hyperbolic": "Hyperbolic $H^2$"}
_COLUMN_TITLES = ["Ground truth", "SRM", "MLP"]
# Plain-text manifold name for the LaTeX table captions/labels this module prints (reuses
# intro_figure's own file-naming slug, e.g. "lorentz_hyperbolic" -> "hyperbolic").
_ROW_DISPLAY_NAMES = _FILE_SLUGS

PANEL_SIZE = (4.6, 4.6)  # inches, per grid cell

# Grid resolution to *render* the Poincaré-disk (curvilinear-chart) row at, decoupled from that
# run's own cfg.resolution (training/scoring) — see module docstring for why pcolormesh needs this
# and imshow doesn't.
_CHART_RENDER_RESOLUTION = 500


def _load(cfg: Config):
    """Load ``cfg``'s trained checkpoint. Returns ``(env, saved_cfg, backend, splat)``."""
    saved_cfg, env, backend, splat = _load_splat(cfg)
    return env, saved_cfg, backend, splat


def _eval(env, saved_cfg: Config, backend, splat, resolution: int):
    """Evaluate the trained splat's field on ``env``'s dense grid at ``resolution`` scalars/point —
    ``resolution`` need not equal ``saved_cfg.resolution``; more evaluation points isn't a retrain.
    Returns ``(thetas, shape, pred)``."""
    thetas, shape = env.grid(resolution)
    pred = _predict_field(saved_cfg, backend, splat, thetas, env).reshape(shape)
    return thetas, shape, pred


def _query_training_run(out_dir: str) -> tuple[float, float]:
    """(final training loss, training wall-clock seconds) for ``out_dir``'s most recent finished
    mlflow run. ``splat.pkl`` (see ``srms/run.py``'s ``main``) never saves either — only the trained
    params — so these two figures have to come from the run log that produced the checkpoint rather
    than the checkpoint itself."""
    runs = mlflow.search_runs(
        experiment_names=["srms"],
        filter_string=f"params.out_dir = '{out_dir}' and attributes.status = 'FINISHED'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        raise RuntimeError(
            f"no finished mlflow run logged for out_dir={out_dir!r} — the results table needs its "
            "final loss and training time, which only the run log carries (retrain via srms.run "
            "if the checkpoint predates its mlflow run, e.g. after clearing mlruns/)"
        )
    row = runs.iloc[0]
    loss = row.get("metrics.loss")
    if loss is None or (isinstance(loss, float) and math.isnan(loss)):
        raise RuntimeError(f"mlflow run {row['run_id']} for out_dir={out_dir!r} logged no 'loss' metric")
    duration_s = (row["end_time"] - row["start_time"]).total_seconds()
    return float(loss), float(duration_s)


def _run_stats(saved_cfg: Config, backend, splat, gt: np.ndarray, pred: np.ndarray, inside: np.ndarray) -> dict:
    """Everything one column of a comparison table needs for one trained run."""
    err = np.where(inside, np.nan, pred - gt)
    loss, time_s = _query_training_run(saved_cfg.out_dir)
    return {
        "loss": loss,
        "error": float(np.sqrt(np.nanmean(err**2))),
        "params": int(backend.num_params(splat)),
        "time_s": time_s,
        "steps": saved_cfg.steps,
    }


def _fmt_num(v: float) -> str:
    """3 significant figures; scientific notation for very small magnitudes (losses/errors)."""
    if v == 0:
        return "0"
    if abs(v) < 1e-3:
        mantissa, exp = f"{v:.2e}".split("e")
        return rf"{float(mantissa):.2g} \times 10^{{{int(exp)}}}"
    return f"{float(f'{v:.3g}'):g}"


def _fmt_time(seconds: float) -> str:
    return f"{seconds / 60:.1f}\\,\\text{{min}}" if seconds >= 60 else f"{seconds:.1f}\\,\\text{{s}}"


def _row(label: str, srm_v: float, mlp_v: float, fmt) -> str:
    """One table row, bolding whichever of the two values is smaller (better)."""
    srm_wins = srm_v <= mlp_v
    srm_s = fmt(srm_v)
    mlp_s = fmt(mlp_v)
    srm_cell = f"$\\boldsymbol{{{srm_s}}}$" if srm_wins else f"${srm_s}$"
    mlp_cell = f"$\\boldsymbol{{{mlp_s}}}$" if not srm_wins else f"${mlp_s}$"
    return f"        {label} & {srm_cell} & {mlp_cell} \\\\"


def _print_table(display_name: str, slug: str, srm: dict, mlp: dict) -> None:
    """Print one LaTeX comparison table (RSRM vs MLP) for one manifold, after printing and checking
    the pair's iteration counts. Both must match — a table comparing runs trained for different
    numbers of steps isn't a fair comparison, so this raises instead of printing one."""
    print(f"% {display_name}: RSRM trained for {srm['steps']} iterations, MLP trained for {mlp['steps']} iterations")
    if srm["steps"] != mlp["steps"]:
        raise ValueError(
            f"{display_name}: RSRM ran {srm['steps']} iterations but MLP ran {mlp['steps']} — "
            "comparison is unfair (mismatched training budgets); fix out_dir/steps before comparing"
        )
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"    \caption{{Comparison of our approach with MLP on the {display_name} manifold.}}",
        rf"    \label{{tab:{slug}-comparison}}",
        r"    \begin{tabular}{lcc}",
        r"        \toprule",
        r"        & \textbf{RSRM (Ours)} & \textbf{MLP} \\",
        r"        \midrule",
        _row("Final Loss", srm["loss"], mlp["loss"], _fmt_num),
        _row("Final Error", srm["error"], mlp["error"], _fmt_num),
        _row("Parameters", srm["params"], mlp["params"], lambda v: f"{int(v)}"),
        # rf"        Training Time & ${_fmt_time(srm['time_s'])}$ & ${_fmt_time(mlp['time_s'])}$ \\",
        r"        \bottomrule",
        r"    \end{tabular}",
        r"\end{table}",
    ]
    print("\n".join(lines))
    print()


def main() -> None:
    for pair in RUNS.values():
        for cfg in pair.values():
            _ensure_trained(cfg)

    n_rows = len(RUNS)
    fig, axes = plt.subplots(n_rows, 3, figsize=(PANEL_SIZE[0] * 3, PANEL_SIZE[1] * n_rows))

    for row, (name, pair) in enumerate(RUNS.items()):
        # srm/mlp share one scene (see module docstring), so obstacles/chart geometry are computed
        # once here rather than once per column.
        env, saved_cfg, srm_backend, srm_splat = _load(pair["srm"])
        _, mlp_cfg, mlp_backend, mlp_splat = _load(pair["mlp"])

        # Scoring grid — each run's own cfg.resolution, exactly what it was trained/scored at. The
        # comparison table is computed from this, untouched by whatever resolution renders below.
        thetas, shape, srm_pred = _eval(env, saved_cfg, srm_backend, srm_splat, saved_cfg.resolution)
        _, _, mlp_pred = _eval(env, mlp_cfg, mlp_backend, mlp_splat, mlp_cfg.resolution)
        inside = np.asarray(env.sdf(thetas)).reshape(shape) < 0.0
        gt = np.asarray(env.ground_truth(saved_cfg.resolution)).reshape(shape)

        srm_stats = _run_stats(saved_cfg, srm_backend, srm_splat, gt, srm_pred, inside)
        mlp_stats = _run_stats(mlp_cfg, mlp_backend, mlp_splat, gt, mlp_pred, inside)
        _print_table(_ROW_DISPLAY_NAMES[name], _FILE_SLUGS[name], srm_stats, mlp_stats)

        # Render grid — curvilinear (Poincaré-disk) charts redraw at _CHART_RENDER_RESOLUTION so
        # their pcolormesh doesn't pixelate (see module docstring); torus/sphere's imshow already
        # looks smooth at the scoring resolution, so they render unchanged.
        render_resolution = _CHART_RENDER_RESOLUTION if hasattr(env, "render_grid_xy") else saved_cfg.resolution
        if render_resolution != saved_cfg.resolution:
            thetas, shape, srm_pred = _eval(env, saved_cfg, srm_backend, srm_splat, render_resolution)
            _, _, mlp_pred = _eval(env, mlp_cfg, mlp_backend, mlp_splat, render_resolution)
            inside = np.asarray(env.sdf(thetas)).reshape(shape) < 0.0
            gt = np.asarray(env.ground_truth(render_resolution)).reshape(shape)

        blocked = inside.astype(float)
        gt_img = np.where(inside, np.nan, gt)
        vmax = float(np.nanmax(gt_img))
        # One shared set of levels (from ground truth) for all three columns of the row, so SRM's
        # and MLP's contours are directly comparable to ground truth's rather than each auto-scaled
        # to its own range.
        levels = _contour_levels(gt_img)
        marker = env.render_marker_deg()
        coords = env.render_grid_xy(render_resolution) if hasattr(env, "render_grid_xy") else None
        edges = env.render_grid_edges_xy(render_resolution) if hasattr(env, "render_grid_edges_xy") else None

        for col, field in enumerate((gt, srm_pred, mlp_pred)):
            ax = axes[row, col]
            img = np.where(inside, np.nan, field)
            if coords is None:
                # Rectangular imshow (torus/sphere) fills the whole panel, so this never shows —
                # skip it for the curvilinear Poincaré-disk chart (hyperbolic), which only covers
                # the disk and would otherwise leave a gray square in its four corners.
                ax.set_facecolor("lightgray")
            handle, mesh1, mesh2 = _draw_field(ax, env, img, shape, coords, edges, "viridis", 0.0, vmax)
            ax.contour(mesh1, mesh2, img, levels=levels, colors="white", linewidths=0.7, alpha=0.75)
            ax.contour(mesh1, mesh2, blocked, levels=[0.5], colors="black", linewidths=1.0)
            ax.plot(*marker, "*", color="red", markersize=10, markeredgecolor="white")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("")
            ax.set_ylabel("")
            if row == 0:
                ax.set_title(_COLUMN_TITLES[col], fontsize=22, fontweight="bold")
            if col == 0:
                ax.set_ylabel(_ROW_TITLES[name], fontsize=22, fontweight="bold")
            if col == 2:
                fig.colorbar(handle, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    out_path = "paper_figures/results_figure.png"
    savefig(fig, out_path, dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
