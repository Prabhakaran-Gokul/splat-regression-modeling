"""Shared matplotlib styling for ``paper_figures/*.py``.

One place to edit the look shared across every figure script here. Currently just LaTeX-rendered
text in Computer Modern (LaTeX's own default serif font), so labels match a paper's own
typesetting instead of matplotlib's default sans-serif. Requires a working TeX installation
(``latex`` + ``dvipng`` on ``PATH``) — both present here (``texlive``/``pdftex``).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def use_latex_fonts() -> None:
    """Route all matplotlib text (titles, tick/colorbar labels, ...) through LaTeX, Computer
    Modern serif. Global and idempotent (``rcParams``) — call once per script, before plotting."""
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
        }
    )


def savefig(fig, out_path: str, dpi: int = 200, pad_inches: float = 0.02) -> None:
    """Save ``fig`` as both a PNG (``out_path``, for quick previewing) and a PDF (same path with a
    ``.pdf`` extension, a vector copy for paper inclusion) — the one save call every figure script
    here uses, so both formats stay in sync automatically. ``out_path`` should end in ``.png``.

    ``bbox_inches="tight"`` + a small ``pad_inches`` crops each save to its actual drawn content
    (not matplotlib's default ~0.1in pad) — panels with axes turned off (see
    ``render_ambient_mpl``'s ``set_axis_off``) crop right down to the object's own silhouette."""
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=pad_inches)
    pdf_path = Path(out_path).with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=pad_inches)
    print(f"saved {out_path} and {pdf_path}")
