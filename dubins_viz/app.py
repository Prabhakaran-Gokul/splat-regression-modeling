#!/usr/bin/env python3
"""
Interactive Dubins optimal time-to-go visualizer.

V(x, y; θ) = minimum path length from (x, y, θ) to the goal state (0, 0, 0°),
computed exactly via Dubins CSC paths (LSL / RSR / LSR / RSL).

Launch:
    python dubins_viz/app.py
    # then open http://127.0.0.1:8050

Controls:
  - "Start heading θ" slider: sets the fixed heading angle for the heatmap.
  - "Turn radius R" slider: sets the minimum turning radius.
  - Hover over the plot: draws the optimal path from cursor → goal in real time.

Heading convention: 0° = east, 90° = north (standard math angle, CCW).
Goal state is always (x=0, y=0, θ=0°), shown as a green marker with east arrow.
"""

from __future__ import annotations

import numpy as np
import dash
from dash import dcc, html, Input, Output, ctx, Patch
import plotly.graph_objects as go

# ── Dubins geometry ────────────────────────────────────────────────────────────

XY_RANGE = 3.0
_INF = np.inf


def _mod2pi(a):
    return a % (2.0 * np.pi)


def _csc_params(alpha: float, beta: float, d: float) -> dict[str, tuple]:
    """Return {type: (t, p, q, total_normalised_length)} for each feasible CSC type."""
    sa, ca = np.sin(alpha), np.cos(alpha)
    sb, cb = np.sin(beta),  np.cos(beta)
    out: dict[str, tuple] = {}

    # LSL
    tmp = np.arctan2(cb - ca, d + sa - sb)
    t, q = _mod2pi(-alpha + tmp), _mod2pi(beta - tmp)
    p2 = 2 + d**2 - 2*(ca*cb + sa*sb) + 2*d*(sa - sb)
    if p2 >= 0:
        p = np.sqrt(p2)
        out["LSL"] = (t, p, q, t + p + q)

    # RSR
    tmp = np.arctan2(ca - cb, d - sa + sb)
    t, q = _mod2pi(alpha - tmp), _mod2pi(-beta + tmp)
    p2 = 2 + d**2 - 2*(ca*cb + sa*sb) + 2*d*(sb - sa)
    if p2 >= 0:
        p = np.sqrt(p2)
        out["RSR"] = (t, p, q, t + p + q)

    # LSR
    p2 = -2 + d**2 + 2*(ca*cb + sa*sb) + 2*d*(sa + sb)
    if p2 >= 0:
        p   = np.sqrt(p2)
        tmp = np.arctan2(-ca - cb, d + sa + sb) - np.arctan2(-2.0, p)
        t, q = _mod2pi(-alpha + tmp), _mod2pi(-beta + tmp)
        out["LSR"] = (t, p, q, t + p + q)

    # RSL
    p2 = -2 + d**2 + 2*(ca*cb + sa*sb) - 2*d*(sa + sb)
    if p2 >= 0:
        p   = np.sqrt(p2)
        tmp = np.arctan2(ca + cb, d - sa - sb) - np.arctan2(2.0, p)
        t, q = _mod2pi(alpha - tmp), _mod2pi(beta - tmp)
        out["RSL"] = (t, p, q, t + p + q)

    return out


def dubins_dist_grid(xx: np.ndarray, yy: np.ndarray,
                     theta: float, R: float = 1.0) -> np.ndarray:
    """Vectorised Dubins minimum-time field over a 2-D grid."""
    x, y  = xx.ravel(), yy.ravel()
    r     = np.sqrt(x**2 + y**2)
    d     = r / R
    psi   = np.where(r > 1e-8, np.arctan2(-y, -x), 0.0)
    alpha = _mod2pi(theta - psi)
    beta  = _mod2pi(0.0   - psi)
    sa, ca = np.sin(alpha), np.cos(alpha)
    sb, cb = np.sin(beta),  np.cos(beta)

    def _len(t, p2, q):
        p = np.sqrt(np.maximum(p2, 0.0))
        return np.where(p2 >= 0, (t + p + q) * R, _INF)

    tmp = np.arctan2(cb - ca, d + sa - sb)
    L1  = _len(_mod2pi(-alpha + tmp),
               2 + d**2 - 2*(ca*cb + sa*sb) + 2*d*(sa - sb),
               _mod2pi(beta - tmp))

    tmp = np.arctan2(ca - cb, d - sa + sb)
    L2  = _len(_mod2pi(alpha - tmp),
               2 + d**2 - 2*(ca*cb + sa*sb) + 2*d*(sb - sa),
               _mod2pi(-beta + tmp))

    p2 = -2 + d**2 + 2*(ca*cb + sa*sb) + 2*d*(sa + sb)
    p_ = np.sqrt(np.maximum(p2, 0.0))
    tmp = np.arctan2(-ca - cb, d + sa + sb) - np.arctan2(-2.0, p_)
    L3  = np.where(p2 >= 0,
                   (_mod2pi(-alpha + tmp) + p_ + _mod2pi(-beta + tmp)) * R, _INF)

    p2 = -2 + d**2 + 2*(ca*cb + sa*sb) - 2*d*(sa + sb)
    p_ = np.sqrt(np.maximum(p2, 0.0))
    tmp = np.arctan2(ca + cb, d - sa - sb) - np.arctan2(2.0, p_)
    L4  = np.where(p2 >= 0,
                   (_mod2pi(alpha - tmp) + p_ + _mod2pi(beta - tmp)) * R, _INF)

    best = np.min(np.stack([L1, L2, L3, L4]), axis=0)
    best[r < 1e-8] = 0.0
    return best.reshape(xx.shape)


def _arc_pts(x0: float, y0: float, th0: float,
             angle: float, turn: str, R: float, n: int
             ) -> tuple[np.ndarray, np.ndarray]:
    """Sample n points along an arc; turn ∈ {'L', 'R'}."""
    if turn == "L":
        cx, cy = x0 - R * np.sin(th0), y0 + R * np.cos(th0)
        phi0   = th0 - np.pi / 2
        phis   = np.linspace(phi0, phi0 + angle, n)
    else:
        cx, cy = x0 + R * np.sin(th0), y0 - R * np.cos(th0)
        phi0   = th0 + np.pi / 2
        phis   = np.linspace(phi0, phi0 - angle, n)
    return cx + R * np.cos(phis), cy + R * np.sin(phis)


def _straight_pts(x0: float, y0: float, th0: float,
                  length: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    s = np.linspace(0.0, length, n)
    return x0 + s * np.cos(th0), y0 + s * np.sin(th0)


def dubins_path(x0: float, y0: float, theta0: float,
                R: float = 1.0, n_pts: int = 350
                ) -> tuple[np.ndarray, np.ndarray, str, float]:
    """
    Trace optimal Dubins path from (x0, y0, theta0) to (0, 0, 0).
    Returns (path_x, path_y, type_str, total_length).
    """
    r2 = x0**2 + y0**2
    if r2 < (0.05 * R) ** 2:
        return np.array([x0]), np.array([y0]), "at goal", 0.0

    d    = np.sqrt(r2) / R
    psi  = np.arctan2(-y0, -x0)
    paths = _csc_params(_mod2pi(theta0 - psi), _mod2pi(-psi), d)
    if not paths:
        return np.array([x0, 0.0]), np.array([y0, 0.0]), "none", _INF

    best_type = min(paths, key=lambda k: paths[k][3])
    t, p, q, total = paths[best_type]
    total_len = total * R
    arc1_len, str_len, arc2_len = t * R, p * R, q * R

    def npts(l: float) -> int:
        return max(2, int(n_pts * l / total_len)) if l > 1e-6 else 0

    n1, n2, n3 = npts(arc1_len), npts(str_len), npts(arc2_len)
    c1, c2 = best_type[0], best_type[2]

    xb: list[np.ndarray] = []
    yb: list[np.ndarray] = []
    x, y, th = x0, y0, theta0

    if n1 > 0:
        xs, ys = _arc_pts(x, y, th, t, c1, R, n1)
        xb.append(xs); yb.append(ys)
        x, y, th = float(xs[-1]), float(ys[-1]), (th + t if c1 == "L" else th - t)

    if n2 > 0:
        xs, ys = _straight_pts(x, y, th, str_len, n2)
        xb.append(xs); yb.append(ys)
        x, y = float(xs[-1]), float(ys[-1])

    if n3 > 0:
        xs, ys = _arc_pts(x, y, th, q, c2, R, n3)
        xb.append(xs); yb.append(ys)

    if not xb:
        return np.array([x0, 0.0]), np.array([y0, 0.0]), best_type, total_len

    return np.concatenate(xb), np.concatenate(yb), best_type, total_len


# ── Static grid ────────────────────────────────────────────────────────────────

N_GRID = 160
x_grid = np.linspace(-XY_RANGE, XY_RANGE, N_GRID)
y_grid = np.linspace(-XY_RANGE, XY_RANGE, N_GRID)
XX, YY = np.meshgrid(x_grid, y_grid)
x_list = x_grid.tolist()
y_list = y_grid.tolist()

ARROW_LEN = 0.4   # heading-indicator length in data units

# Trace indices — must stay in sync with build_full_figure()
_T_HEATMAP  = 0
_T_CONTOURS = 1
_T_PATH     = 2
_T_CURSOR   = 3
_T_HEADING  = 4
_T_GOAL     = 5
_T_GOAL_HDG = 6


# ── Figure construction ────────────────────────────────────────────────────────

def build_full_figure(theta_deg: float, R: float,
                      hover_x: float | None = None,
                      hover_y: float | None = None) -> go.Figure:
    """Rebuild the complete figure (heatmap + static overlays + optional path)."""
    theta = np.deg2rad(theta_deg)
    Z     = dubins_dist_grid(XX, YY, theta, R)

    finite = Z[np.isfinite(Z)]
    vmax   = float(np.nanpercentile(finite, 97)) if len(finite) else 10.0
    n_lvl  = 18
    lvl_sz = vmax / n_lvl

    # ── Path + cursor traces ──────────────────────────────────────────────────
    path_x: list = []
    path_y: list = []
    cur_x:  list = []
    cur_y:  list = []
    hdg_x:  list = []
    hdg_y:  list = []

    if hover_x is not None and hover_y is not None:
        px, py, _, _ = dubins_path(hover_x, hover_y, theta, R)
        path_x, path_y = px.tolist(), py.tolist()
        cur_x, cur_y   = [hover_x], [hover_y]
        hx2 = hover_x + ARROW_LEN * np.cos(theta)
        hy2 = hover_y + ARROW_LEN * np.sin(theta)
        hdg_x, hdg_y   = [hover_x, hx2], [hover_y, hy2]

    # ── Goal heading arrow (east = θ=0°) ─────────────────────────────────────
    goal_alen = ARROW_LEN * min(R, 1.0)   # scale arrow with R but cap at 1 m

    traces = [
        # 0 – smooth heatmap background
        go.Heatmap(
            x=x_list, y=y_list, z=Z.tolist(),
            colorscale="plasma", zmin=0, zmax=vmax,
            zsmooth="best",
            colorbar=dict(
                title=dict(text="V [s]", font=dict(color="#ccc")),
                thickness=14, len=0.85,
                tickfont=dict(color="#ccc"),
            ),
            hovertemplate="x=%{x:.2f}  y=%{y:.2f}  V=%{z:.3f}<extra></extra>",
            name="TTG",
        ),
        # 1 – white contour lines overlay
        go.Contour(
            x=x_list, y=y_list, z=Z.tolist(),
            contours=dict(start=lvl_sz, end=vmax - lvl_sz, size=lvl_sz,
                          coloring="none"),
            line=dict(color="rgba(255,255,255,0.28)", width=0.8),
            showscale=False, hoverinfo="skip", name="",
        ),
        # 2 – optimal path
        go.Scatter(
            x=path_x, y=path_y,
            mode="lines",
            line=dict(color="#ffd700", width=2.8),
            name="Optimal path",
            hoverinfo="skip",
        ),
        # 3 – cursor marker (start position)
        go.Scatter(
            x=cur_x, y=cur_y,
            mode="markers",
            marker=dict(
                symbol="circle", size=11, color="white",
                line=dict(width=2, color="#999"),
            ),
            name="Start",
            hoverinfo="skip",
        ),
        # 4 – cursor heading indicator
        go.Scatter(
            x=hdg_x, y=hdg_y,
            mode="lines",
            line=dict(color="white", width=2.5),
            showlegend=False,
            hoverinfo="skip",
        ),
        # 5 – goal marker (lime circle at origin)
        go.Scatter(
            x=[0], y=[0],
            mode="markers",
            marker=dict(
                symbol="circle", size=13, color="#00e676",
                line=dict(width=2, color="white"),
            ),
            name="Goal (0, 0, 0°)",
            hoverinfo="skip",
        ),
        # 6 – goal heading arrow (east direction = θ=0°)
        go.Scatter(
            x=[0, goal_alen], y=[0, 0],
            mode="lines",
            line=dict(color="#00e676", width=3),
            showlegend=False,
            hoverinfo="skip",
        ),
    ]

    return go.Figure(
        data=traces,
        layout=go.Layout(
            paper_bgcolor="#111",
            plot_bgcolor="#111",
            font=dict(color="#ddd"),
            xaxis=dict(
                title="x [m]", range=[-XY_RANGE, XY_RANGE],
                scaleanchor="y", scaleratio=1,
                showgrid=False, zeroline=False,
                tickcolor="#444", linecolor="#444",
            ),
            yaxis=dict(
                title="y [m]", range=[-XY_RANGE, XY_RANGE],
                showgrid=False, zeroline=False,
                tickcolor="#444", linecolor="#444",
            ),
            showlegend=False,
            margin=dict(l=55, r=10, t=10, b=55),
            hovermode="closest",
        ),
    )


# ── Dash app ───────────────────────────────────────────────────────────────────

def _default_info() -> list:
    return [
        html.Span("Hover the plot to trace", style={"display": "block"}),
        html.Span("the optimal Dubins path.", style={"display": "block"}),
    ]


def _hover_info(hx: float, hy: float,
                ptype: str, plen: float, theta_deg: float) -> list:
    return [
        html.Span(f"Start:  ({hx:+.2f}, {hy:+.2f})", style={"display": "block"}),
        html.Span(f"θ:      {theta_deg}°",             style={"display": "block"}),
        html.Span(f"Type:   {ptype}",                  style={"display": "block"}),
        html.Span(f"Length: {plen:.3f} m",             style={"display": "block"}),
    ]


_SLIDER_MARKS_STYLE = {"color": "#777", "fontSize": "10px"}

app = dash.Dash(__name__, title="Dubins Time-to-Go")
server = app.server  # for deployment with gunicorn

app.layout = html.Div(
    style={
        "fontFamily": "'Courier New', monospace",
        "backgroundColor": "#111", "color": "#eee",
        "display": "flex", "flexDirection": "column",
        "height": "100vh", "padding": "10px 12px",
    },
    children=[
        html.H3(
            "Dubins Optimal Time-to-Go",
            style={
                "margin": "0 0 8px 0", "textAlign": "center",
                "color": "#fff", "fontFamily": "sans-serif",
                "fontWeight": "300", "letterSpacing": "2px",
            },
        ),

        html.Div(
            style={"display": "flex", "flex": "1", "gap": "14px", "minHeight": 0},
            children=[
                # ── Main heatmap plot ─────────────────────────────────────────
                dcc.Graph(
                    id="main-plot",
                    style={"flex": "1"},
                    config={
                        "displayModeBar": True,
                        "scrollZoom": True,
                        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                        "toImageButtonOptions": {"format": "png", "scale": 2},
                    },
                    figure=build_full_figure(0.0, 1.0),
                ),

                # ── Controls sidebar ──────────────────────────────────────────
                html.Div(
                    style={
                        "width": "235px",
                        "display": "flex", "flexDirection": "column",
                        "gap": "20px", "padding": "16px",
                        "backgroundColor": "#1a1a1a", "borderRadius": "8px",
                    },
                    children=[
                        # Heading slider
                        html.Div([
                            html.Label(
                                "Start heading θ",
                                style={"display": "block", "marginBottom": "8px",
                                       "fontSize": "13px", "color": "#bbb"},
                            ),
                            dcc.Slider(
                                id="theta-slider",
                                min=-180, max=180, step=5, value=0,
                                marks={
                                    v: {"label": f"{v}°", "style": _SLIDER_MARKS_STYLE}
                                    for v in [-180, -90, 0, 90, 180]
                                },
                                tooltip={"placement": "bottom", "always_visible": True},
                            ),
                        ]),

                        # Radius slider
                        html.Div([
                            html.Label(
                                "Turn radius R [m]",
                                style={"display": "block", "marginBottom": "8px",
                                       "fontSize": "13px", "color": "#bbb"},
                            ),
                            dcc.Slider(
                                id="radius-slider",
                                min=0.25, max=2.0, step=0.25, value=1.0,
                                marks={
                                    v: {"label": str(v), "style": _SLIDER_MARKS_STYLE}
                                    for v in [0.25, 0.5, 1.0, 1.5, 2.0]
                                },
                                tooltip={"placement": "bottom", "always_visible": True},
                            ),
                        ]),

                        # Info panel
                        html.Div(
                            id="info-panel",
                            style={
                                "backgroundColor": "#252525",
                                "padding": "10px 12px",
                                "borderRadius": "6px",
                                "fontSize": "12px",
                                "lineHeight": "1.9",
                                "color": "#ccc",
                            },
                            children=_default_info(),
                        ),

                        # Legend for path types
                        html.Div(
                            style={
                                "backgroundColor": "#1e1e1e",
                                "padding": "10px 12px",
                                "borderRadius": "6px",
                                "fontSize": "11px",
                                "lineHeight": "1.7",
                                "color": "#888",
                            },
                            children=[
                                html.Span("Path types:", style={"display": "block", "color": "#aaa", "marginBottom": "2px"}),
                                html.Span("L = left arc (CCW)", style={"display": "block"}),
                                html.Span("R = right arc (CW)", style={"display": "block"}),
                                html.Span("S = straight",       style={"display": "block"}),
                            ],
                        ),

                        # Footer notes
                        html.Div(
                            style={
                                "fontSize": "10px", "color": "#444",
                                "marginTop": "auto", "lineHeight": "1.7",
                            },
                            children=[
                                html.P("Goal: (0, 0, θ=0°)",                      style={"margin": "0 0 2px"}),
                                html.P("PDE: (cosθ ∂ₓV + sinθ ∂ᵧV)² + (∂_θV)² = 1", style={"margin": "0 0 2px"}),
                                html.P("Exact: Dubins CSC paths",                  style={"margin": 0}),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)


# ── Callbacks ──────────────────────────────────────────────────────────────────

@app.callback(
    Output("main-plot", "figure"),
    Output("info-panel", "children"),
    Input("theta-slider", "value"),
    Input("radius-slider", "value"),
    Input("main-plot", "hoverData"),
)
def update(theta_deg: float, radius: float, hover_data: dict | None):
    theta     = np.deg2rad(theta_deg)
    triggered = ctx.triggered_id

    # Parse hover position (only accept events from the heatmap trace)
    hover_x = hover_y = None
    if hover_data and hover_data.get("points"):
        pt = hover_data["points"][0]
        if pt.get("curveNumber", -1) == _T_HEATMAP:
            hover_x = pt.get("x")
            hover_y = pt.get("y")

    # ── Hover-only update: use Patch() to avoid re-sending the heatmap ───────
    if triggered == "main-plot":
        fig = Patch()
        if hover_x is not None:
            px, py, ptype, plen = dubins_path(hover_x, hover_y, theta, radius)
            hx2 = hover_x + ARROW_LEN * np.cos(theta)
            hy2 = hover_y + ARROW_LEN * np.sin(theta)
            fig["data"][_T_PATH   ]["x"] = px.tolist()
            fig["data"][_T_PATH   ]["y"] = py.tolist()
            fig["data"][_T_CURSOR ]["x"] = [hover_x]
            fig["data"][_T_CURSOR ]["y"] = [hover_y]
            fig["data"][_T_HEADING]["x"] = [hover_x, hx2]
            fig["data"][_T_HEADING]["y"] = [hover_y, hy2]
            info = _hover_info(hover_x, hover_y, ptype, plen, theta_deg)
        else:
            for t_idx in (_T_PATH, _T_CURSOR, _T_HEADING):
                fig["data"][t_idx]["x"] = []
                fig["data"][t_idx]["y"] = []
            info = _default_info()
        return fig, info

    # ── Theta / radius changed: full figure rebuild ───────────────────────────
    fig = build_full_figure(theta_deg, radius, hover_x, hover_y)
    if hover_x is not None:
        _, _, ptype, plen = dubins_path(hover_x, hover_y, theta, radius)
        info = _hover_info(hover_x, hover_y, ptype, plen, theta_deg)
    else:
        info = _default_info()
    return fig, info


if __name__ == "__main__":
    app.run(debug=False, port=7000, host="127.0.0.1")
