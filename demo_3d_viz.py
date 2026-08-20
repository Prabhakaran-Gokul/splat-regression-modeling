#!/usr/bin/env python3
"""Demo script: generate all 4 types of 3D visualizations (sphere + torus).

Features demonstrated:
  1. Integration into workflow (generates on sphere/torus data)
  2. Interactive Plotly versions (HTML files you can rotate/zoom)
  3. Path/trajectory visualization (gradient descent paths on surfaces)
  4. Multi-angle views (front/back/top/isometric for papers)

Usage:
  python demo_3d_viz.py
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

from visualize_3d import (
    Sphere3DView,
    Torus3DView,
    plotly_sphere,
    plotly_torus,
    extract_path_torus,
    extract_path_sphere,
    render_sphere_comparison,
    render_torus_comparison,
)

OUT_DIR = Path("figures/demo_3d")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def demo_sphere() -> None:
    """Demonstrate 3D sphere visualization with synthetic data."""
    print("\n" + "=" * 60)
    print("SPHERE DEMONSTRATION")
    print("=" * 60)

    # Create synthetic ground-truth: great-circle distance from north pole
    resolution = 120
    lon = np.linspace(-np.pi, np.pi, resolution)
    lat = np.linspace(-np.pi / 2, np.pi / 2, resolution)
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # Great-circle distance from north pole (0, π/2)
    cos_angle = np.clip(
        np.sin(lat_grid),  # cos(angle) for great-circle from north pole
        -1.0,
        1.0,
    )
    gt_field = np.arccos(cos_angle)

    # Synthetic prediction with some noise and bias
    pred_field = gt_field + 0.15 * np.sin(lon_grid) * np.cos(lat_grid)

    start_xyz = np.array([0.0, 0.0, 1.0])  # north pole

    print("1. Single-angle rendering (ground-truth)...")
    Sphere3DView(elev=30, azim=45).render(
        gt_field,
        start=start_xyz,
        title="Sphere: Great-Circle Distance from North Pole",
        out_path=str(OUT_DIR / "sphere_gt_single.png"),
    )

    print("2. Multi-angle rendering (prediction with 4 views)...")
    Sphere3DView().render_multiangle(
        pred_field,
        start=start_xyz,
        title="Sphere Prediction",
        out_dir=str(OUT_DIR),
    )

    print("3. Interactive Plotly rendering (ground-truth)...")
    plotly_sphere(
        gt_field,
        start=start_xyz,
        title="Sphere GT - Interactive (rotate with mouse)",
        out_html=str(OUT_DIR / "sphere_interactive_gt.html"),
    )

    print("4. Interactive Plotly with prediction...")
    plotly_sphere(
        pred_field,
        start=start_xyz,
        title="Sphere Prediction - Interactive (rotate with mouse)",
        out_html=str(OUT_DIR / "sphere_interactive_pred.html"),
    )

    print("5. Comparison figure (GT | Pred | Error)...")
    render_sphere_comparison(gt_field, pred_field, start=start_xyz, out_dir=str(OUT_DIR))

    print(f"✓ Sphere demos saved to {OUT_DIR}/")


def demo_torus() -> None:
    """Demonstrate 3D torus visualization with synthetic data."""
    print("\n" + "=" * 60)
    print("TORUS DEMONSTRATION")
    print("=" * 60)

    # Create synthetic torus field: distance-to-goal
    resolution = 120
    theta1 = np.linspace(-np.pi, np.pi, resolution)
    theta2 = np.linspace(-np.pi, np.pi, resolution)
    t1_grid, t2_grid = np.meshgrid(theta1, theta2)

    # Geodesic distance on flat torus from start
    start = np.array([-1.5, -1.5])
    disp1 = np.sin((t1_grid - start[0]) / 2) * 2
    disp2 = np.sin((t2_grid - start[1]) / 2) * 2
    gt_field = np.sqrt(disp1**2 + disp2**2)

    # Synthetic prediction with obstacles (high cost regions)
    obstacle1_dist = np.sqrt((t1_grid - 0.5) ** 2 + (t2_grid - 0.5) ** 2)
    obstacle2_dist = np.sqrt((t1_grid + 1.0) ** 2 + (t2_grid + 1.0) ** 2)
    obstacle_cost = 1.5 * (np.exp(-10 * obstacle1_dist) + np.exp(-10 * obstacle2_dist))
    pred_field = gt_field + 0.3 * obstacle_cost

    obstacles = [
        (0.5, 0.5, 0.4),
        (-1.0, -1.0, 0.35),
    ]

    print("1. Single-angle rendering (ground-truth)...")
    Torus3DView(elev=25, azim=45).render(
        gt_field,
        start=tuple(start),
        obstacles=obstacles,
        title="Torus: Geodesic Distance (with obstacles)",
        out_path=str(OUT_DIR / "torus_gt_single.png"),
    )

    print("2. Multi-angle rendering (prediction with 4 views)...")
    Torus3DView().render_multiangle(
        pred_field,
        start=tuple(start),
        obstacles=obstacles,
        title="Torus Prediction",
        out_dir=str(OUT_DIR),
    )

    print("3. Extracting optimal path via gradient descent...")
    goal = np.array([1.5, 1.5])
    path = extract_path_torus(pred_field, start=tuple(start), goal=tuple(goal), num_steps=80)
    print(f"   Path has {len(path)} points")

    print("4. Multi-angle with path overlay...")
    Torus3DView().render_multiangle(
        pred_field,
        start=tuple(start),
        path=path,
        obstacles=obstacles,
        title="Torus with Optimal Path",
        out_dir=str(OUT_DIR / "with_path"),
    )

    print("5. Interactive Plotly (ground-truth)...")
    plotly_torus(
        gt_field,
        start=tuple(start),
        obstacles=obstacles,
        title="Torus GT - Interactive (rotate with mouse)",
        out_html=str(OUT_DIR / "torus_interactive_gt.html"),
    )

    print("6. Interactive Plotly (prediction with path)...")
    plotly_torus(
        pred_field,
        start=tuple(start),
        path=path,
        obstacles=obstacles,
        title="Torus Prediction with Path - Interactive (rotate with mouse)",
        out_html=str(OUT_DIR / "torus_interactive_pred_path.html"),
    )

    print("7. Comparison figure (GT | Pred | Error)...")
    render_torus_comparison(
        gt_field, pred_field, start=tuple(start), obstacles=obstacles, out_dir=str(OUT_DIR)
    )

    print(f"✓ Torus demos saved to {OUT_DIR}/")


def demo_custom_colormaps() -> None:
    """Show different colormaps for different use cases."""
    print("\n" + "=" * 60)
    print("COLORMAP VARIATIONS")
    print("=" * 60)

    # Simple torus field
    resolution = 80
    theta1 = np.linspace(-np.pi, np.pi, resolution)
    theta2 = np.linspace(-np.pi, np.pi, resolution)
    t1_grid, t2_grid = np.meshgrid(theta1, theta2)
    field = np.sqrt(np.cos(t1_grid) ** 2 + np.sin(t2_grid) ** 2)

    colormaps = ["viridis", "plasma", "inferno", "coolwarm", "RdYlBu"]
    print(f"Generating single-angle renders with {len(colormaps)} colormaps...")

    for cmap in colormaps:
        viewer = Torus3DView(cmap=cmap, elev=20, azim=45)
        viewer.render(
            field,
            title=f"Torus - {cmap} colormap",
            out_path=str(OUT_DIR / f"torus_cmap_{cmap}.png"),
        )

    print(f"✓ Colormap demos saved to {OUT_DIR}/")


def print_summary() -> None:
    """Print what was generated and next steps."""
    print("\n" + "=" * 60)
    print("SUMMARY: What was generated")
    print("=" * 60)

    print("\n📊 STATIC IMAGES (for papers):")
    print("   • Multi-angle views: front, back, top, isometric")
    print("   • High-resolution PNG (150 DPI, publication-ready)")
    print("   • Supports custom camera angles and colormaps")

    print("\n🔄 INTERACTIVE HTML (for presentations):")
    print("   • Rotate with mouse, zoom with scroll")
    print("   • Pan by dragging")
    print("   • Share via HTML file")
    print(f"   • Look for: {OUT_DIR}/*_interactive_*.html")

    print("\n🛤️  PATH VISUALIZATION:")
    print("   • Gradient descent paths overlaid on field")
    print("   • Cyan lines on surfaces showing optimal flow")
    print("   • Works for both sphere and torus")

    print("\n🎨 FEATURES:")
    print("   • Start/goal/obstacle markers")
    print("   • Contour lines on surfaces")
    print("   • Colorbar with field value scale")
    print("   • NaN masking for obstacles")

    print("\n✨ Next steps:")
    print("   1. Check out the interactive HTML files")
    print("   2. Use PNGs in your research paper")
    print("   3. Customize colors/angles: edit demo_3d_viz.py")
    print("   4. Integrate into torus.py: python torus.py --method roadmap")
    print(f"\n   All outputs: {OUT_DIR}/")


if __name__ == "__main__":
    demo_sphere()
    demo_torus()
    demo_custom_colormaps()
    print_summary()
