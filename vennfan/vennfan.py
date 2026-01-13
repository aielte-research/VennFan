#!/usr/bin/env python3
"""
"Venn fan" circular version of the sine-curve Venn diagrams.

This maps the rectangular half-plane picture to a disc:
    * y = 0   → circle of radius R
    * y > 0   → inside the circle
    * y < 0   → outside the circle

Exposed API
-----------
- vennfan(...): main plotting function
"""

from typing import Sequence, Optional, Union, Tuple, Dict, Callable, List
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.ndimage import distance_transform_edt

from .colors import _rgb
from .defaults import (
    default_palette_for_n,
    default_fontsize,
    default_linewidth,
)
from .curves import get_sine_curve, get_cosine_curve, vennfan_find_extrema
from .utils import (
    disjoint_region_masks,
    visual_center,
    arc_angle_for_region,
    halfplane_to_disc,
    second_radial_intersection,
    radial_segment_center_for_region,
    normalize_angle_90,
    class_label_angles,
    resolve_color_mixing,
    region_label_mode_for_key,
    shrink_text_font_to_region,
    text_color_for_region,
)


def vennfan(
    # Core data
    values,
    class_names: Sequence[str],
    title: Optional[str] = None,
    outfile: Optional[str] = None,
    # Colors
    colors: Optional[Sequence[Union[str, tuple]]] = None,
    outline_colors: Optional[Sequence[Union[str, tuple]]] = None,
    color_mixing: Union[str, Callable] = "average",
    text_color: Optional[str] = None,
    highlight_colors: Optional[float] = None,
    # Boundary curves / geometry
    curve_mode: str = "cosine",
    p: float = 0.33,
    decay: str = "linear",  # "linear" or "exponential"
    epsilon: Optional[float] = None,
    delta: Optional[float] = None,
    b: float = 0.8,
    y_min: float = -1.0,
    y_max: float = 1.0,
    # Fonts & layout
    dpi: Optional[int] = None,
    region_fontsize: Optional[float] = None,
    radial_region_fontsize: Optional[float] = None,
    class_label_fontsize: Optional[float] = None,
    complement_fontsize: float = 8.0,
    linewidth: Optional[float] = None,
    region_label_placement: Optional[str] = None,
    region_radial_offset_inside: Optional[float] = None,
    region_radial_offset_outside: Optional[float] = None,
    last_constant_label_offset: Tuple[float, float] = (0.0, 0.0),
    # Radial behaviour / extras
    radial_bias: Optional[float] = None,
    draw_tight_factor: Optional[float] = None,
    # Visual-center extras
    visual_center_rotate_toward_radial: bool = False,
    # Rectangle-based visual-text-center width (kept as parameter)
    visual_text_center_width: Optional[float] = None,
    # Area fraction target for erosion in visual_text_center mode
    visual_text_center_area_fraction: float = 0.5,
    # Disc grid resolution control
    disc_points_per_class: int = 200,
    # Debugging for visual_text_center (yellow cores, red boxes, cyan line)
    visual_text_center_debug: bool = False,
) -> Optional[Figure]:
    """
    Draw a "Venn fan" diagram (circular version of sine/cosine Venn diagrams).

    Parameters
    ----------
    values :
        N-dimensional array of shape (2, 2, ..., 2) (length-N) containing
        the values/labels for each region. The index is a length-N 0/1 tuple,
        e.g. (1,0,1,...) means “in sets 0 and 2 but not in others”.
        If a cell is None, that region is left unlabeled.

    class_names : Sequence[str]
        Names of the N sets. Used as class labels on the outer ring.
        Empty strings ("") suppress individual class labels.

    title : str, optional
        Figure title. If None, no title is added.

    outfile : str, optional
        If None, the function returns the Matplotlib Figure.
        If a path is provided, the figure is saved there and the function
        returns None.

    Colors
    ------
    colors : sequence of colors, optional
        Per-class fill colors. If shorter than N, it is repeated cyclically.
        If None, uses `default_palette_for_n(N)[0]`.

    outline_colors : sequence of colors, optional
        Per-class outline colors for the boundary curves and class labels.
        If shorter than N, it is repeated cyclically.
        If None, uses `default_palette_for_n(N)[1]`.

    color_mixing : {"average", "subtractive", "hue_average", "alpha_stack"} or callable
        How to combine the per-class colors into the region color.
        - If a string, selects one of the built-in modes.
        - If a callable, it must behave like:
              mixed = f(colors_for_key, present)
          where
              colors_for_key : list of RGB triples for the *present* classes,
              present        : full boolean membership list of length N
                               for that region (True if the class is present).

    text_color : str, optional
        If not None, overrides all automatically chosen region text colors.
        If None, region label text color is derived from the region fill color
        via `text_color_for_region`.

    highlight_colors : float in [0, 1] or None
        Optional radial "halo" / highlight effect inside each region using
        multiple erosions by area fractions 0.9, 0.8, ..., 0.1:

            - None:
                * no extra highlighting overlay is drawn.
            - 0.0:
                * draw all erosion levels, but always with the original region
                  color (no whitening), so visually it changes almost nothing.
            - 1.0:
                * deepest erosion (0.1 area) goes all the way to white,
                  intermediate erosions become progressively lighter mixes
                  between region color and white.

        For intermediate values in (0, 1), the maximum whiteness is scaled
        accordingly.

    Boundary curves / geometry
    --------------------------
    curve_mode : {"cosine", "sine"}
        Base trigonometric curve family used for the classes.

    p : float
        Exponent controlling how amplitude/shape varies with harmonic index.

    decay : {"linear", "exponential"}
        Amplitude-decay mode for the curves.

    epsilon : float or None
        Small vertical offset passed to the curve function.

    delta : float or None
        Optional decay parameter passed to the curve function.

    b : float
        Optional decay base parameter passed to the curve function.

    y_min, y_max : float
        Vertical extent of the rectangular half-plane before mapping to
        the disc.

    Fonts & layout
    --------------
    dpi : int, optional
        Figure DPI. If None, defaults to `100 * N`.

    region_fontsize : float, optional
        Base font size for region labels placed using visual centers.

    radial_region_fontsize : float, optional
        Font size for region labels placed in radial / hybrid modes.

    class_label_fontsize : float, optional
        Font size for the class names drawn around the outside of the fan.

    complement_fontsize : float
        Font size for the complement label.

    linewidth : float, optional
        Line width for the boundary curves. If None, uses `default_linewidth(N)`.

    region_label_placement :
        {"visual_center", "visual_text_center", "radial", "hybrid"} or None

        - None:
            * "visual_center" if `decay == "linear"`,
            * "radial" otherwise.
        - "visual_center":
            classic visual-center placement.
        - "visual_text_center":
            rectangle-based visual text center:
              * erode region until its area is
                ~visual_text_center_area_fraction * original_area
              * inside that eroded core, find the longest line
              * anchor label at its midpoint with that line's orientation
              * (optionally) draw:
                    - eroded cores in whitish-yellow
                    - the longest line in cyan
              * IMPORTANT: shrink_to_fit is always tested against the
                FULL region, not the eroded core.
        - "radial":
            always use radial placement along a ray.
        - "hybrid":
            per-region decision via `region_label_mode_for_key(...)` between
            "visual_center" and "radial".

    visual_center_rotate_toward_radial : bool
        If True and placement is "visual_center", multi-character labels get
        rotated toward that region's radial anchor on the main circle.

    visual_text_center_width : float, optional
        Fixed rectangle width (data units). Currently used only as a gate
        (if <= 0, we skip the rectangle-based search and fall back to
        plain visual-center).

    visual_text_center_area_fraction : float
        Target area fraction for the eroded region in visual_text_center mode.
        Must be in (0, 1]. Default is 0.5.

    disc_points_per_class : int
        Controls the resolution of the disc grid used for region masks.
        Actual grid resolution along each axis is:

            grid_size = N * disc_points_per_class

        Increasing this (e.g. 200, 250, 300) refines the mask and can reduce
        over-aggressive shrinking in small regions at the cost of runtime
        and memory.

    visual_text_center_debug : bool
        If True, enables visual debugging for visual_text_center:
            - eroded-region cores overdrawn in whitish-yellow
            - oriented text bounding boxes (from shrink_text_font_to_region)
              drawn in red
            - longest line segment used for anchoring drawn in cyan.

        If False, all these debug elements are suppressed.

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        If `outfile` is None, returns the Figure. Otherwise, saves the figure
        to `outfile` and returns None.
    """
    # Validate decay mode
    decay = str(decay).lower()
    if decay not in ("linear", "exponential"):
        raise ValueError("decay must be 'linear' or 'exponential'.")

    # Linear vs decaying amplitude regime
    linear_decay = decay == "linear"

    if curve_mode not in ("cosine", "sine"):
        raise ValueError(f"Unsupported curve_mode {curve_mode!r}; use 'cosine' or 'sine'.")

    # Validate / normalize radial_bias
    if radial_bias is not None:
        if not (0.0 <= float(radial_bias) <= 1.0):
            raise ValueError("radial_bias must be in the range [0, 1] or None.")
        radial_bias = float(radial_bias)

    # Validate visual_text_center_area_fraction
    frac = float(visual_text_center_area_fraction)
    if not (0.0 < frac <= 1.0):
        raise ValueError("visual_text_center_area_fraction must be in (0, 1].")
    visual_text_center_area_fraction = frac

    # Validate disc_points_per_class
    disc_points_per_class = int(disc_points_per_class)
    if disc_points_per_class <= 0:
        raise ValueError("disc_points_per_class must be a positive integer.")

    # Validate highlight_colors
    if highlight_colors is not None:
        highlight_factor = float(highlight_colors)
        if not (0.0 <= highlight_factor <= 1.0):
            raise ValueError("highlight_colors must be in [0, 1] or None.")
    else:
        highlight_factor = None

    if curve_mode == "sine":
        curve_fn = get_sine_curve
    else:
        curve_fn = get_cosine_curve

    # ---- Input checks ------------------------------------------------------
    arr = np.asarray(values, dtype=object)
    if arr.ndim < 1 or arr.ndim > 9:
        raise ValueError("Only N in {1,2,...,9} are supported.")
    N = arr.ndim
    if dpi is None:
        dpi = 100 * N
    expected_shape = (2,) * N
    if arr.shape != expected_shape:
        raise ValueError(f"values must have shape {expected_shape}, got {arr.shape}.")
    if len(class_names) != N:
        raise ValueError(f"class_names must have length {N}.")
    if N > 9:
        raise ValueError("N>9 not supported.")

    zeros = (0,) * N
    ones = (1,) * N  # kept for completeness if needed later

    # ---- Region label placement mode ---------------------------------------
    if region_label_placement is None:
        region_label_placement = "visual_center" if linear_decay else "radial"
    else:
        region_label_placement = str(region_label_placement).lower()
    if region_label_placement not in (
        "radial",
        "visual_center",
        "visual_text_center",
        "hybrid",
    ):
        raise ValueError(
            "region_label_placement must be one of "
            "'radial', 'visual_center', 'visual_text_center', or 'hybrid'."
        )

    # Default rectangle width for visual_text_center (kept for API)
    if visual_text_center_width is None:
        visual_text_center_width = float(np.sin(4.0 * np.pi / (2.0 ** N)))

    # Offsets: only float or None
    if region_radial_offset_inside is not None and not isinstance(
        region_radial_offset_inside, (int, float)
    ):
        raise TypeError("region_radial_offset_inside must be a float or None.")
    if region_radial_offset_outside is not None and not isinstance(
        region_radial_offset_outside, (int, float)
    ):
        raise TypeError("region_radial_offset_outside must be a float or None.")

    # ---- Linewidth & colors ------------------------------------------------
    if linewidth is None:
        linewidth = default_linewidth(N)

    default_fills, default_outlines = default_palette_for_n(N)

    # Region fill colors
    if colors is None:
        colors = default_fills
    elif len(colors) < N:
        colors = [colors[i % len(colors)] for i in range(N)]
    rgbs = list(map(_rgb, colors))

    # Outline & label colors
    if outline_colors is None:
        outline_colors = default_outlines

    if len(outline_colors) < N:
        line_colors = [outline_colors[i % len(outline_colors)] for i in range(N)]
    else:
        line_colors = list(outline_colors)
    label_rgbs = [_rgb(c) for c in line_colors]

    # ---- Font sizes --------------------------------------------------------
    if region_fontsize is None or class_label_fontsize is None:
        base_fs_region, base_fs_class = default_fontsize(N, linear_decay, curve_mode)
        if region_fontsize is None:
            region_fontsize = base_fs_region
        if class_label_fontsize is None:
            class_label_fontsize = base_fs_class

    # Radial region labels: explicit size or fall back to region_fontsize
    if radial_region_fontsize is None:
        radial_region_fontsize = region_fontsize
    fs_radial = float(radial_region_fontsize)
    fs_region = float(region_fontsize)

    # ---- Color mixing callback ---------------------------------------------
    mixing_cb = resolve_color_mixing(color_mixing, N)

    # ---- Base domain & disc grid -------------------------------------------
    x_min, x_max = 0.0, 2.0 * np.pi

    R = 1.0
    R_out = 2.0 * R

    grid_n = max(50, disc_points_per_class)  # safety lower bound
    us = np.linspace(-R_out, R_out, N * grid_n)
    vs = np.linspace(-R_out, R_out, N * grid_n)
    U, V = np.meshgrid(us, vs)

    rho = np.sqrt(U * U + V * V)
    theta = np.mod(np.arctan2(V, U), 2.0 * np.pi)

    # Map disc grid back to half-plane coordinates (x_old, y_old)
    x_old = theta.copy()
    y_old = np.full_like(U, y_min - 1.0)

    y_pos_max = float(max(0.0, y_max))
    y_neg_min = float(min(0.0, y_min))

    inside_disc = (rho <= R)
    ring = (rho > R) & (rho <= R_out)

    t_in = np.zeros_like(rho)
    if y_pos_max != 0:
        t_in[inside_disc] = rho[inside_disc] / R
        y_old[inside_disc] = y_pos_max * (1.0 - t_in[inside_disc])
    else:
        y_old[inside_disc] = 0.0

    t_out = np.zeros_like(rho)
    denom = (R_out - R)
    if y_neg_min != 0:
        t_out[ring] = (rho[ring] - R) / denom
        y_old[ring] = y_neg_min * t_out[ring]
    else:
        y_old[ring] = y_min

    # ---- Membership masks on the disc grid --------------------------------
    membership: List[np.ndarray] = []

    for i in range(N):
        curve = curve_fn(
            x_old,
            i,
            N,
            p=p,
            decay=decay,
            epsilon=epsilon,
            delta=delta,
            b=b,
        )
        mask = y_old >= curve
        membership.append(mask)

    # ---- Disjoint region masks ---------------------------------------------
    region_masks = disjoint_region_masks(membership)
    H, W = U.shape

    # ---- Region areas (only for completeness) ------------------------------
    if us.size > 1:
        du = us[1] - us[0]
    else:
        du = (R_out - (-R_out)) / max(W - 1, 1)
    if vs.size > 1:
        dv = vs[1] - vs[0]
    else:
        dv = (R_out - (-R_out)) / max(H - 1, 1)
    pixel_area = abs(du * dv)  # currently unused, but conceptually there

    # For class labels we still want a small outward offset by default
    if region_radial_offset_outside is None:
        base_offset_outside_cls = 0.05
    else:
        base_offset_outside_cls = float(region_radial_offset_outside)

    # ---- Region RGBA image -------------------------------------------------
    rgba = np.zeros((H, W, 4), float)
    region_rgbs: Dict[Tuple[int, ...], np.ndarray] = {}

    # Optional highlight overlay (multi-level erosion-based whitening)
    highlight_rgba = None
    if highlight_factor is not None:
        highlight_rgba = np.zeros_like(rgba)

    for key, mask in region_masks.items():
        if not any(key):
            continue  # complement skipped
        if not mask.any():
            continue

        colors_for_key = [rgbs[i] for i, bit in enumerate(key) if bit]
        present = [bool(bit) for bit in key]

        mixed_rgb = np.asarray(mixing_cb(colors_for_key, present), float)
        if mixed_rgb.shape != (3,):
            raise ValueError("color_mixing callback must return an RGB array of shape (3,).")

        region_rgbs[key] = mixed_rgb
        rgba[mask, 0] = mixed_rgb[0]
        rgba[mask, 1] = mixed_rgb[1]
        rgba[mask, 2] = mixed_rgb[2]
        rgba[mask, 3] = 1.0

        # --- Optional multi-level highlight for this region -----------------
        if highlight_rgba is not None:
            dist = distance_transform_edt(mask)
            orig_area = int(mask.sum())
            if orig_area > 0:
                dvals = dist[mask].ravel()
                if dvals.size > 0:
                    dsorted = np.sort(dvals)  # ascending
                    # area fractions 0.9, 0.8, ..., 0.1
                    fracs = np.arange(0.98, 0.0, -0.02)
                    f_max = fracs[0]
                    f_min = fracs[-1]
                    rng = f_max - f_min if f_max > f_min else 1.0

                    base_rgb = mixed_rgb
                    white = np.array([1.0, 1.0, 1.0], float)

                    for f in fracs:
                        # keep fraction f of the most interior pixels
                        target_pixels = max(1, int(round(orig_area * f)))
                        idx = max(0, min(orig_area - 1, orig_area - target_pixels))
                        threshold = float(dsorted[idx])
                        core_mask = (dist >= threshold) & mask
                        if not core_mask.any():
                            continue

                        # whiteness grows as f decreases (more interior)
                        t = (f_max - f) / rng
                        whiteness = highlight_factor * t

                        new_rgb = (1.0 - whiteness) * base_rgb + whiteness * white

                        highlight_rgba[core_mask, 0] = new_rgb[0]
                        highlight_rgba[core_mask, 1] = new_rgb[1]
                        highlight_rgba[core_mask, 2] = new_rgb[2]
                        highlight_rgba[core_mask, 3] = 1.0

    # Optional eroded-overlay image for visual_text_center testing
    eroded_rgba = None
    if region_label_placement == "visual_text_center" and visual_text_center_debug:
        eroded_rgba = np.zeros_like(rgba)

    # ---- Figure and axes ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.set_dpi(dpi)
    ax.imshow(
        rgba,
        origin="lower",
        extent=[-R_out, R_out, -R_out, R_out],
        interpolation="nearest",
        zorder=1,
    )

    # Overlay highlight colors, if requested
    if highlight_rgba is not None and np.any(highlight_rgba[..., 3] > 0):
        ax.imshow(
            highlight_rgba,
            origin="lower",
            extent=[-R_out, R_out, -R_out, R_out],
            interpolation="nearest",
            zorder=1.2,
        )

    ax.set_xlim(-R_out, R_out)
    ax.set_ylim(-R_out, R_out)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.margins(0.0, 0.0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # ---- Optional: tighten drawing range using analytic extrema ------------
    if draw_tight_factor is not None:
        xmn, xmx, ymn, ymx = vennfan_find_extrema(
            curve_mode=curve_mode,
            p=p,
            decay=decay,
            N=N,
            epsilon=epsilon,
            delta=delta,
            b=b,
        )
        f = float(draw_tight_factor)
        ax.set_xlim(xmn * f, xmx * f)
        ax.set_ylim(ymn * f, ymx * f)

    # ---- Class boundaries in vennfan plane ---------------------------------
    x_plot = np.linspace(x_min, x_max, 1000 * N)
    curves: List[np.ndarray] = []
    disc_u: List[np.ndarray] = []
    disc_v: List[np.ndarray] = []

    for i in range(N):
        y_plot = curve_fn(
            x_plot,
            i,
            N,
            p=p,
            decay=decay,
            epsilon=epsilon,
            delta=delta,
            b=b,
        )
        curves.append(y_plot)

        u_curve, v_curve = halfplane_to_disc(x_plot, y_plot, R, y_min, y_max)
        disc_u.append(u_curve)
        disc_v.append(v_curve)

    # Draw each class outline twice: alpha 1.0 then 0.5
    for pass_alpha in (1.0, 0.5):
        for i in range(N):
            u_curve = disc_u[i]
            v_curve = disc_v[i]
            ax.plot(
                u_curve,
                v_curve,
                color=line_colors[i],
                linewidth=linewidth,
                alpha=pass_alpha,
                zorder=4,
            )
            # Close curve if endpoints don't coincide
            if u_curve.size > 1:
                du_c = u_curve[0] - u_curve[-1]
                dv_c = v_curve[0] - v_curve[-1]
                if du_c * du_c + dv_c * dv_c > 1e-10:
                    ax.plot(
                        [u_curve[-1], u_curve[0]],
                        [v_curve[-1], v_curve[0]],
                        color=line_colors[i],
                        linewidth=linewidth,
                        alpha=pass_alpha,
                        zorder=4,
                    )

    # ---- Precompute for region label placement -----------------------------
    rho = np.sqrt(U * U + V * V)
    circle_band = np.abs(rho - R) <= (0.03 * R)

    # Ensure renderer exists for text extent calculations
    fig.canvas.draw()

    erosion_radius_pix = linewidth * 1.0

    # ---- Region labels (visual_center / visual_text_center / radial) -------
    for key, mask in region_masks.items():
        if not mask.any():
            continue
        if key == zeros:
            continue  # complement handled separately

        value = arr[key]
        if value is None:
            continue

        # Decide per-region mode
        if region_label_placement == "visual_text_center":
            placement_mode = "visual_text_center"
        else:
            placement_mode = region_label_mode_for_key(
                key=key,
                N=N,
                region_label_placement=region_label_placement,
            )

        if placement_mode == "visual_center":
            # --- Plain visual-center placement in disc coordinates -----------
            pos = visual_center(mask, U, V)
            if pos is None:
                continue

            this_color = text_color_for_region(key, region_rgbs, text_color)
            fs_here = fs_region
            u_lab, v_lab = pos

            # Default: horizontal
            rot = 0.0

            # Optional: rotate toward radial anchor on the main circle
            if visual_center_rotate_toward_radial and len(str(value)) > 1:
                angle_raw = arc_angle_for_region(mask, circle_band, theta, U, V)
                if angle_raw is not None:
                    u_anchor = R * np.cos(angle_raw)
                    v_anchor = R * np.sin(angle_raw)
                    du_vec = u_anchor - u_lab
                    dv_vec = v_anchor - v_lab
                    if not (du_vec == 0.0 and dv_vec == 0.0):
                        deg_raw = np.degrees(np.arctan2(dv_vec, du_vec))
                        rot = normalize_angle_90(deg_raw)

            ha = "center"
            va = "center"

            # Shrink against the ORIGINAL region `mask`
            fs_adj = shrink_text_font_to_region(
                fig,
                ax,
                f"{value}",
                u_lab,
                v_lab,
                fs_here,
                mask,
                U,
                V,
                rotation=rot,
                ha=ha,
                va=va,
                erosion_radius_pix=erosion_radius_pix,
                debug_mode=visual_text_center_debug,
            )

            ax.text(
                u_lab,
                v_lab,
                f"{value}",
                ha=ha,
                va=va,
                fontsize=fs_adj,
                color=this_color,
                zorder=5,
                rotation=rot,
                rotation_mode="anchor",
            )

        elif placement_mode == "visual_text_center":
            # --- Area-based erosion + longest-line anchor, with debug overlay ---
            this_color = text_color_for_region(key, region_rgbs, text_color)
            fs_here = fs_region

            # Default fallback: plain visual center
            pos = visual_center(mask, U, V)
            if pos is None:
                continue
            u_opt, v_opt = pos
            rot_opt = 0.0
            debug_line_segment = None  # ((x_minus, y_minus), (x_plus, y_plus)) if available

            width = float(visual_text_center_width)
            if width > 0.0:
                # Distance transform inside the region
                dist = distance_transform_edt(mask)
                orig_area = int(mask.sum())
                if orig_area > 0:
                    # Choose threshold so that eroded area ≈ fraction * original_area
                    target_pixels = max(1, int(round(orig_area * visual_text_center_area_fraction)))
                    dvals = dist[mask].ravel()
                    if dvals.size > 0:
                        dsorted = np.sort(dvals)  # ascending
                        # keep the pixels with the largest distances
                        idx = max(0, min(orig_area - 1, orig_area - target_pixels))
                        threshold = float(dsorted[idx])
                        core_mask = (dist >= threshold) & mask

                        # Plot eroded core in whitish-yellow for debugging
                        if eroded_rgba is not None and core_mask.any():
                            eroded_rgba[core_mask, 0] = 1.0  # R
                            eroded_rgba[core_mask, 1] = 1.0  # G
                            eroded_rgba[core_mask, 2] = 0.7  # B
                            eroded_rgba[core_mask, 3] = 0.8  # A

                        if core_mask.any():
                            # Within core, find the most interior pixel (by dist)
                            dist_core = dist.copy()
                            dist_core[~core_mask] = -1.0
                            max_core = dist_core.max()
                            if max_core > 0.0:
                                iy0, ix0 = np.unravel_index(
                                    np.argmax(dist_core),
                                    dist_core.shape,
                                )
                                x_anchor = float(U[iy0, ix0])
                                y_anchor = float(V[iy0, ix0])

                                xs_grid = U[0, :]
                                ys_grid = V[:, 0]

                                if xs_grid.size > 1:
                                    dxg = xs_grid[1] - xs_grid[0]
                                    x0_grid = xs_grid[0]
                                else:
                                    dxg = 1.0
                                    x0_grid = float(xs_grid[0]) if xs_grid.size else 0.0

                                if ys_grid.size > 1:
                                    dyg = ys_grid[1] - ys_grid[0]
                                    y0_grid = ys_grid[0]
                                else:
                                    dyg = 1.0
                                    y0_grid = float(ys_grid[0]) if ys_grid.size else 0.0

                                step_x = abs(dxg)
                                step_y = abs(dyg)
                                step = float(min(step_x, step_y))
                                if step <= 0.0:
                                    step = 1.0

                                span_x = float(U.max() - U.min()) if U.size else 1.0
                                span_y = float(V.max() - V.min()) if V.size else 1.0
                                max_span = float(np.hypot(span_x, span_y))
                                max_steps = max(1, int(max_span / step) + 3)

                                n_angles = 72
                                angles = np.linspace(0.0, np.pi, int(n_angles), endpoint=False)

                                best_length = -1.0
                                best_center_x = x_anchor
                                best_center_y = y_anchor
                                best_angle = 0.0
                                best_plus_x = x_anchor
                                best_plus_y = y_anchor
                                best_minus_x = x_anchor
                                best_minus_y = y_anchor

                                for theta_line in angles:
                                    ct = float(np.cos(theta_line))
                                    st = float(np.sin(theta_line))

                                    # Forward (+)
                                    x_plus = x_anchor
                                    y_plus = y_anchor
                                    for _ in range(max_steps):
                                        x_try = x_plus + step * ct
                                        y_try = y_plus + step * st
                                        ix = int(round((x_try - x0_grid) / dxg)) if xs_grid.size > 1 else ix0
                                        iy = int(round((y_try - y0_grid) / dyg)) if ys_grid.size > 1 else iy0
                                        if ix < 0 or ix >= W or iy < 0 or iy >= H:
                                            break
                                        if not core_mask[iy, ix]:
                                            break
                                        x_plus = x_try
                                        y_plus = y_try

                                    # Backward (-)
                                    x_minus = x_anchor
                                    y_minus = y_anchor
                                    for _ in range(max_steps):
                                        x_try = x_minus - step * ct
                                        y_try = y_minus - step * st
                                        ix = int(round((x_try - x0_grid) / dxg)) if xs_grid.size > 1 else ix0
                                        iy = int(round((y_try - y0_grid) / dyg)) if ys_grid.size > 1 else iy0
                                        if ix < 0 or ix >= W or iy < 0 or iy >= H:
                                            break
                                        if not core_mask[iy, ix]:
                                            break
                                        x_minus = x_try
                                        y_minus = y_try

                                    cx = 0.5 * (x_plus + x_minus)
                                    cy = 0.5 * (y_plus + y_minus)
                                    length = float(np.hypot(x_plus - x_minus, y_plus - y_minus))

                                    if length > best_length:
                                        best_length = length
                                        best_center_x = cx
                                        best_center_y = cy
                                        best_angle = theta_line
                                        best_plus_x = x_plus
                                        best_plus_y = y_plus
                                        best_minus_x = x_minus
                                        best_minus_y = y_minus

                                if best_length > 0.0:
                                    u_opt = best_center_x
                                    v_opt = best_center_y
                                    angle_deg = float(np.degrees(best_angle))
                                    rot_opt = normalize_angle_90(angle_deg)
                                    debug_line_segment = (
                                        (best_minus_x, best_minus_y),
                                        (best_plus_x, best_plus_y),
                                    )

            ha = "center"
            va = "center"

            # Shrink against ORIGINAL `mask`, not the eroded core.
            fs_adj = shrink_text_font_to_region(
                fig,
                ax,
                f"{value}",
                u_opt,
                v_opt,
                fs_here,
                mask,  # full region
                U,
                V,
                rotation=rot_opt,
                ha=ha,
                va=va,
                erosion_radius_pix=erosion_radius_pix,
                debug_mode=visual_text_center_debug,
            )

            ax.text(
                u_opt,
                v_opt,
                f"{value}",
                ha=ha,
                va=va,
                fontsize=fs_adj,
                color=this_color,
                zorder=5,
                rotation=rot_opt if len(value) > 1 else 0,
                rotation_mode="anchor",
            )

            # Debug: draw the longest line in cyan
            if visual_text_center_debug and debug_line_segment is not None:
                (xm, ym), (xp, yp) = debug_line_segment
                ax.plot(
                    [xm, xp],
                    [ym, yp],
                    color="cyan",
                    linewidth=0.8,
                    zorder=4.5,
                )

        else:
            # --- Radial placement branch ------------------------------------
            last_bit = key[-1]

            angle_raw = arc_angle_for_region(mask, circle_band, theta, U, V)
            if angle_raw is None:
                continue

            v_out = np.array([np.cos(angle_raw), np.sin(angle_raw)], float)

            if radial_bias is None:
                # Fixed-offset behaviour near the main circle (legacy style)
                off_in = (
                    float(region_radial_offset_inside)
                    if region_radial_offset_inside is not None
                    else 0.05
                )
                off_out = (
                    float(region_radial_offset_outside)
                    if region_radial_offset_outside is not None
                    else 0.05
                )
                r_lab = R * (1.0 - off_in) if last_bit == 1 else R * (1.0 + off_out)
            else:
                # Radial centering using radial_bias
                r_mid = radial_segment_center_for_region(
                    mask=mask,
                    angle_rad=angle_raw,
                    u_min=us[0],
                    v_min=vs[0],
                    du_val=du,
                    dv_val=dv,
                    H_val=H,
                    W_val=W,
                    R_max=R_out,
                    radial_bias=radial_bias,
                    n_samples=1024,
                )

                if r_mid is None:
                    pos_vc = visual_center(mask, U, V)
                    if pos_vc is not None:
                        r_mid = float(np.hypot(pos_vc[0], pos_vc[1]))
                    else:
                        r_mid = R

                offset = 0.0
                if last_bit == 1 and region_radial_offset_inside is not None:
                    offset = -float(region_radial_offset_inside)
                elif last_bit == 0 and region_radial_offset_outside is not None:
                    offset = float(region_radial_offset_outside)

                r_lab = r_mid + offset

            u_lab = r_lab * v_out[0]
            v_lab = r_lab * v_out[1]

            deg_raw = np.degrees(angle_raw)
            rot = normalize_angle_90(deg_raw)
            rot_rad = np.deg2rad(rot)
            v_base = np.array([np.cos(rot_rad), np.sin(rot_rad)], float)

            if last_bit == 1:
                v_circle = v_out
            else:
                v_circle = -v_out

            d_align = float(np.dot(v_circle, v_base))

            # Alignment:
            # - In hybrid + biased mode with no extra offsets, center-align text.
            # - Otherwise, keep original right/left rule.
            if (
                region_label_placement == "hybrid"
                and radial_bias is not None
                and (
                    (last_bit == 1 and region_radial_offset_inside is None)
                    or (last_bit == 0 and region_radial_offset_outside is None)
                )
            ):
                ha = "center"
            else:
                ha = "right" if d_align >= 0 else "left"

            va = "center"

            this_color = text_color_for_region(key, region_rgbs, text_color)
            fs_here = fs_radial  # radial labels: fixed size (no shrink)

            ax.text(
                u_lab,
                v_lab,
                f"{value}",
                ha=ha,
                va=va,
                fontsize=fs_here,
                color=this_color,
                zorder=5,
                rotation=rot,
                rotation_mode="anchor",
            )

    # ---- Complement (all zeros) --------------------------------------------
    comp_mask = region_masks.get(zeros)
    if comp_mask is not None and comp_mask.any():
        val_comp = arr[zeros]
        if val_comp is not None:
            this_color = text_color if text_color is not None else "black"
            fs_comp = float(complement_fontsize)

            u_lab = R_out - 0.1
            v_lab = -R_out + 0.1
            rot = 0.0
            ha = "right"
            va = "bottom"

            # Complement label: protect with shrink-to-fit (original comp_mask)
            fs_adj = shrink_text_font_to_region(
                fig,
                ax,
                f"{val_comp}",
                u_lab,
                v_lab,
                fs_comp,
                comp_mask,
                U,
                V,
                rotation=rot,
                ha=ha,
                va=va,
                erosion_radius_pix=erosion_radius_pix,
                debug_mode=visual_text_center_debug,
            )

            ax.text(
                u_lab,
                v_lab,
                f"{val_comp}",
                ha=ha,
                va=va,
                fontsize=fs_adj,
                color=this_color,
                zorder=5,
                rotation=rot,
                rotation_mode="anchor",
            )

    # ---- Overlay eroded regions (for visual_text_center) -------------------
    if visual_text_center_debug and eroded_rgba is not None and np.any(eroded_rgba[..., 3] > 0):
        ax.imshow(
            eroded_rgba,
            origin="lower",
            extent=[-R_out, R_out, -R_out, R_out],
            interpolation="nearest",
            zorder=2,  # above base fill/highlight, below curves & text
        )

    # ---- Class labels on vennfan -------------------------------------------
    dx_const, dy_const = last_constant_label_offset
    extra_radial_offset = 0.05

    label_angle_degs = class_label_angles(N, curve_mode)

    for i, (name, label_col) in enumerate(zip(class_names, label_rgbs)):
        if not name:
            continue

        angle_deg_radial = label_angle_degs[i % len(label_angle_degs)]
        angle_anchor = np.deg2rad(angle_deg_radial)
        v_out = np.array([np.cos(angle_anchor), np.sin(angle_anchor)], float)

        u_curve = disc_u[i]
        v_curve = disc_v[i]
        inter = second_radial_intersection(u_curve, v_curve, angle_anchor)
        if inter is not None:
            u_int, v_int = inter
            r_anchor = float(np.sqrt(u_int * u_int + v_int * v_int))
        else:
            r_anchor = R

        r_lab = r_anchor + base_offset_outside_cls * R

        # Slightly different radial offsets for first few labels vs others
        if i >= 3:
            r_lab += extra_radial_offset * R
        else:
            r_lab += extra_radial_offset * R * 2

        # Optional shift for last (constant) label
        u_lab = r_lab * v_out[0] + (dx_const if i == N - 1 else 0.0)
        v_lab = r_lab * v_out[1] + (dy_const if i == N - 1 else 0.0)

        if i < 3:
            rot_cls = normalize_angle_90(angle_deg_radial - 90.0)
            ha = "center"
            va = "center"
        else:
            rot_cls = normalize_angle_90(angle_deg_radial)
            ha = "left"
            va = "center"

        ax.text(
            u_lab,
            v_lab,
            name,
            ha=ha,
            va=va,
            fontsize=class_label_fontsize,
            color=tuple(label_col),
            fontweight="bold",
            rotation=rot_cls,
            rotation_mode="anchor",
            zorder=6,
        )

    if title:
        ax.set_title(title)

    if outfile:
        fig.savefig(outfile, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return None

    return fig