#!/usr/bin/env python3
"""
Sine-curve "Venn" diagrams (rectangular version, over [0, 2π] × [-1, 1]).

This module exposes:

- venntrig(...): main plotting function
- simple test/demo code under `if __name__ == "__main__":`
"""

from typing import Sequence, Optional, Union, Tuple, Dict, Callable, List
import os
import yaml

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.ndimage import distance_transform_edt

from .colors import _rgb
from .defaults import default_palette_for_n
from .curves import get_sine_curve, get_cosine_curve
from .utils import (
    disjoint_region_masks,
    visual_center_margin,
    visual_center_inset,
    region_constant_line_bisector,
    exclusive_curve_bisector,
    shrink_text_font_to_region,
    harmonic_info_for_index,
    compute_region_fontsizes,
    resolve_color_mixing,
    text_color_for_region,
)

# ---- YAML defaults (non-color) ---------------------------------------------
DEFAULTS_YAML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venntrig_defaults.yaml")
if not os.path.exists(DEFAULTS_YAML_PATH):
    raise FileNotFoundError(f"Missing defaults YAML: {DEFAULTS_YAML_PATH}")

with open(DEFAULTS_YAML_PATH, "r", encoding="utf-8") as _f:
    DEFAULTS = yaml.safe_load(_f) or {}


def venntrig(
    values,
    class_names: Optional[Sequence[str]] = None,
    colors: Optional[Sequence[Union[str, tuple]]] = None,
    outline_colors: Optional[Sequence[Union[str, tuple]]] = None,
    title: Optional[str] = None,
    outfile: Optional[str] = None,
    dpi: int = 600,
    color_mixing: Union[str, Callable[[Sequence[np.ndarray]], np.ndarray]] = "alpha_stack",
    text_color: Optional[str] = None,
    region_label_placement: Optional[str] = None,
    region_label_fontsize: Optional[float] = None,
    class_label_fontsize: Optional[float] = None,
    complement_fontsize: float = 8.0,
    adaptive_fontsize: Optional[bool] = None,
    adaptive_fontsize_range: Optional[Tuple[float, float]] = None,
    sample_res_x: int = 3142,
    sample_res_y: int = 1000,
    include_constant_last: bool = True,
    p: float = 0.33,
    decay: str = "linear",  # "linear" or "exponential"
    epsilon: Optional[float] = None,
    delta: Optional[float] = None,
    b: float = 0.8,
    linewidth: Optional[float] = None,
    curve_mode: str = "sine",
    height_scale: float = 2.0,
    visual_text_center_width: Optional[float] = None,
    visual_text_center_area_fraction: float = 0.5,
    visual_text_center_debug: bool = False,
) -> Optional[Figure]:
    """
    Rectangular version of the sine-curve Venn diagram.
    """
    # ---- Basic input checks -------------------------------------------------
    arr = np.asarray(values, dtype=object)
    if arr.ndim < 1 or arr.ndim > 9:
        raise ValueError("Only N in {1,2,...,9} are supported.")
    N = arr.ndim
    expected_shape = (2,) * N
    if arr.shape != expected_shape:
        raise ValueError(f"values must have shape {expected_shape}, got {arr.shape}.")

    if class_names is None:
        class_names = ["" for _ in range(N)]
    if len(class_names) != N:
        raise ValueError(f"class_names must have length {N}.")
    if N > 9:
        raise ValueError("N>9 not supported.")

    curve_mode = str(curve_mode).lower()
    if curve_mode not in ("sine", "cosine"):
        raise ValueError("curve_mode must be 'sine' or 'cosine'.")

    zeros = (0,) * N
    ones = (1,) * N

    # linear_scale is redundant; infer it from decay
    linear_scale = (decay == "linear")

    # Validate visual_text_center_area_fraction
    frac = float(visual_text_center_area_fraction)
    if not (0.0 < frac <= 1.0):
        raise ValueError("visual_text_center_area_fraction must be in (0, 1].")
    visual_text_center_area_fraction = frac

    # ---- Region label placement mode ---------------------------------------
    if region_label_placement is None:
        region_label_placement = "visual_text_center" if linear_scale else "radial"
    else:
        region_label_placement = str(region_label_placement).lower()
    if region_label_placement not in ("radial", "visual_center", "visual_text_center"):
        raise ValueError(
            "region_label_placement must be one of 'radial', 'visual_center', 'visual_text_center'."
        )

    # Default rectangle width gate for visual_text_center (kept as parameter)
    if visual_text_center_width is None:
        visual_text_center_width = float(np.sin(4.0 * np.pi / (2.0 ** N)))

    # ---- Linewidth & colors -------------------------------------------------
    if linewidth is None:
        linewidth = float(DEFAULTS["linewidths"][N])

    # Default palette for this N (color defaults stay in defaults.py)
    default_fills, default_outlines = default_palette_for_n(N)

    # Fill colors for regions
    if colors is None:
        colors = default_fills
    elif len(colors) < N:
        colors = [colors[i % len(colors)] for i in range(N)]
    rgbs = list(map(_rgb, colors))

    # Outline colors for curves + class labels
    if outline_colors is None:
        outline_colors = default_outlines

    if len(outline_colors) < N:
        line_colors = [outline_colors[i % len(outline_colors)] for i in range(N)]
    else:
        line_colors = list(outline_colors)
    label_rgbs = [_rgb(c) for c in line_colors]

    # ---- Font sizes ---------------------------------------------------------
    if region_label_fontsize is None or class_label_fontsize is None:
        # YAML uses linear/nonlinear; decay uses linear/exponential -> map
        region_tbl = DEFAULTS["fontsizes"]["region"][curve_mode]
        if decay == "linear":
            scale_key = "linear"
        else:
            scale_key = "nonlinear" if "nonlinear" in region_tbl else decay

        base_fs_region = float(region_tbl[scale_key][N])
        base_fs_class = float(DEFAULTS["fontsizes"]["class"][N])

        if region_label_fontsize is None:
            region_label_fontsize = base_fs_region
        if class_label_fontsize is None:
            class_label_fontsize = base_fs_class

    if adaptive_fontsize_range is None:
        lo, hi = DEFAULTS["adaptive_fontsize_range"][N]
        adaptive_fontsize_range = (float(lo), float(hi))

    # ---- Color mixing callback ---------------------------------------------
    mixing_cb = resolve_color_mixing(color_mixing, N)

    # ---- Sampling grid in the universe rectangle ---------------------------
    x_min, x_max = 0.0, 2.0 * np.pi
    y_min, y_max = -1.0, 1.0
    xs = np.linspace(x_min, x_max, int(sample_res_x))
    ys = np.linspace(y_min, y_max, int(sample_res_y))
    X, Y = np.meshgrid(xs, ys)

    # ---- Membership masks & per-class 1D curves on xs ----------------------
    membership: List[np.ndarray] = []
    curve_1d_list: List[np.ndarray] = []

    if curve_mode == "sine":
        curve_fn = get_sine_curve
    else:
        curve_fn = get_cosine_curve

    for i in range(N):
        curve_full = curve_fn(
            X,
            i,
            N,
            p=p,
            decay=decay,
            epsilon=epsilon,
            delta=delta,
            b=b,
        )
        mask = Y >= curve_full
        curve_1d = curve_full[0, :]
        membership.append(mask)
        curve_1d_list.append(curve_1d)

    # ---- Disjoint region masks ---------------------------------------------
    region_masks = disjoint_region_masks(membership)
    H, W = X.shape

    # ---- Region areas & adaptive font sizes --------------------------------
    if xs.size > 1:
        dx = xs[1] - xs[0]
    else:
        dx = (x_max - x_min) / max(W - 1, 1)
    if ys.size > 1:
        dy = ys[1] - ys[0]
    else:
        dy = (y_max - y_min) / max(H - 1, 1)
    pixel_area = abs(dx * dy)

    region_fontsizes, fs_min, fs_max, adaptive_fontsize_flag = compute_region_fontsizes(
        region_masks=region_masks,
        pixel_area=pixel_area,
        complement_key=zeros,
        base_region_fontsize=float(region_label_fontsize),
        N=N,
        linear_scale=linear_scale,
        adaptive_fontsize=adaptive_fontsize,
        adaptive_fontsize_range=adaptive_fontsize_range,
    )
    adaptive_fontsize = adaptive_fontsize_flag  # in case caller inspects it later

    # ---- Region RGBA image (for imshow) ------------------------------------
    rgba = np.zeros((H, W, 4), float)
    region_rgbs: Dict[Tuple[int, ...], np.ndarray] = {}

    for key, mask in region_masks.items():
        if not any(key):
            continue  # complement skipped
        if not mask.any():
            continue
        colors_for_key = [rgbs[i] for i, bit in enumerate(key) if bit]
        mixed_rgb = np.asarray(mixing_cb(colors_for_key), float)
        if mixed_rgb.shape != (3,):
            raise ValueError("color_mixing callback must return an RGB array of shape (3,).")
        region_rgbs[key] = mixed_rgb
        rgba[mask, 0] = mixed_rgb[0]
        rgba[mask, 1] = mixed_rgb[1]
        rgba[mask, 2] = mixed_rgb[2]
        rgba[mask, 3] = 1.0

    # Optional eroded-overlay image for visual_text_center debugging
    eroded_rgba = None
    if region_label_placement == "visual_text_center" and visual_text_center_debug:
        eroded_rgba = np.zeros_like(rgba)

    # ---- Figure and axes ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(5 + 2.5 * N, height_scale * (5 + 2.5 * N) / np.pi))
    fig.set_dpi(dpi)
    ax.imshow(
        rgba,
        origin="lower",
        extent=[x_min, x_max, y_min, y_max],
        interpolation="nearest",
        zorder=1,
    )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("auto")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.margins(0.0, 0.0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # ---- Class boundary curves (analytic, for outlines & labels) ----------
    x_plot = np.linspace(x_min, x_max, 1200)
    curves: List[np.ndarray] = []
    harmonics_for_class: List[Optional[float]] = []

    if curve_mode == "sine":
        curve_fn_plot = get_sine_curve
    else:
        curve_fn_plot = get_cosine_curve

    for i in range(N):
        h_i, _ = harmonic_info_for_index(i, N, include_constant_last)
        harmonics_for_class.append(h_i)

        y_plot = curve_fn_plot(
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

    # Draw class outlines in two passes: alpha 1.0 then 0.5 (for subtle halo)
    for pass_alpha in (1.0, 0.5):
        for i in range(N):
            ax.plot(
                x_plot,
                curves[i],
                color=line_colors[i],
                linewidth=linewidth,
                alpha=pass_alpha,
                zorder=4,
            )

    # ---- Last local maximum for last non-constant class (fallback anchor) ---
    last_max_x = None
    non_const_indices = [i for i, h in enumerate(harmonics_for_class) if h is not None]
    if non_const_indices:
        last_idx = non_const_indices[-1]
        y_last = curves[last_idx]
        dy_last = np.diff(y_last)
        sign_last = np.sign(dy_last)
        idx_max = None
        for j in range(1, len(sign_last)):
            if sign_last[j - 1] > 0 and sign_last[j] < 0:
                idx_max = j
        if idx_max is None:
            idx_max = int(np.argmax(y_last))
        last_max_x = x_plot[idx_max]

    # Ensure renderer exists for text extent calculations
    fig.canvas.draw()

    # ---- Region labels -----------------------------------------------------
    const_y = 0.0
    region_offset = 0.02 * (y_max - y_min)
    erosion_radius_pix = linewidth * 1.5

    def _normalize_angle_90(deg: float) -> float:
        # inline normalization to [-90, 90]
        while deg > 90.0:
            deg -= 180.0
        while deg < -90.0:
            deg += 180.0
        return deg

    # Helper precomputes for the visual_text_center stepping
    dxg = float(xs[1] - xs[0]) if xs.size > 1 else 1.0
    dyg = float(ys[1] - ys[0]) if ys.size > 1 else 1.0
    x0_grid = float(xs[0]) if xs.size else 0.0
    y0_grid = float(ys[0]) if ys.size else 0.0
    step = float(min(abs(dxg), abs(dyg))) if (dxg != 0.0 and dyg != 0.0) else 1.0
    max_span = float(np.hypot((x_max - x_min), (y_max - y_min)))
    max_steps = max(1, int(max_span / step) + 3)

    for key, mask in region_masks.items():
        # Skip complement (all zeros) and all-sets intersection here;
        # these get special handling further below.
        if key == zeros or key == ones:
            continue
        value = arr[key]
        if value is None or not mask.any():
            continue

        this_color = text_color_for_region(key, region_rgbs, text_color)
        fs_here = region_fontsizes.get(key, float(region_label_fontsize))

        if region_label_placement == "visual_center":
            pos = visual_center_inset(mask, X, Y, x_min, x_max, y_min, y_max, n_pix=2)
            if pos is None:
                continue
            x_lab, y_lab = pos
            rot = 0.0
            ha = "center"
            va = "center"

        elif region_label_placement == "visual_text_center":
            # --- Area-based erosion + longest-line anchor (rectangular version) ---
            # Fallback anchor: inset visual center
            pos = visual_center_inset(mask, X, Y, x_min, x_max, y_min, y_max, n_pix=2)
            if pos is None:
                continue
            x_opt, y_opt = pos
            rot_opt = 0.0
            debug_line_segment = None

            width = float(visual_text_center_width) if visual_text_center_width is not None else 1.0
            if width > 0.0:
                dist = distance_transform_edt(mask)
                orig_area = int(mask.sum())
                if orig_area > 0:
                    target_pixels = max(1, int(round(orig_area * visual_text_center_area_fraction)))
                    dvals = dist[mask].ravel()
                    if dvals.size > 0:
                        dsorted = np.sort(dvals)  # ascending
                        idx = max(0, min(orig_area - 1, orig_area - target_pixels))
                        threshold = float(dsorted[idx])
                        core_mask = (dist >= threshold) & mask

                        if eroded_rgba is not None and core_mask.any():
                            eroded_rgba[core_mask, 0] = 1.0
                            eroded_rgba[core_mask, 1] = 1.0
                            eroded_rgba[core_mask, 2] = 0.7
                            eroded_rgba[core_mask, 3] = 0.8

                        if core_mask.any():
                            dist_core = dist.copy()
                            dist_core[~core_mask] = -1.0
                            max_core = float(dist_core.max())
                            if max_core > 0.0:
                                iy0, ix0 = np.unravel_index(np.argmax(dist_core), dist_core.shape)
                                x_anchor = float(X[iy0, ix0])
                                y_anchor = float(Y[iy0, ix0])

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
                                    x_plus, y_plus = x_anchor, y_anchor
                                    for _ in range(max_steps):
                                        x_try = x_plus + step * ct
                                        y_try = y_plus + step * st
                                        ix = int(round((x_try - x0_grid) / dxg)) if dxg != 0.0 else ix0
                                        iy = int(round((y_try - y0_grid) / dyg)) if dyg != 0.0 else iy0
                                        if ix < 0 or ix >= W or iy < 0 or iy >= H:
                                            break
                                        if not core_mask[iy, ix]:
                                            break
                                        x_plus, y_plus = x_try, y_try

                                    # Backward (-)
                                    x_minus, y_minus = x_anchor, y_anchor
                                    for _ in range(max_steps):
                                        x_try = x_minus - step * ct
                                        y_try = y_minus - step * st
                                        ix = int(round((x_try - x0_grid) / dxg)) if dxg != 0.0 else ix0
                                        iy = int(round((y_try - y0_grid) / dyg)) if dyg != 0.0 else iy0
                                        if ix < 0 or ix >= W or iy < 0 or iy >= H:
                                            break
                                        if not core_mask[iy, ix]:
                                            break
                                        x_minus, y_minus = x_try, y_try

                                    cx = 0.5 * (x_plus + x_minus)
                                    cy = 0.5 * (y_plus + y_minus)
                                    length = float(np.hypot(x_plus - x_minus, y_plus - y_minus))

                                    if length > best_length:
                                        best_length = length
                                        best_center_x, best_center_y = cx, cy
                                        best_angle = float(theta_line)
                                        best_plus_x, best_plus_y = x_plus, y_plus
                                        best_minus_x, best_minus_y = x_minus, y_minus

                                if best_length > 0.0:
                                    x_opt, y_opt = best_center_x, best_center_y
                                    angle_deg = float(np.degrees(best_angle))
                                    rot_opt = _normalize_angle_90(angle_deg)
                                    debug_line_segment = ((best_minus_x, best_minus_y), (best_plus_x, best_plus_y))

            x_lab, y_lab = x_opt, y_opt
            rot = rot_opt
            ha = "center"
            va = "center"

            # Debug: draw the longest line segment in cyan
            if visual_text_center_debug and debug_line_segment is not None:
                (xm, ym), (xp, yp) = debug_line_segment
                ax.plot([xm, xp], [ym, yp], color="cyan", linewidth=0.8, zorder=4.5)

        else:
            # "radial" (legacy non-linear placement): constant-line bisector, rotated 90°
            last_bit = key[-1]
            bis = region_constant_line_bisector(mask, X, Y)
            if bis is not None:
                x_mid, y0 = bis  # y0 is 0.0
                x_lab = x_mid
                if last_bit == 1:
                    y_lab = y0 + region_offset
                    ha = "left"
                else:
                    y_lab = y0 - region_offset
                    ha = "right"
            else:
                pos = visual_center_inset(mask, X, Y, x_min, x_max, y_min, y_max, n_pix=2)
                if pos is None:
                    continue
                x_lab, y_lab = pos
                ha = "center"
            rot = 90.0
            va = "center"

        fs_adj = shrink_text_font_to_region(
            fig,
            ax,
            f"{value}",
            x_lab,
            y_lab,
            fs_here,
            mask,
            X,
            Y,
            rotation=rot,
            ha=ha,
            va=va,
            erosion_radius_pix=erosion_radius_pix,
            debug_mode=visual_text_center_debug,
        )

        ax.text(
            x_lab,
            y_lab,
            f"{value}",
            ha=ha,
            va=va,
            fontsize=fs_adj,
            color=this_color,
            zorder=5,
            rotation=rot if (region_label_placement != "visual_text_center" or len(str(value)) > 1) else 0.0,
            rotation_mode="anchor",
        )

    # ---- All-sets intersection (ones) --------------------------------------
    all_mask = np.logical_and.reduce(membership)
    if all_mask.any():
        val_all = arr[ones]
        if val_all is not None:
            fs_all = region_fontsizes.get(ones, float(region_label_fontsize))
            this_color = text_color_for_region(ones, region_rgbs, text_color)

            if region_label_placement == "visual_center":
                pos = visual_center_margin(all_mask, X, Y, margin_frac=0.05)
                if pos is None:
                    pos = visual_center_inset(all_mask, X, Y, x_min, x_max, y_min, y_max, n_pix=2)
                if pos is not None:
                    x_lab, y_lab = pos
                    rot = 0.0
                    ha = "center"
                    va = "center"

            elif region_label_placement == "visual_text_center":
                # Use the same visual_text_center logic as above (with fallback)
                pos = visual_center_margin(all_mask, X, Y, margin_frac=0.05)
                if pos is None:
                    pos = visual_center_inset(all_mask, X, Y, x_min, x_max, y_min, y_max, n_pix=2)
                if pos is None:
                    pos = (0.5 * (x_min + x_max), 0.0)

                x_opt, y_opt = float(pos[0]), float(pos[1])
                rot_opt = 0.0
                debug_line_segment = None

                width = float(visual_text_center_width) if visual_text_center_width is not None else 1.0
                if width > 0.0:
                    dist = distance_transform_edt(all_mask)
                    orig_area = int(all_mask.sum())
                    if orig_area > 0:
                        target_pixels = max(1, int(round(orig_area * visual_text_center_area_fraction)))
                        dvals = dist[all_mask].ravel()
                        if dvals.size > 0:
                            dsorted = np.sort(dvals)
                            idx = max(0, min(orig_area - 1, orig_area - target_pixels))
                            threshold = float(dsorted[idx])
                            core_mask = (dist >= threshold) & all_mask

                            if eroded_rgba is not None and core_mask.any():
                                eroded_rgba[core_mask, 0] = 1.0
                                eroded_rgba[core_mask, 1] = 1.0
                                eroded_rgba[core_mask, 2] = 0.7
                                eroded_rgba[core_mask, 3] = 0.8

                            if core_mask.any():
                                dist_core = dist.copy()
                                dist_core[~core_mask] = -1.0
                                max_core = float(dist_core.max())
                                if max_core > 0.0:
                                    iy0, ix0 = np.unravel_index(np.argmax(dist_core), dist_core.shape)
                                    x_anchor = float(X[iy0, ix0])
                                    y_anchor = float(Y[iy0, ix0])

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

                                        x_plus, y_plus = x_anchor, y_anchor
                                        for _ in range(max_steps):
                                            x_try = x_plus + step * ct
                                            y_try = y_plus + step * st
                                            ix = int(round((x_try - x0_grid) / dxg)) if dxg != 0.0 else ix0
                                            iy = int(round((y_try - y0_grid) / dyg)) if dyg != 0.0 else iy0
                                            if ix < 0 or ix >= W or iy < 0 or iy >= H:
                                                break
                                            if not core_mask[iy, ix]:
                                                break
                                            x_plus, y_plus = x_try, y_try

                                        x_minus, y_minus = x_anchor, y_anchor
                                        for _ in range(max_steps):
                                            x_try = x_minus - step * ct
                                            y_try = y_minus - step * st
                                            ix = int(round((x_try - x0_grid) / dxg)) if dxg != 0.0 else ix0
                                            iy = int(round((y_try - y0_grid) / dyg)) if dyg != 0.0 else iy0
                                            if ix < 0 or ix >= W or iy < 0 or iy >= H:
                                                break
                                            if not core_mask[iy, ix]:
                                                break
                                            x_minus, y_minus = x_try, y_try

                                        cx = 0.5 * (x_plus + x_minus)
                                        cy = 0.5 * (y_plus + y_minus)
                                        length = float(np.hypot(x_plus - x_minus, y_plus - y_minus))

                                        if length > best_length:
                                            best_length = length
                                            best_center_x, best_center_y = cx, cy
                                            best_angle = float(theta_line)
                                            best_plus_x, best_plus_y = x_plus, y_plus
                                            best_minus_x, best_minus_y = x_minus, y_minus

                                    if best_length > 0.0:
                                        x_opt, y_opt = best_center_x, best_center_y
                                        rot_opt = _normalize_angle_90(float(np.degrees(best_angle)))
                                        debug_line_segment = ((best_minus_x, best_minus_y), (best_plus_x, best_plus_y))

                x_lab, y_lab = x_opt, y_opt
                rot = rot_opt
                ha = "center"
                va = "center"

                if visual_text_center_debug and debug_line_segment is not None:
                    (xm, ym), (xp, yp) = debug_line_segment
                    ax.plot([xm, xp], [ym, yp], color="cyan", linewidth=0.8, zorder=4.5)

            else:
                bis = region_constant_line_bisector(all_mask, X, Y)
                if bis is not None:
                    x_mid, y0 = bis
                    x_lab = x_mid
                    y_lab = y0 + region_offset
                else:
                    pos = visual_center_inset(all_mask, X, Y, x_min, x_max, y_min, y_max, n_pix=2)
                    if pos is None:
                        x_lab, y_lab = (0.5 * (x_min + x_max), const_y + region_offset)
                    else:
                        x_lab, y_lab = pos
                rot = 90.0
                ha = "left"
                va = "center"

            fs_adj = shrink_text_font_to_region(
                fig,
                ax,
                f"{val_all}",
                x_lab,
                y_lab,
                fs_all,
                all_mask,
                X,
                Y,
                rotation=rot,
                ha=ha,
                va=va,
                erosion_radius_pix=erosion_radius_pix,
                debug_mode=visual_text_center_debug,
            )

            ax.text(
                x_lab,
                y_lab,
                f"{val_all}",
                ha=ha,
                va=va,
                fontsize=fs_adj,
                color=this_color,
                zorder=5,
                rotation=rot if (region_label_placement != "visual_text_center" or len(str(val_all)) > 1) else 0.0,
                rotation_mode="anchor",
            )

    # ---- Complement (all zeros) – fixed bottom-right corner label ----------
    comp_mask = np.logical_not(np.logical_or.reduce(membership))
    if comp_mask.any():
        val_comp = arr[zeros]
        if val_comp is not None:
            this_color = text_color if text_color is not None else "black"
            fs_comp = float(complement_fontsize)

            x_lab = x_max - 0.1
            y_lab = y_min + 0.1
            rot = 0.0
            ha = "right"
            va = "bottom"

            ax.text(
                x_lab,
                y_lab,
                f"{val_comp}",
                ha=ha,
                va=va,
                fontsize=fs_comp,
                color=this_color,
                zorder=5,
                rotation=rot,
                rotation_mode="anchor",
            )

    # ---- Overlay eroded regions (for visual_text_center debug) -------------
    if visual_text_center_debug and eroded_rgba is not None and np.any(eroded_rgba[..., 3] > 0):
        ax.imshow(
            eroded_rgba,
            origin="lower",
            extent=[x_min, x_max, y_min, y_max],
            interpolation="nearest",
            zorder=2,  # above base fill, below curves & text
        )

    # ---- Class labels for rectangular version ------------------------------
    label_offset = 0.06
    dx_const, dy_const = 0.0, 0.0
    base_rotations = [2.825, 5.625, 11.25, 22.5, 45.0, 90.0]

    for i, (name, label_col) in enumerate(zip(class_names, label_rgbs)):
        if not name:
            continue
        h_i = harmonics_for_class[i]

        bis = exclusive_curve_bisector(
            i,
            x_plot,
            curves,
            N,
            y_min,
            y_max,
        )

        if bis is not None:
            x_bis, y_bis = bis
            x_lab = x_bis
            y_lab = y_bis - label_offset
            if y_lab < y_min + 0.05:
                y_lab = y_min + 0.05
            if h_i is None and N > 4:
                x_lab += dx_const
                y_lab += dy_const
        else:
            y_plot = curves[i]
            if h_i is None:
                x_lab = 0.5 * (x_min + x_max) if last_max_x is None else last_max_x
                y_lab = -label_offset
                if N > 4:
                    x_lab += dx_const
                    y_lab += dy_const
            else:
                dyc = np.diff(y_plot)
                signc = np.sign(dyc)
                i_min_loc = None
                for j in range(1, len(signc)):
                    if signc[j - 1] < 0 and signc[j] > 0:
                        i_min_loc = j
                if i_min_loc is None:
                    i_min_loc = int(np.argmin(y_plot))
                x_lab = x_plot[i_min_loc]
                y_lab = y_plot[i_min_loc] - label_offset
                if y_lab < y_min + 0.05:
                    y_lab = y_min + 0.05

        if h_i is None and i == N - 1:
            if N < 5:
                rot_cls = 0.0
                ha = "center"
                va = "top"
            else:
                rot_cls = 90.0
                ha = "right"
                va = "top"
                y_lab -= 0.02 * (y_max - y_min)
        else:
            rot_cls = base_rotations[i] if i < len(base_rotations) else 90.0
            ha = "center"
            va = "top"

        ax.text(
            x_lab,
            y_lab,
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
