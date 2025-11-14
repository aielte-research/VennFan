#!/usr/bin/env python3
"""
Sine-curve "Venn" diagrams (rectangular version, over [0, 2π] × [-1, 1]).

This module exposes:

- venntrig(...): main plotting function
- simple test/demo code under `if __name__ == "__main__":`
"""

from typing import Sequence, Optional, Union, Tuple, Dict, Callable, List
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from colors import _rgb
from defaults import (
    _default_palette_for_n,
    _default_fontsize,
    _default_linewidth,
)
from curves import get_sine_curve, get_cosine_curve
from utils import (
    _disjoint_region_masks,
    _visual_center_margin,
    _visual_center_inset,
    _region_constant_line_bisector,
    _exclusive_curve_bisector,
    _shrink_text_font_to_region,
    _harmonic_info_for_index,
    compute_region_fontsizes,
    resolve_color_mixing,
    _text_color_for_region,
    _make_demo_values,
)


def venntrig(
    values,
    class_names: Sequence[str],
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
    amp_decay_base: float = 0.75,
    linewidth: Optional[float] = None,
    curve_mode: str = "sine",
    height_scale: float = 2.0,
    linear_scale: bool = True,
) -> Optional[Figure]:
    """
    Rectangular version of the sine-curve Venn diagram.

    Parameters are mostly as in the original monolithic script; this function
    now delegates common logic to helpers in utils.py.
    """
    # ---- Basic input checks -------------------------------------------------
    arr = np.asarray(values, dtype=object)
    if arr.ndim < 1 or arr.ndim > 9:
        raise ValueError("Only N in {1,2,...,9} are supported.")
    N = arr.ndim
    expected_shape = (2,) * N
    if arr.shape != expected_shape:
        raise ValueError(f"values must have shape {expected_shape}, got {arr.shape}.")
    if len(class_names) != N:
        raise ValueError(f"class_names must have length {N}.")
    if N > 9:
        raise ValueError("N>9 not supported.")

    zeros = (0,) * N
    ones = (1,) * N

    # ---- Linewidth & colors -------------------------------------------------
    if linewidth is None:
        linewidth = _default_linewidth(N)

    # Default palette for this N
    default_fills, default_outlines = _default_palette_for_n(N)

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
        base_fs_region, base_fs_class = _default_fontsize(N, linear_scale, curve_mode)
        if region_label_fontsize is None:
            region_label_fontsize = base_fs_region
        if class_label_fontsize is None:
            class_label_fontsize = base_fs_class

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
            lmbd=amp_decay_base,
            linear=linear_scale,
        )
        mask = Y >= curve_full
        curve_1d = curve_full[0, :]
        membership.append(mask)
        curve_1d_list.append(curve_1d)

    # ---- Disjoint region masks ---------------------------------------------
    region_masks = _disjoint_region_masks(membership)
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
    ax.set_ylim(y_min+1/7, y_max)
    ax.set_aspect("auto")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.margins(0.0, 0.0)
    # Remove visible rectangle border for venntrig
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
        h_i, _ = _harmonic_info_for_index(i, N, include_constant_last)
        harmonics_for_class.append(h_i)

        y_plot = curve_fn_plot(
            x_plot,
            i,
            N,
            p=p,
            lmbd=amp_decay_base,
            linear=linear_scale,
        )
        curves.append(y_plot)

    # Draw class outlines in two passes: alpha 1.0 then 0.5 (for subtle halo)
    for pass_alpha in (1.0, 0.5):
        for i in range(N):
            y_plot = curves[i]
            ax.plot(
                x_plot,
                y_plot,
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
    
    if region_label_placement is None:
        region_label_placement = "visual_center" if linear_scale else "radial"
    else:
        region_label_placement = str(region_label_placement).lower()
    if region_label_placement not in ("radial", "visual_center"):
        raise ValueError(
            "region_label_placement must be one of 'radial', 'visual_center'."
        )

    for key, mask in region_masks.items():
        # Skip complement (all zeros) and all-sets intersection here;
        # these get special handling further below.
        if key == zeros or key == ones:
            continue
        value = arr[key]
        if value is None or not mask.any():
            continue

        this_color = _text_color_for_region(key, region_rgbs, text_color)
        fs_here = region_fontsizes.get(key, float(region_label_fontsize))
        
        

        if region_label_placement == "visual_center":
            # Linear: visual centers, inset to avoid bbox (no rotation)
            pos = _visual_center_inset(mask, X, Y, x_min, x_max, y_min, y_max, n_pix=2)
            if pos is None:
                continue
            x_lab, y_lab = pos
            rot = 0.0
            ha = "center"
            va = "center"
        else:
            # Nonlinear: anchor at constant-line intersection bisector, rotated 90°
            last_bit = key[-1]
            bis = _region_constant_line_bisector(mask, X, Y)
            if bis is not None:
                x_mid, y0 = bis  # y0 is 0.0
                x_lab = x_mid
                if last_bit == 1:
                    # Above constant line (inside last class): just above, left-justified
                    y_lab = y0 + region_offset
                    ha = "left"
                else:
                    # Below constant line: just below, right-justified
                    y_lab = y0 - region_offset
                    ha = "right"
            else:
                # Fallback: inset visual center, still rotated
                pos = _visual_center_inset(mask, X, Y, x_min, x_max, y_min, y_max, n_pix=2)
                if pos is None:
                    continue
                x_lab, y_lab = pos
                ha = "center"

            rot = 90.0
            va = "center"

        # Shrink fontsize if needed so text stays inside region
        fs_adj = _shrink_text_font_to_region(
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
            rotation=rot,
            rotation_mode="anchor",
        )

    # ---- All-sets intersection (ones) --------------------------------------
    all_mask = np.logical_and.reduce(membership)
    if all_mask.any():
        val_all = arr[ones]
        if val_all is not None:
            fs_all = region_fontsizes.get(ones, float(region_label_fontsize))

            if region_label_placement == "visual_center":
                # Linear: use margin-based visual center, no rotation
                pos = _visual_center_margin(all_mask, X, Y, margin_frac=0.05)
                if pos is not None:
                    this_color = _text_color_for_region(ones, region_rgbs, text_color)
                    x_lab, y_lab = pos
                    rot = 0.0
                    ha = "center"
                    va = "center"

                    fs_adj = _shrink_text_font_to_region(
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
                        rotation=rot,
                        rotation_mode="anchor",
                    )
            else:
                # Nonlinear: same constant-line bisector rule, rotated 90°
                this_color = _text_color_for_region(ones, region_rgbs, text_color)

                bis = _region_constant_line_bisector(all_mask, X, Y)
                if bis is not None:
                    x_mid, y0 = bis  # y0 is 0.0
                    x_lab = x_mid
                    y_lab = y0 + region_offset
                else:
                    pos = _visual_center_inset(
                        all_mask, X, Y, x_min, x_max, y_min, y_max, n_pix=2
                    )
                    if pos is None:
                        x_lab, y_lab = (0.5 * (x_min + x_max), const_y + region_offset)
                    else:
                        x_lab, y_lab = pos

                rot = 90.0
                ha = "left"
                va = "center"

                fs_adj = _shrink_text_font_to_region(
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
                    rotation=rot,
                    rotation_mode="anchor",
                )

    # ---- Complement (all zeros) – fixed bottom-right corner label ----------
    comp_mask = np.logical_not(np.logical_or.reduce(membership))
    if comp_mask.any():
        val_comp = arr[zeros]
        if val_comp is not None:
            this_color = text_color if text_color is not None else "black"
            fs_comp = float(complement_fontsize)

            # Bottom-right corner of the rectangular canvas, inset by 0.1 in both x and y.
            x_lab = x_max - 0.1
            y_lab = y_min + 0.1
            rot = 0.0
            ha = "right"
            va = "bottom"

            # Keep complement fontsize as requested, no shrink-to-fit
            fs_adj = fs_comp

            ax.text(
                x_lab,
                y_lab,
                f"{val_comp}",
                ha=ha,
                va=va,
                fontsize=fs_adj,
                color=this_color,
                zorder=5,
                rotation=rot,
                rotation_mode="anchor",
            )

    # ---- Class labels for rectangular version ------------------------------
    label_offset = 0.06
    dx_const, dy_const = 0.0, 0.0  # offset for constant class in some cases
    base_rotations = [2.825, 5.625, 11.25, 22.5, 45.0, 90.0]

    for i, (name, label_col) in enumerate(zip(class_names, label_rgbs)):
        if not name:
            continue  # skip empty labels
        h_i = harmonics_for_class[i]

        # Preferred: analytic "exclusive-region" bisector along the class curve
        bis = _exclusive_curve_bisector(
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
            # Fallback: previous min-based anchors
            y_plot = curves[i]
            if h_i is None:
                if last_max_x is None:
                    x_lab = 0.5 * (x_min + x_max)
                else:
                    x_lab = last_max_x
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

        # --- Rotation and alignment for class labels ---
        if h_i is None and i == N - 1:
            # Last / constant class
            if N < 5:
                # For small N: do NOT rotate the last label
                rot_cls = 0.0
                ha = "center"
                va = "top"
            else:
                # For N >= 5: rotate to vertical, right-justified,
                # and offset slightly downwards.
                rot_cls = 90.0
                ha = "right"   # "right" becomes "up" after 90° rotation
                va = "top"
                y_lab -= 0.02 * (y_max - y_min)
        else:
            # Other classes: original fixed rotation sequence
            if i < len(base_rotations):
                rot_cls = base_rotations[i]
            else:
                rot_cls = 90.0
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


if __name__ == "__main__":
    # Simple test/demo: generate a grid of venntrig diagrams for N=1..8
    os.makedirs("venntrig_img", exist_ok=True)
    greek_names = [
        "Alpha", "Beta", "Gamma", "Delta", "Epsilon",
        "Zeta", "Eta", "Theta", "Iota", "Kappa",
    ]

    for curve_mode in ["cosine", "sine"]:
        for linear_scale in [True]:
            for N in range(6, 7):
                print(
                    f"Generating venntrig diagram for "
                    f"curve_mode={curve_mode} linear_scale={linear_scale} N={N} ..."
                )
                values = _make_demo_values(N)
                class_names = greek_names[:N]
                class_names = [""]*N

                outfile = f"venntrig_img/{curve_mode}{'_linear' if linear_scale else ''}_N{N}.pdf"
                venntrig(
                    values,
                    class_names,
                    outfile=outfile,
                    height_scale=2.0,
                    p=1/2,
                    amp_decay_base=0.5,
                    include_constant_last=True,
                    curve_mode=curve_mode,
                    linear_scale=linear_scale,
                    color_mixing="average",
                    region_label_fontsize=20,
                    adaptive_fontsize=False,
                    region_label_placement="visual_center"
                )
