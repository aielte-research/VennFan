#!/usr/bin/env python3
"""
"Venn fan" circular version of the sine-curve Venn diagrams.

- Maps the rectangular half-plane picture to a disc:
    * y = 0   → circle of radius R
    * y > 0   → inside the circle
    * y < 0   → outside the circle

This module exposes:

- vennfan(...): main plotting function
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
from curves import get_sine_curve, get_cosine_curve, vennfan_find_extrema
from utils import (
    _disjoint_region_masks,
    _visual_center,
    _arc_angle_for_region,
    _halfplane_to_disc,
    _second_radial_intersection,
    _radial_segment_center_for_region,
    _normalize_angle_90,
    class_label_angles,
    compute_region_fontsizes,
    resolve_color_mixing,
    region_label_mode_for_key,
    _shrink_text_font_to_region,
    _text_color_for_region,
    _make_demo_values,
)


def vennfan(
    values,
    class_names: Sequence[str],
    colors: Optional[Sequence[Union[str, tuple]]] = None,
    outline_colors: Optional[Sequence[Union[str, tuple]]] = None,
    title: Optional[str] = None,
    outfile: Optional[str] = None,
    dpi: Optional[int] = None,
    color_mixing: Union[str, Callable] = "alpha_stack",
    text_color: Optional[str] = None,
    region_label_fontsize: Optional[float] = None,
    class_label_fontsize: Optional[float] = None,
    complement_fontsize: float = 8.0,
    adaptive_fontsize: Optional[bool] = None,
    adaptive_fontsize_range: Optional[Tuple[float, float]] = None,
    height_scale: float = 2.0,
    p: float = 0.2,
    amp_decay_base: float = 0.8,
    last_constant_label_offset: Tuple[float, float] = (0.0, 0.0),
    region_radial_offset_inside: Optional[float] = None,
    region_radial_offset_outside: Optional[float] = None,
    linewidth: Optional[float] = None,
    curve_mode: str = "cosine",
    linear_scale: bool = True,
    y_min: float = -1.0,
    y_max: float = 1.0,
    region_label_placement: Optional[str] = None,
    radial_bias: float = 0.5,
    draw_tight_factor: Optional[float] = None,
) -> Optional[Figure]:
    """
    vennfan variant of the sine/cosine diagram.

    region_label_placement:
        * None           → "visual_center" if linear_scale else "radial"
        * "visual_center"→ use visual-center-based labels
        * "radial"       → use pure radial labels anchored near the main circle
        * "hybrid"       → choose per-region between radial vs visual-center

    region_radial_offset_inside / region_radial_offset_outside:
        * None (default) → (hybrid radial) labels are centered along the ray
                           (no offset), and text is horizontally centered.
        * float          → shift the radial label by that amount along the ray
                           relative to the computed center.

    radial_bias:
        * Controls the radial "centering" (HYBRID mode) when we intersect the
          region with a ray and get two intersections (r_near, r_far).
        * radial_bias = 0.5 → simple midpoint.
        * radial_bias = 0.6 → center = 0.6 * r_near + 0.4 * r_far, etc.

    draw_tight_factor:
        * If provided, compute tight bounds via
          `vennfan_find_extrema(curve_mode, p=p, lmbd=amp_decay_base, N=N, linear=linear_scale)`
          and scale each of (x_min, x_max, y_min, y_max) by this factor for axes.
    """
    if curve_mode not in ("cosine", "sine"):
        raise ValueError(f"Unsupported curve_mode {curve_mode!r}; use 'cosine' or 'sine'.")

    if not (0.0 <= float(radial_bias) <= 1.0):
        raise ValueError("radial_bias must be in the range [0, 1].")
    radial_bias = float(radial_bias)

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
    ones = (1,) * N

    # ---- Region label placement mode ---------------------------------------
    if region_label_placement is None:
        region_label_placement = "visual_center" if linear_scale else "radial"
    else:
        region_label_placement = str(region_label_placement).lower()
    if region_label_placement not in ("radial", "visual_center", "hybrid"):
        raise ValueError(
            "region_label_placement must be one of 'radial', 'visual_center', or 'hybrid'."
        )

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
        linewidth = _default_linewidth(N)

    default_fills, default_outlines = _default_palette_for_n(N)

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
    if region_label_fontsize is None or class_label_fontsize is None:
        base_fs_region, base_fs_class = _default_fontsize(N, linear_scale, curve_mode)
        if region_label_fontsize is None:
            region_label_fontsize = base_fs_region
        if class_label_fontsize is None:
            class_label_fontsize = base_fs_class

    # ---- Color mixing callback ---------------------------------------------
    # NOTE: resolve_color_mixing now returns a function
    #       mixing_cb(colors, present=None) -> RGB
    mixing_cb = resolve_color_mixing(color_mixing, N)

    # ---- Base domain & disc grid -------------------------------------------
    x_min, x_max = 0.0, 2.0 * np.pi

    R = 1.0
    R_out = 2.0 * R
    us = np.linspace(-R_out, R_out, N * 150)
    vs = np.linspace(-R_out, R_out, N * 150)
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
            lmbd=amp_decay_base,
            linear=linear_scale,
        )
        mask = y_old >= curve
        membership.append(mask)

    # ---- Disjoint region masks ---------------------------------------------
    region_masks = _disjoint_region_masks(membership)
    H, W = U.shape

    # ---- Region areas & adaptive font sizes (vennfan plane) ----------------
    if us.size > 1:
        du = us[1] - us[0]
    else:
        du = (R_out - (-R_out)) / max(W - 1, 1)
    if vs.size > 1:
        dv = vs[1] - vs[0]
    else:
        dv = (R_out - (-R_out)) / max(H - 1, 1)
    pixel_area = abs(du * dv)

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
    adaptive_fontsize = adaptive_fontsize_flag  # normalized flag

    # For class labels we still want a small outward offset by default;
    # only override if region_radial_offset_outside is explicitly given.
    if region_radial_offset_outside is None:
        base_offset_outside_cls = 0.05
    else:
        base_offset_outside_cls = float(region_radial_offset_outside)

    # ---- Region RGBA image -------------------------------------------------
    rgba = np.zeros((H, W, 4), float)
    region_rgbs: Dict[Tuple[int, ...], np.ndarray] = {}

    for key, mask in region_masks.items():
        if not any(key):
            continue  # complement skipped
        if not mask.any():
            continue

        # list of RGBs actually used for this region
        colors_for_key = [rgbs[i] for i, bit in enumerate(key) if bit]
        # full true/false membership list for the region
        present = [bool(bit) for bit in key]

        mixed_rgb = np.asarray(mixing_cb(colors_for_key, present), float)
        if mixed_rgb.shape != (3,):
            raise ValueError("color_mixing callback must return an RGB array of shape (3,).")

        region_rgbs[key] = mixed_rgb
        rgba[mask, 0] = mixed_rgb[0]
        rgba[mask, 1] = mixed_rgb[1]
        rgba[mask, 2] = mixed_rgb[2]
        rgba[mask, 3] = 1.0

    # ---- Figure and axes ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 8 * height_scale / 2.0))
    fig.set_dpi(dpi)
    ax.imshow(
        rgba,
        origin="lower",
        extent=[-R_out, R_out, -R_out, R_out],
        interpolation="nearest",
        zorder=1,
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
        try:
            xmn, xmx, ymn, ymx = vennfan_find_extrema(
                curve_mode=curve_mode,
                p=p,
                lmbd=amp_decay_base,
                N=N,
                linear=linear_scale,
            )
            f = float(draw_tight_factor)
            ax.set_xlim(xmn * f, xmx * f)
            ax.set_ylim(ymn * f, ymx * f)
        except Exception:
            # Fail silently; fall back to full disc bounds
            pass

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
            lmbd=amp_decay_base,
            linear=linear_scale,
        )
        curves.append(y_plot)

        u_curve, v_curve = _halfplane_to_disc(x_plot, y_plot, R, y_min, y_max)
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
    fs_fixed_radial = float(fs_min)  # fixed font size for radial-type placements

    # ---- Region labels ------------------------------------------------------
    for key, mask in region_masks.items():
        if not mask.any():
            continue
        if key == zeros:
            continue  # complement handled separately

        value = arr[key]
        if value is None:
            continue

        placement_mode = region_label_mode_for_key(
            key=key,
            N=N,
            region_label_placement=region_label_placement,
        )

        if placement_mode == "visual_center":
            # Visual-center placement in disc coordinates
            pos = _visual_center(mask, U, V)
            if pos is None:
                continue

            this_color = _text_color_for_region(key, region_rgbs, text_color)
            fs_here = region_fontsizes.get(key, float(region_label_fontsize))
            u_lab, v_lab = pos
            rot = 0.0
            ha = "center"
            va = "center"

            fs_adj = _shrink_text_font_to_region(
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

        elif placement_mode == "radial":
            # Radial mode anchored near the main circle (original behavior)
            last_bit = key[-1]

            angle_raw = _arc_angle_for_region(mask, circle_band, theta, U, V)
            if angle_raw is None:
                continue

            v_out = np.array([np.cos(angle_raw), np.sin(angle_raw)], float)

            # Default offsets if not specified
            off_in = float(region_radial_offset_inside) if region_radial_offset_inside is not None else 0.05
            off_out = float(region_radial_offset_outside) if region_radial_offset_outside is not None else 0.05

            r_lab = R * (1.0 - off_in) if last_bit == 1 else R * (1.0 + off_out)

            u_lab = r_lab * v_out[0]
            v_lab = r_lab * v_out[1]

            deg_raw = np.degrees(angle_raw)
            rot = _normalize_angle_90(deg_raw)
            rot_rad = np.deg2rad(rot)
            v_base = np.array([np.cos(rot_rad), np.sin(rot_rad)], float)

            v_circle = v_out if last_bit == 1 else -v_out
            d_align = float(np.dot(v_circle, v_base))

            ha = "right" if d_align >= 0 else "left"
            va = "center"

            this_color = _text_color_for_region(key, region_rgbs, text_color)
            fs_here = fs_fixed_radial  # fixed size, no shrink

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

        else:
            # HYBRID radial mode with biased radial center
            last_bit = key[-1]

            angle_raw = _arc_angle_for_region(mask, circle_band, theta, U, V)
            if angle_raw is None:
                continue

            v_out = np.array([np.cos(angle_raw), np.sin(angle_raw)], float)

            r_mid = _radial_segment_center_for_region(
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
                pos_vc = _visual_center(mask, U, V)
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
            rot = _normalize_angle_90(deg_raw)
            rot_rad = np.deg2rad(rot)
            v_base = np.array([np.cos(rot_rad), np.sin(rot_rad)], float)

            if last_bit == 1:
                v_circle = v_out
            else:
                v_circle = -v_out

            d_align = float(np.dot(v_circle, v_base))

            # Default (no offsets) → center justification; with offsets → original rule
            if (last_bit == 1 and region_radial_offset_inside is None) or (
                last_bit == 0 and region_radial_offset_outside is None
            ):
                ha = "center"
            else:
                ha = "right" if d_align >= 0 else "left"

            va = "center"

            this_color = _text_color_for_region(key, region_rgbs, text_color)
            fs_here = fs_fixed_radial  # fixed size, no shrink

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
            this_color = (
                text_color if text_color is not None else "black"
            )
            fs_comp = float(complement_fontsize)

            u_lab = R_out - 0.1
            v_lab = -R_out + 0.1
            rot = 0.0
            ha = "right"
            va = "bottom"

            if adaptive_fontsize and not linear_scale:
                fs_adj = _shrink_text_font_to_region(
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
                )
            else:
                fs_adj = fs_comp

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

    # ---- Class labels on vennfan -------------------------------------------
    label_offset = 0.18 * height_scale
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
        inter = _second_radial_intersection(u_curve, v_curve, angle_anchor)
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
            rot_cls = _normalize_angle_90(angle_deg_radial - 90.0)
            ha = "center"
            va = "center"
        else:
            rot_cls = _normalize_angle_90(angle_deg_radial)
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


if __name__ == "__main__":
    os.makedirs("vennfan_img", exist_ok=True)
    greek_names = [
        "Alpha", "Beta", "Gamma", "Delta", "Epsilon",
        "Zeta", "Eta", "Theta", "Iota", "Kappa",
    ]

    for curve_mode in ["cosine", "sine"]:
        for linear_scale in [True]:
            for N in range(6, 7):  # tweak as desired
                print(
                    f"Generating vennfan diagram for "
                    f"curve_mode={curve_mode} linear_scale={linear_scale} N={N}..."
                )
                shape = (2,) * N
                values = np.empty(shape, dtype=object)
                class_names = ["" for _ in range(N)]

                # Demo values: label each region by subset of A,B,C,...
                values = _make_demo_values(N)
                class_names = greek_names[:N]
                class_names = [""]*N

                outfile = f"vennfan_img/vennfan_{curve_mode}{'_linear' if linear_scale else ''}_N{N}.pdf"
                vennfan(
                    values,
                    class_names,
                    outfile=outfile,
                    height_scale=2.0,
                    p=1 / 2,
                    amp_decay_base=4 / 5,
                    curve_mode=curve_mode,
                    linear_scale=linear_scale,
                    draw_tight_factor=1.02,
                    color_mixing="average"
                )
