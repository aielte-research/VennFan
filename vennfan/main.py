#!/usr/bin/env python3
"""
Sine-curve "Venn" diagrams up to N=10, plus a "vennfan" circular variant.

- `venntrig(...)` draws the rectangular version over [0, 2π] × [-1, 1].
- `vennfan(...)` does the same, but it maps the half-plane picture onto a circle:
    * y = 0  →  circle of radius `radius`
    * y > 0  →  inside the circle
    * y < 0  →  outside the circle
"""

from typing import Sequence, Optional, Union, Tuple, Dict, Callable, List
import itertools
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.ndimage import distance_transform_edt

from colors import (
    _default_palette_for_n,
    _rgb,
    _auto_text_color_from_rgb,
    _color_mix_subtractive,
    _color_mix_average,
)
from curves import get_sine_curve, get_cosine_curve


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _disjoint_region_masks(masks_list: Sequence[np.ndarray]) -> Dict[Tuple[int, ...], np.ndarray]:
    """
    Given a list of boolean membership masks for N sets (each shaped HxW),
    return a dict mapping every binary tuple key of length N (e.g., (1,0,1,0))
    to the corresponding disjoint region mask.
    """
    memb = np.stack(masks_list, axis=-1).astype(bool)  # (H, W, N)
    N = memb.shape[-1]
    keys = list(itertools.product((0, 1), repeat=N))   # all 2^N keys
    key_arr = np.array(keys, dtype=bool)               # (K, N)

    maskK = (memb[..., None, :] == key_arr[None, None, :, :]).all(axis=-1)
    return {tuple(map(int, k)): maskK[..., i] for i, k in enumerate(keys)}


def _visual_center(mask: np.ndarray, X: np.ndarray, Y: np.ndarray):
    """Visual center via Euclidean distance transform (SciPy)."""
    if not mask.any():
        return None
    dist = distance_transform_edt(mask)
    yy, xx = np.unravel_index(np.argmax(dist), dist.shape)
    return float(X[yy, xx]), float(Y[yy, xx])


def _centroid(mask: np.ndarray, X: np.ndarray, Y: np.ndarray):
    """Simple centroid of True pixels in mask."""
    if not mask.any():
        return None
    yy, xx = np.where(mask)
    return float(X[yy, xx].mean()), float(Y[yy, xx].mean())


def _visual_center_margin(mask: np.ndarray, X: np.ndarray, Y: np.ndarray, margin_frac: float = 0.05):
    """
    Visual center, but ignore a small margin near the rectangular box edges.
    Used for complement / all-sets center.
    """
    if not mask.any():
        return None

    H, W = mask.shape
    margin_y = max(1, int(margin_frac * H))
    margin_x = max(1, int(margin_frac * W))

    m2 = mask.copy()
    m2[:margin_y, :] = False
    m2[-margin_y:, :] = False
    m2[:, :margin_x] = False
    m2[:, -margin_x:] = False

    if not m2.any():
        return _visual_center(mask, X, Y)

    dist = distance_transform_edt(m2)
    yy, xx = np.unravel_index(np.argmax(dist), dist.shape)
    return float(X[yy, xx]), float(Y[yy, xx])


def _normalize_angle_90(deg: float) -> float:
    """Map any angle (deg) to an equivalent in about [-90, +90] for legible text."""
    a = float(deg)
    while a > 95.0:
        a -= 180.0
    while a < -85.0:
        a += 180.0
    return a


def _visual_center_inset(
    mask: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    n_pix: int = 2,
):
    """
    Visual center, but computed inside an inset of the bounding box:
    - Intersect the region with a rectangle inset by n_pix grid steps
      from each side, then run distance transform.
    - If that intersection is empty, fall back to the full region, but
      clamp the final coordinates back into the inset box.

    This is to avoid labels landing exactly on the bounding box,
    especially for cosine + linear_scale.
    """
    if not mask.any():
        return None

    H, W = mask.shape
    xs = X[0, :]
    ys = Y[:, 0]

    if xs.size > 1:
        dx = xs[1] - xs[0]
    else:
        dx = (x_max - x_min) / max(W - 1, 1)

    if ys.size > 1:
        dy = ys[1] - ys[0]
    else:
        dy = (y_max - y_min) / max(H - 1, 1)

    inset_x_min = x_min + n_pix * dx
    inset_x_max = x_max - n_pix * dx
    inset_y_min = y_min + n_pix * dy
    inset_y_max = y_max - n_pix * dy

    mask_inset = mask & (X > inset_x_min) & (X < inset_x_max) & (Y > inset_y_min) & (Y < inset_y_max)

    if mask_inset.any():
        pos = _visual_center(mask_inset, X, Y)
    else:
        pos = _visual_center(mask, X, Y)

    if pos is None:
        return None

    x_lab, y_lab = pos
    # Clamp into inset box so we never land exactly on bbox edges.
    x_lab = min(max(x_lab, inset_x_min), inset_x_max)
    y_lab = min(max(y_lab, inset_y_min), inset_y_max)
    return x_lab, y_lab


def _region_label_orientation(
    mask: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    anchor_x: float,
    anchor_y: float,
    n_angles: int = 72,
    seg_frac: float = 0.25,
) -> float:
    """
    Original orientation helper (kept for rectangular version if needed).
    Not used in the new region logic, but left here for completeness.
    """
    if not mask.any():
        return 0.0

    H, W = mask.shape
    xs = X[0, :]
    ys = Y[:, 0]
    if xs.size < 2 or ys.size < 2:
        return 0.0

    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    if dx == 0 or dy == 0:
        return 0.0

    span_x = xs[-1] - xs[0]
    span_y = ys[-1] - ys[0]
    L = seg_frac * max(abs(span_x), abs(span_y))

    num_samples = 201
    t_vals = np.linspace(-L, L, num_samples)
    center_idx = num_samples // 2

    best_score = -1.0
    best_theta = 0.0

    for k in range(n_angles):
        theta = np.pi * k / n_angles
        ct = np.cos(theta)
        st = np.sin(theta)

        x_s = anchor_x + t_vals * ct
        y_s = anchor_y + t_vals * st

        ix = np.round((x_s - xs[0]) / dx).astype(int)
        iy = np.round((y_s - ys[0]) / dy).astype(int)

        valid = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        inside = np.zeros_like(valid, dtype=bool)
        inside[valid] = mask[iy[valid], ix[valid]]

        if not inside[center_idx]:
            continue

        len_pos = 0
        for j in range(center_idx, num_samples):
            if inside[j]:
                len_pos += 1
            else:
                break

        len_neg = 0
        for j in range(center_idx, -1, -1):
            if inside[j]:
                len_neg += 1
            else:
                break

        score = float(len_pos * len_neg)
        if score > best_score:
            best_score = score
            best_theta = theta

    if best_score <= 0:
        return 0.0

    deg = np.degrees(best_theta)
    return _normalize_angle_90(deg)


def _arc_angle_for_region(
    mask: np.ndarray,
    circle_band: np.ndarray,
    theta: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    n_bins: int = 720,
) -> Optional[float]:
    """
    Robust angle on the main circle for a region (used by vennfan nonlinear).
    """
    if mask is None or not mask.any():
        return None

    arc_mask = mask & circle_band
    if arc_mask.any():
        angs = theta[arc_mask]
        if angs.size > 0:
            two_pi = 2.0 * np.pi
            idx = np.floor(angs / two_pi * n_bins).astype(int)
            idx = np.clip(idx, 0, n_bins - 1)
            bins = np.zeros(n_bins, dtype=bool)
            bins[idx] = True

            if bins.any():
                segments: List[Tuple[int, int]] = []
                in_seg = False
                start = 0
                for i in range(n_bins * 2):
                    b = bins[i % n_bins]
                    if b and not in_seg:
                        in_seg = True
                        start = i
                    elif not b and in_seg:
                        end = i
                        segments.append((start, end))
                        in_seg = False
                if in_seg:
                    segments.append((start, n_bins * 2))

                best_len = -1
                best_center = None
                for s, e in segments:
                    length = e - s
                    if length <= 0:
                        continue
                    if s >= n_bins and e <= n_bins * 2:
                        continue
                    if length > best_len:
                        best_len = length
                        best_center = 0.5 * (s + e)

                if best_center is not None and best_len > 0:
                    center_idx = best_center % n_bins
                    angle = two_pi * center_idx / n_bins
                    return float(angle)

    # Fallback: angle of visual center
    pos_vc = _visual_center(mask, U, V)
    if pos_vc is None:
        return None
    return float(np.arctan2(pos_vc[1], pos_vc[0]))


def _harmonic_info_for_index(i: int, N: int, include_constant_last: bool) -> Tuple[Optional[float], Optional[float]]:
    """
    For a set index i = 0,1,...,N-1, return (h_i, h_max) where

        h_i = 2^i   for non-constant sets,
        h_i = None  for the constant "∞" set (if include_constant_last and i == N-1).

    h_max is the largest harmonic used among the sine-like classes, i.e.
    h_max = 2^{N-2} if include_constant_last, else 2^{N-1}.
    """
    if include_constant_last and N >= 1 and i == N - 1:
        return None, None

    i_index = i
    if include_constant_last:
        max_index = max(N - 2, 0)
    else:
        max_index = max(N - 1, 0)

    h_i = 2.0 ** i_index
    if max_index > 0:
        h_max = 2.0 ** max_index
    else:
        h_max = h_i

    return h_i, h_max


def _exclusive_curve_bisector(
    i: int,
    x_plot: np.ndarray,
    curves: Sequence[np.ndarray],
    N: int,
    y_min: float,
    y_max: float,
) -> Optional[Tuple[float, float]]:
    """
    For class i, find the midpoint (x, y) on its boundary curve that borders
    its *exclusive* region, using the analytic curves only.

    Bisector is defined with respect to *arc length* along that curve segment.
    """
    key_bits = np.array([1 if k == i else 0 for k in range(N)], dtype=int)
    key_code = int(sum(int(b) << k for k, b in enumerate(key_bits)))

    y_i = curves[i]  # shape (M,)
    if y_i.size == 0:
        return None

    eps_y = 0.01 * (y_max - y_min)
    y_probe = y_i + eps_y

    # Membership codes at each x_plot
    M = []
    for k in range(N):
        y_k = curves[k]
        M.append(y_probe >= y_k)
    M = np.stack(M, axis=0)  # (N, M)

    codes = np.zeros(y_i.size, dtype=int)
    for k in range(N):
        codes |= (M[k].astype(int) << k)
    inside = (codes == key_code)

    Mlen = inside.size
    best_len = 0
    best_start = None
    best_end = None
    j = 0
    while j < Mlen:
        if inside[j]:
            s = j
            while j < Mlen and inside[j]:
                j += 1
            e = j - 1
            length = e - s + 1
            if length > best_len:
                best_len = length
                best_start = s
                best_end = e
        else:
            j += 1

    if best_len <= 0 or best_start is None or best_end is None:
        return None

    # Arc-length-based midpoint along the segment [best_start .. best_end]
    s = best_start
    e = best_end

    x_seg = x_plot[s:e + 1].astype(float)
    y_seg = y_i[s:e + 1].astype(float)

    if x_seg.size == 1:
        return float(x_seg[0]), float(y_seg[0])

    dx = np.diff(x_seg)
    dy = np.diff(y_seg)
    seg_len = np.sqrt(dx * dx + dy * dy)
    total_len = float(seg_len.sum())

    if total_len == 0.0:
        # Degenerate; just return the middle sample in index-space
        mid_idx = (s + e) // 2
        return float(x_plot[mid_idx]), float(y_i[mid_idx])

    cum_len = np.concatenate(([0.0], np.cumsum(seg_len)))
    half_len = 0.5 * total_len

    # Find segment where the half-length falls
    k = int(np.searchsorted(cum_len, half_len) - 1)
    if k < 0:
        k = 0
    if k >= seg_len.size:
        k = seg_len.size - 1

    l0 = cum_len[k]
    l1 = cum_len[k + 1]
    if l1 <= l0:
        t = 0.0
    else:
        t = (half_len - l0) / (l1 - l0)

    x_mid = x_seg[k] + t * (x_seg[k + 1] - x_seg[k])
    y_mid = y_seg[k] + t * (y_seg[k + 1] - y_seg[k])

    return float(x_mid), float(y_mid)


def _tangent_angle_from_curve_fn(
    curve_fn: Callable,
    i: int,
    N: int,
    x0: float,
    curve_exponent: float,
    amp_decay_base: float,
    linear_scale: bool,
) -> float:
    """
    Numeric derivative of curve i at x0 to determine tangent angle in degrees.
    (Not used in vennfan now, left for reference.)
    """
    if i == N - 1:
        return 0.0

    h = max(1e-3, (2.0 * np.pi) / 2000.0)
    x1 = x0 - h
    x2 = x0 + h

    y1 = curve_fn(
        np.array([x1]),
        i,
        N,
        p=curve_exponent,
        lmbd=amp_decay_base,
        linear=linear_scale,
    )[0]
    y2 = curve_fn(
        np.array([x2]),
        i,
        N,
        p=curve_exponent,
        lmbd=amp_decay_base,
        linear=linear_scale,
    )[0]

    dx = x2 - x1
    if dx == 0.0:
        return 0.0

    slope = float(y2 - y1) / dx
    angle = np.degrees(np.arctan2(slope, 1.0))
    return _normalize_angle_90(angle)


def _region_constant_line_bisector(mask: np.ndarray, X: np.ndarray, Y: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    For a given region mask, find the midpoint along the intersection with y ≈ 0.
    """
    if not mask.any():
        return None

    H, W = mask.shape
    xs = X[0, :]
    ys = Y[:, 0]

    if H < 2 or W < 1:
        return None

    # Grid spacing in y
    if ys.size > 1:
        dy = abs(ys[1] - ys[0])
    else:
        dy = float(abs(Y.max() - Y.min()) / max(H - 1, 1))

    # Band around y=0
    band = (np.abs(Y) <= dy)
    band_mask = mask & band
    if not band_mask.any():
        return None

    # Collapse in y to see which x-columns touch the band
    col_mask = band_mask.any(axis=0)  # shape (W,)
    if not col_mask.any():
        return None

    best_len = 0
    best_start = None
    best_end = None
    j = 0
    while j < W:
        if col_mask[j]:
            s = j
            while j < W and col_mask[j]:
                j += 1
            e = j - 1
            length = e - s + 1
            if length > best_len:
                best_len = length
                best_start = s
                best_end = e
        else:
            j += 1

    if best_len <= 0 or best_start is None or best_end is None:
        return None

    mid = 0.5 * (best_start + best_end)
    idxs = np.arange(W, dtype=float)
    x_mid = float(np.interp(mid, idxs, xs.astype(float)))
    y_mid = 0.0
    return x_mid, y_mid


# ---------------------------------------------------------------------------
# Main rectangular plotting function (venntrig)
# ---------------------------------------------------------------------------

def venntrig(
    values,
    class_names: Sequence[str],
    colors: Optional[Sequence[Union[str, tuple]]] = None,
    outline_colors: Optional[Sequence[Union[str, tuple]]] = None,
    title: Optional[str] = None,
    outfile: Optional[str] = None,
    dpi: int = 600,
    color_mixing: Union[str, Callable[[Sequence[np.ndarray]], np.ndarray]] = "average",
    text_color: Optional[str] = None,
    region_label_fontsize: int = 10,
    class_label_fontsize: int = 12,
    sample_res_x: int = 900,
    sample_res_y: int = 900,
    include_constant_last: bool = True,
    curve_exponent: float = 0.33,
    amp_decay_base: float = 0.75,
    last_constant_label_offset: Tuple[float, float] = (0.0, 0.0),
    linewidth: float = 2.0,
    curve_mode: str = "sine",
    height_scale: float = 2.0,
    linear_scale: bool = True,
) -> Optional[Figure]:
    """
    Rectangular version.
    """
    arr = np.asarray(values, dtype=object)
    if arr.ndim < 1 or arr.ndim > 10:
        raise ValueError("Only N in {1,2,...,10} are supported.")
    N = arr.ndim
    expected_shape = (2,) * N
    if arr.shape != expected_shape:
        raise ValueError(f"values must have shape {expected_shape}, got {arr.shape}.")
    if len(class_names) != N:
        raise ValueError(f"class_names must have length {N}.")
    if N > 10:
        raise ValueError("N>10 not supported.")

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

    # Color mixing callback (uses fill colors)
    if isinstance(color_mixing, str):
        if color_mixing == "subtractive":
            mixing_cb = _color_mix_subtractive
        elif color_mixing == "average":
            mixing_cb = _color_mix_average
        else:
            raise ValueError(f"Unrecognized color_mixing string: {color_mixing!r}")
    elif callable(color_mixing):
        mixing_cb = color_mixing
    else:
        raise TypeError("color_mixing must be either a string or a callable.")

    # Sampling grid in the universe rectangle
    x_min, x_max = 0.0, 2.0 * np.pi
    y_min, y_max = -1.0, 1.0
    xs = np.linspace(x_min, x_max, int(sample_res_x))
    ys = np.linspace(y_min, y_max, int(sample_res_y))
    X, Y = np.meshgrid(xs, ys)

    # Membership masks & per-class 1D curves on xs
    membership: List[np.ndarray] = []
    curve_1d_list: List[np.ndarray] = []

    if curve_mode == "sine":
        curve_fn = get_sine_curve
    else:
        curve_fn = get_cosine_curve

    for i in range(N):
        if i == N - 1:
            mask = Y >= 0.0
            curve_1d = np.zeros_like(xs)
        else:
            curve_full = curve_fn(
                X,
                i,
                N,
                p=curve_exponent,
                lmbd=amp_decay_base,
                linear=linear_scale,
            )
            mask = Y >= curve_full
            curve_1d = curve_full[0, :]
        membership.append(mask)
        curve_1d_list.append(curve_1d)

    # Disjoint region masks and region colors
    region_masks = _disjoint_region_masks(membership)
    H, W = X.shape
    rgba = np.zeros((H, W, 4), float)
    region_rgbs: Dict[Tuple[int, ...], np.ndarray] = {}

    for key, mask in region_masks.items():
        if not any(key):
            continue
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

    # Figure and axes
    fig, ax = plt.subplots(figsize=(15, 5 * height_scale))
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

    # Class boundaries (store curves for plotting & later class-label logic)
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

        if i == N - 1:
            y_plot = np.zeros_like(x_plot)
        else:
            y_plot = curve_fn_plot(
                x_plot,
                i,
                N,
                p=curve_exponent,
                lmbd=amp_decay_base,
                linear=linear_scale,
            )

        curves.append(y_plot)

        ax.plot(
            x_plot,
            y_plot,
            color=line_colors[i],
            linewidth=linewidth,
            zorder=4,
        )

    # Last local maximum for last non-constant class (kept as a fallback anchor)
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

    zeros = (0,) * N
    ones = (1,) * N

    # --- Region labels ---
    const_y = 0.0
    region_offset = 0.05 * (y_max - y_min)

    for key, mask in region_masks.items():
        if key == zeros or key == ones:
            continue  # complement handled separately; all-sets handled below
        value = arr[key]
        if value is None:
            continue
        if not mask.any():
            continue

        if text_color is None:
            if key in region_rgbs:
                rgb = region_rgbs[key]
                this_color = _auto_text_color_from_rgb(rgb)
            else:
                this_color = "black"
        else:
            this_color = text_color

        if linear_scale:
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

        ax.text(
            x_lab,
            y_lab,
            f"{value}",
            ha=ha,
            va=va,
            fontsize=region_label_fontsize,
            color=this_color,
            zorder=5,
            rotation=rot,
            rotation_mode="anchor",
        )

    # All-sets intersection
    all_mask = np.logical_and.reduce(membership)
    if all_mask.any():
        val_all = arr[ones]
        if val_all is not None:
            if linear_scale:
                # Linear: use margin-based visual center, no rotation
                pos = _visual_center_margin(all_mask, X, Y, margin_frac=0.05)
                if pos is not None:
                    if text_color is None:
                        rgb = region_rgbs.get(ones)
                        this_color = _auto_text_color_from_rgb(rgb) if rgb is not None else "black"
                    else:
                        this_color = text_color

                    ax.text(
                        pos[0],
                        pos[1],
                        f"{val_all}",
                        ha="center",
                        va="center",
                        fontsize=region_label_fontsize,
                        color=this_color,
                        zorder=5,
                        rotation=0.0,
                        rotation_mode="anchor",
                    )
            else:
                # Nonlinear: same constant-line bisector rule, rotated 90°
                if text_color is None:
                    rgb = region_rgbs.get(ones)
                    this_color = _auto_text_color_from_rgb(rgb) if rgb is not None else "black"
                else:
                    this_color = text_color

                bis = _region_constant_line_bisector(all_mask, X, Y)
                if bis is not None:
                    x_mid, y0 = bis  # y0 is 0.0
                    x_lab = x_mid
                    y_lab = y0 + region_offset
                else:
                    pos = _visual_center_inset(all_mask, X, Y, x_min, x_max, y_min, y_max, n_pix=2)
                    if pos is None:
                        x_lab, y_lab = (0.5 * (x_min + x_max), const_y + region_offset)
                    else:
                        x_lab, y_lab = pos

                ax.text(
                    x_lab,
                    y_lab,
                    f"{val_all}",
                    ha="left",
                    va="center",
                    fontsize=region_label_fontsize,
                    color=this_color,
                    zorder=5,
                    rotation=90.0,
                    rotation_mode="anchor",
                )

    # Complement (all zeros) – visual center with margin, no rotation
    comp_mask = np.logical_not(np.logical_or.reduce(membership))
    if comp_mask.any():
        val_comp = arr[zeros]
        if val_comp is not None:
            pos = _visual_center_margin(comp_mask, X, Y, margin_frac=0.05)
            if pos is not None:
                if text_color is None:
                    this_color = "black"
                else:
                    this_color = text_color

                ax.text(
                    pos[0],
                    pos[1],
                    f"{val_comp}",
                    ha="center",
                    va="center",
                    fontsize=region_label_fontsize,
                    color=this_color,
                    zorder=5,
                    rotation=0.0,
                    rotation_mode="anchor",
                )

    # --- Class labels for rectangular version ---
    label_offset = 0.06
    dx_const, dy_const = last_constant_label_offset

    base_rotations = [2.825, 5.625, 11.25, 22.5, 45.0, 90.0]

    for i, (name, label_col) in enumerate(zip(class_names, label_rgbs)):
        if not name:
            continue  # skip empty labels
        h_i = harmonics_for_class[i]

        x_lab = None
        y_lab = None

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

        # Rotation: fixed sequence, no tangents
        if h_i is None and i == N - 1:
            # Last "constant" class: always full 90°
            rot_cls = 90.0
        else:
            if i < len(base_rotations):
                rot_cls = base_rotations[i]
            else:
                rot_cls = 90.0

        ax.text(
            x_lab,
            y_lab,
            name,
            ha="center",
            va="top",
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


# ---------------------------------------------------------------------------
# Half-plane → disc mapping
# ---------------------------------------------------------------------------

def _halfplane_to_disc(
    x: np.ndarray,
    y: np.ndarray,
    radius: float,
    y_min: float,
    y_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map (x,y) in the rectangular half-plane to (u,v) in the vennfan plane.

    - y = 0 maps to radius = R.
    - y > 0 maps inside the circle (radius < R).
    - y < 0 maps outside the circle (radius > R, up to 2R).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    R = float(radius)
    R_out = 2.0 * R

    y_pos_max = float(max(0.0, y_max))
    y_neg_min = float(min(0.0, y_min))  # negative

    rho = np.empty_like(y)

    pos = (y >= 0.0)
    neg = ~pos

    # Map inside
    rho[pos] = R * (1.0 - y[pos] / y_pos_max) if y_pos_max != 0 else R

    # Map outside
    denom = (R_out - R)
    rho[neg] = R + denom * (y[neg] / y_neg_min) if y_neg_min != 0 else R_out

    theta = x
    u = rho * np.cos(theta)
    v = rho * np.sin(theta)
    return u, v


# ---------------------------------------------------------------------------
# vennfan version
# ---------------------------------------------------------------------------

def vennfan(
    values,
    class_names: Sequence[str],
    colors: Optional[Sequence[Union[str, tuple]]] = None,
    outline_colors: Optional[Sequence[Union[str, tuple]]] = None,
    title: Optional[str] = None,
    outfile: Optional[str] = None,
    dpi: Optional[int] = None,
    color_mixing: Union[str, Callable[[Sequence[np.ndarray]], np.ndarray]] = "average",
    text_color: Optional[str] = "black",
    region_label_fontsize: int = 10,
    class_label_fontsize: int = 12,
    height_scale: float = 2.0,
    include_constant_last: bool = True,
    radius: Optional[float] = 4.0,
    curve_exponent: float = 0.2,
    amp_decay_base: float = 0.8,
    last_constant_label_offset: Tuple[float, float] = (0.0, 0.0),
    region_radial_offset_inside: float = 0.05,
    region_radial_offset_outside: float = 0.05,
    linewidth: float = 2.0,
    curve_mode: str = "cosine",
    linear_scale: bool = True,
) -> Optional[Figure]:
    """
    vennfan variant of the sine diagram.
    """

    if curve_mode not in ("cosine", "sine"):
        raise ValueError(f"Unsupported curve_mode {curve_mode!r}; use 'cosine' or 'sine'.")

    if curve_mode == "sine":
        curve_fn = get_sine_curve
    else:
        curve_fn = get_cosine_curve

    arr = np.asarray(values, dtype=object)
    if arr.ndim < 1 or arr.ndim > 10:
        raise ValueError("Only N in {1,2,...,10} are supported.")
    N = arr.ndim
    if dpi is None:
        dpi = 100 * N
    expected_shape = (2,) * N
    if arr.shape != expected_shape:
        raise ValueError(f"values must have shape {expected_shape}, got {arr.shape}.")
    if len(class_names) != N:
        raise ValueError(f"class_names must have length {N}.")
    if N > 10:
        raise ValueError("N>10 not supported.")

    # Default palette for this N
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

    # Color mixing callback
    if isinstance(color_mixing, str):
        if color_mixing == "subtractive":
            mixing_cb = _color_mix_subtractive
        elif color_mixing == "average":
            mixing_cb = _color_mix_average
        else:
            raise ValueError(f"Unrecognized color_mixing string: {color_mixing!r}")
    elif callable(color_mixing):
        mixing_cb = color_mixing
    else:
        raise TypeError("color_mixing must be either a string or a callable.")

    x_min, x_max = 0.0, 2.0 * np.pi
    y_min, y_max = -1 + 1 / N, 1 - 1 / N

    R = 1.0
    R_out = 2.0 * R
    us = np.linspace(-R_out, R_out, N * 150)
    vs = np.linspace(-R_out, R_out, N * 150)
    U, V = np.meshgrid(us, vs)

    rho = np.sqrt(U * U + V * V)
    theta = np.mod(np.arctan2(V, U), 2.0 * np.pi)

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

    membership: List[np.ndarray] = []

    for i in range(N):
        if i == N - 1:
            mask = y_old >= 0.0
        else:
            curve = curve_fn(
                x_old,
                i,
                N,
                p=curve_exponent,
                lmbd=amp_decay_base,
                linear=linear_scale,
            )
            mask = y_old >= curve
        membership.append(mask)

    region_masks = _disjoint_region_masks(membership)
    H, W = U.shape
    rgba = np.zeros((H, W, 4), float)
    region_rgbs: Dict[Tuple[int, ...], np.ndarray] = {}

    for key, mask in region_masks.items():
        if not any(key):
            continue
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

    fig, ax = plt.subplots(figsize=(8, 8))
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

    # Class boundaries in vennfan plane
    x_plot = np.linspace(x_min, x_max, 1000 * N)
    curves: List[np.ndarray] = []
    harmonics_for_class: List[Optional[float]] = []

    for i in range(N):
        h_i, _ = _harmonic_info_for_index(i, N, include_constant_last)
        harmonics_for_class.append(h_i)

        if i == N - 1:
            y_plot = np.zeros_like(x_plot)
        else:
            y_plot = curve_fn(
                x_plot,
                i,
                N,
                p=curve_exponent,
                lmbd=amp_decay_base,
                linear=linear_scale,
            )

        curves.append(y_plot)

        u_curve, v_curve = _halfplane_to_disc(x_plot, y_plot, R, y_min, y_max)
        ax.plot(
            u_curve,
            v_curve,
            color=line_colors[i],
            linewidth=linewidth,
            zorder=4,
        )
        # If the mapped curve is not closed (start and end differ),
        # explicitly connect the endpoints with a straight segment.
        if u_curve.size > 1:
            du = u_curve[0] - u_curve[-1]
            dv = v_curve[0] - v_curve[-1]
            if du * du + dv * dv > 1e-10:
                ax.plot(
                    [u_curve[-1], u_curve[0]],
                    [v_curve[-1], v_curve[0]],
                    color=line_colors[i],
                    linewidth=linewidth,
                    zorder=4,
                )

    zeros = (0,) * N
    ones = (1,) * N

    rho = np.sqrt(U * U + V * V)
    circle_band = np.abs(rho - R) <= (0.03 * R)
    theta = np.mod(np.arctan2(V, U), 2.0 * np.pi)

    # --- Region labels ---
    if linear_scale:
        # Simple visual-center labels, no rotation
        for key, mask in region_masks.items():
            if not mask.any():
                continue
            if key == zeros:
                continue  # complement handled separately

            value = arr[key]
            if value is None:
                continue

            pos = _visual_center(mask, U, V)
            if pos is None:
                continue

            if text_color is None:
                rgb = region_rgbs.get(key)
                this_color = _auto_text_color_from_rgb(rgb) if rgb is not None else "black"
            else:
                this_color = text_color

            ax.text(
                pos[0],
                pos[1],
                f"{value}",
                ha="center",
                va="center",
                fontsize=region_label_fontsize,
                color=this_color,
                zorder=5,
                rotation=0.0,
                rotation_mode="anchor",
            )
    else:
        # Nonlinear mode: radial region labels on the circle
        for key, mask in region_masks.items():
            if not mask.any():
                continue
            if key == zeros:
                continue  # complement handled separately

            value = arr[key]
            if value is None:
                continue

            last_bit = key[-1]

            angle_raw = _arc_angle_for_region(mask, circle_band, theta, U, V)
            if angle_raw is None:
                continue

            v_out = np.array([np.cos(angle_raw), np.sin(angle_raw)], float)

            if last_bit == 1:
                r_lab = R * (1.0 - float(region_radial_offset_inside))
            else:
                r_lab = R * (1.0 + float(region_radial_offset_outside))

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

            d = float(np.dot(v_circle, v_base))
            ha = "right" if d >= 0 else "left"
            va = "center"

            if text_color is None:
                rgb = region_rgbs.get(key)
                this_color = _auto_text_color_from_rgb(rgb) if rgb is not None else "black"
            else:
                this_color = text_color

            ax.text(
                u_lab,
                v_lab,
                f"{value}",
                ha=ha,
                va=va,
                fontsize=region_label_fontsize,
                color=this_color,
                zorder=5,
                rotation=rot,
                rotation_mode="anchor",
            )

    # Complement (all zeros) – visual center, no rotation
    comp_mask = region_masks.get(zeros)
    if comp_mask is not None and comp_mask.any():
        val_comp = arr[zeros]
        if val_comp is not None:
            pos = _visual_center_margin(comp_mask, U, V, margin_frac=0.05)
            if pos is not None:
                if text_color is None:
                    this_color = "black"
                else:
                    this_color = text_color

                ax.text(
                    pos[0],
                    pos[1],
                    f"{val_comp}",
                    ha="center",
                    va="center",
                    fontsize=region_label_fontsize,
                    color=this_color,
                    zorder=5,
                    rotation=0.0,
                    rotation_mode="anchor",
                )

    # --- Class labels on vennfan ---
    label_offset = 0.18 * height_scale
    dx_const, dy_const = last_constant_label_offset  # currently unused
    extra_radial_offset = 0.05  # your requested extra radial offset for i >= 4

    for i, (name, label_col) in enumerate(zip(class_names, label_rgbs)):
        if not name:
            continue  # skip empty labels
        h_i = harmonics_for_class[i]

        # --- Compute anchor geometry (angle_anchor, v_out, r_lab, u_lab, v_lab) ---
        if h_i is None and i == N - 1:
            # Constant last class: fixed radial angle, outside circle
            angle_anchor = -2.0 * np.pi / (2.0 ** (N))
            r_anchor = R  # conceptually on the circle
            v_out = np.array([np.cos(angle_anchor), np.sin(angle_anchor)], float)
            # base radial label radius
            r_lab = R * (1.0 + float(region_radial_offset_outside))
        else:
            # Non-constant classes: true bisector on boundary, mapped outwards
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
            else:
                # Fallback to curve minimum in half-plane
                y_plot = curves[i]
                dyc = np.diff(y_plot)
                signc = np.sign(dyc)
                i_min = None
                for j in range(1, len(signc)):
                    if signc[j - 1] < 0 and signc[j] > 0:
                        i_min = j
                if i_min is None:
                    i_min = int(np.argmin(y_plot))
                x_bis = x_plot[i_min]
                y_bis = y_plot[i_min]

            u_anchor_arr, v_anchor_arr = _halfplane_to_disc(
                np.array([x_bis]),
                np.array([y_bis]),
                R,
                y_min,
                y_max,
            )
            u_anchor = float(u_anchor_arr[0])
            v_anchor = float(v_anchor_arr[0])

            angle_anchor = float(np.arctan2(v_anchor, u_anchor))
            r_anchor = float(np.sqrt(u_anchor * u_anchor + v_anchor * v_anchor))
            if r_anchor == 0.0:
                v_out = np.array([1.0, 0.0], float)
            else:
                v_out = np.array([u_anchor, v_anchor], float) / r_anchor

            r_lab = r_anchor + float(region_radial_offset_outside) * R

        # Extra radial offset for 5th+ classes
        if i >= 3:
            r_lab += extra_radial_offset * R
        else:
            r_lab += extra_radial_offset * R *2

        u_lab = r_lab * v_out[0]
        v_lab = r_lab * v_out[1]

        # --- Orientation & alignment ---
        angle_deg = np.degrees(angle_anchor)+5

        if i < 3:
            # Small indices: tangential, anchor at center
            deg_tangent = _normalize_angle_90(angle_deg + 90.0)
            rot_cls = deg_tangent
            ha = "center"
            va = "center"
        else:
            # i >= 4: radial, anchor at left side
            deg_radial = _normalize_angle_90(angle_deg)
            rot_cls = deg_radial
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


# ---------------------------------------------------------------------------
# Simple helper for demos
# ---------------------------------------------------------------------------

def _make_demo_values(N: int) -> np.ndarray:
    """
    Label each region by which sets it belongs to, e.g. "", "A", "BC", "ABCDE", etc.
    For testing, the complement (all zeros) is labeled None.
    """
    letters = [chr(ord("A") + i) for i in range(N)]
    shape = (2,) * N
    arr = np.empty(shape, dtype=object)
    for idx in np.ndindex(shape):
        s = "".join(letters[i] for i, bit in enumerate(idx) if bit)
        arr[idx] = s
    arr[(0,) * N] = None
    return arr


if __name__ == "__main__":
    os.makedirs("img", exist_ok=True)
    greek_names = [
        "Alpha", "Beta", "Gamma", "Delta", "Epsilon",
        "Zeta", "Eta", "Theta", "Iota", "Kappa"
    ]

    # # Rectangular version (N=6) in various modes
    # for curve_mode in ["cosine", "sine"]:
    #     for linear_scale in [True, False]:
    #         for N in range(6, 7):
    #             print(f"Generating diagram for curve_mode={curve_mode} linear_scale={linear_scale} N={N} ...")
    #             values = _make_demo_values(N)
    #             class_names = greek_names[:N]

    #             outfile = f"img/{curve_mode}{'_linear' if linear_scale else ''}_N{N}.png"
    #             venntrig(
    #                 values,
    #                 class_names,
    #                 outfile=outfile,
    #                 height_scale=2.0,
    #                 curve_exponent=0.2,
    #                 amp_decay_base=0.8,
    #                 include_constant_last=True,
    #                 curve_mode=curve_mode,
    #                 linear_scale=linear_scale,
    #             )
    fontsizes = {
        2: (16, 20),
        3: (14, 18),
        4: (12, 14),
        5: (11, 13),
        6: (9, 12),
        7: (6, 10),
        8: (4, 6),
        9: (2, 3),
        10: (0.5, 1),
    }
    linewidths = {
        2: 4.0,
        3: 3.5,
        4: 3.0,
        5: 2.5,
        6: 2.0,
        7: 1.5,
        8: 1.2,
        9: 0.8,
        10: 0.3
    }
    for curve_mode in ["cosine", "sine"]:
        for linear_scale in [True, False]:
            for N in range(6, 7):
                print(f"Generating vennfan diagram for curve_mode={curve_mode} linear_scale={linear_scale} N={N}...")
                # shape = (2,) * N
                # values = np.empty(shape, dtype=object)
                # class_names = ["" for _ in range(N)]
                values = _make_demo_values(N)
                class_names = greek_names[:N]

                outfile = f"img/vennfan_{curve_mode}{'_linear' if linear_scale else ''}_N{N}.png"
                vennfan(
                    values,
                    class_names,
                    outfile=outfile,
                    height_scale=2.0,
                    include_constant_last=True,
                    region_label_fontsize=fontsizes[N][0],
                    class_label_fontsize=fontsizes[N][1],
                    text_color="black",
                    curve_exponent=0.2,
                    amp_decay_base=0.8,
                    region_radial_offset_inside=0.02,
                    region_radial_offset_outside=0.02,
                    linewidth=linewidths[N],
                    curve_mode=curve_mode,
                    linear_scale=linear_scale,
                )
