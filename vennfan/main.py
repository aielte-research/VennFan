#!/usr/bin/env python3
"""
Sine-curve "Venn" diagrams up to N=9, plus a "vennfan" circular variant.

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
    _rgb,
    _auto_text_color_from_rgb,
    _color_mix_subtractive,
    _color_mix_average,
    _color_mix_hue_average,
    _color_mix_alpha_stack,
)
from defaults import (
    _default_palette_for_n,
    _default_fontsize,
    _default_adaptive_fontsize,
    _default_linewidth,
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
    yy, xx = np.unravel_index(np.argmax(dist), mask.shape)
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
    yy, xx = np.unravel_index(np.argmax(dist), m2.shape)
    return float(X[yy, xx]), float(Y[yy, xx])


def _normalize_angle_90(deg: float) -> float:
    """Map any angle (deg) to an equivalent in about [-90, +90] for legible text."""
    a = float(deg)
    while a > 95.0:
        a -= 180.0
    while a < -85.0:
        a += 180.0
    return a


def class_label_angles(N: int, curve_mode: str) -> List[float]:
    """Generate N angular positions (degrees) with halving differences."""
    terms: List[float] = []
    
    angle = 90.0
    diff = 135.0
    if curve_mode=="cosine":
        terms.append(90)
        angle = 180
        diff = 90
    for _ in range(N - 1):
        terms.append(angle)
        angle += diff
        diff *= 0.5
    # Last (constant) class
    if curve_mode=="sine":
        terms.append(-360.0 / (2.0 ** N))
    else:
        terms[-1]=(-360.0 / (2.0 ** N))
    return terms

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


def _shrink_text_font_to_region(
    fig: Figure,
    ax,
    text: str,
    x: float,
    y: float,
    base_fontsize: float,
    mask: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    rotation: float = 0.0,
    ha: str = "center",
    va: str = "center",
    shrink_factor: float = 0.9,
    min_fraction: float = 0.25,
    max_iterations: int = 12,
    erosion_radius_pix: Optional[float] = None,
) -> float:
    """
    Given a region mask on grid (X, Y), shrink the fontsize by 10% steps
    until a sample of points inside the text's bounding box all lie inside
    the region. If it never fits, returns the last tried size.

    Before testing, the region mask is uniformly eroded by a distance
    (in grid cells) ≈ linewidth * 1.5, passed as `erosion_radius_pix`,
    to approximate the curve linewidth.
    """
    if base_fontsize <= 0.0:
        return base_fontsize
    if mask is None or not isinstance(mask, np.ndarray) or not mask.any():
        return base_fontsize

    H, W = mask.shape

    # --- Uniform erosion in pixel units (≈ linewidth * 1.5) ---
    if erosion_radius_pix is not None and erosion_radius_pix > 0.0:
        margin_pix = int(round(float(erosion_radius_pix)))
        if margin_pix > 0:
            dist_reg = distance_transform_edt(mask)
            mask_eroded = dist_reg >= margin_pix
            if mask_eroded.any():
                mask = mask_eroded

    canvas = fig.canvas
    renderer = canvas.get_renderer()

    xs_grid = X[0, :]
    ys_grid = Y[:, 0]

    if xs_grid.size > 1:
        dx = xs_grid[1] - xs_grid[0]
        x0_grid = xs_grid[0]
    else:
        x0_grid = float(X.min())
        dx = float((X.max() - X.min()) / max(W - 1, 1))

    if ys_grid.size > 1:
        dy = ys_grid[1] - ys_grid[0]
        y0_grid = ys_grid[0]
    else:
        y0_grid = float(Y.min())
        dy = float((Y.max() - Y.min()) / max(H - 1, 1))

    fs = float(base_fontsize)
    min_fs = max(0.1, float(base_fontsize) * float(min_fraction))

    inv_trans = ax.transData.inverted()

    for _ in range(max_iterations):
        if fs <= 0.0:
            break

        # Create a temporary text object
        t = ax.text(
            x,
            y,
            text,
            ha=ha,
            va=va,
            fontsize=fs,
            rotation=rotation,
            rotation_mode="anchor",
        )
        t.set_clip_on(False)

        # Compute bbox in display coords, then transform to data coords
        t.draw(renderer)
        bbox_disp = t.get_window_extent(renderer=renderer)
        t.remove()

        bbox_data = bbox_disp.transformed(inv_trans)
        x0d, y0d = bbox_data.x0, bbox_data.y0
        x1d, y1d = bbox_data.x1, bbox_data.y1

        # Sample a small grid of points inside the bbox
        nx, ny = 9, 5
        xs_samp = np.linspace(x0d, x1d, nx)
        ys_samp = np.linspace(y0d, y1d, ny)

        fits = True
        for yy in ys_samp:
            if not fits:
                break
            for xx in xs_samp:
                ix = int(round((xx - x0_grid) / dx))
                iy = int(round((yy - y0_grid) / dy))
                if ix < 0 or ix >= W or iy < 0 or iy >= H:
                    fits = False
                    break
                if not mask[iy, ix]:
                    fits = False
                    break

        if fits or fs <= min_fs:
            return max(fs, min_fs)

        fs *= float(shrink_factor)

    return max(fs, min_fs)


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
    color_mixing: Union[str, Callable[[Sequence[np.ndarray]], np.ndarray]] = "alpha_stack",
    text_color: Optional[str] = None,
    region_label_fontsize: Optional[float] = None,
    class_label_fontsize: Optional[float] = None,
    complement_fontsize: float = 6.0,
    adaptive_fontsize: Optional[bool] = None,
    adaptive_fontsize_range: Optional[Tuple[float, float]] = None,
    sample_res_x: int = 900,
    sample_res_y: int = 900,
    include_constant_last: bool = True,
    curve_exponent: float = 0.33,
    amp_decay_base: float = 0.75,
    linewidth: Optional[float] = None,
    curve_mode: str = "sine",
    height_scale: float = 2.0,
    linear_scale: bool = True,
) -> Optional[Figure]:
    """
    Rectangular version.
    """
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

    # Default linewidth per N if not provided
    if linewidth is None:
        linewidth = _default_linewidth(N)

    zeros = (0,) * N
    ones = (1,) * N

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

    # Default font sizes if None
    if region_label_fontsize is None or class_label_fontsize is None:
        base_fs_region, base_fs_class = _default_fontsize(N, linear_scale, curve_mode)
        if region_label_fontsize is None:
            region_label_fontsize = base_fs_region
        if class_label_fontsize is None:
            class_label_fontsize = base_fs_class

    # Color mixing callback (uses fill colors)
    if isinstance(color_mixing, str):
        if color_mixing == "subtractive":
            mixing_cb = _color_mix_subtractive
        elif color_mixing == "average":
            mixing_cb = _color_mix_average
        elif color_mixing == "hue_average":
            mixing_cb = lambda x: _color_mix_hue_average(x, N)
        elif color_mixing == "alpha_stack":
            mixing_cb = _color_mix_alpha_stack
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

    # --- Compute region areas (for adaptive font sizes) ---
    if xs.size > 1:
        dx = xs[1] - xs[0]
    else:
        dx = (x_max - x_min) / max(W - 1, 1)
    if ys.size > 1:
        dy = ys[1] - ys[0]
    else:
        dy = (y_max - y_min) / max(H - 1, 1)
    pixel_area = abs(dx * dy)

    region_areas: Dict[Tuple[int, ...], float] = {
        key: float(mask.sum()) * pixel_area for key, mask in region_masks.items()
    }

    noncomp_keys = [k for k in region_masks.keys() if k != zeros and region_areas.get(k, 0.0) > 0.0]
    if noncomp_keys:
        area_min = min(region_areas[k] for k in noncomp_keys)
        area_max = max(region_areas[k] for k in noncomp_keys)
    else:
        area_min = area_max = 0.0

    # Decide whether adaptive fontsize is on
    if adaptive_fontsize is None:
        adaptive_fontsize = bool(linear_scale)
    else:
        adaptive_fontsize = bool(adaptive_fontsize)

    # Determine font size range  (TREATED AS (fs_min, fs_max))
    if adaptive_fontsize and area_max > 0.0:
        if adaptive_fontsize_range is not None:
            fs_min, fs_max = adaptive_fontsize_range
            if fs_min > fs_max:
                fs_min, fs_max = fs_max, fs_min
        else:
            # Default adaptive range based on N / linear_scale
            fs_min, fs_max = _default_adaptive_fontsize(N, linear_scale)
    else:
        fs_min = fs_max = float(region_label_fontsize)

    region_fontsizes: Dict[Tuple[int, ...], float] = {}
    if adaptive_fontsize and area_max > 0.0 and area_max >= area_min:
        denom = (area_max - area_min) if area_max > area_min else 1.0
        for key, area in region_areas.items():
            if area_max > area_min:
                t = (area - area_min) / denom
            else:
                t = 0.5
            t = max(0.0, min(1.0, t))
            # Larger area → larger fontsize
            fs = fs_min + t * (fs_max - fs_min)
            region_fontsizes[key] = fs
    else:
        for key in region_masks.keys():
            region_fontsizes[key] = float(region_label_fontsize)

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
    fig, ax = plt.subplots(figsize=(5 + 2.5 * N, 5 * height_scale))
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

    # Class boundaries (store curves for plotting & later class-label logic)
    x_plot = np.linspace(x_min, x_max, 1200)
    curves: List[np.ndarray] = []
    harmonics_for_class: List[Optional[float]] = []

    if curve_mode == "sine":
        curve_fn_plot = get_sine_curve
    else:
        curve_fn_plot = get_cosine_curve

    # First: compute curves once
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

    # Then: draw class outlines in two passes: alpha 1.0 then 0.5
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

    # Make sure a renderer exists for text extent calculations
    fig.canvas.draw()

    # --- Region labels ---
    const_y = 0.0
    region_offset = 0.02 * (y_max - y_min)
    erosion_radius_pix = linewidth * 1.5

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

        fs_here = region_fontsizes.get(key, float(region_label_fontsize))

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

    # All-sets intersection
    all_mask = np.logical_and.reduce(membership)
    if all_mask.any():
        val_all = arr[ones]
        if val_all is not None:
            fs_all = region_fontsizes.get(ones, float(region_label_fontsize))
            if linear_scale:
                # Linear: use margin-based visual center, no rotation
                pos = _visual_center_margin(all_mask, X, Y, margin_frac=0.05)
                if pos is not None:
                    if text_color is None:
                        rgb = region_rgbs.get(ones)
                        this_color = _auto_text_color_from_rgb(rgb) if rgb is not None else "black"
                    else:
                        this_color = text_color

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
                fs_comp = float(complement_fontsize)

                x_lab, y_lab = pos
                rot = 0.0
                ha = "center"
                va = "center"

                fs_adj = _shrink_text_font_to_region(
                    fig,
                    ax,
                    f"{val_comp}",
                    x_lab,
                    y_lab,
                    fs_comp,
                    comp_mask,
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
                    f"{val_comp}",
                    ha=ha,
                    va=va,
                    fontsize=fs_adj,
                    color=this_color,
                    zorder=5,
                    rotation=rot,
                    rotation_mode="anchor",
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


def _second_radial_intersection(
    u_curve: np.ndarray,
    v_curve: np.ndarray,
    angle_rad: float,
) -> Optional[Tuple[float, float]]:
    """
    Find the *second* intersection (by increasing radius) of a radial ray
    at angle `angle_rad` with a polyline (u_curve, v_curve).

    Returns (u, v) of the chosen intersection, or None if no intersection.
    """
    u = np.asarray(u_curve, float)
    v = np.asarray(v_curve, float)
    if u.size < 2:
        return None

    # Unit direction of the ray
    dir_x = float(np.cos(angle_rad))
    dir_y = float(np.sin(angle_rad))

    # Cross product sign and dot products with the ray direction
    # cross(dir, P) = dir_x * v - dir_y * u
    s = dir_x * v - dir_y * u
    dot = dir_x * u + dir_y * v

    intersections: List[Tuple[float, float]] = []

    for k in range(u.size - 1):
        s0 = s[k]
        s1 = s[k + 1]
        d0 = dot[k]
        d1 = dot[k + 1]

        # Skip segment completely behind the origin along the ray
        if d0 <= 0.0 and d1 <= 0.0:
            continue

        # Exact hit on a vertex
        if s0 == 0.0 and d0 > 0.0:
            intersections.append((u[k], v[k]))
            continue

        # Sign change → crossing
        if s0 * s1 < 0.0:
            denom_s = s0 - s1
            if denom_s == 0.0:
                continue
            t_seg = s0 / denom_s  # in [0,1]
            if t_seg < 0.0 or t_seg > 1.0:
                continue
            u_int = u[k] + (u[k + 1] - u[k]) * t_seg
            v_int = v[k] + (v[k + 1] - v[k]) * t_seg
            d_int = dir_x * u_int + dir_y * v_int
            if d_int <= 0.0:
                continue
            intersections.append((u_int, v_int))

    if not intersections:
        return None

    # Sort by radius and pick the *second* intersection if it exists
    radii = [ui * ui + vi * vi for (ui, vi) in intersections]
    idxs = np.argsort(radii)
    if idxs.size >= 2:
        j = int(idxs[1])
    else:
        j = int(idxs[0])

    return intersections[j]


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
    color_mixing: Union[str, Callable[[Sequence[np.ndarray]], np.ndarray]] = "alpha_stack",
    text_color: Optional[str] = None,
    region_label_fontsize: Optional[float] = None,
    class_label_fontsize: Optional[float] = None,
    complement_fontsize: float = 8.0,
    adaptive_fontsize: Optional[bool] = None,
    adaptive_fontsize_range: Optional[Tuple[float, float]] = None,
    height_scale: float = 2.0,
    include_constant_last: bool = True,
    curve_exponent: float = 0.2,
    amp_decay_base: float = 0.8,
    last_constant_label_offset: Tuple[float, float] = (0.0, 0.0),
    region_radial_offset_inside: float = 0.05,
    region_radial_offset_outside: float = 0.05,
    linewidth: Optional[float] = None,
    curve_mode: str = "cosine",
    linear_scale: bool = True,
    y_min: float = -1.0,
    y_max: float = 1.0,
) -> Optional[Figure]:
    """
    vennfan variant of the sine diagram.

    The underlying rectangular diagram is considered over [0, 2π] × [y_min, y_max],
    where y_min / y_max can be tweaked via rect_ymin / rect_ymax. If they are
    None, the original defaults (-1+1/N, 1-1/N) are used.
    """

    if curve_mode not in ("cosine", "sine"):
        raise ValueError(f"Unsupported curve_mode {curve_mode!r}; use 'cosine' or 'sine'.")

    if curve_mode == "sine":
        curve_fn = get_sine_curve
    else:
        curve_fn = get_cosine_curve

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

    # Default linewidth per N if not provided
    if linewidth is None:
        linewidth = _default_linewidth(N)

    zeros = (0,) * N
    ones = (1,) * N

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

    # Default font sizes if None
    if region_label_fontsize is None or class_label_fontsize is None:
        base_fs_region, base_fs_class = _default_fontsize(N, linear_scale, curve_mode)
        if region_label_fontsize is None:
            region_label_fontsize = base_fs_region
        if class_label_fontsize is None:
            class_label_fontsize = base_fs_class

    # Color mixing callback
    if isinstance(color_mixing, str):
        if color_mixing == "subtractive":
            mixing_cb = _color_mix_subtractive
        elif color_mixing == "average":
            mixing_cb = _color_mix_average
        elif color_mixing == "hue_average":
            mixing_cb = lambda x: _color_mix_hue_average(x, N)
        elif color_mixing == "alpha_stack":
            mixing_cb = _color_mix_alpha_stack
        else:
            raise ValueError(f"Unrecognized color_mixing string: {color_mixing!r}")
    elif callable(color_mixing):
        mixing_cb = color_mixing
    else:
        raise TypeError("color_mixing must be either a string or a callable.")

    x_min, x_max = 0.0, 2.0 * np.pi

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

    # --- Compute region areas (vennfan plane) for adaptive fonts ---
    if us.size > 1:
        du = us[1] - us[0]
    else:
        du = (R_out - (-R_out)) / max(W - 1, 1)
    if vs.size > 1:
        dv = vs[1] - vs[0]
    else:
        dv = (R_out - (-R_out)) / max(H - 1, 1)
    pixel_area = abs(du * dv)

    region_areas: Dict[Tuple[int, ...], float] = {
        key: float(mask.sum()) * pixel_area for key, mask in region_masks.items()
    }

    noncomp_keys = [k for k in region_masks.keys() if k != zeros and region_areas.get(k, 0.0) > 0.0]
    if noncomp_keys:
        area_min = min(region_areas[k] for k in noncomp_keys)
        area_max = max(region_areas[k] for k in noncomp_keys)
    else:
        area_min = area_max = 0.0

    # Decide whether adaptive fontsize is on (default: ON only if linear_scale=True)
    if adaptive_fontsize is None:
        adaptive_fontsize = bool(linear_scale)
    else:
        adaptive_fontsize = bool(adaptive_fontsize)

    # Determine font size range  (TREATED AS (fs_min, fs_max))
    if adaptive_fontsize and area_max > 0.0:
        if adaptive_fontsize_range is not None:
            fs_min, fs_max = adaptive_fontsize_range
            if fs_min > fs_max:
                fs_min, fs_max = fs_max, fs_min
        else:
            fs_min, fs_max = _default_adaptive_fontsize(N, linear_scale)
    else:
        fs_min = fs_max = float(region_label_fontsize)

    region_fontsizes: Dict[Tuple[int, ...], float] = {}
    if adaptive_fontsize and area_max > 0.0 and area_max >= area_min:
        denom = (area_max - area_min) if area_max > area_min else 1.0
        for key, area in region_areas.items():
            if area_max > area_min:
                t = (area - area_min) / denom
            else:
                t = 0.5
            t = max(0.0, min(1.0, t))
            # Larger area → larger fontsize
            fs = fs_min + t * (fs_max - fs_min)
            region_fontsizes[key] = fs
    else:
        for key in region_masks.keys():
            region_fontsizes[key] = float(region_label_fontsize)

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
    disc_u: List[np.ndarray] = []
    disc_v: List[np.ndarray] = []

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
        disc_u.append(u_curve)
        disc_v.append(v_curve)

    # Draw each class outline twice: alpha 1.0 then alpha 0.5
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
            # If the mapped curve is not closed (start and end differ),
            # explicitly connect the endpoints with a straight segment.
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

    # Precompute some stuff for label placement
    rho = np.sqrt(U * U + V * V)
    circle_band = np.abs(rho - R) <= (0.03 * R)
    theta = np.mod(np.arctan2(V, U), 2.0 * np.pi)

    # Make sure a renderer exists for text extent calculations
    fig.canvas.draw()

    erosion_radius_pix = linewidth * 1.5

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

            fs_here = region_fontsizes.get(key, float(region_label_fontsize))
            u_lab, v_lab = pos
            rot = 0.0
            ha = "center"
            va = "center"

            # In vennfan linear_scale=True we keep shrink-to-fit
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

            fs_here = region_fontsizes.get(key, float(region_label_fontsize))

            # IMPORTANT: for linear_scale=False, we ONLY shrink if adaptive_fontsize is enabled.
            if adaptive_fontsize:
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
            else:
                fs_adj = fs_here

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

    # Complement (all zeros) – fixed bottom-right corner label
    comp_mask = region_masks.get(zeros)
    if comp_mask is not None and comp_mask.any():
        val_comp = arr[zeros]
        if val_comp is not None:
            if text_color is None:
                this_color = "black"
            else:
                this_color = text_color
            fs_comp = float(complement_fontsize)

            # Bottom-right corner of the vennfan canvas, inset by 0.1 in both x and y.
            u_lab = R_out - 0.1
            v_lab = -R_out + 0.1
            rot = 0.0
            ha = "right"
            va = "bottom"

            # Same rule: only shrink in nonlinear vennfan if adaptive_fontsize is on
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

    # --- Class labels on vennfan ---
    label_offset = 0.18 * height_scale  # currently unused, kept for compatibility
    dx_const, dy_const = last_constant_label_offset  # currently unused
    extra_radial_offset = 0.05  # extra radial offset

    # Precomputed radial angles (in degrees) for all class labels
    label_angle_degs = class_label_angles(N, curve_mode)

    for i, (name, label_col) in enumerate(zip(class_names, label_rgbs)):
        if not name:
            continue  # skip empty labels

        # Radial angle for this label
        angle_deg_radial = label_angle_degs[i % len(label_angle_degs)]
        angle_anchor = np.deg2rad(angle_deg_radial)
        v_out = np.array([np.cos(angle_anchor), np.sin(angle_anchor)], float)

        # --- Position: second intersection with its own boundary + small offset ---
        u_curve = disc_u[i]
        v_curve = disc_v[i]
        inter = _second_radial_intersection(u_curve, v_curve, angle_anchor)
        if inter is not None:
            u_int, v_int = inter
            r_anchor = float(np.sqrt(u_int * u_int + v_int * v_int))
        else:
            # Fallback: on the main circle
            r_anchor = R

        # Base radius: just outside that anchor point
        r_lab = r_anchor + float(region_radial_offset_outside) * R

        # Extra radial offset (slightly more for the first few, as before)
        if i >= 3:
            r_lab += extra_radial_offset * R
        else:
            r_lab += extra_radial_offset * R * 2

        u_lab = r_lab * v_out[0]
        v_lab = r_lab * v_out[1]

        # --- Rotation: normalize(angle-90) for i<3, normalize(angle) otherwise ---
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


# ---------------------------------------------------------------------------
# Simple helper for demos
# ---------------------------------------------------------------------------

def _make_demo_values(N: int) -> np.ndarray:
    """
    Label each region by which sets it belongs to, e.g. "", "A", "BC", "ABCDE", etc.
    For testing, the complement (all zeros) is labeled with a copyright notice.
    """
    letters = [chr(ord("A") + i) for i in range(N)]
    shape = (2,) * N
    arr = np.empty(shape, dtype=object)
    for idx in np.ndindex(shape):
        s = "".join(letters[i] for i, bit in enumerate(idx) if bit)
        arr[idx] = s
    arr[(0,) * N] = "© Bálint Csanády\nhttps://github.com/aielte-research/VennFan"
    return arr


if __name__ == "__main__":
    os.makedirs("img", exist_ok=True)
    greek_names = [
        "Alpha", "Beta", "Gamma", "Delta", "Epsilon",
        "Zeta", "Eta", "Theta", "Iota", "Kappa"
    ]
  

    for curve_mode in ["cosine", "sine"]:
        for linear_scale in [False, True]:
            for N in range(1, 10):  # 8, 9 only
                print(f"Generating vennfan diagram for curve_mode={curve_mode} linear_scale={linear_scale} N={N}...")
                # shape = (2,) * N
                # values = np.empty(shape, dtype=object)
                # class_names = ["" for _ in range(N)]
                values = _make_demo_values(N)
                class_names = greek_names[:N]

                outfile = f"img/vennfan_{curve_mode}{'_linear' if linear_scale else ''}_N{N}.pdf"
                vennfan(
                    values,
                    class_names,
                    outfile=outfile,
                    height_scale=2.0,
                    include_constant_last=True,
                    curve_exponent=0.33 if linear_scale else 0.15,
                    amp_decay_base=0.85,
                    region_radial_offset_inside=0.05,
                    region_radial_offset_outside=0.05,
                    curve_mode=curve_mode,
                    linear_scale=linear_scale,
                )
    
    for curve_mode in ["cosine","sine"]:
        for linear_scale in [False, True]:
            for N in range(1, 9):
                print(f"Generating diagram for curve_mode={curve_mode} linear_scale={linear_scale} N={N} ...")
                values = _make_demo_values(N)
                class_names = greek_names[:N]
    
                outfile = f"img/{curve_mode}{'_linear' if linear_scale else ''}_N{N}.pdf"
                venntrig(
                    values,
                    class_names,
                    outfile=outfile,
                    height_scale=2.0,
                    curve_exponent=0.33 if linear_scale else 0.2,
                    amp_decay_base=0.8,
                    include_constant_last=True,
                    curve_mode=curve_mode,
                    linear_scale=linear_scale,
                )

                
    # for curve_mode in ["sine", "cosine"]:
    #     for linear_scale in [False]:
    #         for N in range(1, 10):  # 8, 9 only
    #             print(f"Generating vennfan diagram for curve_mode={curve_mode} linear_scale={linear_scale} N={N}...")
    #             # shape = (2,) * N
    #             # values = np.empty(shape, dtype=object)
    #             # class_names = ["" for _ in range(N)]
    #             values = _make_demo_values(N)
    #             class_names = greek_names[:N]

    #             outfile = f"img/Edvards_{curve_mode}{'_linear' if linear_scale else ''}_N{N}.png"
    #             vennfan(
    #                 values,
    #                 class_names,
    #                 outfile=outfile,
    #                 height_scale=2.0,
    #                 include_constant_last=True,
    #                 curve_exponent=1.0,
    #                 amp_decay_base=0.5,
    #                 region_radial_offset_inside=0.05,
    #                 region_radial_offset_outside=0.05,
    #                 curve_mode=curve_mode,
    #                 linear_scale=linear_scale,
    #             )
