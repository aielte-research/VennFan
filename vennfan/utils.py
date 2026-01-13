#!/usr/bin/env python3
"""
Shared helper functions for venntrig.py and vennfan.py.

This module contains:
- region & mask utilities
- visual / centroid / inset centers
- distance-transform-based text sizing
- basic curve / harmonic helpers
- half-plane → disc mapping helpers for vennfan
- reusable logic for color mixing and adaptive font sizes
- logic for region label placement modes (radial vs visual-center)
- a rectangle-based visual_text_center placement helper
"""

from typing import (
    Sequence,
    Optional,
    Union,
    Tuple,
    Dict,
    Callable,
    List,
)

import itertools

import numpy as np
from matplotlib.figure import Figure
from scipy.ndimage import distance_transform_edt

from .colors import (
    auto_text_color_from_rgb,
    color_mix_subtractive,
    color_mix_average,
    color_mix_hue_average,
    color_mix_alpha_stack,
)
from .defaults import _default_adaptive_fontsize


# ---------------------------------------------------------------------------
# Region & mask helpers
# ---------------------------------------------------------------------------


def disjoint_region_masks(masks_list: Sequence[np.ndarray]) -> Dict[Tuple[int, ...], np.ndarray]:
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


def visual_center(mask: np.ndarray, X: np.ndarray, Y: np.ndarray) -> Optional[Tuple[float, float]]:
    """Visual center via Euclidean distance transform (SciPy)."""
    if not mask.any():
        return None
    dist = distance_transform_edt(mask)
    yy, xx = np.unravel_index(np.argmax(dist), mask.shape)
    return float(X[yy, xx]), float(Y[yy, xx])


def centroid(mask: np.ndarray, X: np.ndarray, Y: np.ndarray) -> Optional[Tuple[float, float]]:
    """Simple centroid of True pixels in mask."""
    if not mask.any():
        return None
    yy, xx = np.where(mask)
    return float(X[yy, xx].mean()), float(Y[yy, xx].mean())


def visual_center_margin(
    mask: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    margin_frac: float = 0.05,
) -> Optional[Tuple[float, float]]:
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
        return visual_center(mask, X, Y)

    dist = distance_transform_edt(m2)
    yy, xx = np.unravel_index(np.argmax(dist), m2.shape)
    return float(X[yy, xx]), float(Y[yy, xx])


def visual_center_inset(
    mask: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    n_pix: int = 2,
) -> Optional[Tuple[float, float]]:
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
        pos = visual_center(mask_inset, X, Y)
    else:
        pos = visual_center(mask, X, Y)

    if pos is None:
        return None

    x_lab, y_lab = pos
    # Clamp into inset box so we never land exactly on bbox edges.
    x_lab = min(max(x_lab, inset_x_min), inset_x_max)
    y_lab = min(max(y_lab, inset_y_min), inset_y_max)
    return x_lab, y_lab


def region_constant_line_bisector(
    mask: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
) -> Optional[Tuple[float, float]]:
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
# Text placement & sizing
# ---------------------------------------------------------------------------

def shrink_text_font_to_region(
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
    shrink_factor: float = 0.90,
    min_fraction: float = 0.25,  # lower bound for how small we allow shrinking
    max_iterations: int = 12,
    erosion_radius_pix: Optional[float] = None,
    debug_mode: bool = False,
) -> float:
    """
    Shrink fontsize so that the *rotated* text's bounding parallelogram lies
    entirely inside `mask` (0 tolerance for crossing), up to sampling.

    Implementation details
    ----------------------
    - Render UNROTATED text to get a tight bbox in display coords.
    - Compute anchor position in display coords.
    - Rotate bbox corners around anchor by `rotation` degrees.
    - Transform rotated corners back to data coords → oriented parallelogram.
    - Sample points inside that parallelogram and check that *all* those
      samples lie inside `mask` (zero tolerance). If any sample lies outside,
      shrink and retry.
    - Optional safety margin:
        * If `erosion_radius_pix` > 0, we erode `mask` uniformly by that
          amount (in grid cells, approximating linewidth) using a distance
          transform, and perform the inside-test against this eroded mask.
          If erosion removes the region completely, we fall back to the
          original mask.

    If `debug_mode` is True, the accepted rotated bbox is drawn in red.
    """
    if base_fontsize <= 0.0:
        return base_fontsize
    if mask is None or not isinstance(mask, np.ndarray) or not mask.any():
        return base_fontsize

    # --- Effective mask, possibly eroded by erosion_radius_pix -------------
    mask_eff = mask
    if erosion_radius_pix is not None and erosion_radius_pix > 0.0:
        margin_pix = int(round(float(erosion_radius_pix)))
        if margin_pix > 0:
            dist_reg = distance_transform_edt(mask)
            mask_eroded = dist_reg >= margin_pix
            if mask_eroded.any():
                mask_eff = mask_eroded

    H, W = mask_eff.shape

    # Grid spacing / origin in data coordinates
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

    if dx == 0.0:
        dx = 1.0
    if dy == 0.0:
        dy = 1.0

    canvas = fig.canvas
    renderer = canvas.get_renderer()
    inv_trans = ax.transData.inverted()

    # Anchor position in display coordinates
    anchor_disp = ax.transData.transform((x, y))

    fs = float(base_fontsize)
    min_fs = max(0.1, float(base_fontsize) * float(min_fraction))

    # Sampling resolution inside oriented bbox
    nx, ny = 21, 11

    accepted_corners_data = None

    # Rotation matrix in display coordinates
    theta = np.deg2rad(rotation)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    R_mat = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=float)

    for _ in range(max_iterations):
        if fs <= 0.0:
            break

        # 1) Render UNROTATED text at this fontsize to measure its bbox
        t = ax.text(
            x,
            y,
            text,
            ha=ha,
            va=va,
            fontsize=fs,
            rotation=0.0,          # measure without rotation
            rotation_mode="anchor",
        )
        t.set_clip_on(False)

        t.draw(renderer)
        bbox_disp = t.get_window_extent(renderer=renderer)
        t.remove()

        # Axis-aligned bbox corners in display coords (unrotated)
        x0d, y0d = bbox_disp.x0, bbox_disp.y0
        x1d, y1d = bbox_disp.x1, bbox_disp.y1
        corners_disp = np.array(
            [
                [x0d, y0d],
                [x1d, y0d],
                [x1d, y1d],
                [x0d, y1d],
            ],
            dtype=float,
        )

        # 2) Rotate corners around anchor in display coords
        A = anchor_disp
        rel = corners_disp - A[None, :]
        rel_rot = rel @ R_mat.T
        corners_disp_rot = A[None, :] + rel_rot

        # 3) Transform rotated corners back to data coords
        corners_data = inv_trans.transform(corners_disp_rot)
        p0 = corners_data[0]
        p1 = corners_data[1]
        p3 = corners_data[3]

        v_w = p1 - p0
        v_h = p3 - p0

        if np.allclose(v_w, 0) or np.allclose(v_h, 0):
            fs = max(fs, min_fs)
            accepted_corners_data = corners_data
            break

        # 4) Sample points inside oriented parallelogram
        s_vals = np.linspace(0.0, 1.0, nx)
        t_vals = np.linspace(0.0, 1.0, ny)
        S, T = np.meshgrid(s_vals, t_vals)
        P = (
            p0[None, None, :]
            + S[..., None] * v_w[None, None, :]
            + T[..., None] * v_h[None, None, :]
        )
        xs_flat = P[..., 0].ravel()
        ys_flat = P[..., 1].ravel()

        # 5) Map sample points to mask grid indices
        fx = (xs_flat - x0_grid) / dx
        fy = (ys_flat - y0_grid) / dy

        ix0 = np.floor(fx).astype(int)
        iy0 = np.floor(fy).astype(int)
        ix1 = ix0 + 1
        iy1 = iy0 + 1

        inside = np.ones_like(xs_flat, dtype=bool)

        # Only sample where fully in bounds
        in_bounds = (
            (ix0 >= 0)
            & (iy0 >= 0)
            & (ix1 < W)
            & (iy1 < H)
        )

        # If any sample point falls outside the grid at all, we already know
        # this bbox is not acceptable under 0-tolerance:
        if not np.all(in_bounds):
            inside[:] = False
        else:
            fx_in = fx[in_bounds]
            fy_in = fy[in_bounds]
            ix0_in = ix0[in_bounds]
            iy0_in = iy0[in_bounds]
            ix1_in = ix1[in_bounds]
            iy1_in = iy1[in_bounds]

            tx = fx_in - ix0_in
            ty = fy_in - iy0_in

            w00 = (1.0 - tx) * (1.0 - ty)
            w10 = tx * (1.0 - ty)
            w01 = (1.0 - tx) * ty
            w11 = tx * ty

            m00 = mask_eff[iy0_in, ix0_in].astype(float)
            m10 = mask_eff[iy0_in, ix1_in].astype(float)
            m01 = (mask_eff[i1_in := iy1_in, ix0_in]).astype(float)
            m11 = (mask_eff[i1_in, ix1_in]).astype(float)

            val = w00 * m00 + w10 * m10 + w01 * m01 + w11 * m11
            inside[in_bounds] = val >= 0.5

        # Zero tolerance: all sampled points must be inside
        all_inside = bool(inside.all())

        if all_inside or fs <= min_fs:
            fs = max(fs, min_fs)
            accepted_corners_data = corners_data
            break

        fs *= float(shrink_factor)

    if fs < min_fs:
        fs = min_fs

    # Debug: plot accepted rotated bbox in red
    if debug_mode and accepted_corners_data is not None:
        xs_box = accepted_corners_data[:, 0]
        ys_box = accepted_corners_data[:, 1]
        xs_box_closed = np.concatenate([xs_box, xs_box[:1]])
        ys_box_closed = np.concatenate([ys_box, ys_box[:1]])
        ax.plot(
            xs_box_closed,
            ys_box_closed,
            linestyle="-",
            linewidth=0.5,
            color="red",
            zorder=10,
        )

    return fs


def visual_text_center(
    mask: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    rectangle_width: float,
    n_angles: int = 72,
) -> Optional[Tuple[float, float, float]]:
    """
    Rectangle-based "visual text center" placement.

    Given:
        - mask : boolean array (H, W) defining the region,
        - (X, Y) : sampling grid in data coordinates,
        - rectangle_width : width of a rectangle in *data units*,

    this function approximates the placement of the *longest* rectangle with
    that fixed width that can fit inside the region:

        * First, erode the region by rectangle_width / 2 (in data units),
          via distance_transform_edt in pixel space, giving a "core" region
          where the rectangle can safely fit laterally.
        * Choose an interior anchor at the pixel with the largest distance
          to the boundary in this eroded region.
        * For a set of angles theta in [0, pi), extend a line segment through
          that anchor in both directions until it leaves the eroded region.
        * For each angle, this gives a maximal segment; pick the one with
          the greatest length.
        * Return (x_center, y_center, rotation_degrees) where:
              - (x_center, y_center) is the midpoint of that segment
              - rotation_degrees is the angle of the segment, normalized to
                about [-90, +90] via normalize_angle_90.

    The returned rotation is intended as the text rotation; the text can then
    be shrunk to fit the region using shrink_text_font_to_region.
    """
    if not mask.any():
        return None

    # Degenerate width → fall back to plain visual center with no rotation
    if rectangle_width <= 0.0:
        vc = visual_center(mask, X, Y)
        if vc is None:
            return None
        return vc[0], vc[1], 0.0

    H, W = mask.shape

    # Distance transform on the raw mask (pixel units)
    dist = distance_transform_edt(mask)

    xs_grid = X[0, :]
    ys_grid = Y[:, 0]

    if xs_grid.size > 1:
        dx = xs_grid[1] - xs_grid[0]
        x0_grid = xs_grid[0]
    else:
        dx = 1.0
        x0_grid = float(xs_grid[0]) if xs_grid.size else 0.0

    if ys_grid.size > 1:
        dy = ys_grid[1] - ys_grid[0]
        y0_grid = ys_grid[0]
    else:
        dy = 1.0
        y0_grid = float(ys_grid[0]) if ys_grid.size else 0.0

    # Approximate data-units → pixel-units scale
    scale_data_per_pixel = max(float(min(abs(dx), abs(dy))), 1e-9)

    # Margin in pixels for half-width
    margin_pix = float(rectangle_width) / (2.0 * scale_data_per_pixel)

    # Core region where the rectangle's width can fit
    core_mask = dist >= margin_pix

    if not core_mask.any():
        # Width too large for this region; fall back to plain visual center
        vc = visual_center(mask, X, Y)
        if vc is None:
            return None
        return vc[0], vc[1], 0.0

    # Anchor: the most interior point of the core region
    dist_core = dist.copy()
    dist_core[~core_mask] = -1.0
    max_core = dist_core.max()
    if max_core <= 0.0:
        vc = visual_center(mask, X, Y)
        if vc is None:
            return None
        return vc[0], vc[1], 0.0

    iy0, ix0 = np.unravel_index(np.argmax(dist_core), dist_core.shape)
    x_anchor = float(X[iy0, ix0])
    y_anchor = float(Y[iy0, ix0])

    # Step size in data units (roughly one pixel)
    step_x = abs(dx)
    step_y = abs(dy)
    step = float(min(step_x, step_y))
    if step <= 0.0:
        step = 1.0

    # Rough upper bound on how many steps we might need
    span_x = float(X.max() - X.min()) if X.size else 1.0
    span_y = float(Y.max() - Y.min()) if Y.size else 1.0
    max_span = float(np.hypot(span_x, span_y))
    max_steps = max(1, int(max_span / step) + 3)

    # Angle sampling (0..pi because theta and theta+pi give the same line)
    angles = np.linspace(0.0, np.pi, int(n_angles), endpoint=False)

    best_length = -1.0
    best_center_x = x_anchor
    best_center_y = y_anchor
    best_angle = 0.0

    for theta in angles:
        ct = float(np.cos(theta))
        st = float(np.sin(theta))

        # Forward direction (+)
        steps_plus = 0
        x_plus = x_anchor
        y_plus = y_anchor
        for _ in range(max_steps):
            x_try = x_plus + step * ct
            y_try = y_plus + step * st
            ix = int(round((x_try - x0_grid) / dx)) if xs_grid.size > 1 else ix0
            iy = int(round((y_try - y0_grid) / dy)) if ys_grid.size > 1 else iy0
            if ix < 0 or ix >= W or iy < 0 or iy >= H:
                break
            if not core_mask[iy, ix]:
                break
            x_plus = x_try
            y_plus = y_try
            steps_plus += 1

        # Backward direction (-)
        steps_minus = 0
        x_minus = x_anchor
        y_minus = y_anchor
        for _ in range(max_steps):
            x_try = x_minus - step * ct
            y_try = y_minus - step * st
            ix = int(round((x_try - x0_grid) / dx)) if xs_grid.size > 1 else ix0
            iy = int(round((y_try - y0_grid) / dy)) if ys_grid.size > 1 else iy0
            if ix < 0 or ix >= W or iy < 0 or iy >= H:
                break
            if not core_mask[iy, ix]:
                break
            x_minus = x_try
            y_minus = y_try
            steps_minus += 1

        # Segment endpoints and length
        cx = 0.5 * (x_plus + x_minus)
        cy = 0.5 * (y_plus + y_minus)
        length = float(np.hypot(x_plus - x_minus, y_plus - y_minus))

        if length > best_length:
            best_length = length
            best_center_x = cx
            best_center_y = cy
            best_angle = theta

    # Normalize angle into [-90, +90] for text rotation
    angle_deg = float(np.degrees(best_angle))
    angle_deg_norm = normalize_angle_90(angle_deg)

    return best_center_x, best_center_y, angle_deg_norm


def text_color_for_region(
    key: Tuple[int, ...],
    region_rgbs: Dict[Tuple[int, ...], np.ndarray],
    text_color: Optional[str],
    default: str = "black",
) -> str:
    """
    Helper to decide text color for a region:
    - if text_color is given, use that,
    - otherwise auto-pick based on the region RGB if available,
      falling back to `default`.
    """
    if text_color is not None:
        return text_color
    rgb = region_rgbs.get(key)
    if rgb is None:
        return default
    return auto_text_color_from_rgb(rgb)


# ---------------------------------------------------------------------------
# Harmonics, class-label angles, and curve-related helpers
# ---------------------------------------------------------------------------


def harmonic_info_for_index(
    i: int,
    N: int,
    include_constant_last: bool,
) -> Tuple[Optional[float], Optional[float]]:
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


def exclusive_curve_bisector(
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


def normalize_angle_90(deg: float) -> float:
    """
    Map any angle (deg) to an equivalent in about [-90, +90] for legible text.
    """
    a = float(deg)
    while a > 95.0:
        a -= 180.0
    while a < -85.0:
        a += 180.0
    return a


def class_label_angles(N: int, curve_mode: str) -> List[float]:
    """
    Generate N angular positions (degrees) with halving differences
    for class labels in vennfan.
    """
    terms: List[float] = []

    angle = 90.0
    diff = 135.0
    if curve_mode == "cosine":
        terms.append(90.0)
        angle = 180.0
        diff = 90.0
    for _ in range(N - 1):
        terms.append(angle)
        angle += diff
        diff *= 0.5
    # Last (constant) class
    if curve_mode == "sine":
        terms.append(-360.0 / (2.0 ** N))
    else:
        terms[-1] = (-360.0 / (2.0 ** N))
    return terms


# ---------------------------------------------------------------------------
# Color-mixing and adaptive font-size helpers
# ---------------------------------------------------------------------------

def resolve_color_mixing(
    color_mixing: Union[str, Callable],
    N: int,
) -> Callable[[Sequence[np.ndarray], Optional[Sequence[bool]]], np.ndarray]:
    """
    Convert the `color_mixing` parameter into a callable:

        mixing_cb(colors, present=None) -> RGB array

    where
        colors  : sequence of RGB np.ndarrays
        present : full True/False membership list for this region
                  (length N); may be None if not used.
    """
    # User-supplied callable: try (colors, present) first, then fall back to (colors)
    if callable(color_mixing) and not isinstance(color_mixing, str):
        def wrapped(colors: Sequence[np.ndarray], present: Optional[Sequence[bool]] = None) -> np.ndarray:
            try:
                return color_mixing(colors, present)
            except TypeError:
                return color_mixing(colors)
        return wrapped

    if not isinstance(color_mixing, str):
        raise TypeError("color_mixing must be either a string or a callable.")

    # Built-in mixing modes from colors.py
    if color_mixing == "subtractive":
        def cb(colors: Sequence[np.ndarray], present: Optional[Sequence[bool]] = None) -> np.ndarray:
            return color_mix_subtractive(colors, present)
        return cb

    if color_mixing == "average":
        def cb(colors: Sequence[np.ndarray], present: Optional[Sequence[bool]] = None) -> np.ndarray:
            return color_mix_average(colors, present)
        return cb

    if color_mixing == "hue_average":
        def cb(colors: Sequence[np.ndarray], present: Optional[Sequence[bool]] = None) -> np.ndarray:
            return color_mix_hue_average(colors, float(N), present)
        return cb

    if color_mixing == "alpha_stack":
        def cb(colors: Sequence[np.ndarray], present: Optional[Sequence[bool]] = None) -> np.ndarray:
            return color_mix_alpha_stack(colors, present)
        return cb

    raise ValueError(f"Unrecognized color_mixing string: {color_mixing!r}")

def compute_region_fontsizes(
    region_masks: Dict[Tuple[int, ...], np.ndarray],
    pixel_area: float,
    complement_key: Tuple[int, ...],
    base_region_fontsize: float,
    N: int,
    linear_scale: bool,
    adaptive_fontsize: Optional[bool],
    adaptive_fontsize_range: Optional[Tuple[float, float]],
) -> Tuple[Dict[Tuple[int, ...], float], float, float, bool]:
    """
    Shared logic for adaptive region font sizes.

    Returns:
        (region_fontsizes, fs_min, fs_max, adaptive_flag)
    """
    # --- Compute region areas ---
    region_areas: Dict[Tuple[int, ...], float] = {
        key: float(mask.sum()) * float(pixel_area) for key, mask in region_masks.items()
    }

    noncomp_keys = [
        k for k in region_masks.keys()
        if k != complement_key and region_areas.get(k, 0.0) > 0.0
    ]
    if noncomp_keys:
        area_min = min(region_areas[k] for k in noncomp_keys)
        area_max = max(region_areas[k] for k in noncomp_keys)
    else:
        area_min = area_max = 0.0

    # Decide whether adaptive fontsize is on (default: ON only if linear_scale=True)
    if adaptive_fontsize is None:
        adaptive_flag = bool(linear_scale)
    else:
        adaptive_flag = bool(adaptive_fontsize)

    # Determine font size range  (treated as (fs_min, fs_max))
    if adaptive_flag and area_max > 0.0:
        if adaptive_fontsize_range is not None:
            fs_min, fs_max = adaptive_fontsize_range
            if fs_min > fs_max:
                fs_min, fs_max = fs_max, fs_min
        else:
            fs_min, fs_max = _default_adaptive_fontsize(N, linear_scale)
    else:
        fs_min = fs_max = float(base_region_fontsize)

    region_fontsizes: Dict[Tuple[int, ...], float] = {}
    if adaptive_flag and area_max > 0.0 and area_max >= area_min:
        denom = (area_max - area_min) if area_max > area_min else 1.0
        for key, area in region_areas.items():
            if area_max > area_min:
                t = (area - area_min) / denom
            else:
                t = 0.5
            t = max(0.0, min(1.0, t))
            fs = fs_min + t * (fs_max - fs_min)
            region_fontsizes[key] = fs
    else:
        for key in region_masks.keys():
            region_fontsizes[key] = float(base_region_fontsize)

    return region_fontsizes, float(fs_min), float(fs_max), adaptive_flag


# ---------------------------------------------------------------------------
# Half-plane → disc mapping + arc helpers (vennfan)
# ---------------------------------------------------------------------------


def halfplane_to_disc(
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
    if y_pos_max != 0:
        rho[pos] = R * (1.0 - y[pos] / y_pos_max)
    else:
        rho[pos] = R

    # Map outside
    denom = (R_out - R)
    if y_neg_min != 0:
        rho[neg] = R + denom * (y[neg] / y_neg_min)
    else:
        rho[neg] = R_out

    theta = x
    u = rho * np.cos(theta)
    v = rho * np.sin(theta)
    return u, v


def second_radial_intersection(
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


def arc_angle_for_region(
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
    pos_vc = visual_center(mask, U, V)
    if pos_vc is None:
        return None
    return float(np.arctan2(pos_vc[1], pos_vc[0]))


def radial_segment_center_for_region(
    mask: np.ndarray,
    angle_rad: float,
    u_min: float,
    v_min: float,
    du_val: float,
    dv_val: float,
    H_val: int,
    W_val: int,
    R_max: float,
    radial_bias: float,
    n_samples: int = 1024,
) -> Optional[float]:
    """
    Sample along a radial ray at angle `angle_rad` and find a "center" radius
    for the intersection with `mask`.

    The intersection closer to radius R (assumed to be in [0, R_max]) gets
    weight `radial_bias`, the other gets weight (1 - radial_bias), giving
    a biased center between the two extremes.
    """
    d_x = float(np.cos(angle_rad))
    d_y = float(np.sin(angle_rad))

    rs = np.linspace(0.0, R_max, n_samples)
    inside = np.zeros_like(rs, dtype=bool)

    for j, r_val in enumerate(rs):
        u = r_val * d_x
        v = r_val * d_y
        ix = int(round((u - u_min) / du_val))
        iy = int(round((v - v_min) / dv_val))
        if ix < 0 or ix >= W_val or iy < 0 or iy >= H_val:
            continue
        inside[j] = mask[iy, ix]

    if not inside.any():
        return None

    idx_true = np.where(inside)[0]
    r_min = float(rs[idx_true[0]])
    r_max = float(rs[idx_true[-1]])

    # Choose which end is closer to R=1 (in vennfan we usually use R=1)
    # but the caller can pick R_max to be the disc outer radius; here we
    # just bias toward the smaller radius.
    if abs(r_min - 1.0) <= abs(r_max - 1.0):
        r_closer = r_min
        r_other = r_max
    else:
        r_closer = r_max
        r_other = r_min

    r_center = radial_bias * r_closer + (1.0 - radial_bias) * r_other
    return r_center


def region_label_mode_for_key(
    key: Tuple[int, ...],
    N: int,
    region_label_placement: str,
    pivot_index_outside: int = 5,
    pivot_index_inside: int = 3,
) -> str:
    """
    Decide label placement mode ('radial' vs 'visual_center') for a given region.

    When region_label_placement is:
        - 'radial'        → always 'radial'
        - 'visual_center' → always 'visual_center'
        - 'hybrid'        → choose depending on key & N (vennfan's heuristic)
    """
    if region_label_placement == "radial":
        return "radial"
    if region_label_placement == "visual_center":
        return "visual_center"

    # Hybrid mode:
    last_bit = key[-1] if len(key) > 0 else 0

    has_high_any_outside = False
    has_missing_high_any_inside = False

    if N > pivot_index_outside + 1:
        for idx in range(pivot_index_outside, N - 1):
            if key[idx]:
                has_high_any_outside = True
                break

    if N > pivot_index_inside + 1:
        for idx in range(pivot_index_inside, N - 1):
            if not key[idx]:
                has_missing_high_any_inside = True
                break

    cond1 = (last_bit == 0 and has_high_any_outside)
    cond4 = (last_bit == 1 and has_missing_high_any_inside)

    if cond1 or cond4:
        return "radial"
    return "visual_center"


# ---------------------------------------------------------------------------
# Small demo helper (used by test code)
# ---------------------------------------------------------------------------


def make_demo_values(N: int) -> np.ndarray:
    """
    Label each region by which sets it belongs to, e.g. "", "A", "BC", "ABCDE", etc.
    For testing, the complement (all zeros) is normally left as None.
    """
    letters = [chr(ord("A") + i) for i in range(N)]
    shape = (2,) * N
    arr = np.empty(shape, dtype=object)
    for idx in np.ndindex(shape):
        s = "".join(letters[i] for i, bit in enumerate(idx) if bit)
        arr[idx] = s
    # Complement (all zeros) left as None by default
    arr[(0,) * N] = None
    return arr
