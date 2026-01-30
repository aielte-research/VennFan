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

from typing import Sequence, Optional, Union, Tuple, Dict, Callable, List, Iterable
import os
import yaml

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.ndimage import distance_transform_edt

from .colors import _rgb, default_palette_for_n
from .curves import get_sine_curve, get_cosine_curve, vennfan_find_extrema
from .utils import (
    disjoint_region_masks,
    visual_center,
    arc_angle_for_region,
    halfplane_to_disc,
    second_radial_intersection,
    normalize_angle_90,
    class_label_angles,
    resolve_color_mixing,
    region_label_mode_for_key,
    shrink_text_font_to_region,
    text_color_for_region,
)


# ---- YAML defaults ---------------------------------------------

class VennfanDefaultSettings:
    """Thin wrapper around vennfan_defaults.yaml to keep access centralized."""

    def __init__(self, yaml_path: str):
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Missing defaults YAML: {yaml_path}")
        with open(yaml_path, "r", encoding="utf-8") as f:
            self._d = yaml.safe_load(f) or {}

    @staticmethod
    def _as_float(x, *, path: str) -> float:
        # Accept only numeric scalars; do not interpret strings like "1/4".
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            raise TypeError(
                f"Expected a numeric (int/float) at {path}, got {type(x).__name__}."
            )
        return float(x)

    def _per_n(self, table, N: int, *, path: str) -> float:
        if not isinstance(table, dict):
            raise TypeError(f"Expected a mapping at {path}, got {type(table).__name__}.")
        if N in table:
            return self._as_float(table[N], path=f"{path}[{N}]")
        if str(N) in table:
            return self._as_float(table[str(N)], path=f"{path}['{N}']")
        raise KeyError(f"Missing key {N} in {path}.")

    def _as_pair(self, x, *, path: str) -> Tuple[float, float]:
        if isinstance(x, (list, tuple)) and len(x) == 2:
            a = self._as_float(x[0], path=f"{path}[0]")
            b = self._as_float(x[1], path=f"{path}[1]")
            return (a, b)
        raise TypeError(f"Expected a 2-sequence of floats at {path}, got {type(x).__name__}.")

    def _per_n_pair(self, table, N: int, *, path: str) -> Tuple[float, float]:
        if not isinstance(table, dict):
            raise TypeError(f"Expected a mapping at {path}, got {type(table).__name__}.")
        if N in table:
            return self._as_pair(table[N], path=f"{path}[{N}]")
        if str(N) in table:
            return self._as_pair(table[str(N)], path=f"{path}['{N}']")
        raise KeyError(f"Missing key {N} in {path}.")

    def linewidth(self, N: int) -> float:
        return self._per_n(self._d["linewidths"], N, path="DEFAULTS.linewidths")

    def class_fontsize(self, N: int) -> float:
        return self._per_n(self._d["fontsizes"]["class"], N, path="DEFAULTS.fontsizes.class")

    def class_label_offset(self, N: int) -> float:
        return self._per_n(self._d["class_label_offset"], N, path="DEFAULTS.class_label_offset")

    def region_fontsize(self, curve_mode: str, decay: str, N: int) -> float:
        region_tbl = self._d["fontsizes"]["region"][curve_mode]
        if decay in region_tbl:
            scale_key = decay
        elif decay == "exponential" and "nonlinear" in region_tbl:
            scale_key = "nonlinear"
        else:
            raise KeyError(f"Missing DEFAULTS.fontsizes.region.{curve_mode}.{decay}")
        return self._per_n(
            region_tbl[scale_key],
            N,
            path=f"DEFAULTS.fontsizes.region.{curve_mode}.{scale_key}",
        )

    def region_radial_offset(self, N: int) -> Tuple[float, float]:
        """
        Return (inside_offset, outside_offset) for region radial labels.

        Expected YAML shape:
            region_radial_offset:
              1: [0.05, 0.05]
              2: [0.05, 0.05]
              ...
        """
        tbl = self._d.get("region_radial_offset")
        if tbl is None:
            raise KeyError("Missing DEFAULTS.region_radial_offset")
        return self._per_n_pair(tbl, N, path="DEFAULTS.region_radial_offset")

    def setting(self, curve_mode: str, decay: str, name: str, N: int, *, required: bool) -> Optional[float]:
        settings = (self._d.get("settings") or {})
        cm = settings.get(curve_mode)
        if not isinstance(cm, dict):
            if required:
                raise KeyError(f"Missing DEFAULTS.settings.{curve_mode}")
            return None
        dec = cm.get(decay)
        if not isinstance(dec, dict):
            if required:
                raise KeyError(f"Missing DEFAULTS.settings.{curve_mode}.{decay}")
            return None
        tbl = dec.get(name)
        if tbl is None:
            if required:
                raise KeyError(f"Missing DEFAULTS.settings.{curve_mode}.{decay}.{name}")
            return None
        return self._per_n(tbl, N, path=f"DEFAULTS.settings.{curve_mode}.{decay}.{name}")


DEFAULTS_YAML_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "vennfan_defaults.yaml",
)

DEFAULTS = VennfanDefaultSettings(DEFAULTS_YAML_PATH)


def vennfan(
    # Core data
    values,
    class_names: Sequence[str],
    title: Optional[str] = None,
    outfile: Optional[Union[str, os.PathLike, Iterable[Union[str, os.PathLike]]]] = None,
    # Boundary curves / geometry
    curve_mode: str = "cosine",  # "sine" or "cosine"
    decay: str = "linear",       # "linear" or "exponential"
    p: Optional[float] = None,
    epsilon: Optional[float] = None,
    delta: Optional[float] = None,
    b: Optional[float] = None,
    # Colors
    colors: Optional[Sequence[Union[str, tuple]]] = None,
    outline_colors: Optional[Sequence[Union[str, tuple]]] = None,
    color_mixing: Union[str, Callable] = "average",  # "average", "subtractive", "hue_average" or "alpha_stack"
    text_color: Optional[str] = None,
    # Highlight
    highlight_factor: Optional[float] = None,
    highlight_color: Union[str, tuple] = "#FFFFFF",
    highlight_gamma: float = 0.5,
    # Fontsizes
    region_fontsize: Optional[float] = None,
    class_label_fontsize: Optional[float] = None,
    complement_fontsize: Optional[float] = None,
    # Text fitting
    shrink_region_text: bool = True,
    # Layout
    dpi: Optional[int] = None,
    linewidth: Optional[float] = None,
    region_label_placement: str = "visual_text_center",  # "visual_text_center", "visual_center", "radial" or "hybrid"
    region_radial_offset: Optional[Union[float, Iterable]] = None,
    last_constant_label_offset: Tuple[float, float] = (0.0, 0.0),
    class_label_offset: Optional[float] = None,
    # Disc grid resolution control
    disc_points_per_class: int = 200,
    # Debugging for visual_text_center (yellow cores, red boxes, cyan line)
    debug: bool = False,
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

    outfile : str or iterable of str, optional
        If None, the function returns the Matplotlib Figure.
        If a single path is provided, the figure is saved there and the function
        returns None.
        If an iterable of paths is provided, the figure is saved to all of them
        and the function returns None.

    Boundary curves / geometry
    --------------------------
    curve_mode : {"cosine", "sine"}
        Base trigonometric curve family used for the classes.

    decay : {"linear", "exponential"}
        Amplitude-decay mode for the curves.

    p : float, optional
        Exponent controlling how amplitude/shape varies with harmonic index.
        If None, a per-N default is taken from the defaults YAML.

    epsilon : float or None
        Small vertical offset passed to the curve function.
        If None, a per-N default is taken from the defaults YAML (if present).

    delta : float or None
        Optional decay parameter passed to the curve function.
        If None, a per-N default is taken from the defaults YAML (if present).

    b : float, optional
        Optional decay base parameter passed to the curve function.
        If None, a per-N default is taken from the defaults YAML.

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

    Highlight
    ---------
    highlight_factor : float in [0, 1] or None
        Optional radial "halo" / highlight effect inside each region.
            - None (default): no extra highlight overlay is drawn.
            - 0.0: highlight is computed but has no visual impact.
            - 1.0: highlight goes all the way to `highlight_color`.

    highlight_color : color
        Matplotlib color spec, e.g. "white" (default) or an RGB tuple, used as the highlight target.
        This is only interpreted when `highlight_factor` is not None.

    highlight_gamma : float
        Gamma applied to the normalized interior distance profile before scaling by `highlight_factor`.
        Default is 0.5 (square root).

    Fontsizes
    ---------
    region_fontsize : float, optional
        Base font size used for all region labels (regardless of placement mode).
        If None, uses a per-N default from the defaults YAML.

    class_label_fontsize : float, optional
        Font size for the class names drawn around the outside of the fan.
        If None, uses a per-N default from the defaults YAML.

    complement_fontsize : float, optional
        Base font size for the complement label (all-zeros region). If None, defaults to
        `region_fontsize` (after `region_fontsize` has been resolved, including YAML defaults).

    Text fitting
    ------------
    shrink_region_text : bool
        If True (default), region labels are shrink-to-fit against their region masks using
        `shrink_text_font_to_region(...)`. This applies in all placement modes ("visual_center",
        "visual_text_center", "radial", and "hybrid") and is also used for the complement label.
        If False, the base font sizes are used without any shrink-to-fit.

    Layout
    ------
    dpi : int, optional
        Figure DPI. If None, defaults to `100 * N`.

    linewidth : float, optional
        Line width for the boundary curves. If None, uses the per-N default from
        the defaults YAML. This also controls the erosion radius used by
        `shrink_text_font_to_region` (via `erosion_radius_pix = linewidth`).

    region_label_placement : {"visual_text_center", "visual_center", "radial", "hybrid"}
        Controls how region labels are positioned.

        Default is "visual_text_center".

        - "visual_center":
            Classic visual-center placement.
        - "visual_text_center":
            Area-based erosion + longest-line anchor (with optional debug overlay).
        - "radial":
            Always use radial placement along a ray.
        - "hybrid":
            Per-region decision via `region_label_mode_for_key(...)` between
            "visual_center" and "radial".

    region_radial_offset : float or (float, float) or [float, float] or None
        Radial offsets used by the radial placement branch, interpreted as (inside, outside).

        - If None, a per-N default pair is loaded from the defaults YAML key `region_radial_offset`.
        - If a float, the same value is used for both inside and outside.
        - If a length-2 tuple/list, it is interpreted as (inside, outside).

        Offsets are always interpreted as **fractions of R** using the rule near the main circle:
            - inside regions:  r = R * (1 - inside_offset)
            - outside regions: r = R * (1 + outside_offset)

    last_constant_label_offset : (float, float)
        Additional (dx, dy) shift applied only to the last class label (i == N-1).

    class_label_offset : float, optional
        Radial offset (in units of R) applied to all class labels. If None, uses a per-N
        default from the defaults YAML.

    Resolution control
    ------------------
    disc_points_per_class : int
        Controls the resolution of the disc grid used for region masks. The grid has
        approximately (N * disc_points_per_class) samples along each axis (with a safety
        lower bound of 50 per class). Higher values refine region boundaries and can reduce
        over-aggressive shrink-to-fit in small regions, at the cost of runtime and memory.

    Debugging
    ---------
    debug : bool
        Enables visual debugging overlays.
        If False (default), all these debug elements are suppressed.

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        If `outfile` is None, returns the Figure. Otherwise, saves the figure
        to `outfile` (or all paths in `outfile` if iterable) and returns None.
    """
    visual_text_center_area_fraction = 0.10

    # Validate decay mode
    decay = str(decay).lower()
    if decay not in ("linear", "exponential"):
        raise ValueError("decay must be 'linear' or 'exponential'.")

    if curve_mode not in ("cosine", "sine"):
        raise ValueError(f"Unsupported curve_mode {curve_mode!r}; use 'cosine' or 'sine'.")

    if not isinstance(shrink_region_text, bool):
        raise TypeError("shrink_region_text must be a bool.")

    # Validate visual_text_center_area_fraction
    frac = float(visual_text_center_area_fraction)
    if not (0.0 < frac <= 1.0):
        raise ValueError("visual_text_center_area_fraction must be in (0, 1].")
    visual_text_center_area_fraction = frac

    # Validate disc_points_per_class
    disc_points_per_class = int(disc_points_per_class)
    if disc_points_per_class <= 0:
        raise ValueError("disc_points_per_class must be a positive integer.")

    # Validate highlight_factor
    if highlight_factor is not None:
        highlight_factor = float(highlight_factor)
        if not (0.0 <= highlight_factor <= 1.0):
            raise ValueError("highlight_factor must be in [0, 1] or None.")

    # Normalize highlight_color (only if highlight is enabled)
    highlight_rgb: Optional[np.ndarray] = None
    if highlight_factor is not None:
        try:
            highlight_rgb = np.asarray(_rgb(highlight_color), float)
        except Exception as e:
            raise TypeError(
                "highlight_color must be a valid Matplotlib color spec (e.g., 'white' or an RGB tuple)."
            ) from e
        if highlight_rgb.shape != (3,):
            raise ValueError("highlight_color must resolve to an RGB triple of shape (3,).")

    curve_fn = get_sine_curve if curve_mode == "sine" else get_cosine_curve

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

    # ---- Curve defaults from YAML -----------------------------------------
    if p is None:
        p = DEFAULTS.setting(curve_mode, decay, "p", N, required=True)
    if epsilon is None:
        epsilon = DEFAULTS.setting(curve_mode, decay, "epsilon", N, required=False)
    if delta is None:
        delta = DEFAULTS.setting(curve_mode, decay, "delta", N, required=False)
    if b is None:
        b_yaml = DEFAULTS.setting(curve_mode, decay, "b", N, required=False)
        b = 0.8 if b_yaml is None else b_yaml

    # ---- Analytic extents (used for complement anchor) ---------------------
    xmn_ext, xmx_ext, ymn_ext, ymx_ext = vennfan_find_extrema(
        curve_mode=curve_mode,
        p=p,
        decay=decay,
        N=N,
        epsilon=epsilon,
        delta=delta,
        b=b,
    )

    # ---- Class-label offset (separate from region radial offsets) ----------
    if class_label_offset is None:
        class_label_offset = DEFAULTS.class_label_offset(N)
    else:
        if isinstance(class_label_offset, bool) or not isinstance(class_label_offset, (int, float)):
            raise TypeError("class_label_offset must be a float or None.")
        class_label_offset = float(class_label_offset)

    zeros = (0,) * N

    # ---- Region label placement mode ---------------------------------------
    region_label_placement = str(region_label_placement).lower()
    if region_label_placement not in ("radial", "visual_center", "visual_text_center", "hybrid"):
        raise ValueError(
            "region_label_placement must be one of "
            "'radial', 'visual_center', 'visual_text_center', or 'hybrid'."
        )

    # Internal gate for visual_text_center algorithm
    visual_text_center_width = float(np.sin(4.0 * np.pi / (2.0 ** N)))

    # ---- Resolve region_radial_offset into (inside, outside) ----------------
    if region_radial_offset is None:
        off_in, off_out = DEFAULTS.region_radial_offset(N)
    else:
        if isinstance(region_radial_offset, bool):
            raise TypeError("region_radial_offset must be a float, a length-2 tuple/list of floats, or None.")
        if isinstance(region_radial_offset, (int, float)):
            f = float(region_radial_offset)
            off_in, off_out = f, f
        elif isinstance(region_radial_offset, (list, tuple)) and len(region_radial_offset) == 2:
            a, b2 = region_radial_offset[0], region_radial_offset[1]
            if isinstance(a, bool) or not isinstance(a, (int, float)):
                raise TypeError("region_radial_offset[0] must be a float.")
            if isinstance(b2, bool) or not isinstance(b2, (int, float)):
                raise TypeError("region_radial_offset[1] must be a float.")
            off_in, off_out = float(a), float(b2)
        else:
            raise TypeError(
                "region_radial_offset must be a float, a length-2 tuple/list of floats, or None."
            )

    # ---- Linewidth & colors ------------------------------------------------
    if linewidth is None:
        linewidth = DEFAULTS.linewidth(N)

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
    base_fs_region = DEFAULTS.region_fontsize(curve_mode, decay, N)
    base_fs_class = DEFAULTS.class_fontsize(N)

    if region_fontsize is None:
        region_fontsize = base_fs_region
    if class_label_fontsize is None:
        class_label_fontsize = base_fs_class
    if complement_fontsize is None:
        complement_fontsize = region_fontsize

    fs_region = float(region_fontsize)
    fs_comp = float(complement_fontsize)

    # ---- Color mixing callback ---------------------------------------------
    mixing_cb = resolve_color_mixing(color_mixing, N)

    # ---- Base domain & disc grid -------------------------------------------
    x_min, x_max = 0.0, 2.0 * np.pi
    R = 1.0
    R_out = 2.0 * R

    grid_n = max(50, disc_points_per_class)
    us = np.linspace(-R_out, R_out, N * grid_n)
    vs = np.linspace(-R_out, R_out, N * grid_n)
    U, V = np.meshgrid(us, vs)

    rho = np.sqrt(U * U + V * V)
    theta = np.mod(np.arctan2(V, U), 2.0 * np.pi)

    # Map disc grid back to half-plane coordinates (x_old, y_old)
    y_min = -1.0
    y_max = 1.0

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
        membership.append(y_old >= curve)

    # ---- Disjoint region masks ---------------------------------------------
    region_masks = disjoint_region_masks(membership)
    H, W = U.shape

    # ---- Region RGBA image -------------------------------------------------
    rgba = np.zeros((H, W, 4), float)
    region_rgbs: Dict[Tuple[int, ...], np.ndarray] = {}

    for key, mask in region_masks.items():
        if not any(key):
            continue
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

        if highlight_factor is not None:
            dist = distance_transform_edt(mask)
            dist_vals = dist[mask]
            if dist_vals.size > 0:
                dmax = float(dist_vals.max())
                if dmax > 0.0:
                    if dmax <= 1.0:
                        t = np.ones_like(dist_vals, float)
                    else:
                        t = (dist_vals - 1.0) / (dmax - 1.0)
                        t = np.clip(t, 0.0, 1.0)

                    prof = t ** float(highlight_gamma)
                    whiteness = float(highlight_factor) * prof
                    hc = highlight_rgb  # type: ignore[assignment]

                    new_rgb = (
                        (1.0 - whiteness)[:, None] * mixed_rgb[None, :]
                        + whiteness[:, None] * hc[None, :]
                    )
                    rgba[mask, 0] = new_rgb[:, 0]
                    rgba[mask, 1] = new_rgb[:, 1]
                    rgba[mask, 2] = new_rgb[:, 2]
                    rgba[mask, 3] = 1.0

    # Optional eroded-overlay image for visual_text_center testing
    eroded_rgba = None
    if region_label_placement == "visual_text_center" and debug:
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

    ax.set_xlim(float(xmn_ext), float(xmx_ext))
    ax.set_ylim(float(ymn_ext), float(ymx_ext))
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.margins(0.0, 0.0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # ---- Class boundaries in vennfan plane ---------------------------------
    x_plot = np.linspace(x_min, x_max, 1000 * N)
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
        u_curve, v_curve = halfplane_to_disc(x_plot, y_plot, R, y_min, y_max)
        disc_u.append(u_curve)
        disc_v.append(v_curve)

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
    circle_band = np.abs(rho - R) <= (0.03 * R)

    fig.canvas.draw()
    erosion_radius_pix = linewidth * 1.0

    # ---- Region labels (visual_center / visual_text_center / radial) -------
    for key, mask in region_masks.items():
        if (not mask.any()) or (key == zeros):
            continue

        value = arr[key]
        if value is None:
            continue

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
            u_lab, v_lab = pos
            rot = 0.0

            ha = "center"
            va = "center"

            if shrink_region_text:
                fs_use = shrink_text_font_to_region(
                    fig, ax, f"{value}", u_lab, v_lab, fs_region,
                    mask, U, V,
                    rotation=rot, ha=ha, va=va,
                    erosion_radius_pix=erosion_radius_pix,
                    debug_mode=debug,
                )
            else:
                fs_use = fs_region

            ax.text(
                u_lab, v_lab, f"{value}",
                ha=ha, va=va, fontsize=fs_use,
                color=this_color, zorder=5,
                rotation=rot, rotation_mode="anchor",
            )

        elif placement_mode == "visual_text_center":
            # --- Area-based erosion + longest-line anchor, with debug overlay ---
            this_color = text_color_for_region(key, region_rgbs, text_color)

            pos = visual_center(mask, U, V)
            if pos is None:
                continue
            u_opt, v_opt = pos
            rot_opt = 0.0
            debug_line_segment = None

            width = float(visual_text_center_width)
            if width > 0.0:
                dist = distance_transform_edt(mask)
                orig_area = int(mask.sum())
                if orig_area > 0:
                    # Choose threshold so that eroded area ≈ fraction * original_area
                    target_pixels = max(1, int(round(orig_area * visual_text_center_area_fraction)))
                    dvals = dist[mask].ravel()
                    if dvals.size > 0:
                        dsorted = np.sort(dvals)
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
                                iy0, ix0 = np.unravel_index(np.argmax(dist_core), dist_core.shape)
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

                                step = float(min(abs(dxg), abs(dyg)))
                                if step <= 0.0:
                                    step = 1.0

                                span_x = float(U.max() - U.min()) if U.size else 1.0
                                span_y = float(V.max() - V.min()) if V.size else 1.0
                                max_span = float(np.hypot(span_x, span_y))
                                max_steps = max(1, int(max_span / step) + 3)

                                angles = np.linspace(0.0, np.pi, 72, endpoint=False)

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
                                    rot_opt = normalize_angle_90(float(np.degrees(best_angle)))
                                    debug_line_segment = (
                                        (best_minus_x, best_minus_y),
                                        (best_plus_x, best_plus_y),
                                    )

            ha = "center"
            va = "center"

            # FIX: shrink + draw must use the SAME rotation
            rot_draw = rot_opt if len(str(value)) > 1 else 0.0

            if shrink_region_text:
                fs_use = shrink_text_font_to_region(
                    fig, ax, f"{value}", u_opt, v_opt, fs_region,
                    mask, U, V,
                    rotation=rot_draw, ha=ha, va=va,
                    erosion_radius_pix=erosion_radius_pix,
                    debug_mode=debug,
                )
            else:
                fs_use = fs_region

            ax.text(
                u_opt, v_opt, f"{value}",
                ha=ha, va=va, fontsize=fs_use,
                color=this_color, zorder=5,
                rotation=rot_draw,
                rotation_mode="anchor",
            )

            if debug and debug_line_segment is not None:
                (xm, ym), (xp, yp) = debug_line_segment
                ax.plot([xm, xp], [ym, yp], color="cyan", linewidth=0.8, zorder=4.5)

        else:
            # --- Radial placement branch ---
            last_bit = key[-1]

            angle_raw = arc_angle_for_region(mask, circle_band, theta, U, V)
            if angle_raw is None:
                continue

            v_out = np.array([np.cos(angle_raw), np.sin(angle_raw)], float)
            r_lab = R * (1.0 - float(off_in)) if last_bit == 1 else R * (1.0 + float(off_out))

            u_lab = r_lab * v_out[0]
            v_lab = r_lab * v_out[1]

            deg_raw = np.degrees(angle_raw)
            rot = normalize_angle_90(deg_raw)
            rot_rad = np.deg2rad(rot)
            v_base = np.array([np.cos(rot_rad), np.sin(rot_rad)], float)

            v_circle = v_out if last_bit == 1 else -v_out
            d_align = float(np.dot(v_circle, v_base))

            ha = "right" if d_align >= 0 else "left"
            va = "center"

            this_color = text_color_for_region(key, region_rgbs, text_color)

            if shrink_region_text:
                fs_use = shrink_text_font_to_region(
                    fig, ax, f"{value}", u_lab, v_lab, fs_region,
                    mask, U, V,
                    rotation=rot, ha=ha, va=va,
                    erosion_radius_pix=erosion_radius_pix,
                    debug_mode=debug,
                )
            else:
                fs_use = fs_region

            ax.text(
                u_lab, v_lab, f"{value}",
                ha=ha, va=va, fontsize=fs_use,
                color=this_color, zorder=5,
                rotation=rot, rotation_mode="anchor",
            )

    # ---- Complement (all zeros) --------------------------------------------
    comp_mask = region_masks.get(zeros)
    if comp_mask is not None and comp_mask.any():
        val_comp = arr[zeros]
        if val_comp is not None:
            this_color = text_color if text_color is not None else "black"

            u_lab = float(xmx_ext)
            v_lab = float(ymn_ext)
            rot = 0.0
            ha = "right"
            va = "bottom"

            if shrink_region_text:
                fs_use = shrink_text_font_to_region(
                    fig, ax, f"{val_comp}", u_lab, v_lab, fs_comp,
                    comp_mask, U, V,
                    rotation=rot, ha=ha, va=va,
                    erosion_radius_pix=erosion_radius_pix,
                    debug_mode=debug,
                )
            else:
                fs_use = fs_comp

            ax.text(
                u_lab, v_lab, f"{val_comp}",
                ha=ha, va=va, fontsize=fs_use,
                color=this_color, zorder=5,
                rotation=rot, rotation_mode="anchor",
            )

    # ---- Overlay eroded regions (for visual_text_center) -------------------
    if debug and eroded_rgba is not None and np.any(eroded_rgba[..., 3] > 0):
        ax.imshow(
            eroded_rgba,
            origin="lower",
            extent=[-R_out, R_out, -R_out, R_out],
            interpolation="nearest",
            zorder=2,
        )

    # ---- Class labels on vennfan -------------------------------------------
    dx_const, dy_const = last_constant_label_offset
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

        r_lab = r_anchor + float(class_label_offset) * R

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
        if isinstance(outfile, (str, os.PathLike)):
            fig.savefig(outfile, dpi=dpi, bbox_inches="tight")
        else:
            for out in outfile:
                fig.savefig(out, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return None

    return fig