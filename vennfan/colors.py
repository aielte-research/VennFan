#!/usr/bin/env python3
import numpy as np
from typing import Dict, List, Tuple, Union, Sequence, Optional
from matplotlib.colors import to_rgb
import colorsys


def _rgb(color: Union[str, tuple]) -> np.ndarray:
    """Convert any Matplotlib color into an RGB float array in [0,1]."""
    return np.array(to_rgb(color), float)


def auto_text_color_from_rgb(rgb: np.ndarray) -> str:
    """Choose black or white text based on background luminance."""
    lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    return "white" if lum < 0.5 else "black"


def color_mix_average(
    colors: Sequence[np.ndarray],
    present: Optional[Sequence[bool]] = None,
) -> np.ndarray:
    """
    Simple average of the provided RGB colors.

    Parameters
    ----------
    colors :
        Sequence of RGB arrays.
    present :
        True/false list for the whole region (same length as number of sets).
        Currently ignored; reserved for future use.
    """
    if not colors:
        return np.zeros(3, float)
    arr = np.stack([np.array(c, float) for c in colors], axis=0)
    return arr.mean(axis=0)


def color_mix_subtractive(
    colors: Sequence[np.ndarray],
    present: Optional[Sequence[bool]] = None,
) -> np.ndarray:
    """
    Mix colors by subtractive mixing.

    Parameters
    ----------
    colors :
        Sequence of RGB arrays.
    present :
        True/false list for the whole region; currently ignored.
    """
    if not colors:
        return np.zeros(3, float)
    arr = np.stack([np.array(c, float) for c in colors], axis=0)
    return np.abs(1.0 - (np.prod(1 - arr, axis=0)) * (len(colors) ** 0.25))


def color_mix_hue_average(
    colors: Sequence[np.ndarray],
    n: float,
    present: Optional[Sequence[bool]] = None,
) -> np.ndarray:
    """
    Mix colors by:
      1. Simple RGB average.
      2. Convert that average to HSV/HLS space.
      3. Set saturation/ lightness based on number of input colors and N.
      4. Convert back to RGB.

    Parameters
    ----------
    colors :
        Sequence of RGB colors (in [0,1] or [0,255]).
    n :
        Maximum number of colors / sets (e.g. N in an N-set Venn diagram).
    present :
        True/false list for the whole region; currently ignored.

    Returns
    -------
    mixed_rgb : np.ndarray
        Mixed RGB color, in the same scale as the input.
    """
    if not colors:
        return np.zeros(3, float)

    # Stack and cast to float
    rgb = np.stack([np.array(c, float) for c in colors], axis=0)

    # Detect input scale ([0,1] or [0,255]) and normalize if needed
    max_val = rgb.max()
    if max_val > 1.0:
        rgb_norm = rgb / 255.0
        scale_back = 255.0
    else:
        rgb_norm = rgb
        scale_back = 1.0

    # 1) Simple RGB average (in normalized space)
    avg_rgb = rgb_norm.mean(axis=0)

    # 2) Convert the average color to HSV/HLS
    h, s, v = colorsys.rgb_to_hsv(*avg_rgb)

    # 3) Saturation & lightness: depend on k / n
    k = len(colors)
    sat = (1.0 - (k / float(n)))
    sat = float(np.clip(sat, 0.0, 1.0))

    l = 1.0 - (k / float(n)) + (1 / float(n)) / 2
    l = float(np.clip(l, 0.0, 1.0))

    # 4) Back to RGB with adjusted saturation/lightness
    mixed_rgb = np.array(colorsys.hls_to_rgb(h, l, sat), dtype=float)

    # Rescale to original range
    mixed_rgb *= scale_back
    return mixed_rgb


def color_mix_alpha_stack(
    colors: Sequence[np.ndarray],
    present: Optional[Sequence[bool]] = None,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Mix colors by stacking them with a fixed per-layer alpha.

    Parameters
    ----------
    colors :
        Sequence of RGB arrays.
    present :
        True/false list for the whole region.
    alpha :
        Per-layer alpha in [0,1].
    reverse :
        If True, compositing order is reversed.
    """
    if not colors:
        return np.zeros(3, float)
    if present[-1]:
        colors = list(reversed(colors))
    a = float(alpha)
    a = max(0.0, min(1.0, a))
    c = np.array(colors[0], float)
    for col in colors[1:]:
        col_arr = np.array(col, float)
        c = c * (1.0 - a) + col_arr * a
    return c
