import numpy as np
from typing import Dict, List, Tuple, Union, Sequence
from matplotlib.colors import to_rgb
import colorsys


def _rgb(color: Union[str, tuple]) -> np.ndarray:
    """Convert any Matplotlib color into an RGB float array in [0,1]."""
    return np.array(to_rgb(color), float)

def _auto_text_color_from_rgb(rgb: np.ndarray) -> str:
    """Choose black or white text based on background luminance."""
    lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    return "white" if lum < 0.5 else "black"

def _color_mix_average(colors: Sequence[np.ndarray]) -> np.ndarray:
    """Simple average of the provided RGB colors."""
    if not colors:
        return np.zeros(3, float)
    arr = np.stack([np.array(c, float) for c in colors], axis=0)
    return arr.mean(axis=0)


def _color_mix_subtractive(colors):
    """
    Mix colors by subtracive mixing.
    """
    if not colors:
        return np.zeros(3, float)
    arr = np.stack([np.array(c, float) for c in colors], axis=0)
    return np.abs(1.0 - (np.prod(1-arr, axis=0))*(len(colors)**0.25))

# def _color_mix_hue_average(colors: Sequence[np.ndarray], n: float) -> np.ndarray:
#     """
#     Mix colors by averaging only their hue (in HSV space).
    
#     - Hue: circular mean of all input hues.
#     - Saturation: 1 - k/n, where k is the number of input colors.
#     - Value: simple average of all input values.
    
#     The function accepts RGB colors as np.ndarray of shape (3,),
#     either in [0, 1] or [0, 255]. It returns an RGB color in the
#     same range as the inputs.
#     """
#     if not colors:
#         return np.zeros(3, float)

#     # Stack and cast to float
#     rgb = np.stack([np.array(c, float) for c in colors], axis=0)

#     # Detect whether we are in [0, 1] or [0, 255] and normalize if needed
#     max_val = rgb.max()
#     if max_val > 1.0:
#         rgb_norm = rgb / 255.0
#         scale_back = 255.0
#     else:
#         rgb_norm = rgb
#         scale_back = 1.0

#     # Convert each RGB to HSV
#     hsv = np.array([colorsys.rgb_to_hsv(*c) for c in rgb_norm])

#     hues = hsv[:, 0]     # in [0, 1], representing angle on the color wheel
#     values = hsv[:, 2]   # brightness

#     # Circular mean of hue
#     angles = 2.0 * np.pi * hues
#     mean_sin = np.sin(angles).mean()
#     mean_cos = np.cos(angles).mean()
#     mean_angle = np.arctan2(mean_sin, mean_cos)
#     if mean_angle < 0.0:
#         mean_angle += 2.0 * np.pi
#     hue_mean = mean_angle / (2.0 * np.pi)

#     # Saturation: 1 - k/n (clamped to [0, 1])
#     k = len(colors)
#     saturation = 0.9*(1.05 - (k / float(n)))
#     saturation = float(np.clip(saturation, 0.0, 1.0))

#     # Value: simple average of the input values
#     value = float(values.mean())

#     # Back to RGB
#     mixed_rgb = np.array(colorsys.hsv_to_rgb(hue_mean, saturation, value), dtype=float)

#     # Rescale to original range
#     mixed_rgb *= scale_back
#     return mixed_rgb


# def _color_mix_hue_average(colors: Sequence[np.ndarray], n: float) -> np.ndarray:
#     """
#     Mix colors by:
#       - Hue: arithmetic mean of hue values (treated as linear spectrum, not circular).
#       - Saturation: 1 - k/n, where k is the number of input colors.
#       - Value: arithmetic mean of values.

#     Input: sequence of RGB colors as np.ndarray of shape (3,),
#            in either [0, 1] or [0, 255].
#     Output: single RGB color in the same range as the input.
#     """
#     if not colors:
#         return np.zeros(3, float)

#     # Stack and cast to float
#     rgb = np.stack([np.array(c, float) for c in colors], axis=0)

#     # Detect whether we are in [0, 1] or [0, 255] and normalize if needed
#     max_val = rgb.max()
#     if max_val > 1.0:
#         rgb_norm = rgb / 255.0
#         scale_back = 255.0
#     else:
#         rgb_norm = rgb
#         scale_back = 1.0

#     # Convert each RGB to HSV
#     hsv = np.array([colorsys.rgb_to_hsv(*c) for c in rgb_norm])

#     hues = hsv[:, 0]    # [0, 1]
#     values = hsv[:, 2]  # brightness

#     # Hue: simple arithmetic mean on spectrum
#     hue_mean = float(hues.mean())

#     # Saturation: 1 - k/n, clamped to [0, 1]
#     k = len(colors)
#     saturation = 0.7*(1.05 - (k / float(n)))
#     saturation = float(np.clip(saturation, 0.0, 1.0))

#     # Value: average of input values
#     value = float(values.mean())

#     # Back to RGB
#     mixed_rgb = np.array(colorsys.hsv_to_rgb(hue_mean, saturation, value), dtype=float)

#     # Rescale to original range
#     mixed_rgb *= scale_back
#     return mixed_rgb

def _color_mix_hue_average(colors: Sequence[np.ndarray], n: float) -> np.ndarray:
    """
    Mix colors by:
      1. Simple RGB average.
      2. Convert that average to HSV.
      3. Set saturation to 1 - k/n, where k is the number of input colors.
      4. Convert back to RGB.

    Accepts RGB colors in [0, 1] or [0, 255] and returns in the same range.
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

    # 2) Convert the average color to HSV
    h, s, v = colorsys.rgb_to_hsv(*avg_rgb)

    # 3) Saturation: 1 - k/n, clamped to [0, 1]
    k = len(colors)
    sat = (1.0 - (k / float(n)))
    sat = float(np.clip(sat, 0.0, 1.0))
    
    l = 1.0 - (k / float(n)) + (1 / float(n))/2
    l = float(np.clip(l, 0.0, 1.0))

    # 4) Back to RGB with adjusted saturation
    mixed_rgb = np.array(colorsys.hls_to_rgb(h, l, sat), dtype=float)

    # Rescale to original range
    mixed_rgb *= scale_back
    return mixed_rgb

def _color_mix_alpha_stack(colors: Sequence[np.ndarray], alpha: float = 0.5) -> np.ndarray:
    """
    Mix colors by stacking them with a fixed per-layer alpha.
    """
    if not colors:
        return np.zeros(3, float)
    a = float(alpha)
    a = max(0.0, min(1.0, a))
    c = np.array(colors[0], float)
    for col in colors[1:]:
        col_arr = np.array(col, float)
        c = c * (1.0 - a) + col_arr * a
    return c