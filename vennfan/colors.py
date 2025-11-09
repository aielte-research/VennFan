import numpy as np
from typing import Dict, List, Tuple, Union, Sequence
from matplotlib.colors import to_rgb

# ---------------------------------------------------------------------------
# Per-N color palettes (up to 10 sets)
# ---------------------------------------------------------------------------

FILL_COLORS: Dict[int, List[str]] = {
    1:  ["#7FFF7F"],
    2:  ["#75FFFF", "#FF7777"],
    3:  ["#6AFFFF", "#FFFF6A", "#FF7272"],
    4:  ["#7878FF", "#72FF72", "#FFFF70", "#FA7979"],
    5:  ["#7878FA", "#76FBFB", "#76FB76", "#FEFE75", "#F86E6E"],
    6:  ["#8872FF", "#78D0FC", "#77FDBA", "#A0FA73", "#FEE774", "#F97575"],
    7:  ["#743BFF", "#3B8FFF", "#3BFFE3", "#3BFF58", "#AAFF3B", "#FFC73B", "#FF3B3B"],
    8:  ["#8E48FF", "#507BFF", "#3EE6FF", "#3BFF9D", "#54FF3B", "#CFFF45", "#FFB53E", "#FF4B4B"],
    9:  ["#913BFF", "#3B51FF", "#3BBCFF", "#3BFFD3", "#3EFF67", "#7DFF3B", "#E8FF3B", "#FFA83B", "#FF3B3B"],
    10: ["#A042FF", "#4242FF", "#42A0FF", "#42FFFF", "#42FFA0", "#42FF42", "#A0FF42", "#FFFF42", "#FFA042", "#FF4242"],
}

OUTLINE_COLORS: Dict[int, List[str]] = {
    1: ["#0D730D"],
    2: ["#0E7373", "#730D0D"],
    3: ["#0E7373", "#73730D", "#730D0D"],
    4: ["#0D0D73", "#0D730D", "#73730D", "#730D0D"],
    5: ["#0D0D73", "#0D7373", "#0D730D", "#73730D", "#730D0D"],
    6: ["#110067", "#004669", "#006332", "#256E01", "#715E00", "#6D0000"],
    7: ["#2D1173", "#113B73", "#127364", "#11731F", "#497311", "#735711", "#731111"],
    8: ["#361173", "#112973", "#116773", "#117342", "#1B730F", "#5A7311", "#734E11", "#731111"],
    9: ["#3D1173", "#111C73", "#115373", "#12735E", "#117327", "#327311", "#687311", "#734711", "#731111"],
    10: ["#421173","#111173", "#114273", "#117373", "#117342", "#117311", "#427311", "#737311", "#734211", "#731111"],
}

def _default_palette_for_n(N: int) -> Tuple[List[str], List[str]]:
    """
    Return (fill_colors, outline_colors) for a given N, using explicit
    per-N lists. If N not in dict, clamp to nearest defined N.
    """

    keys = sorted(FILL_COLORS.keys())
    N_clamped = max(keys[0], min(keys[-1], N))

    fills = FILL_COLORS.get(N_clamped)
    outlines = OUTLINE_COLORS.get(N_clamped)

    if fills is None or outlines is None:
        raise RuntimeError(f"No palette defined for N={N_clamped}.")

    if len(fills) != N_clamped or len(outlines) != N_clamped:
        raise RuntimeError(f"Palette length mismatch for N={N_clamped}.")

    # If N != N_clamped, map to N colors by index interpolation
    if N != N_clamped:
        idxs = np.linspace(0, N_clamped - 1, N).round().astype(int)
        fills = [fills[i] for i in idxs]
        outlines = [outlines[i] for i in idxs]
    else:
        fills = list(fills)
        outlines = list(outlines)

    return fills, outlines

def _rgb(color: Union[str, tuple]) -> np.ndarray:
    """Convert any Matplotlib color into an RGB float array in [0,1]."""
    return np.array(to_rgb(color), float)


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


def _auto_text_color_from_rgb(rgb: np.ndarray) -> str:
    """Choose black or white text based on background luminance."""
    lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    return "white" if lum < 0.5 else "black"