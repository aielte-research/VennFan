import numpy as np
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Per-N color palettes (up to 10 sets)
# ---------------------------------------------------------------------------

FILL_COLORS: Dict[int, List[str]] = {
    1:  ["#7FFF7F"],
    2:  ["#7dc9ff", "#ffba7d"],
    3:  ["#7dc9ff", "#ffba7d", "#80ff80"],
    4:  ["#78FFFF", "#BC7AFF", "#FF7D7D", "#BCFF78"],
    5:  ["#7BFFFF", "#9B82FF", "#FF82CD", "#FFCB7D", "#98FF7F"],
    6:  ["#BFFF7E", "#FFC082", "#FF7FBF", "#BD7BFF", "#82C1FF", "#7DFFBE"],
    7:  ["#7CE9FF", "#8682FF", "#F081FF", "#FF7FA1", "#FFCB7D", "#C2FF7D", "#84FFAB"],
    8:  ["#86FFED", "#80B3FF", "#AF84FF", "#FF83F3", "#FF8396", "#FFCC80", "#D2FF7F", "#81FF8E"],
    #8:  ["#00A68D", "#003E9B", "#3900A2", "#9F008F", "#9E0018", "#A36200", "#68A100", "#00A310"],
    9:  ["#84FFCE", "#86DFFF", "#848CFF", "#CF87FF", "#FF80DD", "#FF858D", "#FFCE85", "#DDFF7F", "#88FF80"],
}

OUTLINE_COLORS: Dict[int, List[str]] = {
    1: ["#0D730D"],
    2: ["#004c83", "#803c00"],
    3: ["#004c83", "#803c00", "#007c00"],
    4: ["#008383", "#3F007E", "#7B0000", "#3F7E00"],
    5: ["#008585", "#1B0085", "#84004F", "#7D4B00", "#187A00"],
    6: ["#56AB00", "#A55300", "#9D004E", "#5300A5", "#004F9E","#00A151"],
    7: ["#007926", "#006579", "#040079", "#710080", "#850023", "#7D4B00", "#407800"],
    8: ["#00A68D", "#003E9B", "#3900A2", "#9F008F", "#9E0018", "#A36200", "#68A100", "#00A310"],
    9: ["#00814E", "#005E80", "#000985", "#4D0080", "#790058", "#850009", "#814E00", "#638700", "#098300"],
}

def default_palette_for_n(N: int) -> Tuple[List[str], List[str]]:
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

def default_fontsize(N: int, linear_scale:bool=True, curve_mode="cosine") -> Tuple[float, float]:
    class_fontsizes = {
        1: 20,
        2: 20,
        3: 18,
        4: 18,
        5: 16,
        6: 16,
        7: 12,
        8: 6,
        9: 3,
    }
    if curve_mode=="cosine":
        if linear_scale:
            region_fontsizes = {
                1: 16,
                2: 14,
                3: 12,
                4: 10,
                5: 10,
                6: 9.5,
                7: 6,
                8: 4,
                9: 1,
            }
        else:
            region_fontsizes = {
                1: 24,
                2: 22,
                3: 20,
                4: 18,
                5: 12,
                6: 11,
                7: 6,
                8: 3.5,
                9: 2,
            }
    elif curve_mode=="sine":
        if linear_scale:
            region_fontsizes = {
                1: 16,
                2: 16,
                3: 16,
                4: 16,
                5: 14,
                6: 12,
                7: 6,
                8: 4,
                9: 1,
            }
        else:
            region_fontsizes = {
                1: 24,
                2: 22,
                3: 20,
                4: 18,
                5: 12,
                6: 11,
                7: 6,
                8: 3.5,
                9: 2,
            }
    return (region_fontsizes[N],class_fontsizes[N])

def _default_adaptive_fontsize(N: int, linear_scale:bool=True, curve_mode="cosine") -> Tuple[float, float]:
    fontsizes = {
        1: (16, 22),
        2: (14, 20),
        3: (12, 18),
        4: (10, 18),
        5: (8, 16),
        6: (10, 16),
        7: (6, 20),
        8: (4, 8),
        9: (1, 3),
    }
    return fontsizes[N]

def default_linewidth(N: int, linear_scale:bool=True, curve_mode="cosine") -> Tuple[float, float]:
    linewidths = {
        1: 6,
        2: 5.0,
        3: 4.5,
        4: 4.0,
        5: 3.5,
        6: 3.0,
        7: 1.5,
        8: 1.0,
        9: 0.5,
    }
    return linewidths[N]